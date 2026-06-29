"""Passing structure analytics: zone geometry, zone exit timing, and network depth.

This module extends the event-level passing metrics in passing.py with three
new analytical dimensions:

Zone geometry    : how pass direction and length change as the ball moves through
                   each pitch third (defensive / midfield / attacking).
Zone progression : which zone-to-zone transitions dominate each team's style.
Zone exit timing : how many passes and seconds it takes to escape the defensive third.
Pitch network    : player nodes placed at their average pass location on the pitch,
                   edges weighted by completed-pass count.
Betweenness      : per-player betweenness centrality — who sits on the most passing
                   paths and therefore acts as a structural hub.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.passing.passing import (
    _complete_pass_mask,
    enrich_pass_vectors,
    filter_regular_play_passes,
)

_DEF_X = 40.0
_ATT_X = 80.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pitch_zone(x) -> str:
    x = float(x) if pd.notna(x) else np.nan
    if pd.isna(x):
        return "unknown"
    if x < _DEF_X:
        return "defensive"
    if x < _ATT_X:
        return "midfield"
    return "attacking"


def _team_events_sorted(group: pd.DataFrame, team: str) -> pd.DataFrame:
    return (
        group[group["team"] == team]
        .sort_values(["minute", "second"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Zone geometry
# ---------------------------------------------------------------------------

def build_zone_pass_geometry(events_df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """
    For each team and each pitch zone, compute directional and length metrics.

    Answers: are passes forward or lateral? Does direction change by zone?

    Columns
    -------
    team, zone, passes, forward_pct, lateral_pct, backward_pct,
    avg_length, avg_dx, completion_pct
    """
    passes = filter_regular_play_passes(events_df)
    passes = passes[passes["team"].isin(teams)]
    passes["zone"] = passes["location_x"].map(_pitch_zone)

    rows = []
    zone_order = ["defensive", "midfield", "attacking"]
    for team in teams:
        tp = passes[passes["team"] == team]
        for zone in zone_order:
            zp = tp[tp["zone"] == zone]
            if zp.empty:
                continue
            dir_share = zp["direction"].value_counts(normalize=True) * 100
            rows.append(
                {
                    "team": team,
                    "zone": zone,
                    "passes": len(zp),
                    "forward_pct": dir_share.get("forward", 0.0),
                    "lateral_pct": dir_share.get("lateral", 0.0),
                    "backward_pct": dir_share.get("backward", 0.0),
                    "avg_length": float(zp["pass_length"].mean()),
                    "avg_dx": float(zp["dx"].mean()),
                    "completion_pct": float(_complete_pass_mask(zp).mean() * 100),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Zone-to-zone transitions
# ---------------------------------------------------------------------------

def build_zone_transitions(events_df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """
    Count completed Regular Play pass transitions between pitch zones.

    A transition from 'defensive' → 'midfield' means the passer was in the
    defensive third and the ball arrived in the midfield third.

    Use this to see whether each team progresses step-by-step or skips zones.
    """
    passes = filter_regular_play_passes(events_df)
    passes = passes[passes["team"].isin(teams)]
    completed = _complete_pass_mask(passes)
    passes = passes[completed].copy()

    passes["from_zone"] = passes["location_x"].map(_pitch_zone)
    passes["to_zone"] = passes["pass_end_x"].map(_pitch_zone)

    rows = []
    for team in teams:
        tp = passes[passes["team"] == team]
        transitions = (
            tp.groupby(["from_zone", "to_zone"])
            .size()
            .reset_index(name="count")
        )
        total = transitions["count"].sum()
        transitions["pct"] = transitions["count"] / total * 100
        transitions["team"] = team
        rows.append(transitions)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Zone exit timing
# ---------------------------------------------------------------------------

def build_zone_exit_timing(
    events_df: pd.DataFrame,
    teams: list[str],
) -> pd.DataFrame:
    """
    For each Regular Play possession that starts inside the defensive third
    (x < 40), measure how many passes and how many seconds elapse before the
    ball exits to midfield for the first time.

    Possessions that start in midfield or higher are excluded.
    Possessions that never exit the defensive third are flagged with
    `exited = False` and their full pass count / duration are recorded.

    Columns
    -------
    possession_team, possession_id, exited, passes_before_exit,
    seconds_before_exit, total_passes, start_x
    """
    _GROUP = ["match_id", "period", "possession", "possession_team"]
    rows = []

    for keys, group in events_df.groupby(_GROUP, dropna=False, sort=False):
        poss_team = keys[3]
        if poss_team not in teams:
            continue

        te = _team_events_sorted(group, poss_team)
        if te.empty:
            continue

        if "play_pattern" in te.columns and te["play_pattern"].iloc[0] != "Regular Play":
            continue

        if "location_x" not in te.columns:
            continue

        start_x = te["location_x"].dropna().iloc[0] if not te["location_x"].dropna().empty else np.nan
        if pd.isna(start_x) or float(start_x) >= _DEF_X:
            continue

        t0 = float(te["minute"].iloc[0]) * 60 + float(te["second"].iloc[0])
        total_passes = int((te["type"] == "Pass").sum())

        exited = False
        passes_before = 0
        secs_before = 0.0
        pass_count = 0

        for _, row in te.iterrows():
            etype = row.get("type")
            if etype == "Pass":
                pass_count += 1
                end_x = row.get("pass_end_x")
                if pd.notna(end_x) and float(end_x) >= _DEF_X:
                    passes_before = pass_count - 1
                    secs_before = max(
                        float(row["minute"]) * 60 + float(row["second"]) - t0, 0.0
                    )
                    exited = True
                    break
            elif etype == "Carry":
                end_x = row.get("carry_end_x")
                if pd.notna(end_x) and float(end_x) >= _DEF_X:
                    secs_before = max(
                        float(row["minute"]) * 60 + float(row["second"]) - t0, 0.0
                    )
                    exited = True
                    break

        if not exited:
            passes_before = total_passes
            t_end = float(te["minute"].iloc[-1]) * 60 + float(te["second"].iloc[-1])
            secs_before = max(t_end - t0, 0.0)

        rows.append(
            {
                "possession_team": poss_team,
                "possession_id": keys[2],
                "start_x": float(start_x),
                "exited": exited,
                "passes_before_exit": passes_before,
                "seconds_before_exit": secs_before,
                "total_passes": total_passes,
            }
        )

    return pd.DataFrame(rows)


def summarize_zone_exit(zone_exit: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-team summary of zone-exit timing."""
    rows = []
    for team, grp in zone_exit.groupby("possession_team"):
        exited = grp[grp["exited"]]
        rows.append(
            {
                "team": team,
                "def_third_possessions": len(grp),
                "exit_rate_pct": grp["exited"].mean() * 100,
                "median_passes_before_exit": exited["passes_before_exit"].median(),
                "median_seconds_before_exit": exited["seconds_before_exit"].median(),
                "mean_passes_before_exit": exited["passes_before_exit"].mean(),
                "mean_seconds_before_exit": exited["seconds_before_exit"].mean(),
                "direct_exits_pct": (exited["passes_before_exit"] == 0).mean() * 100,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pitch-positioned pass network
# ---------------------------------------------------------------------------

def build_pitch_network(events_df: pd.DataFrame, team: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a pitch-positioned pass network for one team.

    Nodes are placed at each player's average pass location (where they
    typically have the ball when passing). Edge weight = completed pass count
    between the passer–recipient pair.

    Returns
    -------
    nodes : player, avg_x, avg_y, involvements (passes made + received)
    edges : player, pass_recipient, weight
    """
    passes = filter_regular_play_passes(events_df, team)
    completed = _complete_pass_mask(passes)
    cp = passes[
        completed & passes["player"].notna() & passes["pass_recipient"].notna()
    ].copy()

    if cp.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Average pass-origin position per player
    passer_pos = (
        cp.groupby("player")[["location_x", "location_y"]]
        .mean()
        .rename(columns={"location_x": "avg_x", "location_y": "avg_y"})
    )

    # Average receipt position per player (where they stand when receiving)
    recip_pos = (
        cp.groupby("pass_recipient")[["pass_end_x", "pass_end_y"]]
        .mean()
        .rename(columns={"pass_end_x": "avg_x", "pass_end_y": "avg_y"})
    )
    recip_pos.index.name = "player"

    # Merge: prefer passer position over recipient-only position
    all_pos = pd.concat([recip_pos, passer_pos])
    avg_pos = all_pos.groupby(level=0).mean()

    involvements = (
        pd.concat([cp["player"], cp["pass_recipient"]])
        .value_counts()
        .rename("involvements")
    )

    nodes = avg_pos.join(involvements, how="left").fillna(0).reset_index()
    nodes.columns = ["player", "avg_x", "avg_y", "involvements"]
    nodes["involvements"] = nodes["involvements"].astype(int)

    edges = (
        cp.groupby(["player", "pass_recipient"])
        .size()
        .reset_index(name="weight")
    )

    return nodes, edges


# ---------------------------------------------------------------------------
# Betweenness centrality
# ---------------------------------------------------------------------------

def compute_betweenness_centrality(
    events_df: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """
    Compute per-player betweenness centrality in the completed Regular Play
    pass network.

    Betweenness measures how often a player appears on the shortest passing
    path between two other players — the true structural hub metric, as
    opposed to just counting involvements.

    Returns
    -------
    DataFrame with columns: player, betweenness, in_degree, out_degree,
    total_degree, involvement_pct
    """
    import networkx as nx

    passes = filter_regular_play_passes(events_df, team)
    completed = _complete_pass_mask(passes)
    cp = passes[
        completed & passes["player"].notna() & passes["pass_recipient"].notna()
    ]

    if cp.empty:
        return pd.DataFrame(
            columns=["player", "betweenness", "in_degree", "out_degree",
                     "total_degree", "involvement_pct"]
        )

    G = nx.DiGraph()
    for _, row in cp.iterrows():
        p, r = row["player"], row["pass_recipient"]
        if G.has_edge(p, r):
            G[p][r]["weight"] += 1
        else:
            G.add_edge(p, r, weight=1)

    betweenness = nx.betweenness_centrality(G, normalized=True, weight="weight")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    total_involvement = sum(in_deg.values()) + sum(out_deg.values())

    rows = []
    for player, b in betweenness.items():
        ind = in_deg.get(player, 0)
        outd = out_deg.get(player, 0)
        rows.append(
            {
                "player": player,
                "betweenness": b,
                "in_degree": ind,
                "out_degree": outd,
                "total_degree": ind + outd,
                "involvement_pct": (ind + outd) / total_involvement * 100
                if total_involvement
                else np.nan,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("betweenness", ascending=False)
        .reset_index(drop=True)
    )
