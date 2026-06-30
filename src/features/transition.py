"""Transition metrics for StatsBomb event data.

A *transition* is an open-play change of possession — one team winning the ball
back from the other during live play (not from a set-piece restart). StatsBomb
includes the ball-winning action (Ball Recovery / Duel / Interception) as the
first event of the new possession, so the start of an open-play possession that
follows the opponent's possession marks the moment the ball was won.

This module answers four questions:
  1. What do teams do immediately after winning possession?   -> response type
  2. How aggressively do they transition into attack?         -> speed / territory
  3. How do they react after losing possession?               -> defensive transition
  4. Which transitions become dangerous?                      -> outcomes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Possessions that begin in live play. Everything else (throw-in, free kick,
# goal kick, corner, keeper, kick off) is a set-piece restart, not a transition.
OPEN_PLAY_PATTERNS = {"Regular Play", "From Counter"}

# Direction thresholds (StatsBomb x, metres-ish on a 120-long pitch).
FORWARD_DX = 5.0      # >= this is a forward action
BACKWARD_DX = -5.0    # <= this is a backward action
LONG_PASS_M = 30.0

# Time windows (seconds).
ATTACK_WINDOW_SEC = 10.0       # territory gained immediately after winning
DEF_REACTION_SEC = 5.0         # window to react after losing the ball

# Pitch thirds (StatsBomb x).
DEF_THIRD = 40.0
ATT_THIRD = 80.0

ON_BALL = ("Pass", "Carry")
DEFENSIVE_ACTIONS = (
    "Pressure", "Duel", "Ball Recovery", "Interception",
    "Block", "Clearance", "Foul Committed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordered(events_df: pd.DataFrame) -> pd.DataFrame:
    """Return events sorted into true chronological order within the match."""
    e = events_df.copy()
    sort_col = "index" if "index" in e.columns else None
    if sort_col is None:
        e["_t"] = e["minute"] * 60 + e["second"]
        sort_col = "_t"
    return e.sort_values(["period", sort_col]).reset_index(drop=True)


def zone_of_x(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x < DEF_THIRD:
        return "defensive"
    if x < ATT_THIRD:
        return "middle"
    return "attacking"


def _parse_bool(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _possession_meta(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per possession chain, in chronological order, with the
    previous possession's team attached (for detecting open-play turnovers)."""
    e = _ordered(events_df)
    e["_t"] = e["minute"] * 60 + e["second"]

    rows = []
    for (period, poss), g in e.groupby(["period", "possession"], sort=False):
        team = g["possession_team"].dropna().iloc[0] if g["possession_team"].notna().any() else np.nan
        rows.append(
            {
                "match_id": g["match_id"].iloc[0],
                "period": period,
                "possession_id": poss,
                "possession_team": team,
                "play_pattern": g["play_pattern"].dropna().iloc[0] if g["play_pattern"].notna().any() else np.nan,
                "first_order": g.index[0],
                "start_t": g["_t"].iloc[0],
                "end_t": g["_t"].iloc[-1],
            }
        )
    meta = pd.DataFrame(rows).sort_values(["period", "first_order"]).reset_index(drop=True)
    meta["prev_team"] = meta.groupby("period")["possession_team"].shift(1)
    meta["is_open_play_win"] = (
        meta["play_pattern"].isin(OPEN_PLAY_PATTERNS)
        & meta["prev_team"].notna()
        & meta["possession_team"].notna()
        & (meta["prev_team"] != meta["possession_team"])
    )
    return meta


def _team_on_ball(group: pd.DataFrame) -> pd.DataFrame:
    """Possession-team Pass/Carry events with start/end coords and time."""
    g = group[group["team"].eq(group["possession_team"])].copy()
    g = g[g["type"].isin(ON_BALL)]
    if g.empty:
        return g.assign(start_x=[], start_y=[], end_x=[], end_y=[], dx=[], t=[])
    g["start_x"] = pd.to_numeric(g["location_x"], errors="coerce")
    g["start_y"] = pd.to_numeric(g["location_y"], errors="coerce")
    end_x = np.where(
        g["type"].eq("Pass"),
        pd.to_numeric(g.get("pass_end_x"), errors="coerce"),
        pd.to_numeric(g.get("carry_end_x"), errors="coerce"),
    )
    end_y = np.where(
        g["type"].eq("Pass"),
        pd.to_numeric(g.get("pass_end_y"), errors="coerce"),
        pd.to_numeric(g.get("carry_end_y"), errors="coerce"),
    )
    g["end_x"] = end_x
    g["end_y"] = end_y
    g["dx"] = g["end_x"] - g["start_x"]
    g["t"] = g["minute"] * 60 + g["second"]
    return g.dropna(subset=["start_x", "end_x"])


# ---------------------------------------------------------------------------
# 1 + 2. Offensive transitions: response and aggression
# ---------------------------------------------------------------------------

def build_transition_table(
    events_df: pd.DataFrame,
    possession_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per open-play ball win, describing the immediate response and
    how aggressively the team transitioned into attack."""
    e = _ordered(events_df)
    meta = _possession_meta(events_df)
    wins = meta[meta["is_open_play_win"]]

    groups = {poss: g for poss, g in e.groupby("possession", sort=False)}
    rows = []
    for _, m in wins.iterrows():
        poss = m["possession_id"]
        team = m["possession_team"]
        g = groups.get(poss)
        if g is None:
            continue
        on_ball = _team_on_ball(g)
        if on_ball.empty:
            continue

        first = on_ball.iloc[0]
        start_x = first["start_x"]
        start_t = on_ball["t"].iloc[0]
        dx = first["dx"]
        is_long = bool(first["type"] == "Pass" and pd.notna(first.get("pass_length")) and float(first.get("pass_length") or 0) >= LONG_PASS_M)

        if dx >= FORWARD_DX:
            response = "attack"
        elif dx <= BACKWARD_DX:
            response = "recycle_back"
        else:
            response = "secure"

        # Territory gained in the first ATTACK_WINDOW_SEC seconds.
        window = on_ball[on_ball["t"] - start_t <= ATTACK_WINDOW_SEC]
        max_x_window = window["end_x"].max() if not window.empty else start_x
        territory_10s = max_x_window - start_x

        # Time to reach the final third.
        in_ft = on_ball[on_ball["end_x"] >= ATT_THIRD]
        time_to_final_third = (in_ft["t"].iloc[0] - start_t) if not in_ft.empty else np.nan

        rows.append(
            {
                "match_id": m["match_id"],
                "period": m["period"],
                "possession_id": poss,
                "team": team,
                "win_zone": zone_of_x(start_x),
                "start_x": start_x,
                "first_action": first["type"],
                "first_dx": dx,
                "first_is_long": is_long,
                "response": response,
                "territory_10s": territory_10s,
                "time_to_final_third": time_to_final_third,
                "on_ball_actions": int(len(on_ball)),
            }
        )

    trans = pd.DataFrame(rows)
    if trans.empty or possession_summary is None:
        return trans

    keep = [
        "match_id", "period", "possession_id", "possession_team",
        "pass_count", "clock_duration", "net_x_progression", "directness_ratio",
        "final_third_entry", "box_entry", "shot_count", "xg_created",
        "ends_with_shot",
    ]
    ps = possession_summary[[c for c in keep if c in possession_summary.columns]].rename(
        columns={"possession_team": "team"}
    )
    return trans.merge(ps, on=["match_id", "period", "possession_id", "team"], how="left")


def summarize_offensive_transitions(trans_table: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Team-level offensive-transition profile (questions 1 & 2)."""
    rows = []
    for team in teams:
        t = trans_table[trans_table["team"].eq(team)]
        n = len(t)
        if n == 0:
            continue
        rows.append(
            {
                "team": team,
                "transitions": n,
                "attack_pct": t["response"].eq("attack").mean() * 100,
                "secure_pct": t["response"].eq("secure").mean() * 100,
                "recycle_back_pct": t["response"].eq("recycle_back").mean() * 100,
                "first_long_pct": t["first_is_long"].mean() * 100,
                "median_territory_10s": t["territory_10s"].median(),
                "mean_territory_10s": t["territory_10s"].mean(),
                "median_directness": t["directness_ratio"].median() if "directness_ratio" in t else np.nan,
                "reached_final_third_pct": t["final_third_entry"].mean() * 100 if "final_third_entry" in t else np.nan,
                "median_time_to_final_third": t["time_to_final_third"].median(),
            }
        )
    return pd.DataFrame(rows)


def transition_win_zones(trans_table: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Where on the pitch each team wins the ball in open play (counts + pct)."""
    rows = []
    order = ["defensive", "middle", "attacking"]
    for team in teams:
        t = trans_table[trans_table["team"].eq(team)]
        total = len(t)
        if total == 0:
            continue
        counts = t["win_zone"].value_counts()
        for zone in order:
            c = int(counts.get(zone, 0))
            rows.append(
                {"team": team, "win_zone": zone, "count": c, "pct": c / total * 100}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Defensive transitions: reaction after losing the ball
# ---------------------------------------------------------------------------

def build_defensive_transition_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per open-play loss: how the team that just lost the ball reacted
    in the DEF_REACTION_SEC window while the opponent had possession."""
    e = _ordered(events_df)
    e["_t"] = e["minute"] * 60 + e["second"]
    meta = _possession_meta(events_df)
    wins = meta[meta["is_open_play_win"]]

    groups = {poss: g for poss, g in e.groupby("possession", sort=False)}
    rows = []
    for _, m in wins.iterrows():
        winner = m["possession_team"]
        loser = m["prev_team"]
        g = groups.get(m["possession_id"])
        if g is None:
            continue
        g = g.sort_values("_t")
        start_t = g["_t"].iloc[0]

        # Where the loser lost the ball ~ where the winner won it.
        win_start_x = pd.to_numeric(
            g[g["team"].eq(winner)]["location_x"], errors="coerce"
        ).dropna()
        loss_x = win_start_x.iloc[0] if not win_start_x.empty else np.nan
        # Loser's pitch perspective: a high winner-x = loser lost it deep in
        # their own attacking half (good place to counterpress).
        loss_zone = zone_of_x(loss_x)

        # Loser's defensive actions during the opponent possession window.
        window = g[g["_t"] - start_t <= DEF_REACTION_SEC]
        loser_actions = window[window["team"].eq(loser)]

        counterpressed = bool(loser_actions.get("counterpress", pd.Series(dtype=object)).map(_parse_bool).any()) \
            if "counterpress" in loser_actions.columns else False
        def_actions = loser_actions[loser_actions["type"].isin(DEFENSIVE_ACTIONS)]
        any_def_action = not def_actions.empty
        if any_def_action:
            time_to_first_def = float(def_actions["_t"].iloc[0] - start_t)
        else:
            time_to_first_def = np.nan

        # Territory the winner gained in the same window (territory conceded).
        winner_on_ball = _team_on_ball(g)
        winner_on_ball = winner_on_ball[winner_on_ball["t"] - start_t <= DEF_REACTION_SEC]
        if not winner_on_ball.empty:
            territory_conceded = winner_on_ball["end_x"].max() - winner_on_ball["start_x"].iloc[0]
        else:
            territory_conceded = np.nan

        rows.append(
            {
                "match_id": m["match_id"],
                "period": m["period"],
                "possession_id": m["possession_id"],
                "team": loser,
                "loss_zone": loss_zone,
                "counterpressed": counterpressed,
                "any_def_action_5s": any_def_action,
                "time_to_first_def_action": time_to_first_def,
                "territory_conceded_5s": territory_conceded,
            }
        )
    return pd.DataFrame(rows)


def summarize_defensive_transitions(def_table: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    rows = []
    for team in teams:
        t = def_table[def_table["team"].eq(team)]
        n = len(t)
        if n == 0:
            continue
        rows.append(
            {
                "team": team,
                "losses": n,
                "counterpress_pct": t["counterpressed"].mean() * 100,
                "def_action_5s_pct": t["any_def_action_5s"].mean() * 100,
                "median_time_to_def_action": t["time_to_first_def_action"].median(),
                "median_territory_conceded_5s": t["territory_conceded_5s"].median(),
                "mean_territory_conceded_5s": t["territory_conceded_5s"].mean(),
                "lost_in_att_third_pct": t["loss_zone"].eq("attacking").mean() * 100,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Dangerous transitions: outcomes
# ---------------------------------------------------------------------------

def summarize_transition_danger(
    trans_table: pd.DataFrame,
    possession_summary: pd.DataFrame,
    teams: list[str],
) -> pd.DataFrame:
    """Compare danger created by open-play transitions vs. all other
    possessions (set-piece + sustained), per team."""
    trans_keys = set(
        zip(trans_table["period"], trans_table["possession_id"], trans_table["team"])
    )

    rows = []
    for team in teams:
        ps = possession_summary[possession_summary["possession_team"].eq(team)].copy()
        if ps.empty:
            continue
        is_trans = ps.apply(
            lambda r: (r["period"], r["possession_id"], team) in trans_keys, axis=1
        )
        for label, subset in [("transition", ps[is_trans]), ("other", ps[~is_trans])]:
            n = len(subset)
            if n == 0:
                continue
            rows.append(
                {
                    "team": team,
                    "possession_type": label,
                    "possessions": n,
                    "shot_pct": subset["shot_count"].gt(0).mean() * 100,
                    "box_entry_pct": subset["box_entry"].mean() * 100,
                    "final_third_pct": subset["final_third_entry"].mean() * 100,
                    "total_xg": subset["xg_created"].sum(),
                    "xg_per_possession": subset["xg_created"].mean(),
                }
            )
    return pd.DataFrame(rows)


def transition_danger_by_response(trans_table: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Does attacking immediately or securing first create more danger?"""
    rows = []
    for team in teams:
        t = trans_table[trans_table["team"].eq(team)]
        for response in ["attack", "secure", "recycle_back"]:
            sub = t[t["response"].eq(response)]
            n = len(sub)
            if n == 0:
                continue
            rows.append(
                {
                    "team": team,
                    "response": response,
                    "transitions": n,
                    "box_entry_pct": sub["box_entry"].mean() * 100 if "box_entry" in sub else np.nan,
                    "shot_pct": sub["shot_count"].gt(0).mean() * 100 if "shot_count" in sub else np.nan,
                    "total_xg": sub["xg_created"].sum() if "xg_created" in sub else np.nan,
                }
            )
    return pd.DataFrame(rows)
