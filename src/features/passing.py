"""Passing-intent and passing-structure features.

Merged from the former passing/{passing,structure}.py modules.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd

"""Passing-intent metrics for StatsBomb event data."""


import ast
import re

import numpy as np
import pandas as pd


PASS_ANALYSIS_COLS = [
    "id",
    "match_id",
    "team",
    "player",
    "player_id",
    "type",
    "possession",
    "possession_team",
    "play_pattern",
    "location_x",
    "location_y",
    "pass_end_x",
    "pass_end_y",
    "carry_end_x",
    "carry_end_y",
    "pass_length",
    "pass_angle",
    "pass_complete",
    "pass_outcome",
    "pass_height",
    "pass_cross",
    "pass_through_ball",
    "pass_switch",
    "under_pressure",
    "pass_recipient",
    "pass_recipient_id",
    "minute",
    "second",
    "timestamp",
    "position",
    "tactics"
]

FORWARD_X_M = 2.0
LATERAL_MIN_DISTANCE_M = 5.0
SHORT_PASS_M = 15.0
PROGRESSIVE_PASS_X_M = 10.0
BYPASS_MIDFIELD_X = 60.0
FINAL_THIRD_X = 80.0
LEFT_WIDE_Y = 26.67
RIGHT_WIDE_Y = 53.33

DEFENSIVE_ROLES = {"goalkeeper", "fullback", "center_back"}
WIDE_ROLES = {"fullback", "wide_midfield", "winger"}
CENTRAL_ROLES = {"central_midfield", "attacking_midfield"}
ADVANCED_ROLES = {"forward", "winger", "attacking_midfield"}


def filter_regular_play_passes(events: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    """Return Regular Play passes, optionally limited to one team."""
    passes = events[
        events["type"].eq("Pass") & events["play_pattern"].eq("Regular Play")
    ].copy()
    if team is not None:
        passes = passes[passes["team"].eq(team)]
    return enrich_pass_vectors(passes)


def filter_regular_play_events(events: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    """Return Regular Play events, optionally limited to one team."""
    out = events[events["play_pattern"].eq("Regular Play")].copy()
    if team is not None:
        out = out[out["team"].eq(team)]
    return out


def enrich_pass_vectors(passes: pd.DataFrame) -> pd.DataFrame:
    """Add pass vector fields used by geometry, pressure, and progression metrics."""
    out = passes.copy()
    out["dx"] = out["pass_end_x"] - out["location_x"]
    out["dy"] = out["pass_end_y"] - out["location_y"]
    out["distance"] = np.hypot(out["dx"], out["dy"])
    out["verticality"] = out["dx"] / out["distance"].replace(0, np.nan)
    out["direction"] = out["dx"].map(pass_direction)
    angle = out["pass_angle"].astype(float)
    out["angle_0_2pi"] = np.mod(angle, 2 * np.pi)
    out["direction_sector"] = pd.cut(
        out["angle_0_2pi"],
        bins=np.linspace(0, 2 * np.pi, 9),
        labels=False,
        include_lowest=True,
    )
    return out


def pass_direction(dx: float) -> str:
    if pd.isna(dx):
        return "unknown"
    if dx > FORWARD_X_M:
        return "forward"
    if dx < -FORWARD_X_M:
        return "backward"
    return "lateral"


def compute_pass_geometry(passes: pd.DataFrame) -> pd.Series:
    """Summarize vertical, lateral, and directional-variety passing intent."""
    passes = enrich_pass_vectors(passes)
    if passes.empty:
        return _empty_series(
            [
                "passes",
                "verticality",
                "forward_pass_pct",
                "backward_pass_pct",
                "lateral_pass_pct",
                "lateral_circulation_pct",
                "angle_entropy",
            ]
        )

    direction = passes["direction"].value_counts(normalize=True) * 100
    lateral = (
        passes["dy"].abs().gt(passes["dx"].abs())
        & passes["distance"].ge(LATERAL_MIN_DISTANCE_M)
    )

    return pd.Series(
        {
            "passes": len(passes),
            "verticality": passes["verticality"].mean(),
            "forward_pass_pct": direction.get("forward", 0.0),
            "backward_pass_pct": direction.get("backward", 0.0),
            "lateral_pass_pct": direction.get("lateral", 0.0),
            "lateral_circulation_pct": lateral.mean() * 100,
            "angle_entropy": _entropy(passes["direction_sector"].dropna()),
        }
    )


def compute_pass_network_metrics(passes: pd.DataFrame) -> pd.Series:
    """Compute directed pass-network density plus simple undirected clustering."""
    edges = build_pass_network_edges(passes)
    if edges.empty:
        return _empty_series(
            [
                "network_players",
                "centralization",
                "connectivity",
                "clustering",
                "top_player_name",
                "top_player_involvement_pct",
                "top_edge",
                "top_edge_pct",
            ]
        )

    players = pd.Index(pd.unique(edges[["player", "pass_recipient"]].values.ravel("K"))).dropna()
    n_players = len(players)
    involvement = (
        pd.concat([edges["player"], edges["pass_recipient"]])
        .value_counts()
        .reindex(players, fill_value=0)
    )
    share = involvement / involvement.sum()
    unique_edges = edges[["player", "pass_recipient"]].drop_duplicates()
    possible_edges = n_players * (n_players - 1)
    top_player = involvement.idxmax()
    top_edge_row = edges.sort_values("weight", ascending=False).iloc[0]
    top_edge = f"{top_edge_row['player']} -> {top_edge_row['pass_recipient']}"

    return pd.Series(
        {
            "network_players": n_players,
            "centralization": (share**2).sum(),
            "connectivity": len(unique_edges) / possible_edges if possible_edges else np.nan,
            "clustering": _average_clustering(unique_edges, players),
            "top_player_name": top_player,
            "top_player_involvement_pct": share.loc[top_player] * 100,
            "top_edge": top_edge,
            "top_edge_pct": top_edge_row["weight"] / edges["weight"].sum() * 100,
        }
    )


def build_pass_network_edges(passes: pd.DataFrame) -> pd.DataFrame:
    """Return completed player-to-recipient pass edges."""
    passes = passes.copy()
    completed = _complete_pass_mask(passes)
    edges = passes[
        completed & passes["player"].notna() & passes["pass_recipient"].notna()
    ][["team", "player", "pass_recipient"]].copy()
    if edges.empty:
        return edges.assign(weight=pd.Series(dtype=int))
    return (
        edges.groupby(["team", "player", "pass_recipient"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )


def pass_network_adjacency(passes: pd.DataFrame, team: str) -> pd.DataFrame:
    """Build a weighted adjacency matrix for completed Regular Play passes."""
    edges = build_pass_network_edges(passes[passes["team"].eq(team)])
    if edges.empty:
        return pd.DataFrame()
    matrix = edges.pivot_table(
        index="player",
        columns="pass_recipient",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )
    players = sorted(set(matrix.index).union(matrix.columns))
    return matrix.reindex(index=players, columns=players, fill_value=0)


def compute_buildup_patterns(events: pd.DataFrame, team: str) -> pd.Series:
    """Summarize first-pass behavior and positional spread inside open-play buildups."""
    passes = filter_regular_play_passes(events, team)
    if passes.empty:
        return _empty_series(
            [
                "first_forward_pct",
                "first_backward_pct",
                "first_lateral_pct",
                "first_short_pct",
                "first_long_pct",
                "first_from_defensive_third_pct",
                "first_into_midfield_or_better_pct",
                "avg_endpoint_width_std",
                "avg_endpoint_depth_std",
            ]
        )

    first = (
        passes.sort_values(["period", "possession", "minute", "second"])
        .groupby("possession", sort=False)
        .head(1)
    )
    first_dir = first["direction"].value_counts(normalize=True) * 100
    structured = passes.groupby("possession").filter(lambda g: len(g) >= 3)
    spread = structured.groupby("possession").agg(
        endpoint_width_std=("pass_end_y", "std"),
        endpoint_depth_std=("pass_end_x", "std"),
    )

    return pd.Series(
        {
            "first_forward_pct": first_dir.get("forward", 0.0),
            "first_backward_pct": first_dir.get("backward", 0.0),
            "first_lateral_pct": first_dir.get("lateral", 0.0),
            "first_short_pct": first["pass_length"].lt(SHORT_PASS_M).mean() * 100,
            "first_long_pct": first["pass_length"].ge(SHORT_PASS_M).mean() * 100,
            "first_from_defensive_third_pct": first["location_x"].lt(40).mean() * 100,
            "first_into_midfield_or_better_pct": first["pass_end_x"].ge(40).mean() * 100,
            "avg_endpoint_width_std": spread["endpoint_width_std"].mean(),
            "avg_endpoint_depth_std": spread["endpoint_depth_std"].mean(),
        }
    )


def first_pass_direction_table(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    rows = []
    for team in teams:
        passes = filter_regular_play_passes(events, team)
        if passes.empty:
            continue
        first = (
            passes.sort_values(["period", "possession", "minute", "second"])
            .groupby("possession", sort=False)
            .head(1)
        )
        counts = first["direction"].value_counts()
        total = counts.sum()
        for direction in ["forward", "lateral", "backward"]:
            rows.append(
                {
                    "team": team,
                    "direction": direction,
                    "count": int(counts.get(direction, 0)),
                    "pct": counts.get(direction, 0) / total * 100 if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_pressure_passing(events: pd.DataFrame, team: str) -> pd.Series:
    """Measure pass stability and immediate escape options under pressure."""
    passes = filter_regular_play_passes(events, team)
    if passes.empty:
        return _empty_series(
            [
                "pressured_pass_pct",
                "pressured_completion_pct",
                "unpressured_completion_pct",
                "pressured_mean_dx",
                "unpressured_mean_dx",
                "pressure_escape_forward_pass_pct",
                "pressure_escape_positive_carry_pct",
                "pressure_escape_turnover_pct",
            ]
        )

    pressured = passes["under_pressure"].eq(True)
    pressured_passes = passes[pressured]
    unpressured_passes = passes[~pressured]

    escapes = pressure_escape_table(events, team)
    escape_rates = escapes["escape_type"].value_counts(normalize=True) * 100

    return pd.Series(
        {
            "pressured_pass_pct": pressured.mean() * 100,
            "pressured_completion_pct": _complete_pass_mask(pressured_passes).mean() * 100
            if len(pressured_passes)
            else np.nan,
            "unpressured_completion_pct": _complete_pass_mask(unpressured_passes).mean() * 100
            if len(unpressured_passes)
            else np.nan,
            "pressured_mean_dx": pressured_passes["dx"].mean(),
            "unpressured_mean_dx": unpressured_passes["dx"].mean(),
            "pressure_escape_forward_pass_pct": escape_rates.get("complete_forward_pass", 0.0),
            "pressure_escape_positive_carry_pct": escape_rates.get("positive_carry", 0.0),
            "pressure_escape_turnover_pct": escape_rates.get("turnover", 0.0),
        }
    )


def pressure_escape_table(events: pd.DataFrame, team: str) -> pd.DataFrame:
    """Classify the next same-team action after each pressured Regular Play pass."""
    team_events = filter_regular_play_events(events, team)
    team_events = team_events.sort_values(["period", "possession", "minute", "second"]).copy()
    passes = enrich_pass_vectors(team_events[team_events["type"].eq("Pass")])
    pressured_ids = set(passes[passes["under_pressure"].eq(True)]["id"])
    rows = []

    for _, group in team_events.groupby("possession", sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            if row.get("id") not in pressured_ids:
                continue
            if idx + 1 >= len(group):
                rows.append({"team": team, "pass_id": row.get("id"), "escape_type": "turnover"})
                continue
            nxt = group.iloc[idx + 1]
            rows.append(
                {
                    "team": team,
                    "pass_id": row.get("id"),
                    "escape_type": _classify_escape(nxt),
                }
            )

    return pd.DataFrame(rows, columns=["team", "pass_id", "escape_type"])


def compute_progression_mechanism(events: pd.DataFrame, team: str) -> pd.Series:
    """Compare positive x progression generated by passes and carries."""
    team_events = filter_regular_play_events(events, team)
    passes = enrich_pass_vectors(team_events[team_events["type"].eq("Pass")])
    carries = team_events[team_events["type"].eq("Carry")].copy()
    carries["dx"] = carries["carry_end_x"] - carries["location_x"]

    pass_positive = passes["dx"].clip(lower=0).sum()
    carry_positive = carries["dx"].clip(lower=0).sum()
    total_positive = pass_positive + carry_positive

    return pd.Series(
        {
            "positive_pass_progression": pass_positive,
            "positive_carry_progression": carry_positive,
            "pass_progression_share_pct": pass_positive / total_positive * 100
            if total_positive
            else np.nan,
            "carry_progression_share_pct": carry_positive / total_positive * 100
            if total_positive
            else np.nan,
            "progression_per_pass": passes["dx"].mean(),
            "progression_per_carry": carries["dx"].mean(),
            "progressive_pass_pct": passes["dx"].ge(PROGRESSIVE_PASS_X_M).mean() * 100
            if len(passes)
            else np.nan,
            "progressive_carry_pct": carries["dx"].ge(PROGRESSIVE_PASS_X_M).mean() * 100
            if len(carries)
            else np.nan,
        }
    )


def build_team_passing_summary(events: pd.DataFrame, team: str) -> pd.Series:
    """Combine all Regular Play passing-intent metrics for one team."""
    passes = filter_regular_play_passes(events, team)
    return pd.concat(
        [
            compute_pass_geometry(passes),
            compute_pass_network_metrics(passes),
            compute_buildup_patterns(events, team),
            compute_buildup_shape(events, team),
            compute_pressure_passing(events, team),
            compute_progression_mechanism(events, team),
        ]
    )


def _classify_escape(event: pd.Series) -> str:
    if event.get("type") == "Pass":
        dx = event.get("pass_end_x") - event.get("location_x")
        complete = pd.isna(event.get("pass_outcome")) or bool(event.get("pass_complete"))
        if complete and pd.notna(dx) and dx > FORWARD_X_M:
            return "complete_forward_pass"
        if not complete:
            return "turnover"
        return "retained_pass"

    if event.get("type") == "Carry":
        dx = event.get("carry_end_x") - event.get("location_x")
        if pd.notna(dx) and dx > 0:
            return "positive_carry"
        return "retained_carry"

    if event.get("type") in {"Dispossessed", "Miscontrol"}:
        return "turnover"
    return "retained_other"


def _complete_pass_mask(passes: pd.DataFrame) -> pd.Series:
    if passes.empty:
        return pd.Series(dtype=bool, index=passes.index)
    if "pass_complete" in passes.columns:
        return passes["pass_complete"].astype("boolean").fillna(False).astype(bool)
    return passes["pass_outcome"].isna()


def _entropy(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    probs = values.value_counts(normalize=True)
    return float(-(probs * np.log2(probs)).sum())


def _average_clustering(edges: pd.DataFrame, players: pd.Index) -> float:
    if len(players) < 3:
        return np.nan

    neighbors = {player: set() for player in players}
    for _, edge in edges.iterrows():
        a = edge["player"]
        b = edge["pass_recipient"]
        if a == b or pd.isna(a) or pd.isna(b):
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)

    coeffs = []
    edge_set = {
        frozenset((edge["player"], edge["pass_recipient"]))
        for _, edge in edges.iterrows()
        if edge["player"] != edge["pass_recipient"]
    }
    for player, neigh in neighbors.items():
        degree = len(neigh)
        if degree < 2:
            continue
        possible = degree * (degree - 1) / 2
        actual = 0
        neigh_list = list(neigh)
        for i, first in enumerate(neigh_list):
            for second in neigh_list[i + 1 :]:
                if frozenset((first, second)) in edge_set:
                    actual += 1
        coeffs.append(actual / possible)

    return float(np.mean(coeffs)) if coeffs else np.nan


def _empty_series(keys: list[str]) -> pd.Series:
    return pd.Series({key: np.nan for key in keys})

def player_network_involvement(
    passes: pd.DataFrame,
    teams: list[str],
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    """Rank players by share of completed pass-network involvement."""
    passes = filter_regular_play_passes(passes) if "dx" not in passes.columns else passes.copy()
    rows = []
    for team in teams:
        edges = build_pass_network_edges(passes[passes["team"].eq(team)])
        if edges.empty:
            continue
        involvement = (
            pd.concat([edges["player"], edges["pass_recipient"]])
            .value_counts()
            .rename_axis("player")
            .reset_index(name="involvements")
        )
        total = involvement["involvements"].sum()
        involvement["team"] = team
        involvement["involvement_pct"] = involvement["involvements"] / total * 100
        involvement["rank"] = np.arange(1, len(involvement) + 1)
        rows.append(involvement.head(top_n))

    if not rows:
        return pd.DataFrame(
            columns=["player", "involvements", "team", "involvement_pct", "rank"]
        )
    return pd.concat(rows, ignore_index=True)[
        ["team", "rank", "player", "involvements", "involvement_pct"]
    ]


def top_pass_connections(
    passes: pd.DataFrame,
    teams: list[str],
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    """Rank completed passer-to-recipient edges by share of team network passes."""
    passes = filter_regular_play_passes(passes) if "dx" not in passes.columns else passes.copy()
    rows = []
    for team in teams:
        edges = build_pass_network_edges(passes[passes["team"].eq(team)])
        if edges.empty:
            continue
        total = edges["weight"].sum()
        out = edges.sort_values("weight", ascending=False).head(top_n).copy()
        out["edge"] = out["player"] + " -> " + out["pass_recipient"]
        out["edge_pct"] = out["weight"] / total * 100
        out["rank"] = np.arange(1, len(out) + 1)
        rows.append(out[["team", "rank", "player", "pass_recipient", "edge", "weight", "edge_pct"]])

    if not rows:
        return pd.DataFrame(
            columns=["team", "rank", "player", "pass_recipient", "edge", "weight", "edge_pct"]
        )
    return pd.concat(rows, ignore_index=True)

def extract_lineup_snapshots(events: pd.DataFrame) -> pd.DataFrame:
    """Extract Starting XI and Tactical Shift lineup snapshots into player rows."""
    snapshot_events = events[events["type"].isin(["Starting XI", "Tactical Shift"])].copy()
    rows = []
    for _, event in snapshot_events.iterrows():
        tactics = event.get("tactics")
        formation = _parse_formation(tactics)
        for player in _parse_lineup_players(tactics):
            rows.append(
                {
                    "team": event.get("team"),
                    "event_type": event.get("type"),
                    "minute": event.get("minute"),
                    "second": event.get("second"),
                    "timestamp": event.get("timestamp"),
                    "formation": formation,
                    "player": player.get("player"),
                    "player_id": player.get("player_id"),
                    "position": player.get("position"),
                    "role_group": role_group(player.get("position")),
                }
            )
    return pd.DataFrame(rows)


def starting_lineup_table(events: pd.DataFrame) -> pd.DataFrame:
    """Create one row per starting player with formation and normalized role group."""
    lineups = extract_lineup_snapshots(events)
    if lineups.empty:
        return lineups
    return lineups[lineups["event_type"].eq("Starting XI")].reset_index(drop=True)


def role_group(position: str | float | None) -> str:
    """Map StatsBomb position names into broader buildup role groups."""
    if position is None or pd.isna(position):
        return "unknown"

    p = str(position).lower()
    if "goalkeeper" in p:
        return "goalkeeper"
    if "back" in p and "center" not in p:
        return "fullback"
    if "center back" in p:
        return "center_back"
    if "wing" in p:
        return "winger"
    if "wide" in p or ("midfield" in p and ("left" in p or "right" in p) and "center" not in p and "defensive" not in p):
        return "wide_midfield"
    if "attacking midfield" in p:
        return "attacking_midfield"
    if "midfield" in p:
        return "central_midfield"
    if "forward" in p or "striker" in p:
        return "forward"
    return "unknown"


def attach_player_roles(events: pd.DataFrame, lineup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach starter position/role to passers and pass recipients."""
    out = events.copy()
    lineup = starting_lineup_table(out) if lineup is None else lineup.copy()
    if lineup.empty:
        out["player_role"] = out.get("position", pd.Series(index=out.index, dtype=object)).map(role_group)
        out["recipient_role"] = "unknown"
        return out

    role_lookup = lineup.dropna(subset=["team", "player"]).set_index(["team", "player"])["role_group"]
    position_lookup = lineup.dropna(subset=["team", "player"]).set_index(["team", "player"])["position"]

    out["player_position"] = [
        position_lookup.get((team, player), position)
        for team, player, position in zip(
            out.get("team", pd.Series(index=out.index)),
            out.get("player", pd.Series(index=out.index)),
            out.get("position", pd.Series(index=out.index)),
        )
    ]
    out["player_role"] = [
        role_lookup.get((team, player), role_group(position))
        for team, player, position in zip(out["team"], out["player"], out["player_position"])
    ]
    out["recipient_position"] = [
        position_lookup.get((team, recipient), np.nan)
        for team, recipient in zip(out.get("team", pd.Series(index=out.index)), out.get("pass_recipient", pd.Series(index=out.index)))
    ]
    out["recipient_role"] = out["recipient_position"].map(role_group)
    return out


def first_pass_table(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """First completed or attempted pass in each Regular Play possession."""
    lineup = starting_lineup_table(events)
    passes = attach_player_roles(filter_regular_play_passes(events), lineup)
    passes = passes[passes["team"].isin(teams)].copy()
    if passes.empty:
        return pd.DataFrame()

    first = (
        passes.sort_values(["team", "period", "possession", "minute", "second"])
        .groupby(["team", "possession"], sort=False)
        .head(1)
        .copy()
    )
    first["start_x_zone"] = first["location_x"].map(pitch_x_zone)
    first["end_x_zone"] = first["pass_end_x"].map(pitch_x_zone)
    first["start_y_channel"] = first["location_y"].map(pitch_y_channel)
    first["end_y_channel"] = first["pass_end_y"].map(pitch_y_channel)
    first["is_short"] = first["pass_length"].lt(SHORT_PASS_M)
    first["bypasses_midfield_line"] = (
        first["location_x"].lt(40) & first["pass_end_x"].ge(BYPASS_MIDFIELD_X)
    )
    cols = [
        "team",
        "possession",
        "minute",
        "second",
        "player",
        "player_role",
        "pass_recipient",
        "recipient_role",
        "location_x",
        "location_y",
        "pass_end_x",
        "pass_end_y",
        "start_x_zone",
        "end_x_zone",
        "start_y_channel",
        "end_y_channel",
        "pass_length",
        "direction",
        "is_short",
        "bypasses_midfield_line",
    ]
    return first[cols].reset_index(drop=True)


def early_buildup_sequence_table(
    events: pd.DataFrame,
    teams: list[str],
    *,
    n_passes: int = 3,
) -> pd.DataFrame:
    """First n Regular Play passes of each possession with starter roles attached."""
    lineup = starting_lineup_table(events)
    passes = attach_player_roles(filter_regular_play_passes(events), lineup)
    passes = passes[passes["team"].isin(teams)].copy()
    if passes.empty:
        return pd.DataFrame()

    passes = passes.sort_values(["team", "period", "possession", "minute", "second"])
    passes["pass_number_in_possession"] = passes.groupby(["team", "possession"]).cumcount() + 1
    early = passes[passes["pass_number_in_possession"].le(n_passes)].copy()
    early["end_x_zone"] = early["pass_end_x"].map(pitch_x_zone)
    early["end_y_channel"] = early["pass_end_y"].map(pitch_y_channel)
    early["involves_central_midfield"] = early["player_role"].isin(CENTRAL_ROLES) | early["recipient_role"].isin(CENTRAL_ROLES)
    early["targets_wide_role"] = early["recipient_role"].isin(WIDE_ROLES)
    early["targets_advanced_role"] = early["recipient_role"].isin(ADVANCED_ROLES)
    cols = [
        "team",
        "possession",
        "pass_number_in_possession",
        "player",
        "player_role",
        "pass_recipient",
        "recipient_role",
        "location_x",
        "location_y",
        "pass_end_x",
        "pass_end_y",
        "end_x_zone",
        "end_y_channel",
        "pass_length",
        "direction",
        "involves_central_midfield",
        "targets_wide_role",
        "targets_advanced_role",
    ]
    return early[cols].reset_index(drop=True)


def compute_buildup_shape(events: pd.DataFrame, team: str) -> pd.Series:
    """Team-level lineup-aware Regular Play buildup shape metrics."""
    first = first_pass_table(events, [team])
    early = early_buildup_sequence_table(events, [team], n_passes=3)
    if first.empty:
        return _empty_series(
            [
                "regular_play_buildups",
                "first_pass_short_pct",
                "first_pass_forward_pct",
                "first_pass_to_fullback_or_wide_pct",
                "first_pass_to_central_midfield_pct",
                "first_pass_to_forward_pct",
                "first_pass_bypass_midfield_pct",
                "defensive_to_advanced_first_pass_pct",
                "early_width_y_std",
                "early_depth_x_std",
                "early_wide_role_target_pct",
                "early_central_involvement_pct",
                "early_central_lane_pct",
                "avg_passes_to_wide_channel",
                "avg_passes_to_final_third",
            ]
        )

    early_spread = early.groupby("possession").agg(
        width_y_std=("pass_end_y", "std"),
        depth_x_std=("pass_end_x", "std"),
    )
    passes_to_wide = _passes_until_condition(early, early["end_y_channel"].isin(["left_wide", "right_wide"]))
    passes_to_final_third = _passes_until_condition(early, early["pass_end_x"].ge(FINAL_THIRD_X))
    defensive_to_advanced = first["player_role"].isin(DEFENSIVE_ROLES) & first["recipient_role"].isin(ADVANCED_ROLES)

    return pd.Series(
        {
            "regular_play_buildups": len(first),
            "first_pass_short_pct": first["is_short"].mean() * 100,
            "first_pass_forward_pct": first["direction"].eq("forward").mean() * 100,
            "first_pass_to_fullback_or_wide_pct": first["recipient_role"].isin(WIDE_ROLES).mean() * 100,
            "first_pass_to_central_midfield_pct": first["recipient_role"].eq("central_midfield").mean() * 100,
            "first_pass_to_forward_pct": first["recipient_role"].eq("forward").mean() * 100,
            "first_pass_bypass_midfield_pct": first["bypasses_midfield_line"].mean() * 100,
            "defensive_to_advanced_first_pass_pct": defensive_to_advanced.mean() * 100,
            "early_width_y_std": early_spread["width_y_std"].mean(),
            "early_depth_x_std": early_spread["depth_x_std"].mean(),
            "early_wide_role_target_pct": early["targets_wide_role"].mean() * 100,
            "early_central_involvement_pct": early["involves_central_midfield"].mean() * 100,
            "early_central_lane_pct": early["end_y_channel"].eq("central").mean() * 100,
            "avg_passes_to_wide_channel": passes_to_wide.mean(),
            "avg_passes_to_final_third": passes_to_final_third.mean(),
        }
    )


def buildup_shape_table(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Build one lineup-aware buildup-shape row per team."""
    return pd.DataFrame(
        {"team": team, **compute_buildup_shape(events, team).to_dict()} for team in teams
    )


def pitch_x_zone(x: float) -> str | float:
    if pd.isna(x):
        return np.nan
    if x < 40:
        return "defensive_third"
    if x < FINAL_THIRD_X:
        return "middle_third"
    return "final_third"


def pitch_y_channel(y: float) -> str | float:
    if pd.isna(y):
        return np.nan
    if y < LEFT_WIDE_Y:
        return "left_wide"
    if y <= RIGHT_WIDE_Y:
        return "central"
    return "right_wide"


def _parse_formation(tactics) -> float | int | None:
    if isinstance(tactics, dict):
        return tactics.get("formation")
    if isinstance(tactics, str):
        match = re.search(r"'formation':\s*([0-9.]+)", tactics)
        if match:
            value = float(match.group(1))
            return int(value) if value.is_integer() else value
    return np.nan


def _parse_lineup_players(tactics) -> list[dict]:
    if isinstance(tactics, dict):
        lineup = tactics.get("lineup", [])
    elif isinstance(tactics, str):
        lineup = []
        for item in re.findall(r"\{'jersey_number':.*?'position':\s*\{.*?\}\}", tactics, flags=re.DOTALL):
            try:
                lineup.append(ast.literal_eval(item))
            except (SyntaxError, ValueError):
                continue
    else:
        return []

    rows = []
    for item in list(lineup):
        player = item.get("player", {}) if isinstance(item, dict) else {}
        position = item.get("position", {}) if isinstance(item, dict) else {}
        rows.append(
            {
                "player": player.get("name"),
                "player_id": player.get("id"),
                "position": position.get("name"),
                "position_id": position.get("id"),
            }
        )
    return rows


def _passes_until_condition(early: pd.DataFrame, condition: pd.Series) -> pd.Series:
    flagged = early.loc[condition, ["possession", "pass_number_in_possession"]]
    if flagged.empty:
        return pd.Series(dtype=float)
    return flagged.groupby("possession")["pass_number_in_possession"].min()

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


import numpy as np
import pandas as pd


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
