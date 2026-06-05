"""Passing-intent metrics for StatsBomb event data."""

from __future__ import annotations

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
]

FORWARD_X_M = 2.0
LATERAL_MIN_DISTANCE_M = 5.0
SHORT_PASS_M = 15.0
PROGRESSIVE_PASS_X_M = 10.0


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
        return _empty_series(["network_players", "centralization", "connectivity", "clustering"])

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

    return pd.Series(
        {
            "network_players": n_players,
            "centralization": (share**2).sum(),
            "connectivity": len(unique_edges) / possible_edges if possible_edges else np.nan,
            "clustering": _average_clustering(unique_edges, players),
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