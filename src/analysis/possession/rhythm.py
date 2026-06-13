"""Possession rhythm and tempo metrics.

Rhythm describes *how* a possession unfolds in time — the speed of individual
actions, how consistent that speed is, and how the team alternates between
circulating the ball and committing to forward progression.

Key dimensions
--------------
Speed        : median / mean inter-event gap within a possession (seconds)
Consistency  : coefficient of variation of those gaps (lower = more metronomic)
Shape        : how tempo changes from the first event to the last (trend)
Burst play   : longest run of rapid consecutive actions
Circulation  : passes / seconds before the first significant forward move
Progression  : x-gain per progressive action; passes needed per 10 m advanced
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A consecutive pair of team events closer than this is a "rapid" action (burst play)
BURST_GAP_THRESHOLD = 1.5  # seconds

# Minimum x-gain for a single pass or carry to count as a "progressive" action
PROGRESSION_X_GAIN = 10.0  # metres (StatsBomb pitch units)

_GROUP_COLS = ["match_id", "period", "possession", "possession_team"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _team_events(group: pd.DataFrame) -> pd.DataFrame:
    """Own-team events within a possession, sorted by time."""
    poss_team = group["possession_team"].iloc[0]
    return (
        group[group["team"] == poss_team]
        .sort_values(["minute", "second"])
        .reset_index(drop=True)
    )


def _to_seconds(df: pd.DataFrame) -> pd.Series:
    return df["minute"].astype(float) * 60 + df["second"].astype(float)


def _gaps(te: pd.DataFrame) -> np.ndarray:
    """Inter-event gaps in seconds (clipped to ≥ 0)."""
    return np.diff(_to_seconds(te).values).clip(min=0)


def _burst_stats(gaps: np.ndarray) -> tuple[int, float]:
    """Return (longest_burst_length, burst_share)."""
    if len(gaps) == 0:
        return 0, 0.0
    rapid = gaps < BURST_GAP_THRESHOLD
    max_burst = curr = 0
    for b in rapid:
        curr = curr + 1 if b else 0
        max_burst = max(max_burst, curr)
    burst_events = int(rapid.sum())
    return max_burst, burst_events / len(gaps)


def _tempo_trend(gaps: np.ndarray) -> float:
    """
    Linear slope of instantaneous tempo (events/sec) across the possession.
    Positive = team speeds up as possession develops; negative = slows down.
    """
    if len(gaps) < 3:
        return np.nan
    t_norm = np.linspace(0.0, 1.0, len(gaps))
    tempo = 1.0 / np.maximum(gaps, 0.1)
    try:
        return float(np.polyfit(t_norm, tempo, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def _move_table(te: pd.DataFrame) -> pd.DataFrame:
    """Pass and carry rows with x-gain column, sorted by time."""
    parts = []

    passes = te[te["type"] == "Pass"]
    if {"location_x", "pass_end_x"}.issubset(te.columns):
        dx = (
            pd.to_numeric(passes["pass_end_x"], errors="coerce")
            - pd.to_numeric(passes["location_x"], errors="coerce")
        )
        parts.append(
            pd.DataFrame(
                {
                    "minute": passes["minute"].values,
                    "second": passes["second"].values,
                    "x_gain": dx.values,
                    "move_type": "pass",
                }
            )
        )

    carries = te[te["type"] == "Carry"]
    if {"location_x", "carry_end_x"}.issubset(te.columns):
        dx = (
            pd.to_numeric(carries["carry_end_x"], errors="coerce")
            - pd.to_numeric(carries["location_x"], errors="coerce")
        )
        parts.append(
            pd.DataFrame(
                {
                    "minute": carries["minute"].values,
                    "second": carries["second"].values,
                    "x_gain": dx.values,
                    "move_type": "carry",
                }
            )
        )

    if not parts:
        return pd.DataFrame(columns=["minute", "second", "x_gain", "move_type"])

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["minute", "second"])
        .reset_index(drop=True)
    )


def _circulation_stats(te: pd.DataFrame, moves: pd.DataFrame) -> dict:
    """
    Compute the split between the circulation phase and the progression phase.

    Circulation phase: everything before the first action with x_gain ≥ PROGRESSION_X_GAIN.
    Progression phase: the rest of the possession.

    If no progression occurs, the entire possession is treated as circulation.
    """
    passes = te[te["type"] == "Pass"]
    n_passes = len(passes)

    if moves.empty or passes.empty:
        return {
            "pre_prog_pass_count": float(n_passes),
            "pre_prog_duration_sec": np.nan,
            "progressive_action_count": 0,
            "avg_progressive_x_gain": np.nan,
        }

    prog_moves = moves[moves["x_gain"].fillna(0) >= PROGRESSION_X_GAIN]
    prog_count = len(prog_moves)
    avg_prog_x = float(prog_moves["x_gain"].mean()) if prog_count else np.nan

    if prog_moves.empty:
        t0 = float(_to_seconds(te).iloc[0])
        t_end = float(_to_seconds(te).iloc[-1])
        return {
            "pre_prog_pass_count": float(n_passes),
            "pre_prog_duration_sec": max(t_end - t0, 0.0),
            "progressive_action_count": 0,
            "avg_progressive_x_gain": np.nan,
        }

    first_prog_time = (
        float(prog_moves.iloc[0]["minute"]) * 60
        + float(prog_moves.iloc[0]["second"])
    )
    t0 = float(_to_seconds(te).iloc[0])

    pass_times = _to_seconds(passes).values
    pre_prog_passes = int((pass_times < first_prog_time).sum())
    pre_prog_dur = max(first_prog_time - t0, 0.0)

    return {
        "pre_prog_pass_count": float(pre_prog_passes),
        "pre_prog_duration_sec": float(pre_prog_dur),
        "progressive_action_count": prog_count,
        "avg_progressive_x_gain": avg_prog_x,
    }


def _chain_rhythm(group: pd.DataFrame) -> dict:
    te = _team_events(group)

    empty = {
        "gap_median": np.nan,
        "gap_mean": np.nan,
        "gap_std": np.nan,
        "gap_cv": np.nan,
        "gap_p25": np.nan,
        "gap_p75": np.nan,
        "longest_burst": 0,
        "burst_share": 0.0,
        "tempo_trend": np.nan,
        "pre_prog_pass_count": np.nan,
        "pre_prog_duration_sec": np.nan,
        "progressive_action_count": 0,
        "avg_progressive_x_gain": np.nan,
    }

    if len(te) < 2:
        return empty

    gaps = _gaps(te)
    if len(gaps) == 0:
        return empty

    gap_mean = float(np.mean(gaps))
    gap_std = float(np.std(gaps))
    gap_cv = gap_std / gap_mean if gap_mean > 0 else np.nan
    longest_burst, burst_share = _burst_stats(gaps)

    moves = _move_table(te)
    circ = _circulation_stats(te, moves)

    return {
        "gap_median": float(np.median(gaps)),
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "gap_cv": gap_cv,
        "gap_p25": float(np.percentile(gaps, 25)),
        "gap_p75": float(np.percentile(gaps, 75)),
        "longest_burst": longest_burst,
        "burst_share": burst_share,
        "tempo_trend": _tempo_trend(gaps),
        **circ,
    }


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_possession_rhythm(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-chain rhythm metrics for every possession in the match."""
    rows = []
    for keys, group in events_df.groupby(_GROUP_COLS, dropna=False, sort=False):
        metrics = _chain_rhythm(group)
        rows.append(dict(zip(_GROUP_COLS, keys), **metrics))
    return pd.DataFrame(rows).rename(columns={"possession": "possession_id"})


def build_intra_possession_tempo(
    events_df: pd.DataFrame,
    teams: list[str],
) -> pd.DataFrame:
    """
    Record the inter-event gap at every event position within a possession.

    Row structure: (possession_team, event_position, gap_sec)

    event_position = 1 means the gap between the 1st and 2nd own-team event,
    position = 2 means the gap between the 2nd and 3rd, etc.

    Averaging gap_sec grouped by (possession_team, event_position) shows how
    team tempo evolves through a typical possession chain.
    """
    rows = []
    for keys, group in events_df.groupby(_GROUP_COLS, dropna=False, sort=False):
        poss_team = keys[3]
        if poss_team not in teams:
            continue
        te = _team_events(group)
        if len(te) < 2:
            continue
        for pos, gap in enumerate(_gaps(te), start=1):
            rows.append(
                {
                    "possession_team": poss_team,
                    "event_position": pos,
                    "gap_sec": float(gap),
                }
            )
    return pd.DataFrame(rows)


def build_rhythm_summary(
    events_df: pd.DataFrame,
    possession_summary: pd.DataFrame,
    teams: list[str],
) -> pd.DataFrame:
    """
    Per-chain rhythm table joined with possession context (Regular Play only).

    Added derived columns
    ---------------------
    passes_per_10m               : passes per 10 m of net x-progression
    pre_prog_pass_share          : fraction of passes that occur before first progression
    """
    rhythm = build_possession_rhythm(events_df)

    ctx_cols = [
        c for c in [
            "match_id", "period", "possession_id", "possession_team",
            "play_pattern", "pass_count", "clock_duration",
            "directness_ratio", "net_x_progression", "total_x_progression",
            "final_third_entry", "box_entry", "xg_created", "ends_with_shot",
        ]
        if c in possession_summary.columns
    ]

    merged = rhythm.merge(
        possession_summary[ctx_cols],
        on=["match_id", "period", "possession_id", "possession_team"],
        how="left",
    )

    if "play_pattern" in merged.columns:
        merged = merged[merged["play_pattern"] == "Regular Play"].copy()

    if {"pass_count", "net_x_progression"}.issubset(merged.columns):
        merged["passes_per_10m"] = np.where(
            merged["net_x_progression"] > 0,
            merged["pass_count"] / (merged["net_x_progression"] / 10.0),
            np.nan,
        )

    if {"pass_count", "pre_prog_pass_count"}.issubset(merged.columns):
        merged["pre_prog_pass_share"] = (
            merged["pre_prog_pass_count"] / merged["pass_count"].replace(0, np.nan)
        ).clip(0, 1)

    return merged[merged["possession_team"].isin(teams)].reset_index(drop=True)


def summarize_rhythm_by_team(rhythm_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate chain-level rhythm into a one-row-per-team summary table.
    Use this for headline comparison displays.
    """
    rows = []
    for team, grp in rhythm_summary.groupby("possession_team"):
        # Only possessions with meaningful gap data
        valid = grp.dropna(subset=["gap_median"])

        rows.append(
            {
                "team": team,
                "possessions": len(grp),
                # Speed
                "median_gap_sec": valid["gap_median"].median(),
                "mean_gap_sec": valid["gap_mean"].mean(),
                # Consistency
                "median_gap_cv": valid["gap_cv"].median(),
                "pct_consistent": (valid["gap_cv"] < 1.0).mean() * 100,
                # Burst
                "median_longest_burst": valid["longest_burst"].median(),
                "mean_burst_share": valid["burst_share"].mean() * 100,
                # Tempo shape
                "pct_speeding_up": (valid["tempo_trend"] > 0).mean() * 100,
                # Circulation
                "median_pre_prog_passes": grp["pre_prog_pass_count"].median(),
                "median_pre_prog_duration_sec": grp["pre_prog_duration_sec"].median(),
                "median_pre_prog_pass_share": grp["pre_prog_pass_share"].median() * 100
                if "pre_prog_pass_share" in grp.columns else np.nan,
                # Progression efficiency
                "median_progressive_actions": grp["progressive_action_count"].median(),
                "mean_avg_progressive_x_gain": grp["avg_progressive_x_gain"].mean(),
                "median_passes_per_10m": grp["passes_per_10m"].median()
                if "passes_per_10m" in grp.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)
