"""Playstyle metrics for StatsBomb event data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.possession import (
    _possession_stats,
    _possession_stats_outcome,
    _possession_stats_base,
    _possession_stats_pressure,
    _possession_stats_progression,
    _possession_stats_time,
    _possession_directness_ratio,
    _possession_movement_events,
    add_possession_opponent,
)


LONG_PASS_M = 30
ERA_2004 = "2003/2004"
ERA_2016 = "2015/2016"
ERA_A = ERA_2004
ERA_B = ERA_2016

EVENT_COLS = [
    # Core
    "id",
    "match_id",
    "team",
    "team_id",
    "player",
    "player_id",
    "type",
    "period",
    "minute",
    "second",
    "timestamp",
    "duration",

    # Possession
    "possession",
    "possession_team",
    "possession_team_id",
    "play_pattern",

    # Spatial (normalized from StatsBomb location arrays)
    "location_x",
    "location_y",
    "pass_end_x",
    "pass_end_y",
    "carry_end_x",
    "carry_end_y",

    # Passing
    "pass_length",
    "pass_angle",
    "pass_outcome",
    "pass_complete",
    "pass_cross",
    "pass_goal_assist",
    "pass_switch",
    "pass_through_ball",
    "pass_type",
    "pass_height",

    # Pressing / defense
    "under_pressure",
    "counterpress",
    "duel_type",
    "duel_outcome",
    "interception_outcome",
    "ball_recovery_offensive",

    # Physicality
    "foul_committed_type",
    "foul_won_defensive",

    # Chance creation
    "shot_statsbomb_xg",
    "shot_outcome",
    "shot_type",
    "pass_shot_assist",

    # Dribbling
    "dribble_outcome",
]

METRIC_GROUPS = {
    "Possession": [
    "match_id",
    "period",
    "possession_id",
    "possession_team",
    "opponent",
    "play_pattern",

    # Time / control
    "start_time",
    "end_time",
    "clock_duration",
    "action_duration",
    "event_count",
    "pass_count",
    "carry_count",
    "touch_count",

    # Tempo / deliberateness
    "events_per_second",
    "passes_per_second",
    "avg_event_duration",
    "directness_ratio",

    # Progression
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "net_x_progression",
    "total_x_progression",
    "progressive_distance",
    "max_x_reached",
    "final_third_entry",
    "box_entry",

    # Pressure
    "pressure_event_count",
    "under_pressure_count",
    "pressure_rate",
    "pressured_pass_count",
    "pressured_pass_completion",
    "counterpress_faced",

    # Outcome / danger
    "shot_count",
    "xg_created",
    "key_pass_count",
    "turnover_type",
    "ends_with_shot",
    "ends_in_final_third",
    "ends_in_box"
]

}

# INITIAL MATCH METRICS

def build_match_team(analysis_events: pd.DataFrame) -> pd.DataFrame:
    return (
        analysis_events.groupby(["season", "match_id", "team"], sort=False)
        .apply(compute_match_team_metrics, include_groups=False)
        .reset_index()
    )

def build_season_metrics(match_team: pd.DataFrame) -> pd.DataFrame:
    return match_team.groupby("season").mean(numeric_only=True).round(3).sort_index()


def compute_match_team_metrics(df: pd.DataFrame) -> pd.Series:
    passes = df[df["type"] == "Pass"]
    shots = df[df["type"] == "Shot"]
    if "pass_complete" in passes.columns:
        pass_complete = passes["pass_complete"].astype("boolean").fillna(False)
    else:
        pass_complete = passes["pass_outcome"].isna()
    if "pass_end_x" in passes.columns:
        end_x = passes["pass_end_x"]
    else:
        end_x = passes["pass_end_location"].map(parse_pitch_x)
    poss = _possession_stats(df)

    return pd.Series(
        {
            "events": len(df),
            "passes": len(passes),
            "pass_completion": pass_complete.mean() if len(passes) else np.nan,
            "avg_pass_length": passes["pass_length"].mean(),
            "long_pass_pct": (passes["pass_length"] > LONG_PASS_M).mean() if len(passes) else np.nan,
            "crosses": passes["pass_cross"].eq(True).sum(),
            "through_balls": passes["pass_through_ball"].eq(True).sum(),
            "pressure_on_pass_pct": passes["under_pressure"].eq(True).mean() if len(passes) else np.nan,
            "counterpresses": df["counterpress"].eq(True).sum(),
            "dribbles": (df["type"] == "Dribble").sum(),
            "duels": (df["type"] == "Duel").sum(),
            "fouls": (df["type"] == "Foul Committed").sum(),
            "offensive_recoveries": df["ball_recovery_offensive"].eq(True).sum(),
            "shots": len(shots),
            "xg": shots["shot_statsbomb_xg"].sum(skipna=True),
            "regular_play_pct": (df["play_pattern"] == "Regular Play").mean(),
            "passes_into_final_third": (end_x > 80).sum(),
            **poss,
            "avg_possession_chain": poss["avg_possession_chain"],
        }
    )


def parse_pitch_x(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip("[]").strip()
    if not s:
        return np.nan
    # StatsBomb CSV: "[61.4 43.6]" or "[60.2, 35.1]" or "[60.2,]"
    if "," in s:
        part = s.split(",")[0].strip()
    else:
        part = s.split()[0]
    try:
        return float(part)
    except ValueError:
        return np.nan


# MATCH METRICS

# POSSESSION METRICS

