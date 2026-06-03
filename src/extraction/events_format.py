"""Normalize StatsBomb event data into an analysis-ready v4 events table."""

from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
import pandas as pd


CORE_EVENT_COLS = [
    "id",
    "index",
    "match_id",
    "period",
    "timestamp",
    "minute",
    "second",
    "type",
    "possession",
    "possession_team",
    "possession_team_id",
    "play_pattern",
    "team",
    "team_id",
    "player",
    "player_id",
    "position",
    "duration",
    "under_pressure",
    "counterpress",
    "off_camera",
    "out",
]

SPATIAL_DERIVED_COLS = [
    "location_x",
    "location_y",
    "pass_end_x",
    "pass_end_y",
    "carry_end_x",
    "carry_end_y",
    "shot_end_x",
    "shot_end_y",
    "shot_end_z",
]

EVENT_TYPE_COLS = [
    "bad_behaviour_card",
    "ball_receipt_outcome",
    "ball_recovery_offensive",
    "ball_recovery_recovery_failure",
    "block_deflection",
    "block_offensive",
    "block_save_block",
    "carry_end_location",
    "clearance_aerial_won",
    "clearance_body_part",
    "dribble_no_touch",
    "dribble_nutmeg",
    "dribble_outcome",
    "dribble_overrun",
    "duel_outcome",
    "duel_type",
    "foul_committed_advantage",
    "foul_committed_card",
    "foul_committed_offensive",
    "foul_committed_penalty",
    "foul_committed_type",
    "foul_won_advantage",
    "foul_won_defensive",
    "foul_won_penalty",
    "goalkeeper_body_part",
    "goalkeeper_end_location",
    "goalkeeper_outcome",
    "goalkeeper_position",
    "goalkeeper_technique",
    "goalkeeper_type",
    "interception_outcome",
    "miscontrol_aerial_won",
    "pass_angle",
    "pass_assisted_shot_id",
    "pass_body_part",
    "pass_cross",
    "pass_cut_back",
    "pass_deflected",
    "pass_end_location",
    "pass_goal_assist",
    "pass_height",
    "pass_length",
    "pass_miscommunication",
    "pass_no_touch",
    "pass_outcome",
    "pass_recipient",
    "pass_recipient_id",
    "pass_shot_assist",
    "pass_switch",
    "pass_technique",
    "pass_through_ball",
    "pass_type",
    "shot_aerial_won",
    "shot_body_part",
    "shot_deflected",
    "shot_end_location",
    "shot_first_time",
    "shot_follows_dribble",
    "shot_freeze_frame",
    "shot_key_pass_id",
    "shot_open_goal",
    "shot_outcome",
    "shot_statsbomb_xg",
    "shot_technique",
    "shot_type",
    "substitution_outcome",
    "substitution_outcome_id",
    "substitution_replacement",
    "substitution_replacement_id",
]

BOOL_COLS = ["under_pressure", "counterpress", "off_camera", "out"]

LOCATION_SPLITS = {
    "location": ("location_x", "location_y"),
    "pass_end_location": ("pass_end_x", "pass_end_y"),
    "carry_end_location": ("carry_end_x", "carry_end_y"),
    "shot_end_location": ("shot_end_x", "shot_end_y", "shot_end_z"),
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_coord(value: Any, index: int) -> float:
    if _is_missing(value):
        return np.nan
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) <= index:
            return np.nan
        try:
            return float(value[index])
        except (TypeError, ValueError):
            return np.nan
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return np.nan
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)) and len(parsed) > index:
            try:
                return float(parsed[index])
            except (TypeError, ValueError):
                return np.nan

        parts = [part for part in s.strip("[]").replace(",", " ").split() if part]
        if len(parts) > index:
            try:
                return float(parts[index])
            except ValueError:
                return np.nan
    return np.nan


def _coerce_bool(series: pd.Series) -> pd.Series:
    def parse_bool(value: Any) -> bool:
        if _is_missing(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    return series.map(parse_bool)


def _parse_related_events(value: Any) -> list[str] | None:
    if _is_missing(value):
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [s]
    return None


def _serialize_nested(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def split_location_column(df: pd.DataFrame, source: str, targets: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    if source not in out.columns:
        for target in targets:
            out[target] = np.nan
        return out

    for index, target in enumerate(targets):
        out[target] = out[source].map(lambda value, idx=index: _parse_coord(value, idx))
    return out


def normalize_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a flat table that preserves StatsBomb v4 event semantics for analysis."""
    out = df.copy()

    for source, targets in LOCATION_SPLITS.items():
        out = split_location_column(out, source, targets)

    for col in BOOL_COLS:
        out[col] = _coerce_bool(out[col]) if col in out.columns else False

    if "type" in out.columns:
        is_pass = out["type"] == "Pass"
        out["pass_complete"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        if "pass_outcome" in out.columns:
            out.loc[is_pass, "pass_complete"] = out.loc[is_pass, "pass_outcome"].isna()
        else:
            out.loc[is_pass, "pass_complete"] = True

    if "related_events" in out.columns:
        out["related_events"] = out["related_events"].map(_parse_related_events)

    for nested_col in ["tactics", "shot_freeze_frame"]:
        if nested_col in out.columns:
            out[nested_col] = out[nested_col].map(_serialize_nested)

    for col in [*CORE_EVENT_COLS, *SPATIAL_DERIVED_COLS, *EVENT_TYPE_COLS]:
        if col not in out.columns:
            out[col] = np.nan

    return out


def is_normalized(df: pd.DataFrame) -> bool:
    return "location_x" in df.columns and "pass_complete" in df.columns
