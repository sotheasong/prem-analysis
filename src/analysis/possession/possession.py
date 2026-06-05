from __future__ import annotations

import numpy as np
import pandas as pd

'''
    This module contains functions to compute possession-level metrics.
    
'''


def _possession_stats(df: pd.DataFrame) -> dict:
    if df.empty or df["possession"].isna().all():
        return {
            "avg_possession_chain": np.nan,
            "median_possession_events": np.nan,
            "avg_possession_duration": np.nan,
            "passes_per_possession": np.nan,
            "possession_count": 0,
        }

    chains = df.groupby("possession")
    chain_sizes = chains.size()
    n_poss = df["possession"].nunique()
    passes = (df["type"] == "Pass").sum()

    duration_sum = np.nan
    if "duration" in df.columns:
        duration_sum = chains["duration"].sum().mean()

    return {
        "possession_count": n_poss,
        "avg_possession_chain": chain_sizes.mean(),
        "median_possession_events": chain_sizes.median(),
        "avg_possession_duration": duration_sum,
        "passes_per_possession": passes / n_poss if n_poss else np.nan,
    }


def add_possession_opponent(possessions: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    match_teams = (
        events_df.dropna(subset=["team"])
        .groupby("match_id")["team"]
        .unique()
        .to_dict()
    )

    def get_opponent(row):
        teams = match_teams.get(row["match_id"], [])
        opponents = [team for team in teams if team != row["possession_team"]]
        return opponents[0] if opponents else np.nan
    
    out = possessions.copy()
    out["opponent"] = out.apply(get_opponent, axis=1)
    return out


def _possession_stats_base(events_df: pd.DataFrame) -> pd.DataFrame:
    e = events_df.copy()

    grouped = e.groupby(
        ["match_id", "period", "possession", "possession_team"], dropna=False, sort=False,
    )
    
    base = grouped.agg(
        play_pattern=("play_pattern", "first")
    ).reset_index()

    base = base.rename(columns={
        "possession": "possession_id"
    })

    return base


def _possession_stats_time(events_df: pd.DataFrame) -> pd.DataFrame:
    e = events_df.copy()
    grouped = e.groupby(
        ["match_id", "period", "possession", "possession_team"], dropna=False, sort=False,
    )

    time_control = grouped.agg(
        start_minute=("minute", "first"),
        start_second=("second", "first"),
        end_minute=("minute", "last"),
        end_second=("second", "last"),
        event_count=("id", "count"),
        pass_count=("type", lambda s: (s == "Pass").sum()),
        carry_count=("type", lambda s: (s == "Carry").sum()),
        action_duration=("duration", "sum"),
        avg_event_duration=("duration", "mean"),
    ).reset_index()


    time_control["start_time"] = (
        time_control["start_minute"] * 60 + time_control["start_second"]
    )
    time_control["end_time"] = (
        time_control["end_minute"] * 60 + time_control["end_second"]
    )
    time_control["clock_duration"] = (
        time_control["end_time"] - time_control["start_time"]
    ).clip(lower=0)
    time_control["touch_count"] = (
        time_control["pass_count"] + time_control["carry_count"]
    )
    time_control["events_per_second"] = (
        time_control["event_count"] / time_control["clock_duration"].replace(0, pd.NA)
    )
    time_control["passes_per_second"] = (
        time_control["pass_count"] / time_control["clock_duration"].replace(0, pd.NA)
    )

    # directness ratio indicates whether the possession lead to forward progression or not
    # closer to 1 - direct forward movement
    # closer to 0 - recycled possession (sideways passes)
    # negative - backward progression (behind where progression started)
    
    time_control["directness_ratio"] = [
        _possession_directness_ratio(group)
        for _, group in grouped
    ]

    time_control = time_control.rename(columns={
        "possession": "possession_id",
    })
    return time_control


def _possession_stats_progression(events_df: pd.DataFrame) -> pd.DataFrame:
    e = events_df.copy()
    group_cols = ["match_id", "period", "possession", "possession_team"]
    rows = []
    for keys, group in e.groupby(group_cols, dropna=False, sort=False):
        metrics = _compute_possession_progression(group).to_dict()
        rows.append(dict(zip(group_cols, keys), **metrics))

    progression = pd.DataFrame(rows)
    return progression.rename(columns={"possession": "possession_id"})


def _possession_stats_pressure(events_df: pd.DataFrame) -> pd.DataFrame:
    e = events_df.copy()
    group_cols = ["match_id", "period", "possession", "possession_team"]
    rows = []
    for keys, group in e.groupby(group_cols, dropna=False, sort=False):
        metrics = _compute_possession_pressure(group).to_dict()
        rows.append(dict(zip(group_cols, keys), **metrics))

    pressure = pd.DataFrame(rows)
    return pressure.rename(columns={"possession": "possession_id"})


def _possession_stats_outcome(events_df: pd.DataFrame) -> pd.DataFrame:
    e = events_df.copy()
    group_cols = ["match_id", "period", "possession", "possession_team"]
    rows = []
    for keys, group in e.groupby(group_cols, dropna=False, sort=False):
        metrics = _compute_possession_outcome(group).to_dict()
        rows.append(dict(zip(group_cols, keys), **metrics))

    outcome = pd.DataFrame(rows)
    return outcome.rename(columns={"possession": "possession_id"})


def _compute_possession_progression(df: pd.DataFrame) -> pd.Series:
    movement = _possession_movement_events(df)
    locations = _possession_locations(df, movement)

    if locations.empty:
        return pd.Series(
            {
                "start_x": np.nan,
                "start_y": np.nan,
                "end_x": np.nan,
                "end_y": np.nan,
                "net_x_progression": np.nan,
                "total_x_progression": 0.0,
                "progressive_distance": 0.0,
                "max_x_reached": np.nan,
                "final_third_entry": False,
                "box_entry": False,
            }
        )

    start_x = locations["x"].iloc[0]
    start_y = locations["y"].iloc[0]
    end_x = locations["x"].iloc[-1]
    end_y = locations["y"].iloc[-1]

    if movement.empty:
        x_delta = pd.Series(dtype=float)
    else:
        x_delta = movement["end_x"] - movement["start_x"]

    return pd.Series(
        {
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "net_x_progression": end_x - start_x,
            "total_x_progression": x_delta[x_delta > 0].sum() if not x_delta.empty else 0.0,
            "progressive_distance": x_delta[x_delta >= 10].sum() if not x_delta.empty else 0.0,
            "max_x_reached": locations["x"].max(),
            "final_third_entry": locations["x"].ge(80).any(),
            "box_entry": (locations["x"].ge(102) & locations["y"].between(18, 62)).any(),
        }
    )


def _compute_possession_pressure(df: pd.DataFrame) -> pd.Series:
    team_events = _possession_team_events(df)
    opponent_events = _possession_opponent_events(df)

    under_pressure = _bool_series(team_events, "under_pressure")
    team_passes = team_events[team_events["type"].eq("Pass")]
    pressured_passes = team_passes[_bool_series(team_passes, "under_pressure")]

    if pressured_passes.empty:
        pressured_pass_completion = np.nan
    elif "pass_complete" in pressured_passes.columns:
        pressured_pass_completion = pressured_passes["pass_complete"].astype("boolean").fillna(False).mean()
    elif "pass_outcome" in pressured_passes.columns:
        pressured_pass_completion = pressured_passes["pass_outcome"].isna().mean()
    else:
        pressured_pass_completion = np.nan

    pressure_event_count = int(opponent_events["type"].eq("Pressure").sum()) if "type" in opponent_events else 0
    under_pressure_count = int(under_pressure.sum())
    pressure_denominator = len(team_events)

    return pd.Series(
        {
            "pressure_event_count": pressure_event_count,
            "under_pressure_count": under_pressure_count,
            "pressure_rate": under_pressure_count / pressure_denominator if pressure_denominator else np.nan,
            "pressured_pass_count": int(len(pressured_passes)),
            "pressured_pass_completion": pressured_pass_completion,
            "counterpress_faced": int(_bool_series(opponent_events, "counterpress").sum()),
        }
    )


def _compute_possession_outcome(df: pd.DataFrame) -> pd.Series:
    team_events = _possession_team_events(df)
    locations = _possession_locations(df, _possession_movement_events(df))

    shots = team_events[team_events["type"].eq("Shot")]
    key_pass_count = 0
    if "pass_shot_assist" in team_events.columns:
        key_pass_count += int(_bool_series(team_events, "pass_shot_assist").sum())
    if "pass_goal_assist" in team_events.columns:
        key_pass_count += int(_bool_series(team_events, "pass_goal_assist").sum())

    turnover_type = _infer_turnover_type(team_events)
    end_x = locations["x"].iloc[-1] if not locations.empty else np.nan
    end_y = locations["y"].iloc[-1] if not locations.empty else np.nan

    return pd.Series(
        {
            "shot_count": int(len(shots)),
            "xg_created": shots["shot_statsbomb_xg"].sum(skipna=True) if "shot_statsbomb_xg" in shots else 0.0,
            "key_pass_count": key_pass_count,
            "turnover_type": turnover_type,
            "ends_with_shot": bool(len(shots) and team_events.index[-1] in shots.index),
            "ends_in_final_third": bool(pd.notna(end_x) and end_x >= 80),
            "ends_in_box": bool(pd.notna(end_x) and pd.notna(end_y) and end_x >= 102 and 18 <= end_y <= 62),
        }
    )


def _possession_directness_ratio(df: pd.DataFrame) -> float:
    movement = _possession_movement_events(df)
    if movement.empty:
        return np.nan

    total_distance = np.hypot(
        movement["end_x"] - movement["start_x"],
        movement["end_y"] - movement["start_y"],
    ).sum()
    if total_distance == 0:
        return np.nan

    net_x_progression = movement["end_x"].iloc[-1] - movement["start_x"].iloc[0]
    return net_x_progression / total_distance


def _possession_team_events(df: pd.DataFrame) -> pd.DataFrame:
    if {"team", "possession_team"}.issubset(df.columns):
        return df[df["team"].eq(df["possession_team"])]
    return df


def _possession_opponent_events(df: pd.DataFrame) -> pd.DataFrame:
    if {"team", "possession_team"}.issubset(df.columns):
        return df[df["team"].notna() & ~df["team"].eq(df["possession_team"])]
    return df.iloc[0:0]


def _possession_movement_events(df: pd.DataFrame) -> pd.DataFrame:
    df = _possession_team_events(df)

    movement_parts = []

    if {"location_x", "location_y", "pass_end_x", "pass_end_y"}.issubset(df.columns):
        passes = df.loc[
            df["type"].eq("Pass"),
            ["location_x", "location_y", "pass_end_x", "pass_end_y"],
        ].rename(
            columns={
                "location_x": "start_x",
                "location_y": "start_y",
                "pass_end_x": "end_x",
                "pass_end_y": "end_y",
            }
        )
        movement_parts.append(passes)

    if {"location_x", "location_y", "carry_end_x", "carry_end_y"}.issubset(df.columns):
        carries = df.loc[
            df["type"].eq("Carry"),
            ["location_x", "location_y", "carry_end_x", "carry_end_y"],
        ].rename(
            columns={
                "location_x": "start_x",
                "location_y": "start_y",
                "carry_end_x": "end_x",
                "carry_end_y": "end_y",
            }
        )
        movement_parts.append(carries)

    if not movement_parts:
        return pd.DataFrame(columns=["start_x", "start_y", "end_x", "end_y"])

    movement = pd.concat(movement_parts, axis=0).sort_index()
    coord_cols = ["start_x", "start_y", "end_x", "end_y"]
    for col in coord_cols:
        movement[col] = pd.to_numeric(movement[col], errors="coerce")
    return movement.dropna(subset=coord_cols)


def _possession_locations(df: pd.DataFrame, movement: pd.DataFrame) -> pd.DataFrame:
    df = _possession_team_events(df)

    location_parts = []

    if {"location_x", "location_y"}.issubset(df.columns):
        starts = df[["location_x", "location_y"]].rename(
            columns={"location_x": "x", "location_y": "y"}
        )
        location_parts.append(starts)

    if not movement.empty:
        movement_ends = movement[["end_x", "end_y"]].rename(
            columns={"end_x": "x", "end_y": "y"}
        )
        location_parts.append(movement_ends)

    if not location_parts:
        return pd.DataFrame(columns=["x", "y"])

    locations = pd.concat(location_parts, axis=0).sort_index()
    for col in ["x", "y"]:
        locations[col] = pd.to_numeric(locations[col], errors="coerce")
    return locations.dropna(subset=["x", "y"])


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].map(_parse_bool)


def _infer_turnover_type(team_events: pd.DataFrame) -> str | float:
    if team_events.empty:
        return np.nan

    last = team_events.iloc[-1]
    event_type = last.get("type")
    if event_type == "Pass" and pd.notna(last.get("pass_outcome")):
        return f"Pass - {last.get('pass_outcome')}"
    if event_type == "Dribble" and pd.notna(last.get("dribble_outcome")):
        return f"Dribble - {last.get('dribble_outcome')}"
    if event_type == "Shot" and pd.notna(last.get("shot_outcome")):
        return f"Shot - {last.get('shot_outcome')}"
    if _parse_bool(last.get("out")):
        return "Out"
    return event_type


def _parse_bool(value) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


MERGE_COLS = ["match_id", "period", "possession_id", "possession_team"]


def build_possession_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """Merge possession-chain tables into one match-level summary."""
    base = add_possession_opponent(_possession_stats_base(events_df), events_df)
    return (
        base.merge(_possession_stats_time(events_df), on=MERGE_COLS, how="left")
        .merge(_possession_stats_progression(events_df), on=MERGE_COLS, how="left")
        .merge(_possession_stats_pressure(events_df), on=MERGE_COLS, how="left")
        .merge(_possession_stats_outcome(events_df), on=MERGE_COLS, how="left")
    )