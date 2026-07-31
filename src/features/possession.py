"""Possession-chain, team-comparison, and rhythm features.

Merged from the former possession/{possession,compare,rhythm}.py modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.constants import LONG_PASS_M
from src.features.passing import (
    build_team_passing_summary,
    filter_regular_play_passes as _filter_regular_play_passes,
)


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


# ---------------------------------------------------------------------------
# Score-state (game state at the start of each possession)
# ---------------------------------------------------------------------------
def _goal_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Goal-scoring moments, one row per goal, credited to the scoring team.

    A team's tally increments on a ``Shot`` finished as a ``Goal`` or on an
    ``Own Goal For`` (its mirror ``Own Goal Against`` is ignored to avoid
    double counting). Each goal carries the ``possession`` it occurred in, so
    goals can be ordered relative to possessions without timestamp matching.
    Reconciles exactly with official final scores on both open-data seasons.
    """
    is_shot_goal = events_df["type"].eq("Shot")
    if "shot_outcome" in events_df.columns:
        is_shot_goal &= events_df["shot_outcome"].eq("Goal")
    else:
        is_shot_goal &= False
    is_own_goal_for = events_df["type"].eq("Own Goal For")
    goals = events_df.loc[
        is_shot_goal | is_own_goal_for,
        ["match_id", "period", "possession", "team"],
    ]
    return goals.dropna(subset=["team"])


def compute_possession_score_state(events_df: pd.DataFrame) -> pd.DataFrame:
    """Score-state at the *start* of every possession, from the owner's view.

    Returns one row per ``(match_id, period, possession_id, possession_team)``
    with ``score_diff_before`` (signed goal difference, possession_team minus
    opponent) and ``score_state`` in ``{leading, level, trailing}``.

    Definition: goals scored *within* a possession are that possession's
    outcome, not its context, so a possession sees only goals from strictly
    earlier possessions (ordered by ``period`` then ``possession``). The first
    possession of every match is therefore ``level`` by construction, and the
    restart possession after a goal correctly reflects the new scoreline.
    """
    goals = _goal_events(events_df)
    # goals scored per (match, period, possession, scoring team)
    goal_counts = (
        goals.groupby(["match_id", "period", "possession", "team"])
        .size()
        .rename("goals")
        .reset_index()
    )

    rows = []
    owner = (
        events_df.dropna(subset=["possession_team"])
        .groupby(["match_id", "period", "possession"], sort=False)["possession_team"]
        .first()
        .reset_index()
    )
    goals_by_match = {
        mid: grp for mid, grp in goal_counts.groupby("match_id")
    }
    for match_id, mposs in owner.groupby("match_id", sort=False):
        mposs = mposs.sort_values(["period", "possession"])
        teams = list(mposs["possession_team"].unique())
        running = {t: 0 for t in teams}
        mgoals = goals_by_match.get(match_id)
        # (period, possession) -> {team: goals scored in it}
        in_poss = {}
        if mgoals is not None:
            for _, g in mgoals.iterrows():
                in_poss.setdefault((g["period"], g["possession"]), {})[g["team"]] = int(g["goals"])
                running.setdefault(g["team"], 0)  # own-goal-for team may differ
        for _, r in mposs.iterrows():
            pt = r["possession_team"]
            gf = running.get(pt, 0)
            ga = sum(v for t, v in running.items() if t != pt)
            diff = gf - ga
            rows.append(
                {
                    "match_id": match_id,
                    "period": r["period"],
                    "possession_id": r["possession"],
                    "possession_team": pt,
                    "score_diff_before": diff,
                    "score_state": (
                        "leading" if diff > 0 else "trailing" if diff < 0 else "level"
                    ),
                }
            )
            for t, n in in_poss.get((r["period"], r["possession"]), {}).items():
                running[t] = running.get(t, 0) + n
    return pd.DataFrame(rows, columns=MERGE_COLS + ["score_diff_before", "score_state"])


def build_possession_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """Merge possession-chain tables into one match-level summary."""
    base = add_possession_opponent(_possession_stats_base(events_df), events_df)
    return (
        base.merge(_possession_stats_time(events_df), on=MERGE_COLS, how="left")
        .merge(_possession_stats_progression(events_df), on=MERGE_COLS, how="left")
        .merge(_possession_stats_pressure(events_df), on=MERGE_COLS, how="left")
        .merge(_possession_stats_outcome(events_df), on=MERGE_COLS, how="left")
        .merge(compute_possession_score_state(events_df), on=MERGE_COLS, how="left")
    )

"""Team-level possession and event summaries for match comparisons."""


import numpy as np
import pandas as pd


# StatsBomb pitch thirds (attacking direction = increasing x)
DEF_THIRD_X = 40.0
ATT_THIRD_X = 80.0

POSSESSION_SCENARIOS = [
    {"label": "All possessions", "play_pattern": None, "min_passes": None},
    {"label": "Regular Play", "play_pattern": "Regular Play", "min_passes": None},
    {"label": "Regular Play (4+ passes)", "play_pattern": "Regular Play", "min_passes": 4},
    {"label": "Regular Play (6+ passes)", "play_pattern": "Regular Play", "min_passes": 6},
]


def pitch_third(x: float) -> str | float:
    if pd.isna(x):
        return np.nan
    if x < DEF_THIRD_X:
        return "defensive"
    if x < ATT_THIRD_X:
        return "midfield"
    return "attacking"


def filter_possessions(
    possession_summary: pd.DataFrame,
    team: str,
    *,
    play_pattern: str | None = None,
    min_passes: int | None = None,
) -> pd.DataFrame:
    out = possession_summary[possession_summary["possession_team"] == team].copy()
    if play_pattern is not None:
        out = out[out["play_pattern"] == play_pattern]
    if min_passes is not None:
        out = out[out["pass_count"] >= min_passes]
    return out


def summarize_possession_subset(poss: pd.DataFrame) -> pd.Series:
    if poss.empty:
        return pd.Series(dtype=float)

    pressured_completion = poss["pressured_pass_completion"].mean()

    return pd.Series(
        {
            "possessions": len(poss),
            "median_events": poss["event_count"].median(),
            "median_passes": poss["pass_count"].median(),
            "median_duration_sec": poss["clock_duration"].median(),
            "median_events_per_sec": poss["events_per_second"].median(),
            "median_directness": poss["directness_ratio"].median(),
            "mean_directness": poss["directness_ratio"].mean(),
            "median_net_x": poss["net_x_progression"].median(),
            "final_third_entry_pct": poss["final_third_entry"].mean() * 100,
            "box_entry_pct": poss["box_entry"].mean() * 100,
            "ends_final_third_pct": poss["ends_in_final_third"].mean() * 100,
            "ends_box_pct": poss["ends_in_box"].mean() * 100,
            "pressure_rate_pct": poss["pressure_rate"].mean() * 100,
            "pressured_pass_completion_pct": pressured_completion * 100 if pd.notna(pressured_completion) else np.nan,
            "ends_with_shot_pct": poss["ends_with_shot"].mean() * 100,
            "poss_with_shot_pct": (poss["shot_count"] > 0).mean() * 100,
            "total_xg": poss["xg_created"].sum(),
            "mean_start_x": poss["start_x"].mean(),
            "mean_end_x": poss["end_x"].mean(),
        }
    )


def summarize_team_events(events: pd.DataFrame, team: str) -> pd.Series:
    td = events[events["team"] == team]
    passes = td[td["type"] == "Pass"]
    shots = td[td["type"] == "Shot"]
    carries = td[td["type"] == "Carry"]
    dribbles = td[td["type"] == "Dribble"]

    if "pass_complete" in passes.columns:
        complete = passes["pass_complete"].astype("boolean").fillna(False)
    else:
        complete = passes["pass_outcome"].isna()

    pass_height = passes["pass_height"].value_counts(normalize=True, dropna=False) * 100

    return pd.Series(
        {
            "events": len(td),
            "passes": len(passes),
            "pass_completion_pct": complete.mean() * 100 if len(passes) else np.nan,
            "avg_pass_length_m": passes["pass_length"].mean(),
            "long_pass_pct": (passes["pass_length"] > LONG_PASS_M).mean() * 100 if len(passes) else np.nan,
            "crosses": int(passes["pass_cross"].eq(True).sum()),
            "through_balls": int(passes["pass_through_ball"].eq(True).sum()),
            "passes_into_final_third": int((passes["pass_end_x"] > ATT_THIRD_X).sum()),
            "pressure_on_pass_pct": passes["under_pressure"].eq(True).mean() * 100 if len(passes) else np.nan,
            "ground_pass_pct": pass_height.get("Ground Pass", np.nan),
            "high_pass_pct": pass_height.get("High Pass", np.nan),
            "carries": int(len(carries)),
            "dribbles": int(len(dribbles)),
            "dribbles_complete": int(dribbles["dribble_outcome"].eq("Complete").sum()),
            "counterpresses": int(td["counterpress"].eq(True).sum()),
            "duels": int((td["type"] == "Duel").sum()),
            "fouls_committed": int((td["type"] == "Foul Committed").sum()),
            "shots": int(len(shots)),
            "xg": shots["shot_statsbomb_xg"].sum(skipna=True),
            "goals": int(shots["shot_outcome"].eq("Goal").sum()),
        }
    )


def build_possession_style_table(
    possession_summary: pd.DataFrame,
    teams: list[str],
    scenarios: list[dict] | None = None,
) -> pd.DataFrame:
    scenarios = scenarios or POSSESSION_SCENARIOS
    rows = []
    for scenario in scenarios:
        for team in teams:
            poss = filter_possessions(
                possession_summary,
                team,
                play_pattern=scenario.get("play_pattern"),
                min_passes=scenario.get("min_passes"),
            )
            summary = summarize_possession_subset(poss)
            rows.append(
                {
                    "scenario": scenario["label"],
                    "team": team,
                    **summary.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def build_event_style_table(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"team": team, **summarize_team_events(events, team).to_dict()} for team in teams
    )


def build_passing_style_table(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Build a team-level Regular Play passing-intent table."""
    return pd.DataFrame(
        {"team": team, **build_team_passing_summary(events, team).to_dict()}
        for team in teams
    )


def filter_regular_play_passes(events: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    """Expose Regular Play pass filtering for notebooks."""
    return _filter_regular_play_passes(events, team)


def regular_play_start_zones(possession_summary: pd.DataFrame, team: str) -> pd.Series:
    poss = filter_possessions(
        possession_summary, team, play_pattern="Regular Play"
    )
    zones = poss["start_x"].map(pitch_third)
    return zones.value_counts().rename_axis("start_zone").reset_index(name="count")


def top_turnover_types(
    possession_summary: pd.DataFrame, team: str, *, n: int = 8
) -> pd.Series:
    poss = possession_summary[possession_summary["possession_team"] == team]
    return poss["turnover_type"].value_counts().head(n)


def longest_possessions(
    possession_summary: pd.DataFrame,
    team: str,
    *,
    n: int = 5,
    sort_by: str = "pass_count",
) -> pd.DataFrame:
    cols = [
        "possession_id",
        "play_pattern",
        "pass_count",
        "event_count",
        "clock_duration",
        "directness_ratio",
        "net_x_progression",
        "final_third_entry",
        "xg_created",
    ]
    poss = possession_summary[possession_summary["possession_team"] == team]
    return poss.nlargest(n, sort_by)[cols]


def pass_count_distribution(
    possession_summary: pd.DataFrame,
    team: str,
    *,
    play_pattern: str = "Regular Play",
) -> pd.Series:
    poss = filter_possessions(possession_summary, team, play_pattern=play_pattern)
    bins = [0, 1, 2, 3, 5, 8, np.inf]
    labels = ["0", "1", "2", "3-4", "5-7", "8+"]
    return (
        pd.cut(poss["pass_count"], bins=bins, labels=labels, right=False)
        .value_counts()
        .sort_index()
    )


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
