"""Team-level possession and event summaries for match comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.metrics import LONG_PASS_M
from src.analysis.passing.passing import (
    build_team_passing_summary,
    filter_regular_play_passes as _filter_regular_play_passes,
)

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
