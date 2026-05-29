"""Playstyle metrics for StatsBomb event data."""

from __future__ import annotations

import numpy as np
import pandas as pd

LONG_PASS_M = 30

EVENT_COLS = [
    "id",
    "match_id",
    "team",
    "type",
    "play_pattern",
    "possession",
    "duration",
    "minute",
    "second",
    "under_pressure",
    "counterpress",
    "pass_length",
    "pass_outcome",
    "pass_cross",
    "pass_through_ball",
    "pass_end_location",
    "location",
    "player",
    "duel_type",
    "ball_recovery_offensive",
    "dribble_outcome",
    "shot_statsbomb_xg",
]

ERA_A = "2003/2004"
ERA_B = "2015/2016"

METRIC_GROUPS = {
    "Passes & possession (headline)": [
        "passes",
        "avg_possession_chain",
        "median_possession_events",
        "avg_possession_duration",
        "passes_per_possession",
    ],
    "Structure & possession": [
        "avg_pass_length",
        "long_pass_pct",
        "pass_completion",
        "regular_play_pct",
        "passes_into_final_third",
    ],
    "Pressing & organization": [
        "pressure_on_pass_pct",
        "counterpresses",
        "offensive_recoveries",
    ],
    "Direct / individual play": [
        "dribbles",
        "crosses",
        "through_balls",
    ],
    "Physical confrontations": [
        "duels",
        "fouls",
    ],
    "Chance creation": [
        "shots",
        "xg",
    ],
}


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


def compute_match_team_metrics(df: pd.DataFrame) -> pd.Series:
    passes = df[df["type"] == "Pass"]
    shots = df[df["type"] == "Shot"]
    pass_complete = passes["pass_outcome"] == "Complete"
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
            # alias for comparison table
            "avg_possession_chain": poss["avg_possession_chain"],
        }
    )


def build_match_team(analysis_events: pd.DataFrame) -> pd.DataFrame:
    return (
        analysis_events.groupby(["season", "match_id", "team"], sort=False)
        .apply(compute_match_team_metrics, include_groups=False)
        .reset_index()
    )


def build_match_passes(match_team: pd.DataFrame) -> pd.DataFrame:
    return (
        match_team.groupby(["season", "match_id"], as_index=False)
        .agg(
            team_passes_mean=("passes", "mean"),
            team_passes_median=("passes", "median"),
            total_passes=("passes", "sum"),
        )
    )


def passes_season_summary(match_passes: pd.DataFrame) -> pd.DataFrame:
    return (
        match_passes.groupby("season")
        .agg(
            matches=("match_id", "nunique"),
            mean_total_passes_per_match=("total_passes", "mean"),
            median_total_passes_per_match=("total_passes", "median"),
            mean_team_passes_per_match=("team_passes_mean", "mean"),
            median_team_passes_per_match=("team_passes_median", "median"),
        )
        .round(1)
    )


def build_possession_chains(analysis_events: pd.DataFrame) -> pd.DataFrame:
    return (
        analysis_events.groupby(["season", "match_id", "possession"], sort=False)
        .agg(
            event_count=("type", "size"),
            pass_count=("type", lambda s: (s == "Pass").sum()),
            duration_sec=("duration", "sum"),
        )
        .reset_index()
    )


def possession_season_summary(possession_chains: pd.DataFrame) -> pd.DataFrame:
    return (
        possession_chains.groupby("season")
        .agg(
            mean_events_per_possession=("event_count", "mean"),
            median_events_per_possession=("event_count", "median"),
            mean_passes_per_possession=("pass_count", "mean"),
            median_passes_per_possession=("pass_count", "median"),
            mean_duration_sec=("duration_sec", "mean"),
            median_duration_sec=("duration_sec", "median"),
        )
        .round(3)
    )


def build_season_metrics(match_team: pd.DataFrame) -> pd.DataFrame:
    return match_team.groupby("season").mean(numeric_only=True).round(3).sort_index()


def build_era_comparison(season_metrics: pd.DataFrame) -> pd.DataFrame:
    if ERA_A not in season_metrics.index or ERA_B not in season_metrics.index:
        raise ValueError(f"Expected seasons {ERA_A} and {ERA_B}: {season_metrics.index.tolist()}")

    early = season_metrics.loc[ERA_A]
    late = season_metrics.loc[ERA_B]
    rows = []
    for group, metrics in METRIC_GROUPS.items():
        for metric in metrics:
            if metric not in season_metrics.columns:
                continue
            e, l = early[metric], late[metric]
            pct = ((l - e) / e * 100) if e not in (0, np.nan) and pd.notna(e) else np.nan
            rows.append(
                {
                    "theme": group,
                    "metric": metric,
                    ERA_A: e,
                    ERA_B: l,
                    "pct_change": round(pct, 1) if pd.notna(pct) else np.nan,
                }
            )
    return pd.DataFrame(rows).set_index(["theme", "metric"])


def headline_summary(
    passes_summary: pd.DataFrame,
    possession_summary: pd.DataFrame,
) -> str:
    lines = ["### Headline answers (team-match / possession-chain averages)\n"]
    if ERA_A in passes_summary.index and ERA_B in passes_summary.index:
        p_early = passes_summary.loc[ERA_A, "mean_total_passes_per_match"]
        p_late = passes_summary.loc[ERA_B, "mean_total_passes_per_match"]
        pct = (p_late - p_early) / p_early * 100
        lines.append(
            f"- **Passes per match (both teams):** {ERA_A} averaged **{p_early:.0f}** total passes per game; "
            f"{ERA_B} averaged **{p_late:.0f}** ({pct:+.1f}%)."
        )
    if ERA_A in possession_summary.index and ERA_B in possession_summary.index:
        e_early = possession_summary.loc[ERA_A, "mean_events_per_possession"]
        e_late = possession_summary.loc[ERA_B, "mean_events_per_possession"]
        pct_e = (e_late - e_early) / e_early * 100
        d_early = possession_summary.loc[ERA_A, "mean_duration_sec"]
        d_late = possession_summary.loc[ERA_B, "mean_duration_sec"]
        pct_d = (d_late - d_early) / d_early * 100 if pd.notna(d_early) and d_early else np.nan
        lines.append(
            f"- **Possession length (events per possession):** {ERA_A} **{e_early:.2f}** vs {ERA_B} **{e_late:.2f}** ({pct_e:+.1f}%)."
        )
        if pd.notna(d_early) and pd.notna(d_late):
            lines.append(
                f"- **Possession duration (seconds, summed event durations):** {ERA_A} **{d_early:.2f}s** vs "
                f"{ERA_B} **{d_late:.2f}s** ({pct_d:+.1f}%)."
            )
    lines.append(
        "\n*2003/04 sample: 38 matches; 2015/16: 380. Prefer distribution plots over means alone.*"
    )
    return "\n".join(lines)
