"""Batch feature runner for the scaling phase.

Turns a whole season of extracted events into two tidy backbone tables:

* ``match_team_style`` — one row per ``(match_id, team)``; the per-match style
  fingerprint plus context covariates (opponent strength, venue, result, red
  cards). This is the evolution backbone and the future modeling design matrix.
* ``possession_chains`` — ``build_possession_summary`` concatenated across the
  season and tagged with context; the backbone for distributional questions.

Opponent strength is resolved per season via the ``SEASONS`` registry: real
standings are computed when the season's dataset is the full league, and a
static historical table is used when coverage is partial (e.g. the 2003/04
open data, which is Arsenal-only). See ``src/utils/constants.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.utils.constants import (
    CARD_COLS,
    EVENT_COLS,
    RED_CARD_VALUES,
    SEASONS,
)
from src.features.passing import PASS_ANALYSIS_COLS, build_team_passing_summary
from src.features.possession import (
    build_possession_summary,
    build_rhythm_summary,
    filter_possessions,
    summarize_possession_subset,
    summarize_rhythm_by_team,
    summarize_team_events,
)

# ---------------------------------------------------------------------------
# Column taxonomy of the style table
# ---------------------------------------------------------------------------
# ``build_match_team_style`` writes three kinds of column alongside the style
# blocks. Naming them here, next to the code that creates them, is what lets
# downstream analysis ask for "the engineered features" without maintaining its
# own drop-list (see ``style_columns``).

# Row identity and labels. Never predictors, never responses.
KEY_COLS = frozenset({
    "season", "match_id", "date", "match_week", "team", "team_id", "opponent",
})

# Exogenous covariates: fixed before kickoff, so they are safe as predictors.
# ``home`` is the 0/1 encoding of ``venue`` that analysis code derives.
CONTEXT_COLS = frozenset({
    "venue", "home", "opp_final_points", "opp_final_position",
})

# Outcome-derived. Any of these leaks the result into a model of style, so they
# are excluded from the feature universe by default. The score-state shares
# belong here despite looking like style: they are a function of goals scored.
OUTCOME_COLS = frozenset({
    "goals_for", "goals_against", "goals", "result",
    "red_cards_for", "red_cards_against",
    "pct_poss_leading", "pct_time_leading",
    "pct_poss_level", "pct_time_level",
    "pct_poss_trailing", "pct_time_trailing",
})

NON_FEATURE_COLS = KEY_COLS | CONTEXT_COLS | OUTCOME_COLS


def style_columns(df: pd.DataFrame) -> list[str]:
    """The engineered style features of a ``match_team_style`` frame.

    Every numeric column that is not a key, an exogenous covariate or an
    outcome. Non-numeric style columns (``pi_top_player_name``, ``pi_top_edge``)
    are excluded too: they label a match rather than measure it. Column order is
    preserved.
    """
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


# Count columns (from ``summarize_team_events``) reported as per-100-passes so
# matches of different length/game-state are comparable.
_COUNT_COLS_PER100 = (
    "crosses",
    "through_balls",
    "passes_into_final_third",
    "carries",
    "dribbles",
    "dribbles_complete",
    "counterpresses",
    "duels",
    "fouls_committed",
    "shots",
)


# ---------------------------------------------------------------------------
# Opponent strength (per-season strategy)
# ---------------------------------------------------------------------------
def compute_season_standings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Final league table computed from a *full-league* season's results.

    Returns a frame indexed by team with ``points``, ``goal_diff`` and
    ``position``. Only valid when every fixture is present (``coverage=="full"``).
    """
    rows = []
    for _, m in matches_df.iterrows():
        home, away = m["match_home_team"], m["match_away_team"]
        home_goals, away_goals = (int(x) for x in m["match_score"].split(" - "))
        home_pts, away_pts = (
            (3, 0) if home_goals > away_goals
            else (0, 3) if away_goals > home_goals
            else (1, 1)
        )
        rows.append({"team": home, "pts": home_pts, "gf": home_goals, "ga": away_goals})
        rows.append({"team": away, "pts": away_pts, "gf": away_goals, "ga": home_goals})

    table = (
        pd.DataFrame(rows)
        .groupby("team")
        .agg(points=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
        .reset_index()
    )
    table["goal_diff"] = table["gf"] - table["ga"]
    table = table.sort_values(
        ["points", "goal_diff", "gf"], ascending=False
    ).reset_index(drop=True)
    table["position"] = table.index + 1
    return table.set_index("team")


def resolve_opponent_strength(season: str, matches_df: pd.DataFrame) -> dict:
    """Return ``{team: (final_position, final_points)}`` for the season.

    Static historical table for partial-coverage seasons; computed standings
    for full-league seasons. The uniform dict interface means downstream code
    never needs to know which strategy was used.
    """
    static = SEASONS[season]["strength"]
    if static is not None:
        return static
    standings = compute_season_standings(matches_df)
    return {team: (int(r.position), int(r.points)) for team, r in standings.iterrows()}


# ---------------------------------------------------------------------------
# Per-(match, team) context block
# ---------------------------------------------------------------------------
def match_context(match_row: pd.Series, team: str, strength: dict) -> dict:
    """Non-style covariates for one ``(match, team)`` row.

    ``match_row`` is a row of ``matches_<suffix>.csv`` (with a ``season``
    column added); ``strength`` is the output of ``resolve_opponent_strength``.
    Score is parsed home-first (StatsBomb convention: ``"<home> - <away>"``).
    """
    home, away = match_row["match_home_team"], match_row["match_away_team"]
    home_goals, away_goals = (int(x) for x in match_row["match_score"].split(" - "))

    if team == home:
        opponent, venue, goals_for, goals_against = away, "home", home_goals, away_goals
    else:
        opponent, venue, goals_for, goals_against = home, "away", away_goals, home_goals

    result = "W" if goals_for > goals_against else "L" if goals_for < goals_against else "D"
    opp_position, opp_points = strength[opponent]

    return {
        "season": match_row["season"],
        "match_id": match_row["match_id"],
        "date": match_row["match_kickoff"],
        "match_week": match_row["match_week"],
        "team": team,
        "opponent": opponent,
        "venue": venue,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "result": result,
        "opp_final_points": opp_points,
        "opp_final_position": opp_position,
    }


_SCORE_STATES = ("leading", "level", "trailing")


def score_state_shares(team_poss: pd.DataFrame) -> dict:
    """Count- and duration-weighted share (%) of a team's possessions per state.

    ``pct_poss_*`` weights each possession equally; ``pct_time_*`` weights by
    ``clock_duration`` (share of the team's on-ball *time* in each state). The
    trailing shares double as the "chasing" game-state covariate. Empty input
    (a team with no possessions) yields NaNs, not spurious zeros.
    """
    out = {}
    n = len(team_poss)
    total_dur = team_poss["clock_duration"].sum() if n else 0
    state = team_poss.get("score_state")
    for s in _SCORE_STATES:
        if n == 0 or state is None:
            out[f"pct_poss_{s}"] = np.nan
            out[f"pct_time_{s}"] = np.nan
            continue
        mask = state.eq(s)
        out[f"pct_poss_{s}"] = mask.mean() * 100
        out[f"pct_time_{s}"] = (
            team_poss.loc[mask, "clock_duration"].sum() / total_dur * 100
            if total_dur else np.nan
        )
    return out


def count_red_cards(events_df: pd.DataFrame, team: str) -> tuple[int, int]:
    """(red_cards_for, red_cards_against) — reds are Red Card or Second Yellow."""
    present = [c for c in CARD_COLS if c in events_df.columns]
    if not present:
        return 0, 0
    red = pd.Series(False, index=events_df.index)
    for col in present:
        red |= events_df[col].isin(RED_CARD_VALUES)
    carded_teams = events_df.loc[red, "team"]
    return int((carded_teams == team).sum()), int((carded_teams != team).sum())


# ---------------------------------------------------------------------------
# Style-table assembly
# ---------------------------------------------------------------------------
def build_match_team_style(
    events_df: pd.DataFrame, match_row: pd.Series, strength: dict
) -> pd.DataFrame:
    """Two rows (home team, away team) of context + style for one match.

    Feature blocks are namespaced (``pi_``/``poss_``/``rp6_``/``rhythm_``) so
    shared key names across families don't clobber each other on merge.
    """
    home, away = match_row["match_home_team"], match_row["match_away_team"]
    teams = [home, away]

    poss = build_possession_summary(events_df)
    rhythm = summarize_rhythm_by_team(build_rhythm_summary(events_df, poss, teams))
    rhythm = rhythm.set_index("team") if "team" in rhythm.columns else rhythm

    rows = []
    for team in teams:
        reds_for, reds_against = count_red_cards(events_df, team)

        event_style = summarize_team_events(events_df, team)
        passing = build_team_passing_summary(events_df, team).add_prefix("pi_")
        poss_all = summarize_possession_subset(
            filter_possessions(poss, team)
        ).add_prefix("poss_")
        poss_rp6 = summarize_possession_subset(
            filter_possessions(poss, team, play_pattern="Regular Play", min_passes=6)
        ).add_prefix("rp6_")
        rhythm_row = (
            rhythm.loc[team].add_prefix("rhythm_")
            if team in rhythm.index
            else pd.Series(dtype=float)
        )

        rows.append(
            {
                **match_context(match_row, team, strength),
                "red_cards_for": reds_for,
                "red_cards_against": reds_against,
                **score_state_shares(filter_possessions(poss, team)),
                **event_style.to_dict(),
                **passing.to_dict(),
                **poss_all.to_dict(),
                **poss_rp6.to_dict(),
                **rhythm_row.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def build_season_style_table(
    events_df: pd.DataFrame, matches_df: pd.DataFrame, strength: dict
) -> pd.DataFrame:
    """``match_team_style`` for a season: one row per (match, team)."""
    frames = [
        build_match_team_style(
            events_df[events_df["match_id"] == match_row["match_id"]],
            match_row,
            strength,
        )
        for _, match_row in matches_df.iterrows()
    ]
    return pd.concat(frames, ignore_index=True)


def normalize_style_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-100-pass and per-possession rate columns for comparability.

    Raw counts are kept; ``<count>_per100pass`` columns are added alongside.
    """
    out = df.copy()
    passes = out["passes"].replace(0, np.nan)
    for col in _COUNT_COLS_PER100:
        if col in out.columns:
            out[f"{col}_per100pass"] = out[col] / passes * 100
    if "poss_possessions" in out.columns:
        out["xg_per_possession"] = out["xg"] / out["poss_possessions"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Possession-chains table (distributional backbone)
# ---------------------------------------------------------------------------
def _context_frame(matches_df: pd.DataFrame, strength: dict) -> pd.DataFrame:
    """One context row per (match_id, team) — used to tag possession chains."""
    rows = [
        match_context(match_row, team, strength)
        for _, match_row in matches_df.iterrows()
        for team in (match_row["match_home_team"], match_row["match_away_team"])
    ]
    keep = [
        "season", "match_id", "match_week", "date", "team",
        "opponent", "venue", "opp_final_points", "opp_final_position",
    ]
    return pd.DataFrame(rows)[keep]


def build_season_possession_chains(
    events_df: pd.DataFrame, matches_df: pd.DataFrame, strength: dict
) -> pd.DataFrame:
    """``build_possession_summary`` across the season, tagged with context.

    ``build_possession_summary`` is already multi-match safe (it groups by
    ``match_id``), so it is run once on the whole season, then context is merged
    on ``(match_id, possession_team)``.
    """
    chains = build_possession_summary(events_df)
    context = _context_frame(matches_df, strength)
    merged = chains.merge(
        context,
        left_on=["match_id", "possession_team"],
        right_on=["match_id", "team"],
        how="left",
    )
    return merged.drop(columns=["team"])


# ---------------------------------------------------------------------------
# Convenience: one-line season run (registry-driven)
# ---------------------------------------------------------------------------
def load_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (events, matches) for a season using the ``SEASONS`` registry."""
    suffix = SEASONS[season]["suffix"]
    wanted = sorted(set(EVENT_COLS) | set(PASS_ANALYSIS_COLS) | set(CARD_COLS))

    events_path = PROCESSED_DIR / f"events_{suffix}.parquet"
    import pyarrow.parquet as pq  # local import: only needed here

    available = set(pq.ParquetFile(events_path).schema.names)
    cols = [c for c in wanted if c in available]
    events = pd.read_parquet(events_path, columns=cols)

    matches = pd.read_csv(PROCESSED_DIR / f"matches_{suffix}.csv")
    matches["season"] = season
    return events, matches


def build_season_features(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end for one season: returns (match_team_style, possession_chains).

    Registry-driven, so swapping seasons is a one-line change. Persisting the
    frames to ``data/features/`` is left to the driving notebook.
    """
    events, matches = load_season(season)
    strength = resolve_opponent_strength(season, matches)
    style = normalize_style_table(build_season_style_table(events, matches, strength))
    chains = build_season_possession_chains(events, matches, strength)
    return style, chains
