"""An outside opinion on xG.

Every other check in Phase 3e compares the pipeline either to itself or to a
match result. This is the one place a *second model* gets a vote: Understat fits
its own expected-goals model from its own event data, so a disagreement here is
two models disagreeing rather than an arithmetic error.

That also bounds what agreement proves. Two xG models correlating at r ~ 0.9 is
what two competent models look like; it does not make either of them right.

Only seasons from 2014/15 onward can be checked, which is where Understat's
coverage starts.
"""

from __future__ import annotations

import pandas as pd

# Registry season -> (Understat league code, Understat season label).
UNDERSTAT_SEASONS = {
    "2015/2016": ("EPL", "2015"),
    "La Liga 2015/2016": ("La_Liga", "2015"),
    "Serie A 2015/2016": ("Serie_A", "2015"),
    "Ligue 1 2015/2016": ("Ligue_1", "2015"),
    "Barcelona 2014/2015": ("La_Liga", "2014"),
}

# Understat's club spelling -> StatsBomb's. Mostly dropped accents and dropped
# suffixes; none are guessable, all were derived by diffing the two name sets.
TEAM_ALIASES = {
    "EPL": {
        "Bournemouth": "AFC Bournemouth", "Leicester": "Leicester City",
        "Norwich": "Norwich City", "Stoke": "Stoke City",
        "Swansea": "Swansea City", "Tottenham": "Tottenham Hotspur",
        "West Ham": "West Ham United",
    },
    "La_Liga": {
        "Atletico Madrid": "Atlético Madrid", "Almeria": "Almería",
        "Cordoba": "Córdoba CF", "Deportivo La Coruna": "RC Deportivo La Coruña",
        "Levante": "Levante UD", "Malaga": "Málaga",
        "Sporting Gijon": "Sporting Gijón",
    },
    "Serie_A": {
        "Inter": "Inter Milan", "Roma": "AS Roma", "Verona": "Hellas Verona",
    },
    "Ligue_1": {
        "Caen": "Stade Malherbe Caen", "GFC Ajaccio": "Gazélec Ajaccio",
        "Marseille": "Olympique de Marseille", "Monaco": "AS Monaco",
        "Nice": "OGC Nice", "Paris Saint Germain": "Paris Saint-Germain",
        "Reims": "Stade de Reims", "SC Bastia": "Bastia",
        "Saint-Etienne": "Saint-Étienne",
    },
}

# Late kickoffs land on different calendar dates in the two sources, so the join
# tolerates a day. `goals_agree` in the report is what makes that safe.
DATE_TOLERANCE = pd.Timedelta("1D")


def understat_team_matches(league: str, season: str) -> pd.DataFrame:
    """One row per (team, match) with Understat's xG and goals scored."""
    from understatapi import UnderstatClient

    aliases = TEAM_ALIASES.get(league, {})
    with UnderstatClient() as client:
        teams = client.league(league=league).get_team_data(season=season)

    rows = [{"team": aliases.get(t["title"], t["title"]),
             "date": pd.to_datetime(h["date"][:10]),
             "understat_xg": float(h["xG"]),
             "understat_goals": int(h["scored"])}
            for t in teams.values() for h in t["history"]]
    return pd.DataFrame(rows)


def compare_xg(season: str) -> pd.DataFrame:
    """Join this project's per-match xG to Understat's for one season."""
    from src.config import FEATURES_DIR, PROCESSED_DIR
    from src.utils.constants import SEASONS

    if season not in UNDERSTAT_SEASONS:
        raise KeyError(f"Understat has no coverage for {season!r}; "
                       f"known: {sorted(UNDERSTAT_SEASONS)}")
    league, us_season = UNDERSTAT_SEASONS[season]
    suffix = SEASONS[season]["suffix"]

    matches = pd.read_csv(PROCESSED_DIR / f"matches_{suffix}.csv")
    goals = matches["match_score"].str.split(" - ", expand=True).astype(int)
    ours = pd.concat([
        matches.assign(team=matches["match_home_team"], goals=goals[0]),
        matches.assign(team=matches["match_away_team"], goals=goals[1]),
    ])[["match_id", "team", "match_kickoff", "goals"]]
    ours["date"] = pd.to_datetime(ours["match_kickoff"].str[:10])

    style = pd.read_csv(FEATURES_DIR / f"match_team_style_{suffix}.csv")
    ours = ours.merge(style[["match_id", "team", "xg"]], on=["match_id", "team"])

    merged = pd.merge_asof(
        ours.sort_values("date"),
        understat_team_matches(league, us_season).sort_values("date"),
        on="date", by="team", tolerance=DATE_TOLERANCE, direction="nearest")
    merged["goals_agree"] = merged["goals"] == merged["understat_goals"]
    return merged
