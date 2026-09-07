"""Which seasons this pipeline can actually use.

Phase 3a learned that a season label promises nothing: 2003/04 carries twenty
club names and is one club's fixture list. Phase 3e found the same shape in
Bundesliga 2015/16, which reads as a season and is 34 Leverkusen matches. The
cost of missing that is a full feature build against data that cannot support a
league field.

So coverage is computed from the match list. Everything here is a pure function
of a matches frame; the network lives in one thin adapter at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The event types src/features/ actually reads. Coverage is not usability: a
# season can hold a full schedule and still lack a type the pipeline needs, and
# an old collection is exactly where that happens. tests/test_event_types.py
# reads the feature source and fails if this list drifts from it.
REQUIRED_EVENT_TYPES = frozenset({
    "Ball Recovery", "Block", "Carry", "Clearance", "Dispossessed", "Dribble",
    "Duel", "Foul Committed", "Interception", "Miscontrol", "Own Goal For",
    "Pass", "Pressure", "Shot", "Starting XI", "Tactical Shift",
})


# The subset a *single match* is expected to carry, so a sample can be judged
# on it. Measured over 7 PL 2015/16 matches: these fourteen appeared in all
# seven, while "Tactical Shift" appeared in five and "Own Goal For" in two.
# Those two are real dependencies but occasional, so their absence from a
# sample is uninformative rather than disqualifying.
PER_MATCH_EVENT_TYPES = REQUIRED_EVENT_TYPES - {"Own Goal For", "Tactical Shift"}


@dataclass(frozen=True)
class CoveragePolicy:
    """The judgment calls in the classification, in one place.

    ``min_series_matches`` matches one club must have played before its fixture
                    list counts as a season rather than a handful of games.
                    20 sits in the gap between La Liga 2005/06 (17 Barcelona
                    matches, too thin to profile) and 2007/08 (27, the earliest
                    season Phase 3e builds).
    ``min_teams_full``  clubs a competition needs before it can be a league at
                    all. The big five run 18 or 20.
    ``schedule_ratio``  fraction of a complete double round robin that must be
                    present. Ligue 1 2015/16 is 377 of 380, so this cannot
                    demand the whole schedule.
    """

    min_series_matches: int = 20
    min_teams_full: int = 18
    schedule_ratio: float = 0.90


def _appearances(matches: pd.DataFrame) -> pd.Series:
    """Matches played, per club."""
    both = pd.concat([matches["home_team"], matches["away_team"]])
    return both.value_counts()


def classify_coverage(matches: pd.DataFrame,
                     policy: CoveragePolicy = CoveragePolicy()) -> str:
    """Verdict for one season's match list."""
    played = _appearances(matches)
    n_matches = len(matches)
    if played.max() == n_matches:
        if n_matches < policy.min_series_matches:
            return "sparse"
        return "single_club"

    n_teams = len(played)
    complete = n_teams * (n_teams - 1)
    if (n_teams >= policy.min_teams_full
            and n_matches >= policy.schedule_ratio * complete):
        return "full"
    return "sparse"


def missing_event_types(events: pd.DataFrame,
                        required=REQUIRED_EVENT_TYPES) -> list[str]:
    """Required event types this collection does not carry.

    Pass ``PER_MATCH_EVENT_TYPES`` when judging a sample rather than a season:
    the occasional types cannot be ruled out from a handful of matches.
    """
    return sorted(set(required) - set(events["type"].unique()))


def event_type_report(events: pd.DataFrame,
                      required=REQUIRED_EVENT_TYPES) -> dict:
    """What a sample of events carries, and how densely.

    Density is reported, never failed on. A 2003/04 collection logging fewer
    carries per match than a 2015/16 one is drift, which the contrast in
    ``src/representation/era.py`` already removes. A *missing* type is the
    thing that stops a build.
    """
    n_matches = int(events["match_id"].nunique())
    counts = events["type"].value_counts()
    return {
        "n_matches": n_matches,
        "n_events": len(events),
        "density": {t: round(c / n_matches, 2) for t, c in counts.items()},
        "missing": missing_event_types(events, required),
    }


def summarize(matches: pd.DataFrame,
              policy: CoveragePolicy = CoveragePolicy()) -> dict:
    """The verdict, plus the shape a reader needs to disagree with it."""
    played = _appearances(matches)
    return {
        "n_matches": len(matches),
        "n_teams": len(played),
        "top_team": played.index[0],
        "top_team_matches": int(played.iloc[0]),
        "median_matches_per_team": int(played.median()),
        "coverage": classify_coverage(matches, policy),
    }


# --- the network edge ------------------------------------------------------
# Everything above is a pure function of a matches frame. These three are the
# only places the catalogue is fetched, so the judgment stays testable offline.

def competitions() -> pd.DataFrame:
    """Every competition-season StatsBomb publishes."""
    from statsbombpy import sb
    return sb.competitions()


def seasons_for(competition_id: int) -> pd.DataFrame:
    """The seasons published for one competition, newest label first."""
    comps = competitions()
    return comps[comps["competition_id"] == competition_id]


def audit(competition_id: int, season_id: int,
          policy: CoveragePolicy = CoveragePolicy()) -> dict:
    """Fetch one season's match list and report what shape it is."""
    from statsbombpy import sb
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    result = summarize(matches, policy)
    result["competition_id"] = competition_id
    result["season_id"] = season_id
    return result


def check_event_types(competition_id: int, season_id: int, sample_n: int = 5,
                      required=PER_MATCH_EVENT_TYPES) -> dict:
    """Sample a season's matches and report what event types they carry.

    The sample is evenly spaced through the season in date order rather than
    random, so a verdict is reproducible and a season that changed collection
    mid-way cannot hide in one end of the fixture list.
    """
    from statsbombpy import sb
    matches = sb.matches(competition_id=competition_id,
                         season_id=season_id).sort_values("match_date")
    step = max(1, len(matches) // sample_n)
    match_ids = matches["match_id"].iloc[::step].tolist()[:sample_n]
    events = pd.concat([sb.events(match_id=int(m)) for m in match_ids],
                       ignore_index=True)
    report = event_type_report(events, required)
    report["competition_id"] = competition_id
    report["season_id"] = season_id
    report["sampled_match_ids"] = [int(m) for m in match_ids]
    return report
