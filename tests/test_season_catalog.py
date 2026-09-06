"""S1 — coverage is a measured property of a season, not a promise.

Phase 3a discovered the hard way that a season can carry twenty club names and
still be one club's fixture list: 2003/04 holds 76 rows and 20 teams, but
Arsenal appears 38 times and everybody else exactly twice. Phase 3e found the
same trap in Bundesliga 2015/16, which looks like a season and is 34 Leverkusen
matches.

So "is this season usable" has to be computed from the match list rather than
read off a season label. These fixtures are built as exact schedules, so every
verdict is derivable by hand from the construction and never from the code.
"""

import pandas as pd
import pytest

from src.extraction.catalog import audit, classify_coverage, summarize


def double_round_robin(n_teams: int) -> pd.DataFrame:
    """Every club plays every other home and away. n_teams*(n_teams-1) matches."""
    teams = [f"T{i:02d}" for i in range(n_teams)]
    rows = [{"home_team": h, "away_team": a}
            for h in teams for a in teams if h != a]
    return pd.DataFrame(rows)


def one_club_season(n_teams: int, focal: str = "T00") -> pd.DataFrame:
    """One club plays every other home and away; nobody else meets. 2*(n-1) rows.

    This is the 2003/04 shape (20 teams, 38 Arsenal matches) and the Bundesliga
    2015/16 shape (18 teams, 34 Leverkusen matches).
    """
    others = [f"T{i:02d}" for i in range(1, n_teams)]
    rows = ([{"home_team": focal, "away_team": o} for o in others]
            + [{"home_team": o, "away_team": focal} for o in others])
    return pd.DataFrame(rows)


def test_a_complete_double_round_robin_is_a_full_league():
    matches = double_round_robin(20)
    assert len(matches) == 380          # the schedule is what we think it is
    assert classify_coverage(matches) == "full"


def test_one_clubs_fixture_list_is_not_a_full_league():
    """Bundesliga 2015/16: 18 club names, 34 matches, all of them Leverkusen's.

    The season label and the club count both say league. The schedule does not.
    """
    matches = one_club_season(18)
    assert len(matches) == 34
    assert matches.home_team.nunique() == 18   # looks like a league
    assert classify_coverage(matches) == "single_club"


def test_a_one_off_final_is_sparse_not_a_single_club_season():
    """Both finalists appear in every match, because there is one match.

    "One club appears in all of them" is true of a cup final and says nothing.
    A single-club season has to be long enough to profile, which is what
    separates the Champions League finals (1 match) and La Liga 1973/74 from
    Arsenal 2003/04.
    """
    final = pd.DataFrame([{"home_team": "Juventus", "away_team": "Barcelona"}])
    assert classify_coverage(final) == "sparse"


def test_a_partially_collected_league_is_sparse():
    """20 clubs, no club in every match, but only 150 of the 380 fixtures.

    Nothing here is one club's season, so the single-club rule stays silent.
    The season still cannot define a field, and saying "full" would send it
    into a build that assumes a complete schedule.
    """
    full = double_round_robin(20)
    matches = full.iloc[::2].head(150)          # every other fixture, so all
    clubs = pd.concat([matches.home_team, matches.away_team]).nunique()
    assert clubs == 20                          # clubs still turn up
    assert len(matches) == 150                  # the schedule does not
    assert classify_coverage(matches) == "sparse"


def test_a_league_missing_a_few_fixtures_is_still_full():
    """Ligue 1 2015/16 is 377 of 380. A real collection is allowed to be short.

    This is the counterweight to the test above: demanding a complete schedule
    would throw away a season the phase intends to build.
    """
    matches = double_round_robin(20).iloc[:-3]
    assert len(matches) == 377
    assert classify_coverage(matches) == "full"


def test_summarize_reports_the_shape_that_produced_the_verdict():
    """The 2003/04 shape, where every number is fixed by the construction.

    One club plays 38, the other nineteen play 2 each, so the median is 2 and
    the maximum is 38. Reporting those alongside the verdict is what lets a
    reader disagree with the classifier instead of trusting it.
    """
    matches = one_club_season(20, focal="Arsenal")
    assert summarize(matches) == {
        "n_matches": 38,
        "n_teams": 20,
        "top_team": "Arsenal",
        "top_team_matches": 38,
        "median_matches_per_team": 2,
        "coverage": "single_club",
    }


# --- integration: the classifier against the real catalogue ----------------
# The synthetic fixtures above prove the arithmetic. This proves the arithmetic
# lands on the right seasons, which is the claim Phase 3e actually rests on.

REAL_SEASONS = [
    ("Premier League", "2015/2016", "full"),
    ("La Liga",        "2015/2016", "full"),
    ("Serie A",        "2015/2016", "full"),
    ("Premier League", "2003/2004", "single_club"),
    ("1. Bundesliga",  "2015/2016", "single_club"),   # the regression case
    ("La Liga",        "1973/1974", "sparse"),
    ("Champions League", "2015/2016", "sparse"),      # a final, not a season
]


def _statsbomb_reachable(comps_holder: dict) -> bool:
    try:
        from statsbombpy import sb
        comps_holder["comps"] = sb.competitions()
        return True
    except Exception:
        return False


def test_audit_classifies_the_real_seasons_phase_3e_depends_on():
    """Bundesliga 2015/16 is the row that matters: 18 clubs, 34 Leverkusen
    matches. If it ever reads as ``full`` the phase queues a 380-match build
    against one club's fixture list."""
    holder = {}
    if not _statsbomb_reachable(holder):
        pytest.skip("StatsBomb open data unreachable")
    comps = holder["comps"]

    for comp_name, season_name, expected in REAL_SEASONS:
        row = comps[(comps.competition_name == comp_name)
                    & (comps.season_name == season_name)]
        assert len(row) == 1, f"{comp_name} {season_name} not in the catalogue"
        result = audit(int(row.iloc[0].competition_id),
                       int(row.iloc[0].season_id))
        assert result["coverage"] == expected, (
            f"{comp_name} {season_name}: expected {expected}, "
            f"got {result['coverage']} from {result}")
