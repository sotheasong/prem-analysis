"""S7 — the gate: a built season must agree with the league it claims to be.

The strongest check the project has, and the only fully independent one. FBref's
advanced stats are StatsBomb-derived, so agreeing with them proves the
aggregation and not the measurement. A final league table is different: it comes
from the results, it is public, and nothing in this pipeline influenced it.

One rule covers every coverage shape. A club's points deficit against the
official table may not exceed 3 per match the collection is missing for that
club, and no club may hold *more* points than it really finished with. The
strictness then falls out of the coverage rather than being configured:

  * a complete season leaves every club 0 missing, so the deficit must be 0
  * Ligue 1 2015/16 permits exactly its three absent fixtures, 3 points each
  * a Barcelona season binds Barcelona tightly and says little about clubs seen
    twice, which is the honest amount to say about them

The fixtures below are hand-built results, so every expected total is countable
off the construction rather than recomputed the way the code computes it.
"""

import pandas as pd
import pytest

from src.extraction.reconcile import reconcile


def results(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """(home, away, home_goals, away_goals) -> a processed-matches frame."""
    return pd.DataFrame([
        {"match_home_team": h, "match_away_team": a,
         "match_score": f"{hg} - {ag}"} for h, a, hg, ag in rows])


# A whole 3-club double round robin. A beats both, B beats C, C wins nothing.
# A: 2 wins + ... -> counted by hand below.
TINY_LEAGUE = results([
    ("A", "B", 2, 0), ("B", "A", 0, 1),      # A wins both -> A 6, B 0
    ("A", "C", 1, 0), ("C", "A", 0, 3),      # A wins both -> A 12, C 0
    ("B", "C", 2, 1), ("C", "B", 1, 1),      # B win + draw -> B 4, C 1
])
TINY_OFFICIAL = {"A": (1, 12), "B": (2, 4), "C": (3, 1)}


def test_a_complete_season_with_correct_results_reconciles_exactly():
    report = reconcile(TINY_LEAGUE, TINY_OFFICIAL, full_season_matches=4)
    assert list(report["deficit"]) == [0, 0, 0]
    assert report["ok"].all()


def test_a_club_holding_more_points_than_it_finished_with_fails():
    """A negative deficit is not a near miss, it is a structural error.

    It means a match was counted twice, attributed to the wrong season, or
    parsed with the scoreline reversed. No amount of missing data can produce
    it, so the bound is one-sided on purpose.
    """
    inflated = dict(TINY_OFFICIAL, A=(1, 9))     # A really scored 12
    report = reconcile(TINY_LEAGUE, inflated, full_season_matches=4)
    assert report.loc["A", "deficit"] == -3
    assert not report.loc["A", "ok"]


def test_one_absent_fixture_buys_three_points_of_slack_and_no_more():
    """The Ligue 1 case in miniature. Drop A's away win over C: A loses exactly
    the 3 points that fixture was worth, which is the most one match can hide."""
    short = TINY_LEAGUE.drop(index=3).reset_index(drop=True)   # C v A, A won 3-0
    report = reconcile(short, TINY_OFFICIAL, full_season_matches=4)

    assert report.loc["A", "played"] == 3
    assert report.loc["A", "deficit"] == 3
    assert report.loc["A", "max_deficit"] == 3
    assert report["ok"].all()

    # One point more unexplained than a single match can account for, and it fails.
    harsher = dict(TINY_OFFICIAL, A=(1, 13))
    assert not reconcile(short, harsher, full_season_matches=4).loc["A", "ok"]


# --- the gate itself, over every built season ------------------------------

def test_every_built_season_reconciles_against_its_official_table():
    """The acceptance criterion for ticket 07. No season enters the manifest
    until it passes, and a failure names the clubs rather than a count."""
    from src.extraction.reconcile import reconcile_season
    from src.utils.constants import SEASONS

    checked = 0
    for key in SEASONS:
        try:
            report = reconcile_season(key)
        except FileNotFoundError:
            continue                      # not built; data/ is gitignored
        checked += 1
        broken = report[~report["ok"]]
        assert broken.empty, f"{key} fails reconciliation:\n{broken}"
    if checked == 0:
        pytest.skip("no seasons built yet")


def test_the_complete_leagues_reconcile_with_no_slack_at_all():
    """Three of the four 2015/16 leagues hold every fixture, so 'within bound'
    is not good enough for them: the deficit must be exactly zero for all sixty
    clubs. This is what stops the bound quietly excusing a real error."""
    from src.extraction.reconcile import reconcile_season

    for key in ["2015/2016", "La Liga 2015/2016", "Serie A 2015/2016"]:
        try:
            report = reconcile_season(key)
        except FileNotFoundError:
            pytest.skip(f"{key} not built")
        assert (report["played"] == 38).all(), f"{key} is not complete"
        assert (report["deficit"] == 0).all(), (
            f"{key} does not reconcile exactly:\n"
            f"{report[report['deficit'] != 0]}")


FOCAL_CLUB = {"2003/2004": "Arsenal",
              **{f"Barcelona {y - 1}/{y}": "Barcelona" for y in range(2008, 2016)}}


def test_the_focal_club_is_where_a_single_club_season_is_actually_checked():
    """"Within bound" is nearly vacuous for these seasons: an opponent seen
    twice gets a ~108 point allowance, which no error could exceed. The focal
    club is different, because it plays every collected match, so its deficit
    is exact for the subset and the bound closes to zero once the collection
    holds all 38.

    Arsenal 2003/04 and Barcelona 2014/15 are both complete, and both must
    reproduce their real points totals with no slack whatsoever.
    """
    from src.extraction.reconcile import reconcile_season

    checked = 0
    for season, club in FOCAL_CLUB.items():
        try:
            row = reconcile_season(season).loc[club]
        except FileNotFoundError:
            continue
        checked += 1
        assert row["ok"], f"{season}: {club} {row.to_dict()}"
        if row["played"] == 38:
            assert row["deficit"] == 0, (
                f"{season}: {club} holds all 38 matches but reproduces "
                f"{row['points']} against an official {row['official_points']}")
    if checked == 0:
        pytest.skip("no single-club seasons built yet")
