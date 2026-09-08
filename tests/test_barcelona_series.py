"""S6 — the depth series reconciles against the league it played in.

Track 2 is eight consecutive Barcelona seasons, and every one is partial: the
collection holds 27 to 38 of the 38 league matches. That makes the usual
"rebuild the table and diff it" gate impossible, because nineteen of the twenty
clubs appear only in their fixtures against Barcelona.

What *is* checkable is the focal club. Barcelona plays in every collected match,
so its own points are exact for the subset, and the official table bounds them:
the collected total can never exceed the season total, and the shortfall can
never exceed three points per uncollected match. That is enough to catch a
season pointed at the wrong table, a misparsed scoreline, or matches attributed
to the wrong year.

Points are summed here rather than through ``compute_season_standings``, which
documents itself as valid only for full-league seasons, and so that the expected
value comes from the official table rather than from project code.
"""

import pandas as pd
import pytest

from src.config import PROCESSED_DIR
from src.utils.constants import LA_LIGA_TABLES, SEASONS

SEASON_KEYS = [f"Barcelona {y - 1}/{y}" for y in range(2008, 2016)]
FULL_SEASON_MATCHES = 38


def barcelona_points(matches: pd.DataFrame) -> int:
    points = 0
    for _, m in matches.iterrows():
        home, away = m["match_home_team"], m["match_away_team"]
        hg, ag = (int(x) for x in m["match_score"].split(" - "))
        assert "Barcelona" in (home, away), f"non-Barcelona match: {home} v {away}"
        gf, ga = (hg, ag) if home == "Barcelona" else (ag, hg)
        points += 3 if gf > ga else 1 if gf == ga else 0
    return points


def built_seasons() -> list[str]:
    return [k for k in SEASON_KEYS
            if (PROCESSED_DIR / f"matches_{SEASONS[k]['suffix']}.csv").exists()]


def test_barcelona_points_are_bounded_by_the_official_table():
    keys = built_seasons()
    if not keys:
        pytest.skip("Barcelona seasons not built yet (data/ is gitignored)")

    for key in keys:
        season = key.replace("Barcelona ", "")
        matches = pd.read_csv(PROCESSED_DIR / f"matches_{SEASONS[key]['suffix']}.csv")
        collected = barcelona_points(matches)
        official = LA_LIGA_TABLES[season]["Barcelona"][1]
        uncollected = FULL_SEASON_MATCHES - len(matches)

        assert collected <= official, (
            f"{season}: collected {collected} pts from {len(matches)} matches "
            f"but the season total was only {official}")
        assert official - collected <= 3 * uncollected, (
            f"{season}: {official - collected} pts unaccounted for across "
            f"{uncollected} uncollected matches")


def test_a_complete_season_reconciles_exactly():
    """2014/15 is the one season collected in full, so its total must land on
    the official number with no slack. It is the case that proves the bound
    above is not just permissive arithmetic."""
    key = "Barcelona 2014/2015"
    path = PROCESSED_DIR / f"matches_{SEASONS[key]['suffix']}.csv"
    if not path.exists():
        pytest.skip("Barcelona 2014/2015 not built yet")

    matches = pd.read_csv(path)
    assert len(matches) == FULL_SEASON_MATCHES
    assert barcelona_points(matches) == LA_LIGA_TABLES["2014/2015"]["Barcelona"][1]
