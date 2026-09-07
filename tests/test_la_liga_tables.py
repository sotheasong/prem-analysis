"""S4 — opponent strength for seasons that cannot compute their own.

Barcelona 2007/08–2014/15 is one club's fixture list, so the season's results
cannot produce a league table and `compute_season_standings` does not apply.
The strength covariate has to come from a static historical table, which is the
`FINAL_TABLE_2004` pattern repeated eight times.

The failure mode this guards is quiet. `match_context` does `strength[opponent]`,
so a name the table spells differently is a KeyError at build time at best, and
at worst a season silently missing opponents. The integration test at the bottom
is the one that matters: every club StatsBomb records must resolve.
"""

import pytest

from src.utils.constants import LA_LIGA_TABLES

BARCELONA_SEASONS = ["2007/2008", "2008/2009", "2009/2010", "2010/2011",
                     "2011/2012", "2012/2013", "2013/2014", "2014/2015"]


def test_every_barcelona_season_has_a_table():
    assert sorted(LA_LIGA_TABLES) == BARCELONA_SEASONS


def test_each_table_is_a_complete_twenty_club_league():
    for season, table in LA_LIGA_TABLES.items():
        assert len(table) == 20, f"{season} has {len(table)} clubs"
        positions = sorted(pos for pos, _ in table.values())
        assert positions == list(range(1, 21)), f"{season} positions {positions}"


def test_points_never_increase_as_position_falls():
    """A table where 4th has more points than 3rd is a transcription error.

    This is the cheap check that catches a mistyped points value, which would
    otherwise pass silently into the opponent-strength covariate.
    """
    for season, table in LA_LIGA_TABLES.items():
        by_position = sorted(table.values())
        points = [pts for _, pts in by_position]
        assert points == sorted(points, reverse=True), f"{season}: {by_position}"


# --- integration: the tables against the clubs StatsBomb actually records ----

def test_every_club_statsbomb_records_resolves_to_a_position():
    """The acceptance criterion for this ticket: zero unmatched names.

    Spellings diverge in ways that are easy to miss by eye. StatsBomb writes
    `Athletic Club`, `RC Deportivo La Coruña`, `CD Numancia de Soria` and
    `Real Murcia CF` where the source writes `Athletic Bilbao`, `Deportivo
    La Coruña`, `Numancia` and `Murcia`. Nothing catches that except this.
    """
    try:
        from statsbombpy import sb
        comps = sb.competitions()
    except Exception:
        pytest.skip("StatsBomb open data unreachable")

    for season in BARCELONA_SEASONS:
        row = comps[(comps.competition_name == "La Liga")
                    & (comps.season_name == season)]
        assert len(row) == 1, f"La Liga {season} not in the catalogue"
        matches = sb.matches(competition_id=11,
                             season_id=int(row.iloc[0].season_id))
        clubs = set(matches.home_team) | set(matches.away_team)
        unmatched = sorted(clubs - set(LA_LIGA_TABLES[season]))
        assert unmatched == [], f"{season}: {unmatched}"
