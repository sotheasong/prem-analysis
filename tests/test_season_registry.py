"""S3 — the registry cannot lie about a season.

``SEASONS`` is the single place a season is declared, and everything downstream
trusts it: which file suffix to read, where opponent strength comes from,
whether the season can define a league field. A wrong entry is not caught by a
type error, it is caught three hours into a build or, worse, in a result.

So the entries are checked against a fixed vocabulary here, and against the
StatsBomb catalogue itself in the integration test at the bottom.
"""

import pytest

from src.utils.constants import SEASONS

REQUIRED_KEYS = {"competition", "competition_id", "season_id", "suffix",
                 "coverage", "collection_pass", "role", "strength"}

# Coverage uses the classifier's vocabulary, not a second one of its own.
COVERAGE_VALUES = {"full", "single_club", "sparse"}

# How a season is used by Phase 4's fit/project rule. `field` seasons supply
# distinct clubs to fit archetypes on; `series` seasons are repeated measures of
# one club and are projected into that fitted space, never fitted on.
ROLE_VALUES = {"field", "series"}


def test_every_season_declares_the_full_entry():
    for key, entry in SEASONS.items():
        assert REQUIRED_KEYS <= set(entry), f"{key} is missing {REQUIRED_KEYS - set(entry)}"


def test_coverage_and_role_use_the_agreed_vocabulary():
    for key, entry in SEASONS.items():
        assert entry["coverage"] in COVERAGE_VALUES, f"{key}: {entry['coverage']}"
        assert entry["role"] in ROLE_VALUES, f"{key}: {entry['role']}"


def test_a_season_is_identified_by_competition_and_season_together():
    """StatsBomb reuses season_id 27 for 2015/16 in every league, so the
    identifier is the pair. Suffixes must be unique too, because they name the
    files a build writes and a collision silently overwrites a season."""
    pairs = [(e["competition_id"], e["season_id"]) for e in SEASONS.values()]
    suffixes = [e["suffix"] for e in SEASONS.values()]
    assert len(set(pairs)) == len(pairs), "duplicate (competition_id, season_id)"
    assert len(set(suffixes)) == len(suffixes), "duplicate suffix"


def test_only_full_seasons_compute_their_own_standings():
    """A partial season cannot derive a league table from its own results, so it
    must carry a static one. This is the 2003/04 lesson as an invariant."""
    for key, entry in SEASONS.items():
        if entry["coverage"] == "full":
            assert entry["strength"] is None, f"{key} should compute standings"
        else:
            assert entry["strength"] is not None, f"{key} needs a static table"


# --- integration: the registry against the catalogue it claims to describe ---

def test_declared_coverage_matches_what_the_catalogue_actually_holds():
    """Every `coverage` value is a claim about real data, so check it against
    real data. This is what stops Bundesliga 2015/16 being declared `full`
    because the season label looked like a league.
    """
    try:
        from statsbombpy import sb
        sb.competitions()
    except Exception:
        pytest.skip("StatsBomb open data unreachable")

    from src.extraction.catalog import audit
    for key, entry in SEASONS.items():
        result = audit(entry["competition_id"], entry["season_id"])
        assert result["coverage"] == entry["coverage"], (
            f"{key} declares {entry['coverage']} but the catalogue says "
            f"{result['coverage']}: {result}")


def test_the_barcelona_series_is_declared_with_its_static_tables():
    """Track 2 is eight consecutive seasons of one club, so every entry is a
    `series` (projected into a fitted field, never fitted on) and every one
    carries the historical table its own results cannot produce."""
    from src.utils.constants import LA_LIGA_TABLES

    for season in ["2007/2008", "2008/2009", "2009/2010", "2010/2011",
                   "2011/2012", "2012/2013", "2013/2014", "2014/2015"]:
        key = f"Barcelona {season}"
        assert key in SEASONS, f"{key} is not declared"
        entry = SEASONS[key]
        assert entry["role"] == "series"
        assert entry["coverage"] == "single_club"
        assert entry["competition_id"] == 11
        assert entry["strength"] is LA_LIGA_TABLES[season]
