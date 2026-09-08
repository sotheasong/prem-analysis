"""S10 — the frozen manifest, which is what Phase 4 reads instead of guessing.

One row per season, mixing what the registry declares with what the validators
measured. A season that has not passed the gate must not look identical to one
that has.
"""

import pandas as pd
import pytest

from src.validation.manifest import build_manifest


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    m = build_manifest()
    if not m["built"].any():
        pytest.skip("nothing built (data/ is gitignored)")
    return m


def test_absent_fixtures_is_only_reported_where_it_means_something(manifest):
    """A single-club season is missing ~345 of a round robin's 380 fixtures by
    design, so printing that number invites someone to read a healthy Barcelona
    season as catastrophically incomplete. The count belongs to full-league
    coverage, where 3 absences is a real and actionable finding.
    """
    built = manifest[manifest["built"]]
    full = built[built["coverage"] == "full"]
    partial = built[built["coverage"] != "full"]

    assert full["absent_fixtures"].notna().all()
    assert partial["absent_fixtures"].isna().all(), (
        "absent_fixtures is meaningless for single-club coverage and must be "
        "blank rather than a large number")


def test_every_built_season_passed_the_reconciliation_gate(manifest):
    built = manifest[manifest["built"]]
    assert built["reconciles"].all(), built.loc[~built["reconciles"], "season"].tolist()


def test_the_corpus_is_the_89_club_seasons_the_phase_promised(manifest):
    """80 field club-seasons across four 2015/16 leagues, plus 9 series."""
    built = manifest[manifest["built"]]
    field_clubs = int(built.loc[built["role"] == "field", "clubs"].sum())
    series_seasons = int((built["role"] == "series").sum())
    assert field_clubs == 80, field_clubs
    assert series_seasons == 9, series_seasons
