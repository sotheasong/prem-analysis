"""S7 — the Phase 3d gates, on the real 2003/04 and 2015/16 tables.

The synthetic slices prove the arithmetic. These prove the arithmetic does the
job it was built for, against football rather than against a fixture. The gate
was agreed before the numbers were known: the anchors whose cross-era gap is
mostly collection drift must leave the extreme set once the contrast is applied.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.representation.era import confound_ledger, decompose

pytestmark = pytest.mark.skipif(
    not all((FEATURES_DIR / f).exists() for f in
            ["match_team_style_2004.csv", "match_team_style_2016.csv",
             "phase3c_league_anchors_2016.csv"]),
    reason="requires the built feature tables in data/features/",
)


@pytest.fixture(scope="module")
def tables():
    a = pd.read_csv(FEATURES_DIR / "phase3c_league_anchors_2016.csv")
    flag = lambda c: a.loc[a[c].astype(str).str.lower().isin(["true", "1"]), "feature"].tolist()
    return (pd.read_csv(FEATURES_DIR / "match_team_style_2004.csv"),
            pd.read_csv(FEATURES_DIR / "match_team_style_2016.csv"),
            flag("in_tier"), flag("in_spine"))


@pytest.fixture(scope="module")
def ledger(tables):
    old, new, anchors, _ = tables
    return confound_ledger(old, "Arsenal", new, "Arsenal", anchors)


def test_2003_04_has_no_league_in_it_only_arsenal_fixtures(tables):
    """The fact the whole phase is built around. Every other club appears
    exactly twice, being its two games against Arsenal."""
    old = tables[0]
    played = old["team"].value_counts()
    assert played["Arsenal"] == 38
    assert set(played.drop("Arsenal")) == {2}


def test_the_possession_count_gap_is_almost_entirely_the_collection(ledger):
    """Arsenal 2003/04 reads +5.69 sd on `poss_possessions` in the 2015/16
    field, and its own 2003/04 opponents sit at the same inflated level, so
    the contrast leaves nearly nothing behind."""
    row = ledger.loc["poss_possessions"]
    assert row["raw_z"] == pytest.approx(5.69, abs=0.01)
    assert abs(row["contrast_z"]) < 0.30
    assert row["offset_share"] > 1.0
    assert row["verdict"] == "offset-dominated"


def test_the_agreed_gate_offset_dominated_anchors_leave_the_extreme_set(ledger):
    """Phase 3d's pre-registered gate. Of the anchors flagged as drift-driven,
    almost none may remain beyond 2.5 sd once the contrast is applied."""
    flagged = ledger.index[ledger["verdict"] == "offset-dominated"]
    assert len(flagged) > 20
    survivors = (ledger.loc[flagged, "contrast_z"].abs() > 2.5).sum()
    assert survivors <= 1


def test_the_contrast_shrinks_the_extremes_overall(ledger):
    assert (ledger["raw_z"].abs() > 2.5).sum() == 5
    assert (ledger["contrast_z"].abs() > 2.5).sum() == 2
    assert ledger["contrast_z"].abs().max() < ledger["raw_z"].abs().max()


def test_most_of_the_identity_tier_is_untrustworthy_in_the_raw_lens(ledger):
    """The finding that justifies the phase. Two thirds of the anchors carry
    more collection drift than cross-era signal, so a raw percentile
    comparison between 2003/04 and 2015/16 is mostly measuring StatsBomb."""
    assert (ledger["verdict"] == "offset-dominated").sum() >= 30


def test_the_decomposition_identity_holds_on_the_real_tables(tables):
    old, new, anchors, _ = tables
    scale = new.groupby("team")[anchors].mean().std(ddof=0)
    got = decompose(old, "Arsenal", new, "Arsenal", anchors, scale=scale,
                    n_resamples=200, seed=0)
    residual = got["total_change"] - got["league_moved"] - got["team_moved"]
    assert np.allclose(residual.to_numpy(), 0.0, atol=1e-12)


def test_angle_entropy_moved_with_the_league_not_against_it(tables):
    """Phase 3b called `pi_angle_entropy` a universal identity axis. Across the
    eras it shifts +1.93 sd, and the decomposition says the league carried
    nearly all of it while Arsenal's margin over its opponents barely moved."""
    old, new, anchors, _ = tables
    scale = new.groupby("team")[anchors].mean().std(ddof=0)
    got = decompose(old, "Arsenal", new, "Arsenal", anchors, scale=scale,
                    n_resamples=2000, seed=0).loc["pi_angle_entropy"]

    assert got["total_change"] == pytest.approx(1.93, abs=0.01)
    assert got["league_moved"] == pytest.approx(1.75, abs=0.01)
    assert abs(got["team_moved"]) < 0.25
    assert got["team_lo"] < 0 < got["team_hi"]


def test_a_flat_total_hides_two_large_opposing_moves_in_the_real_data(tables):
    """`pi_early_width_y_std` is why the raw column cannot answer "did Arsenal
    change?" on its own: -0.87 total, from +1.88 league and -2.75 Arsenal."""
    old, new, anchors, _ = tables
    scale = new.groupby("team")[anchors].mean().std(ddof=0)
    got = decompose(old, "Arsenal", new, "Arsenal", anchors, scale=scale,
                    n_resamples=200, seed=0).loc["pi_early_width_y_std"]

    assert abs(got["total_change"]) < 1.0
    assert got["league_moved"] > 1.5
    assert got["team_moved"] < -2.5
