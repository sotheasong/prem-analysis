"""S2 — bootstrap intervals on the season summaries.

Resampling the 38 matches says how much of a feature's stability is real and how
much is a 38-match accident. Truth for the mean interval is normal theory,
computed independently of the resampler: mean +/- 1.96 * sd / sqrt(n).
"""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.diagnostics.style import diagnose, team_season

# ``data/`` is gitignored, so these tests run against locally built tables. Skip
# rather than fail when the pipeline has not been run in this checkout.
pytestmark = pytest.mark.skipif(
    not all((FEATURES_DIR / f).exists() for f in ['match_team_style_2016.csv']),
    reason="requires the built feature tables in data/features/",
)


@pytest.fixture(scope="module")
def leicester():
    style = pd.read_csv(FEATURES_DIR / "match_team_style_2016.csv")
    return diagnose(team_season(style, "Leicester City"))


def test_mean_interval_agrees_with_normal_theory(leicester):
    values = leicester.ts.frame["passes"].to_numpy()
    half_width = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    expected_lo, expected_hi = values.mean() - half_width, values.mean() + half_width

    row = leicester.bootstrap(["passes"], seed=0).set_index("feature").loc["passes"]
    assert row["mean_lo"] == pytest.approx(expected_lo, rel=0.05)
    assert row["mean_hi"] == pytest.approx(expected_hi, rel=0.05)


def test_intervals_bracket_the_point_estimates(leicester):
    row = leicester.bootstrap(["passes"], seed=0).set_index("feature").loc["passes"]
    assert row["mean_lo"] < row["mean"] < row["mean_hi"]
    assert row["cv_lo"] < row["cv"] < row["cv_hi"]


def test_an_identity_anchor_has_a_tighter_cv_interval_than_an_output(leicester):
    """The headline Phase 2 claim: stability is separable, not a rounding artifact."""
    out = leicester.bootstrap(
        ["pi_angle_entropy", "poss_ends_box_pct"], seed=0).set_index("feature")
    anchor, output = out.loc["pi_angle_entropy"], out.loc["poss_ends_box_pct"]
    assert anchor["cv_hi"] < output["cv_lo"]      # intervals do not overlap


def test_the_same_seed_gives_the_same_interval(leicester):
    first = leicester.bootstrap(["passes"], seed=7)
    second = leicester.bootstrap(["passes"], seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_gives_a_different_interval(leicester):
    first = leicester.bootstrap(["passes"], seed=7).loc[0, "mean_lo"]
    second = leicester.bootstrap(["passes"], seed=8).loc[0, "mean_lo"]
    assert first != second


def test_missing_matches_are_excluded_not_propagated(leicester):
    """One 2015/16 feature is NaN for a single match; it must still resample."""
    feature = "rp6_pressured_pass_completion_pct"
    assert leicester.ts.frame[feature].isna().sum() == 1
    row = leicester.bootstrap([feature], seed=0).set_index("feature").loc[feature]
    assert np.isfinite([row["mean"], row["mean_lo"], row["mean_hi"]]).all()
