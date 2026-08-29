"""S2 — the context decomposition: z-scored OLS of each feature on the design.

Truth here is hand-derived from the definition of least squares, not from the
implementation. The design is two orthogonal contrasts over n=8 and the response
is built from them with a known orthogonal residual, so every quantity below
(beta, SE, R^2, VIF) has a closed form worked out on paper:

    a = [+1]*4 + [-1]*4        b = [+1,-1]*4        c = a*b   (all mutually orthogonal)
    y = 3a + 4b + c            =>  z(y) = (3a + 4b + c)/sqrt(26)

    beta_a = 3/sqrt(26)        beta_b = 4/sqrt(26)
    SSR    = 8/26              SST = 8        =>  R^2 = 25/26
    s^2    = SSR/(8-3)         SE  = sqrt(s^2 / 8)   (same for both, orthogonal design)
"""

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.style import diagnose, team_season

A = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
B = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
C = A * B
SQRT26 = np.sqrt(26.0)


@pytest.fixture(scope="module")
def synthetic():
    frame = pd.DataFrame({
        "team": "T",
        "date": pd.date_range("2015-08-08", periods=8, freq="7D"),
        "venue": ["home", "away"] * 4,
        "match_id": range(8),
        "y": 3 * A + 4 * B + C,
        "flat": 1.0,                     # degenerate: constant across the season
    })
    ts = team_season(frame, "T")
    design = lambda _ts: pd.DataFrame({"a": A, "b": B})
    return diagnose(ts, design=design)


def test_recovers_known_coefficients(synthetic):
    fit = synthetic.fit("y")
    assert fit.beta["a"] == pytest.approx(3 / SQRT26, abs=1e-9)
    assert fit.beta["b"] == pytest.approx(4 / SQRT26, abs=1e-9)


def test_reports_known_variance_explained(synthetic):
    fit = synthetic.fit("y")
    assert fit.r2 == pytest.approx(25 / 26, abs=1e-9)
    assert fit.dof == 5                                  # n=8, intercept + 2 predictors


def test_reports_known_standard_errors_and_significance(synthetic):
    fit = synthetic.fit("y")
    expected_se = np.sqrt((8 / 26) / 5 / 8)
    assert fit.se["a"] == pytest.approx(expected_se, abs=1e-9)
    assert fit.se["b"] == pytest.approx(expected_se, abs=1e-9)
    # t = beta/se: 6.71 and 8.94 on 5 dof, both well past the 0.05 threshold.
    assert fit.p["a"] < 0.01
    assert fit.p["b"] < 0.001


def test_confidence_intervals_bracket_the_estimate(synthetic):
    lo, hi = synthetic.fit("y").ci["a"]
    assert lo < 3 / SQRT26 < hi


def test_orthogonal_design_has_unit_vif(synthetic):
    assert synthetic.vif == pytest.approx({"a": 1.0, "b": 1.0}, abs=1e-9)


def test_dominant_driver_is_the_largest_absolute_coefficient(synthetic):
    assert synthetic.registry.set_index("feature").loc["y", "dominant"] == "b"


def test_constant_features_are_dropped_not_fitted(synthetic):
    """A feature with no variance is stable but information-free."""
    tiers = synthetic.registry.set_index("feature")["tier"]
    assert tiers["flat"] == "drop"
    assert "flat" not in synthetic.live
