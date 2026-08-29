"""S2 — dispersion: the stability half of the identity/context split.

Coefficient of variation, sample standard deviation over the absolute mean.
Worked by hand for [1..8]:

    mean = 4.5
    sum of squared deviations = 2*(0.5^2 + 1.5^2 + 2.5^2 + 3.5^2) = 42
    std(ddof=1) = sqrt(42/7) = sqrt(6)
    cv = sqrt(6) / 4.5 = 0.5443310540
"""

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.style import diagnose, team_season

RAMP = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
CV_RAMP = np.sqrt(6) / 4.5


def make(**features):
    n = len(next(iter(features.values())))
    frame = pd.DataFrame({
        "team": "T",
        "date": pd.date_range("2015-08-08", periods=n, freq="7D"),
        "venue": ["home", "away"] * (n // 2),
        # Not a linear function of date-index and venue: the three covariates
        # have to be separable or the design is rank deficient.
        "opp_final_position": [3, 12, 7, 1, 15, 5, 9, 18][:n],
        **features,
    })
    return diagnose(team_season(frame, "T"))


def test_cv_matches_the_hand_worked_value():
    d = make(metric=RAMP)
    assert d.dispersion.loc["metric", "cv"] == pytest.approx(CV_RAMP)
    assert d.dispersion.loc["metric", "mean"] == pytest.approx(4.5)


def test_cv_is_scale_free():
    """Doubling a feature's units must not change how stable it looks."""
    d = make(metric=RAMP, doubled=[2 * v for v in RAMP])
    assert d.dispersion.loc["doubled", "cv"] == pytest.approx(
        d.dispersion.loc["metric", "cv"])


def test_features_are_ranked_most_stable_first():
    d = make(steady=[10.0, 10.1, 9.9, 10.0, 10.05, 9.95, 10.0, 10.1],
             jumpy=[1.0, 9.0, 2.0, 8.0, 1.5, 8.5, 2.5, 7.5])
    assert list(d.dispersion.index[:1]) == ["steady"]


def test_mean_zero_feature_has_undefined_cv_not_infinity():
    d = make(centred=[-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -1.5, 1.5])
    assert np.isnan(d.dispersion.loc["centred", "cv"])


def test_registry_carries_the_cv():
    d = make(metric=RAMP)
    assert d.registry.set_index("feature").loc["metric", "cv"] == pytest.approx(CV_RAMP)
