"""S2 — a design the engine cannot fit must fail loudly.

If two covariates are linear combinations of each other, the normal equations
are singular: coefficients are not identified and every standard error comes out
NaN. Silently tiering 135 features off NaN p-values is the worst outcome, so the
diagnostic refuses the design instead.
"""

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.style import diagnose, team_season

H1 = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)


def one_season(**features):
    return team_season(pd.DataFrame({
        "team": "T",
        "date": pd.date_range("2015-08-08", periods=8, freq="7D"),
        "venue": np.where(H1 > 0, "home", "away"),
        **features,
    }), "T")


def test_collinear_design_is_rejected():
    ts = one_season(metric=[1.0, 2, 3, 4, 5, 6, 7, 8])
    design = lambda t: pd.DataFrame({
        "home": t.frame["home"],
        "away": 1 - t.frame["home"],      # exact complement of home
    })
    with pytest.raises(ValueError, match="rank"):
        diagnose(ts, design=design)


def test_design_shorter_than_the_season_is_rejected():
    ts = one_season(metric=[1.0, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="row"):
        diagnose(ts, design=lambda t: pd.DataFrame({"home": [1, 0, 1]}))


def test_a_full_rank_design_is_accepted():
    ts = one_season(metric=[1.0, 2, 3, 4, 5, 6, 7, 8])
    d = diagnose(ts, design=lambda t: pd.DataFrame({"home": t.frame["home"]}))
    assert d.fit("metric").se["home"] > 0
