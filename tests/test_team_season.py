"""S2 — ``team_season``: one team's season, ready to diagnose.

The three validation notebooks each hand-rolled this slice (filter to the team,
sort by date, encode venue, pick the feature columns). It is the entry point of
the diagnostic module.
"""

import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.diagnostics.style import team_season

# ``data/`` is gitignored, so these tests run against locally built tables. Skip
# rather than fail when the pipeline has not been run in this checkout.
pytestmark = pytest.mark.skipif(
    not all((FEATURES_DIR / f).exists() for f in ['match_team_style_2016.csv', 'phase3b_leicester_feature_registry.csv']),
    reason="requires the built feature tables in data/features/",
)


@pytest.fixture(scope="module")
def style_2016():
    return pd.read_csv(FEATURES_DIR / "match_team_style_2016.csv")


def test_selects_one_team_full_season(style_2016):
    ts = team_season(style_2016, "Leicester City")
    assert ts.team == "Leicester City"
    assert ts.n == 38
    assert set(ts.frame["team"]) == {"Leicester City"}


def test_rows_are_in_chronological_order(style_2016):
    ts = team_season(style_2016, "Leicester City")
    dates = ts.frame["date"]
    assert pd.api.types.is_datetime64_any_dtype(dates)
    assert dates.is_monotonic_increasing
    # Leicester's 2015/16 opened away at Everton on the final day it did not.
    assert ts.frame["date"].iloc[0] < ts.frame["date"].iloc[-1]


def test_encodes_venue_as_home_indicator(style_2016):
    ts = team_season(style_2016, "Leicester City")
    home = ts.frame["home"]
    assert set(home.unique()) == {0, 1}
    assert home.sum() == 19          # 19 home, 19 away in a 38-game season
    assert (ts.frame.loc[home == 1, "venue"] == "home").all()


def test_feature_universe_is_the_style_columns(style_2016):
    ts = team_season(style_2016, "Leicester City")
    registry = pd.read_csv(FEATURES_DIR / "phase3b_leicester_feature_registry.csv")
    assert sorted(ts.features) == sorted(registry["feature"])


def test_unknown_team_is_an_error_not_an_empty_frame(style_2016):
    with pytest.raises(ValueError, match="Wigan"):
        team_season(style_2016, "Wigan Athletic")
