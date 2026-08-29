"""S2 — the second context design: the opponent's *measured* style.

Phases 1 to 3b proxy opponent strength with final league position, a number from
outside the data. A full-league season lets the covariate be measured instead:
what the opponent actually does on the ball, averaged over its own season. This
is the adapter Phase 3c needs, and the reason ``design`` is a seam.
"""

import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.diagnostics.style import diagnose, opponent_style_design, team_season

# ``data/`` is gitignored, so these tests run against locally built tables. Skip
# rather than fail when the pipeline has not been run in this checkout.
pytestmark = pytest.mark.skipif(
    not all((FEATURES_DIR / f).exists() for f in ['match_team_style.csv', 'match_team_style_2016.csv']),
    reason="requires the built feature tables in data/features/",
)

FEATURE = "poss_mean_directness"


@pytest.fixture(scope="module")
def style_2016():
    return pd.read_csv(FEATURES_DIR / "match_team_style_2016.csv")


def test_covariate_is_the_opponents_own_season_average(style_2016):
    """Computed here from the raw table, not read back from the design."""
    ts = team_season(style_2016, "Leicester City")
    design = opponent_style_design(style_2016, FEATURE)(ts)

    season_mean = style_2016.groupby("team")[FEATURE].mean()
    expected = ts.frame["opponent"].map(season_mean)
    pd.testing.assert_series_equal(
        design["opp_style"], expected, check_names=False)


def test_design_replaces_league_position_with_measured_style(style_2016):
    ts = team_season(style_2016, "Leicester City")
    design = opponent_style_design(style_2016, FEATURE)(ts)
    assert list(design.columns) == ["date", "home", "opp_style"]
    assert len(design) == 38


def test_it_drives_a_full_diagnosis(style_2016):
    ts = team_season(style_2016, "Leicester City")
    diag = diagnose(ts, design=opponent_style_design(style_2016, FEATURE))
    assert set(diag.registry["dominant"].dropna()) <= {"date", "home", "opp_style"}
    assert len(diag.registry) == 135


def test_partial_coverage_seasons_are_refused(style_2016):
    """2003/04 is Arsenal-only: every opponent has two matches, not a season."""
    style_2004 = pd.read_csv(FEATURES_DIR / "match_team_style.csv")
    ts = team_season(style_2004, "Arsenal")
    with pytest.raises(ValueError, match="coverage"):
        opponent_style_design(style_2004, FEATURE)(ts)


def test_unknown_feature_is_refused(style_2016):
    ts = team_season(style_2016, "Leicester City")
    with pytest.raises(KeyError, match="not_a_feature"):
        opponent_style_design(style_2016, "not_a_feature")(ts)
