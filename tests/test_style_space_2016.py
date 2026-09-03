"""S8 — does the 2015/16 field measure style, or is it just the league table?

An identity-anchor representation earns its name only if two clubs can be close
in it while finishing far apart, and far apart in it while finishing together.
These are Phase 3c's validation gates. The external truth they are checked
against is football rather than the pipeline: Leicester won the title playing
unlike any elite side, direct and low on possession, while Arsenal played the
possession game Chelsea, City and Liverpool were also playing.

The first two gates justify the decision to normalize plain season means rather
than context-adjusted ones, by showing what a full league cancels on its own.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import FEATURES_DIR, PROCESSED_DIR
from src.features.batch import resolve_opponent_strength
from src.representation.style_space import league_anchors, style_space

pytestmark = pytest.mark.skipif(
    not (FEATURES_DIR / "match_team_style_2016.csv").exists(),
    reason="requires the built feature tables in data/features/",
)

TOP_SIX = {"Leicester City", "Arsenal", "Tottenham Hotspur",
           "Manchester City", "Manchester United", "Southampton"}


@pytest.fixture(scope="module")
def style():
    return pd.read_csv(FEATURES_DIR / "match_team_style_2016.csv")


@pytest.fixture(scope="module")
def anchors(style):
    return league_anchors(style)


@pytest.fixture(scope="module")
def space(style, anchors):
    return style_space(style, anchors.features)


@pytest.fixture(scope="module")
def positions():
    matches = pd.read_csv(PROCESSED_DIR / "matches_2016.csv")
    strength = resolve_opponent_strength("2015/2016", matches)
    return pd.Series({team: pos for team, (pos, _) in strength.items()})


# -- why plain season means are enough ---------------------------------------
def test_a_full_league_balances_venue_on_its_own(style, anchors, space):
    """19 home and 19 away, so the season mean is already the venue-balanced one."""
    home = style[style["venue"] == "home"].groupby("team")[anchors.features].mean()
    away = style[style["venue"] == "away"].groupby("team")[anchors.features].mean()
    gap = (space.profiles - (home + away) / 2).abs()

    complete = [f for f in anchors.features if style[f].notna().all()]
    assert gap[complete].to_numpy().max() < 1e-12
    # The one incomplete column is the whole of the remaining discrepancy.
    assert set(anchors.features) - set(complete) == {"rp6_pressured_pass_completion_pct"}


def test_the_schedule_confound_is_one_position_wide(style, positions):
    """A club cannot play itself, and that is the entire imbalance.

    Positions sum to 210, so a club finishing p faces a pool averaging
    (210 - p) / 19: 11.0 for the champion, 10.0 for the bottom club. One
    position on a twenty-position scale, which is why no adjustment is applied.
    """
    faced = style.groupby("team")["opp_final_position"].mean()
    expected = (210 - positions) / 19
    pd.testing.assert_series_equal(
        faced.sort_index(), expected.sort_index(), check_names=False, atol=1e-12)
    assert faced.max() - faced.min() == pytest.approx(1.0)


# -- the gate: style is not quality ------------------------------------------
def test_style_distance_does_not_recover_the_league_table(space, positions):
    """If it did, the anchors would be measuring how good a club is, not how it plays."""
    correlations = []
    for team in space.teams:
        style_gap = space.distances[team].drop(index=team)
        table_gap = (positions.drop(team) - positions[team]).abs()
        correlations.append(
            stats.spearmanr(style_gap, table_gap.reindex(style_gap.index)).statistic)
    assert abs(np.mean(correlations)) < 0.30


def test_the_champion_plays_like_the_bottom_half_not_like_the_elite(space):
    """Leicester's title is the case the representation has to get right."""
    nearest = space.neighbours("Leicester City", 3)
    assert TOP_SIX.isdisjoint(nearest.index)


def test_arsenal_sits_with_the_possession_sides(space):
    nearest = space.neighbours("Arsenal", 4)
    assert len(TOP_SIX & set(nearest.index)) >= 1
    assert {"Chelsea", "Liverpool", "Manchester City"} & set(nearest.index)


def test_the_percentiles_read_out_the_two_styles(space):
    """Directness shows up as needing few passes to reach the final third."""
    route = space.percentiles["pi_avg_passes_to_final_third"]
    assert route["Leicester City"] <= 10      # among the most direct in the league
    assert route["Arsenal"] >= 90             # among the most patient
