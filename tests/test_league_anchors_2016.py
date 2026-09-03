"""S4 on real data — the 2015/16 league-wide identity tier.

The counts below were measured from the 20-team sweep before this module was
written. They are here to pin the tier that Phase 3c normalizes against and,
more importantly, to hold the design decision in place: the league-wide tier is
not either champion's frozen registry, and swapping it for one would break
these tests loudly.
"""

import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.representation.style_space import league_anchors

pytestmark = pytest.mark.skipif(
    not (FEATURES_DIR / "match_team_style_2016.csv").exists(),
    reason="requires the built feature tables in data/features/",
)


@pytest.fixture(scope="module")
def anchors():
    style = pd.read_csv(FEATURES_DIR / "match_team_style_2016.csv")
    return league_anchors(style)


def test_the_field_is_the_whole_league(anchors):
    assert anchors.n_teams == 20


def test_the_tier_is_the_measured_fifteen_of_twenty_set(anchors):
    assert len(anchors.features) == 49
    assert anchors.min_teams == 15


def test_the_spine_is_the_fifteen_features_every_club_agrees_on(anchors):
    assert len(anchors.spine) == 15
    # Phase 3b called these universal identity axes from two champions. At 20 of
    # 20 clubs the claim stops being an n=2 anecdote.
    assert "pi_angle_entropy" in anchors.spine
    assert "rhythm_mean_burst_share" in anchors.spine


def test_the_tier_is_not_either_champions_registry(anchors):
    """The whole reason for measuring it league-wide rather than inheriting it."""
    arsenal = pd.read_csv(FEATURES_DIR / "phase2_arsenal_feature_registry.csv")
    leicester = pd.read_csv(FEATURES_DIR / "phase3b_leicester_feature_registry.csv")
    tier = set(anchors.features)
    for registry in (arsenal, leicester):
        one_team = set(registry.loc[registry["tier"] == "identity", "feature"])
        assert tier != one_team
        assert tier - one_team                    # the league adds axes...
        assert one_team - tier                    # ...and drops team quirks


def test_a_feature_kept_at_the_threshold_is_weaker_than_one_at_full_agreement(anchors):
    marginal = [f for f in anchors.features if anchors.counts[f] == 15]
    assert marginal, "expected some features to sit right on the threshold"
    assert set(marginal).isdisjoint(anchors.spine)


def test_the_arsenal_only_season_cannot_define_a_league(anchors):
    style_2004 = pd.read_csv(FEATURES_DIR / "match_team_style.csv")
    with pytest.raises(ValueError, match="coverage"):
        league_anchors(style_2004)
