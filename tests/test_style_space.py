"""S5 — the era-normalized field: where each club sits among its contemporaries.

A club's season is aggregated to one number per anchor feature, then expressed
two ways against the 20 clubs of its own league. The z-score keeps magnitude and
is what distances are measured in; the percentile is the interpretable read-out.

Worked by hand for a five-club field with season means [1, 2, 3, 4, 5]:

    mean = 3, population sd = sqrt((4+1+0+1+4)/5) = sqrt(2)
    z          = [-sqrt(2), -sqrt(2)/2, 0, +sqrt(2)/2, +sqrt(2)]
    percentile = [20, 40, 60, 80, 100]          (share of the field at or below)

The league is the whole population, not a sample, so the sd uses ddof=0.
"""

import numpy as np
import pandas as pd
import pytest

from src.representation.style_space import style_space

CLUBS = ["Ayton", "Beeches", "Carrow", "Dell", "Elland"]
MATCHES = 38
SQRT2 = np.sqrt(2)


@pytest.fixture(scope="module")
def field():
    """Each club is flat across its season, so season means are exact."""
    rows = []
    for rank, club in enumerate(CLUBS, start=1):
        rows.append(pd.DataFrame({
            "team": club,
            "date": pd.date_range("2015-08-08", periods=MATCHES, freq="7D"),
            "venue": ["home", "away"] * (MATCHES // 2),
            "opp_final_position": [((i * 7) % 20) + 1 for i in range(MATCHES)],
            "climbing": float(rank),
            "falling": float(len(CLUBS) + 1 - rank),
            "ignored": float(rank) * 100,
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def space(field):
    return style_space(field, ["climbing", "falling"])


def test_a_club_season_collapses_to_its_mean_per_feature(space):
    assert space.profiles.loc["Carrow", "climbing"] == pytest.approx(3.0)
    assert space.profiles.loc["Carrow", "falling"] == pytest.approx(3.0)
    assert space.profiles.loc["Elland", "climbing"] == pytest.approx(5.0)


def test_only_the_named_features_enter_the_space(space):
    assert list(space.profiles.columns) == ["climbing", "falling"]


def test_every_club_in_the_league_is_in_the_field(space):
    assert sorted(space.teams) == sorted(CLUBS)


def test_z_matches_the_hand_worked_values(space):
    expected = pd.Series([-SQRT2, -SQRT2 / 2, 0.0, SQRT2 / 2, SQRT2], index=CLUBS)
    pd.testing.assert_series_equal(
        space.z["climbing"], expected, check_names=False, atol=1e-12)


def test_each_feature_is_normalized_on_its_own(space):
    """`falling` is the mirror of `climbing`, so its z must be the negation."""
    pd.testing.assert_series_equal(
        space.z["falling"], -space.z["climbing"], check_names=False)


def test_percentiles_match_the_hand_worked_values(space):
    expected = pd.Series([20.0, 40.0, 60.0, 80.0, 100.0], index=CLUBS)
    pd.testing.assert_series_equal(
        space.percentiles["climbing"], expected, check_names=False)


def test_a_partial_coverage_season_has_no_field(field):
    thin = field[field["team"].isin(CLUBS[:2])].head(20)
    with pytest.raises(ValueError, match="coverage"):
        style_space(thin, ["climbing"])
