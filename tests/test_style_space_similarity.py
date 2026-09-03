"""S7 — distance between clubs in the era-normalized field.

Distance is Euclidean in z-space, not in percentile space: percentiles are ranks
and throw away the gap between clubs, which is the part that says whether a
leader is narrow or runaway.

The fixture makes the geometry checkable by hand. `climbing` and `falling` are
mirror images, so the five clubs lie on a straight line in z-space, and each
step along it is exactly one unit:

    z(climbing) = [-sqrt2, -sqrt2/2, 0, +sqrt2/2, +sqrt2]   z(falling) = -z(climbing)
    consecutive gap = sqrt((sqrt2/2)^2 + (sqrt2/2)^2) = 1

so the distance between any two clubs is the difference in their league position.
"""

import pandas as pd
import pytest

from src.representation.style_space import style_space

CLUBS = ["Ayton", "Beeches", "Carrow", "Dell", "Elland"]
MATCHES = 38


@pytest.fixture(scope="module")
def space():
    rows = []
    for rank, club in enumerate(CLUBS, start=1):
        rows.append(pd.DataFrame({
            "team": club,
            "date": pd.date_range("2015-08-08", periods=MATCHES, freq="7D"),
            "venue": ["home", "away"] * (MATCHES // 2),
            "opp_final_position": [((i * 7) % 20) + 1 for i in range(MATCHES)],
            "climbing": float(rank),
            "falling": float(len(CLUBS) + 1 - rank),
        }))
    return style_space(pd.concat(rows, ignore_index=True), ["climbing", "falling"])


def test_distance_matches_the_hand_worked_geometry(space):
    d = space.distances
    assert d.loc["Ayton", "Beeches"] == pytest.approx(1.0)
    assert d.loc["Ayton", "Elland"] == pytest.approx(4.0)
    assert d.loc["Beeches", "Dell"] == pytest.approx(2.0)


def test_a_club_is_at_no_distance_from_itself(space):
    assert (pd.Series(space.distances.to_numpy().diagonal()) == 0).all()


def test_distance_is_symmetric(space):
    pd.testing.assert_frame_equal(space.distances, space.distances.T)


def test_nearest_neighbours_are_the_closest_other_clubs(space):
    assert list(space.neighbours("Ayton", 2).index) == ["Beeches", "Carrow"]
    assert space.neighbours("Ayton", 2).iloc[0] == pytest.approx(1.0)


def test_a_club_is_never_its_own_neighbour(space):
    assert "Carrow" not in space.neighbours("Carrow", 4).index


def test_asking_for_more_neighbours_than_the_league_holds_returns_the_league(space):
    assert len(space.neighbours("Carrow", 99)) == len(CLUBS) - 1


def test_an_unknown_club_is_refused(space):
    with pytest.raises(KeyError, match="Wigan"):
        space.neighbours("Wigan Athletic", 2)
