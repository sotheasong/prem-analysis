"""S6 — placing an outside team-season in a league's field.

Phase 3d wants to ask where Arsenal 2003/04 would sit among the 2015/16 clubs.
That season is Arsenal-only, so it can never define a field; it can only be
positioned in one. ``place`` does that arithmetic and nothing more. It says
where a set of raw season means falls among 20 contemporaries, which is a
statement about the numbers, not a claim that two eras measured the same thing.

The anchoring property is self-placement: feeding back a club that is already in
the field must return that club's own row exactly. Anything else means the
outside path and the inside path disagree about what a percentile is.
"""

import numpy as np
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


def test_placing_a_club_already_in_the_field_returns_its_own_position(space):
    for club in CLUBS:
        placed = space.place(space.profiles.loc[club])
        pd.testing.assert_series_equal(
            placed["z"], space.z.loc[club], check_names=False)
        pd.testing.assert_series_equal(
            placed["percentile"], space.percentiles.loc[club], check_names=False)


def test_an_outsider_beyond_the_field_tops_it(space):
    placed = space.place(pd.Series({"climbing": 99.0, "falling": 99.0}))
    assert (placed["percentile"] == 100.0).all()
    assert (placed["z"] > 0).all()


def test_an_outsider_below_the_field_bottoms_it(space):
    placed = space.place(pd.Series({"climbing": -99.0, "falling": -99.0}))
    assert (placed["percentile"] == 0.0).all()
    assert (placed["z"] < 0).all()


def test_an_outsider_lands_between_the_clubs_it_falls_between(space):
    """Between Beeches (2) and Carrow (3), so two of five clubs sit at or below."""
    placed = space.place(pd.Series({"climbing": 2.5, "falling": 2.5}))
    assert placed.loc["climbing", "percentile"] == 40.0
    assert placed.loc["climbing", "z"] == pytest.approx(-0.5 / np.sqrt(2))


def test_the_raw_value_travels_with_the_position(space):
    placed = space.place(pd.Series({"climbing": 2.5, "falling": 4.0}))
    assert placed.loc["climbing", "value"] == 2.5
    assert placed.loc["falling", "value"] == 4.0


def test_a_profile_missing_an_axis_is_refused(space):
    with pytest.raises(KeyError, match="falling"):
        space.place(pd.Series({"climbing": 2.5}))
