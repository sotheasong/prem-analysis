"""S1 — the paired within-match contrast.

Phase 3c placed a raw season profile in a league field. That breaks across eras:
the 2003/04 collection segments possessions more finely than the 2015/16 one, so
Arsenal reads +5.69 sd on `poss_possessions` when its own 2003/04 opponents read
the same inflated number. Any bias shared by both teams in a match cancels in
the difference between them, which is what this contrast is for.

The anchoring property is antisymmetry. A match has two rows and one difference,
so the two rows must be exact negatives. A contrast that fails this is not
measuring a difference at all.
"""

import numpy as np
import pandas as pd
import pytest

from src.representation.era import opponent_relative

FEATURES = ["shots", "passes"]


def two_match_frame() -> pd.DataFrame:
    """Two matches, hand-worked so the expected contrasts are readable.

    m1: Ajax 10 shots vs Brugge 4  -> +6 / -6
    m2: Ajax  7 shots vs Brugge 9  -> -2 / +2
    """
    return pd.DataFrame({
        "match_id": [1, 1, 2, 2],
        "team": ["Ajax", "Brugge", "Ajax", "Brugge"],
        "shots": [10.0, 4.0, 7.0, 9.0],
        "passes": [400.0, 300.0, 350.0, 350.0],
    })


def test_contrast_is_the_paired_within_match_difference():
    out = opponent_relative(two_match_frame(), FEATURES)
    got = out.set_index(["match_id", "team"])["shots"]

    assert got[(1, "Ajax")] == pytest.approx(6.0)
    assert got[(1, "Brugge")] == pytest.approx(-6.0)
    assert got[(2, "Ajax")] == pytest.approx(-2.0)
    assert got[(2, "Brugge")] == pytest.approx(2.0)


def test_the_two_rows_of_a_match_are_exact_negatives():
    """Antisymmetry. Holds per match, per feature, with no tolerance to spare."""
    out = opponent_relative(two_match_frame(), FEATURES)
    sums = out.groupby("match_id")[FEATURES].sum()
    assert np.allclose(sums.to_numpy(), 0.0, atol=0.0)


def test_output_keeps_the_shape_style_space_consumes():
    """The whole point of returning a match-level frame: 3c's field builder
    takes `team` plus feature columns, so it composes with no changes."""
    frame = two_match_frame()
    out = opponent_relative(frame, FEATURES)

    assert list(out.columns) == ["match_id", "team"] + FEATURES
    assert len(out) == len(frame)
    assert sorted(out["team"].unique()) == ["Ajax", "Brugge"]


def test_a_match_missing_its_opponent_row_is_refused():
    """2003/04 is Arsenal-only at the club level but every match has both sides.
    A one-sided match would silently produce a contrast against nothing."""
    frame = two_match_frame().drop(index=1)  # remove Brugge from match 1
    with pytest.raises(ValueError, match="two rows"):
        opponent_relative(frame, FEATURES)
