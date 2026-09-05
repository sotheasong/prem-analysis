"""S2 — a contrast field is a style space over different numbers.

`opponent_relative` returns a match-level frame on purpose, so Phase 3c's
`style_space` builds a field from it with no changes at all. This is the whole
reason 3d needs one new function rather than a parallel copy of 3c.

Two properties earn the design. A balanced schedule makes the league mean of
every contrast exactly zero, because each match contributes +d and -d. And a
season-wide collection offset added to every row leaves the contrast field
untouched, which is the drift cancellation stated as a test rather than a hope.
"""

import numpy as np
import pandas as pd
import pytest

from src.representation.era import opponent_relative
from src.representation.style_space import style_space

TEAMS = [f"T{i}" for i in range(8)]
FEATURES = ["tempo", "width"]
# A double round-robin: every club plays every other twice, 14 matches each.
STRENGTH = {t: float(i) for i, t in enumerate(TEAMS)}


def round_robin(offset: float = 0.0) -> pd.DataFrame:
    """Each club's value depends only on itself, so contrasts are predictable.

    ``offset`` is added to every row, standing in for a season-wide collection
    artifact that inflates both sides of every fixture equally.
    """
    rows, match_id = [], 0
    for home in TEAMS:
        for away in TEAMS:
            if home == away:
                continue
            match_id += 1
            for team in (home, away):
                rows.append({
                    "match_id": match_id,
                    "team": team,
                    "tempo": offset + 10.0 * STRENGTH[team],
                    "width": offset + 100.0 - 3.0 * STRENGTH[team],
                })
    return pd.DataFrame(rows)


def contrast_field(offset: float = 0.0):
    return style_space(opponent_relative(round_robin(offset), FEATURES),
                       FEATURES, min_matches=14)


def test_a_balanced_schedule_centres_every_contrast_on_zero():
    profiles = contrast_field().profiles
    assert np.allclose(profiles.mean().to_numpy(), 0.0, atol=1e-12)


def test_a_season_wide_offset_leaves_the_contrast_field_identical():
    """The point of the whole phase. A collection artifact that inflates both
    teams in every match by 40 units changes nothing a contrast can see."""
    plain, inflated = contrast_field(0.0), contrast_field(40.0)
    pd.testing.assert_frame_equal(plain.profiles, inflated.profiles)
    pd.testing.assert_frame_equal(plain.z, inflated.z)
    pd.testing.assert_frame_equal(plain.percentiles, inflated.percentiles)


def test_the_contrast_field_ranks_clubs_by_the_margin_they_generate():
    """Strength was built to run T0 (weakest) to T7 (strongest) on tempo."""
    profiles = contrast_field().profiles
    assert list(profiles["tempo"].sort_values().index) == TEAMS
    assert list(profiles["width"].sort_values().index) == TEAMS[::-1]


def test_a_club_margin_matches_the_hand_worked_value():
    """T7 faces the other seven clubs, mean strength 3.0, so its tempo margin is
    10 * (7 - 3) = 40. Derived from the fixture's construction, not the code."""
    profiles = contrast_field().profiles
    assert profiles.loc["T7", "tempo"] == pytest.approx(40.0)
    assert profiles.loc["T0", "tempo"] == pytest.approx(10.0 * (0 - 4.0))
