"""S3 — measuring collection drift without letting the focal team into it.

The drift estimate exists to be subtracted from the focal team. Estimating it
from a sample that is half focal-team rows subtracts the team's own style away
as if it were an artifact. On the real 2003/04 table the two estimators sit a
median 0.73 sd apart across the 49 anchors, so this is not a fine point.

The rule: the baseline is the focal team's opponents, in the focal team's own
matches, on both sides of the comparison.
"""

import pandas as pd
import pytest

from src.representation.era import era_offset

FEATURES = ["tempo"]


def season(focal: str, opponents: list[str], focal_value: float,
           opponent_value: float, start: int) -> pd.DataFrame:
    rows = []
    for i, opp in enumerate(opponents):
        for team, value in ((focal, focal_value), (opp, opponent_value)):
            rows.append({"match_id": start + i, "team": team, "tempo": value})
    return pd.DataFrame(rows)


OLD = season("Arsenal", [f"O{i}" for i in range(6)], 90.0, 50.0, start=0)
NEW = season("Arsenal", [f"N{i}" for i in range(6)], 70.0, 30.0, start=100)


def test_offset_is_the_gap_between_the_two_opponent_pools():
    """Old opponents sit at 50, new opponents at 30, so the era drift is +20.
    Arsenal's own 90 and 70 play no part in it."""
    offset = era_offset(OLD, "Arsenal", NEW, "Arsenal", FEATURES)
    assert offset["tempo"] == pytest.approx(20.0)


def test_making_the_focal_team_absurd_does_not_move_the_offset():
    """The property the whole estimator exists for. If Arsenal could shift its
    own drift estimate, the correction would erase Arsenal's style."""
    absurd = OLD.copy()
    absurd.loc[absurd["team"] == "Arsenal", "tempo"] = 9999.0

    assert era_offset(absurd, "Arsenal", NEW, "Arsenal", FEATURES)["tempo"] == \
        pytest.approx(era_offset(OLD, "Arsenal", NEW, "Arsenal", FEATURES)["tempo"])


def test_matches_the_focal_team_did_not_play_are_ignored():
    """2003/04 holds only Arsenal fixtures, but 2015/16 holds all 380. The
    baseline has to mean the same thing in both, so it is always restricted to
    the focal team's own matches."""
    intruder = pd.DataFrame({
        "match_id": [900, 900], "team": ["O0", "O1"], "tempo": [0.0, 0.0],
    })
    padded = pd.concat([OLD, intruder], ignore_index=True)
    assert era_offset(padded, "Arsenal", NEW, "Arsenal", FEATURES)["tempo"] == \
        pytest.approx(20.0)


def test_the_two_sides_may_have_different_focal_teams():
    """Notebook 3d-ii compares Arsenal 2003/04 against Leicester 2015/16, so
    each season's baseline is that season's champion's opponents."""
    new_lei = season("Leicester", [f"N{i}" for i in range(6)], 60.0, 25.0, start=200)
    assert era_offset(OLD, "Arsenal", new_lei, "Leicester", FEATURES)["tempo"] == \
        pytest.approx(25.0)


def test_a_focal_team_absent_from_the_frame_is_refused():
    with pytest.raises(KeyError, match="Chelsea"):
        era_offset(OLD, "Chelsea", NEW, "Arsenal", FEATURES)
