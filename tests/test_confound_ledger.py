"""S4 — the ledger that says how much of a cross-era gap is the collection.

Placing a 2003/04 profile in the 2015/16 field produces a number for every
anchor. The ledger's job is to say which of those numbers mean anything. It
divides the drift, measured off other teams entirely, by the raw gap, and the
field's standard deviation cancels out of that ratio, so the share is scale-free.

The fixture is built so every share is hand-derivable: share = offset / gap,
with no reference to how the code computes it.
"""

import numpy as np
import pandas as pd
import pytest

from src.representation.era import confound_ledger

CLUBS = [f"C{i}" for i in range(9)]
FEATURES = ["clean", "mixed", "dominated", "flat"]

# New-season club i sits at 10*i on every feature, so the field mean is 40 and
# C4's opponent pool (indices 0-3, 5-8) averages exactly 4, i.e. 40.
# Old season: opponents carry the drift, Arsenal carries drift plus real style.
#   feature      opponents   Arsenal   offset   gap   share
#   clean            42         60        2      20    0.10
#   mixed            52         64       12      24    0.50
#   dominated        64         65       24      25    0.96
#   flat             50         40       10       0    undefined
OLD_OPPONENT = {"clean": 42.0, "mixed": 52.0, "dominated": 64.0, "flat": 50.0}
OLD_FOCAL = {"clean": 60.0, "mixed": 64.0, "dominated": 65.0, "flat": 40.0}


def new_season() -> pd.DataFrame:
    rows, mid = [], 0
    for home in CLUBS:
        for away in CLUBS:
            if home == away:
                continue
            mid += 1
            for team in (home, away):
                value = 10.0 * CLUBS.index(team)
                rows.append({"match_id": mid, "team": team,
                             **{f: value for f in FEATURES}})
    return pd.DataFrame(rows)


def old_season() -> pd.DataFrame:
    rows = []
    for i in range(16):
        rows.append({"match_id": 1000 + i, "team": "Arsenal", **OLD_FOCAL})
        rows.append({"match_id": 1000 + i, "team": f"P{i % 8}", **OLD_OPPONENT})
    return pd.DataFrame(rows)


def ledger(**kwargs) -> pd.DataFrame:
    return confound_ledger(old_season(), "Arsenal", new_season(), "C4",
                           FEATURES, min_matches=16, **kwargs)


def test_share_is_the_offset_over_the_raw_gap_and_is_scale_free():
    got = ledger()["offset_share"]
    assert got["clean"] == pytest.approx(2 / 20)
    assert got["mixed"] == pytest.approx(12 / 24)
    assert got["dominated"] == pytest.approx(24 / 25)


def test_verdicts_split_at_the_agreed_thresholds():
    got = ledger()["verdict"]
    assert got["clean"] == "clean"
    assert got["mixed"] == "mixed"
    assert got["dominated"] == "offset-dominated"


def test_the_thresholds_are_inclusive_lower_bounds():
    """A share landing exactly on a boundary belongs to the worse bucket.

    The threshold is the ledger's own computed share rather than the literal
    0.10, because the arithmetic lands a ulp below it (0.09999999999999999).
    Feeding the value back is what makes this a real test of `>=` against `>`
    instead of a test of float equality, which no implementation could pass.
    """
    shares = ledger()["offset_share"]
    assert ledger(mixed=shares["clean"])["verdict"]["clean"] == "mixed"
    assert ledger(dominated=shares["mixed"])["verdict"]["mixed"] == "offset-dominated"


def test_a_feature_with_no_raw_gap_gets_no_share_to_divide():
    """An undefined ratio must not be dressed up as a verdict."""
    row = ledger().loc["flat"]
    assert np.isnan(row["offset_share"])
    assert row["verdict"] == "no-gap"


def test_the_contrast_collapses_exactly_where_the_ledger_says_it_should():
    """The ledger's claim, checked against the lens it is a ledger for. On the
    offset-dominated feature the contrast must lose nearly all of the raw gap;
    on the clean one it must keep most of it."""
    got = ledger()
    dominated = abs(got.loc["dominated", "contrast_z"] / got.loc["dominated", "raw_z"])
    clean = abs(got.loc["clean", "contrast_z"] / got.loc["clean", "raw_z"])
    assert dominated < 0.10
    assert clean > 0.70


def test_every_requested_feature_gets_exactly_one_row():
    got = ledger()
    assert list(got.index) == FEATURES
    assert list(got.columns) == ["raw_z", "offset", "offset_sd", "offset_share",
                                 "verdict", "contrast_z"]
