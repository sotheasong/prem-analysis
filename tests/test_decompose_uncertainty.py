"""S6 — error bars on the decomposition, and the multiplicity correction.

"Arsenal moved +3.83 sd" is a point estimate off 38 matches. Without an interval
it gets read as a fact. Matches are resampled whole, so all three terms are
recomputed from the same resample and the identity survives inside the bootstrap
rather than being imposed on top of it.

Benjamini-Hochberg is separate: 49 anchors tested at once means two or three
flag by chance, so the headline spine claims have to clear an FDR correction.
"""

import numpy as np
import pandas as pd
import pytest

from src.representation.era import benjamini_hochberg, decompose

FEATURES = ["shifted", "noise"]
RNG = np.random.default_rng(7)


def season(start: int, focal_shift: float) -> pd.DataFrame:
    """38 matches. `shifted` gives the focal team a real margin of
    ``focal_shift``; `noise` gives it none, only match-to-match scatter."""
    rows = []
    for i in range(38):
        wobble = RNG.normal(0, 1.0, size=4)
        rows.append({"match_id": start + i, "team": "A",
                     "shifted": 50 + focal_shift + wobble[0],
                     "noise": 50 + wobble[1]})
        rows.append({"match_id": start + i, "team": f"X{i}",
                     "shifted": 50 + wobble[2], "noise": 50 + wobble[3]})
    return pd.DataFrame(rows)


OLD, NEW = season(0, focal_shift=0.0), season(500, focal_shift=8.0)


def run(**kwargs):
    return decompose(OLD, "A", NEW, "A", FEATURES, n_resamples=2000, seed=0, **kwargs)


def test_the_same_seed_gives_the_same_interval():
    pd.testing.assert_frame_equal(run(), run())


def test_a_real_shift_produces_an_interval_clear_of_zero():
    got = run().loc["shifted"]
    assert got["team_lo"] > 0
    assert got["team_hi"] > got["team_lo"]
    assert got["team_p"] < 0.01


def test_pure_noise_produces_an_interval_containing_zero():
    got = run().loc["noise"]
    assert got["team_lo"] < 0 < got["team_hi"]
    assert got["team_p"] > 0.05


def test_the_interval_brackets_the_point_estimate():
    got = run()
    for term in ["total", "league", "team"]:
        point = got[f"{term}_moved" if term != "total" else "total_change"]
        assert (got[f"{term}_lo"] <= point).all()
        assert (point <= got[f"{term}_hi"]).all()


def test_scaling_moves_the_interval_with_the_estimate():
    scale = pd.Series({"shifted": 4.0, "noise": 4.0})
    plain, scaled = run(), run(scale=scale)
    assert np.allclose(scaled["team_lo"], plain["team_lo"] / 4.0)
    assert np.allclose(scaled["team_p"], plain["team_p"])


def test_benjamini_hochberg_matches_a_worked_example():
    """Five p-values at alpha=0.05, so the rank thresholds are 0.01 .. 0.05.
    Sorted: 0.001<=0.01, 0.008<=0.02, 0.025<=0.03, 0.045>0.04, 0.9>0.05.
    The largest passing rank is 3, so the three smallest are kept."""
    p = pd.Series({"a": 0.001, "b": 0.008, "c": 0.025, "d": 0.045, "e": 0.9})
    kept = benjamini_hochberg(p, alpha=0.05)
    assert kept.to_dict() == {"a": True, "b": True, "c": True,
                              "d": False, "e": False}


def test_benjamini_hochberg_keeps_a_buried_small_p_below_the_cutoff():
    """The step-up property. Sorted: 0.009<=0.01 passes, 0.021>0.02 FAILS its
    own rank, 0.022<=0.03 and 0.023<=0.04 pass, 0.06>0.05 fails. The largest
    passing rank is 4, so everything at or below 0.023 is kept, which rescues
    the 0.021 that failed on its own. A per-rank filter would drop it."""
    p = pd.Series({"a": 0.009, "b": 0.021, "c": 0.022, "d": 0.023, "e": 0.06})
    kept = benjamini_hochberg(p, alpha=0.05)
    assert kept.to_dict() == {"a": True, "b": True, "c": True,
                              "d": True, "e": False}
