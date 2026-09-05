"""S5 — separating "Arsenal changed" from "football changed".

The identity is algebra, not a model:

    (A_new - A_old) = (O_new - O_old) + (C_new - C_old),  where C = A - O

Total change equals what the league did plus what the team did relative to its
peers. It matters because the raw column lies in both directions on real data.
`pi_angle_entropy` moved +1.93 sd across the eras with only +0.18 of that being
Arsenal, and `pi_early_width_y_std` looks nearly flat at -0.87 while hiding a
-2.75 sd Arsenal move against a league going the other way.
"""

import numpy as np
import pandas as pd
import pytest

from src.representation.era import decompose

FEATURES = ["tempo", "width"]


def season(focal: str, focal_values: dict, opponent_values: dict,
           start: int, n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({"match_id": start + i, "team": focal, **focal_values})
        rows.append({"match_id": start + i, "team": f"X{i}", **opponent_values})
    return pd.DataFrame(rows)


def test_the_identity_holds_to_float_precision_on_random_data():
    """The three terms are computed independently, so their agreeing is a
    real check rather than a restatement."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        old = season("A", {f: rng.normal(50, 9) for f in FEATURES},
                     {f: rng.normal(50, 9) for f in FEATURES}, start=0)
        new = season("A", {f: rng.normal(50, 9) for f in FEATURES},
                     {f: rng.normal(50, 9) for f in FEATURES}, start=500)
        got = decompose(old, "A", new, "A", FEATURES)
        residual = got["total_change"] - got["league_moved"] - got["team_moved"]
        assert np.allclose(residual.to_numpy(), 0.0, atol=1e-12)


def test_a_league_wide_shift_is_credited_to_the_league_not_the_team():
    """The `pi_angle_entropy` case: everyone moved, the team moved with them,
    so the team's margin over its opponents is unchanged."""
    old = season("A", {"tempo": 60.0, "width": 10.0},
                 {"tempo": 50.0, "width": 20.0}, start=0)
    new = season("A", {"tempo": 90.0, "width": 40.0},
                 {"tempo": 80.0, "width": 50.0}, start=500)

    got = decompose(old, "A", new, "A", FEATURES)
    assert got.loc["tempo", "total_change"] == pytest.approx(30.0)
    assert got.loc["tempo", "league_moved"] == pytest.approx(30.0)
    assert got.loc["tempo", "team_moved"] == pytest.approx(0.0)


def test_a_team_only_shift_is_credited_to_the_team():
    old = season("A", {"tempo": 60.0, "width": 10.0},
                 {"tempo": 50.0, "width": 20.0}, start=0)
    new = season("A", {"tempo": 75.0, "width": 10.0},
                 {"tempo": 50.0, "width": 20.0}, start=500)

    got = decompose(old, "A", new, "A", FEATURES)
    assert got.loc["tempo", "league_moved"] == pytest.approx(0.0)
    assert got.loc["tempo", "team_moved"] == pytest.approx(15.0)


def test_a_flat_total_can_hide_two_large_opposing_moves():
    """The `pi_early_width_y_std` case, which is the reason the raw column
    alone cannot answer "did this team change?"."""
    old = season("A", {"tempo": 50.0, "width": 0.0},
                 {"tempo": 30.0, "width": 0.0}, start=0)
    new = season("A", {"tempo": 50.0, "width": 0.0},
                 {"tempo": 70.0, "width": 0.0}, start=500)

    got = decompose(old, "A", new, "A", FEATURES)
    assert got.loc["tempo", "total_change"] == pytest.approx(0.0)
    assert got.loc["tempo", "league_moved"] == pytest.approx(40.0)
    assert got.loc["tempo", "team_moved"] == pytest.approx(-40.0)


def test_scaling_divides_every_term_and_keeps_the_identity():
    old = season("A", {"tempo": 60.0, "width": 10.0},
                 {"tempo": 50.0, "width": 20.0}, start=0)
    new = season("A", {"tempo": 90.0, "width": 40.0},
                 {"tempo": 80.0, "width": 50.0}, start=500)
    scale = pd.Series({"tempo": 10.0, "width": 5.0})

    plain = decompose(old, "A", new, "A", FEATURES)
    scaled = decompose(old, "A", new, "A", FEATURES, scale=scale)
    for column in ["total_change", "league_moved", "team_moved"]:
        assert np.allclose(scaled[column], plain[column] / scale)
    residual = scaled["total_change"] - scaled["league_moved"] - scaled["team_moved"]
    assert np.allclose(residual.to_numpy(), 0.0, atol=1e-12)


def test_league_moved_is_the_drift_estimate_negated():
    """Not a coincidence worth discovering later. `league_moved` is `O_new -
    O_old` and `era_offset` is `O_old - O_new`, so the decomposition's league
    term carries exactly the contamination the ledger warns about. Only
    `team_moved` is drift-immune."""
    from src.representation.era import era_offset

    old = season("A", {"tempo": 60.0, "width": 10.0},
                 {"tempo": 50.0, "width": 20.0}, start=0)
    new = season("A", {"tempo": 90.0, "width": 40.0},
                 {"tempo": 80.0, "width": 50.0}, start=500)

    got = decompose(old, "A", new, "A", FEATURES)
    offset = era_offset(old, "A", new, "A", FEATURES)
    assert np.allclose(got["league_moved"], -offset)
