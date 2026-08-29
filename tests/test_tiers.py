"""S2 — the five-tier classification, the diagnostic's headline product.

Each feature lands in exactly one tier:

    drop          no variance at all
    redundant     collinear with a lower-CV feature in the same cluster
    context       well explained by the design, significantly so
    identity      stable and not context-explained
    experimental  everything else

The fixture is built from mutually orthogonal +/-1 contrasts over eight matches,
so every correlation in it is exactly 0 or exactly 1 and each verdict follows
without recomputing the rules. ``h1`` drives venue; ``h2``, ``h5`` and ``h6``
are orthogonal to it and to each other, so nothing clusters by accident.
"""

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.style import diagnose, team_season

H1 = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)   # venue
H2 = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
H4 = H1 * H2
H5 = H1 * np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
H6 = H2 * np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)

# Venue is the only covariate here, so the fixture stays exactly controllable.
VENUE_ONLY = lambda ts: pd.DataFrame({"home": ts.frame["home"]})


def frame(**features):
    return pd.DataFrame({
        "team": "T",
        "date": pd.date_range("2015-08-08", periods=8, freq="7D"),
        "venue": np.where(H1 > 0, "home", "away"),
        **features,
    })


@pytest.fixture(scope="module")
def diagnosis():
    venue_driven = 100 + 5 * H1 + 0.5 * H4     # mostly venue, a little residual
    return diagnose(
        team_season(frame(
            constant=7.0,
            venue_driven=venue_driven,
            venue_driven_rescaled=2 * venue_driven - 195,   # exactly collinear
            steady_one=10 + 0.03 * H2,
            steady_two=4 + 0.02 * H6,
            wild=20 + 15 * H5,
        ), "T"),
        design=VENUE_ONLY,
    )


@pytest.fixture(scope="module")
def tiers(diagnosis):
    return diagnosis.registry.set_index("feature")["tier"].to_dict()


def test_constant_feature_is_dropped(tiers):
    assert tiers["constant"] == "drop"


def test_design_explained_feature_is_context(tiers):
    assert tiers["venue_driven"] == "context"


def test_collinear_copy_is_redundant_not_context(tiers):
    """Same information, higher CV, so it defers to its cluster representative."""
    assert tiers["venue_driven_rescaled"] == "redundant"


def test_stable_unexplained_features_are_identity_anchors(tiers):
    assert tiers["steady_one"] == "identity"
    assert tiers["steady_two"] == "identity"


def test_unstable_unexplained_feature_is_experimental(tiers):
    assert tiers["wild"] == "experimental"


def test_cluster_membership_and_representative_are_recorded(diagnosis):
    reg = diagnosis.registry.set_index("feature")
    pair = ["venue_driven", "venue_driven_rescaled"]
    assert reg.loc[pair, "cluster"].nunique() == 1
    assert reg.loc["venue_driven", "representative"]
    assert not reg.loc["venue_driven_rescaled", "representative"]
    assert reg.loc["steady_one", "cluster"] == -1
    assert reg.loc["steady_one", "representative"]


def test_retained_is_identity_plus_context(diagnosis):
    reg = diagnosis.registry.set_index("feature")
    assert set(reg.index[reg["retained"]]) == {
        "venue_driven", "steady_one", "steady_two"}


def test_every_feature_is_classified_exactly_once(diagnosis):
    reg = diagnosis.registry
    assert len(reg) == len(diagnosis.ts.features)
    assert reg["feature"].is_unique
    assert reg["tier"].notna().all()


def test_a_perfectly_explained_feature_reports_no_significance():
    """Zero residual means no evidence, not a NaN p-value leaking into a tier."""
    d = diagnose(team_season(frame(exact=100 + 10 * H1), "T"), design=VENUE_ONLY)
    fit = d.fit("exact")
    assert fit.se["home"] == 0.0
    assert fit.p["home"] == 1.0
    assert d.registry.set_index("feature").loc["exact", "tier"] != "context"
