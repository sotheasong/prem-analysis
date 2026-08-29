"""S2 — redundancy: collinear features collapsed into clusters.

Features with |r| above the threshold are joined into connected components, so
that one representative per component can be kept before modeling. Truth here is
constructed: a perfect copy, a sign-flipped copy, and a chain that is only
transitively linked.
"""

import numpy as np
import pandas as pd

from src.diagnostics.style import TierPolicy, diagnose, team_season

BASE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
NOISE = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]


def make(policy=None, **features):
    frame = pd.DataFrame({
        "team": "T",
        "date": pd.date_range("2015-08-08", periods=8, freq="7D"),
        "venue": ["home", "away"] * 4,
        "opp_final_position": [3, 12, 7, 1, 15, 5, 9, 18],
        **features,
    })
    kwargs = {"policy": policy} if policy is not None else {}
    return diagnose(team_season(frame, "T"), **kwargs)


def as_sets(clusters):
    return sorted((sorted(c) for c in clusters), key=len, reverse=True)


def test_a_perfect_copy_is_one_cluster():
    d = make(metric=BASE, copy_of_metric=[2 * v + 1 for v in BASE], other=NOISE)
    assert as_sets(d.clusters) == [["copy_of_metric", "metric"]]


def test_sign_flipped_features_are_redundant_too():
    """A mirror image carries the same information as the original."""
    d = make(metric=BASE, mirror=[-v for v in BASE], other=NOISE)
    assert as_sets(d.clusters) == [["metric", "mirror"]]


def test_independent_features_form_no_cluster():
    d = make(metric=BASE, other=NOISE)
    assert d.clusters == []


def test_transitively_linked_features_collapse_into_one_component():
    """a~b and b~c, but a and c are only linked through b.

    Built so the chain is provable: a = b + k*u and c = b + k*v with u and v
    orthogonal to each other and to b, which makes corr(a, c) = corr(a, b)^2.
    Pick k so that correlation lands at 0.85, under the 0.90 threshold, while
    each link sits at sqrt(0.85) = 0.922, over it.
    """
    b = np.array(BASE)
    u = np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=float)
    v = np.array([1, 1, -1, -1, -1, -1, 1, 1], dtype=float)
    k = np.sqrt(b.var() / 0.85 - b.var())
    a, c = b + k * u, b + k * v

    corr = np.corrcoef([a, b, c])
    assert abs(corr[0, 1]) > 0.9 and abs(corr[1, 2]) > 0.9    # links exist
    assert abs(corr[0, 2]) < 0.9                              # ...but not directly

    d = make(a=list(a), b=list(b), c=list(c))
    assert as_sets(d.clusters) == [["a", "b", "c"]]


def test_threshold_is_configurable():
    """Two unrelated features cluster once the bar is low enough."""
    d = make(metric=BASE, other=NOISE, policy=TierPolicy(corr_hi=0.1))
    assert as_sets(d.clusters) == [["metric", "other"]]
