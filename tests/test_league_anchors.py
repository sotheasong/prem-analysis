"""S4 — the identity-anchor tier, measured across a whole league.

Phases 2 and 3b each froze a registry for one team, and "identity anchor" there
means "stable for that team". Arsenal's 54 and Leicester's 52 overlap on only
42, so neither is a league-wide axis. ``league_anchors`` runs the diagnostic on
every team and keeps the features that anchor for at least ``min_teams`` of
them, which makes the tier a measured property of the feature.

The fixture is constructed so no classification is left to chance: every
feature is exactly orthogonal to the context design (context R^2 = 0) and to
every other feature (nothing clusters), so a feature's tier follows from its CV
alone, and its CV is set by an amplitude chosen per team.
"""

import numpy as np
import pandas as pd
import pytest

from src.representation.style_space import league_anchors

TEAMS = ["Ayton", "Beeches", "Carrow", "Dell", "Elland", "Fratton"]
MATCHES = 38
DATES = pd.date_range("2015-08-08", periods=MATCHES, freq="7D")
VENUE = np.where(np.arange(MATCHES) % 2 == 0, "home", "away")
# A fixed non-monotonic schedule: opponent strength must be separable from
# both season drift and venue, or each club's design matrix is rank deficient.
OPP_POSITION = [((i * 7) % 20) + 1 for i in range(MATCHES)]

# Amplitudes, and so CVs, per feature per team. `universal` is tiny everywhere;
# `swing` is tiny for the first three clubs and wild for the last three;
# `never` is wild everywhere. The fillers only pad the feature set so the
# per-team CV median is well defined; their own verdicts are not under test.
TINY, MID, WILD = 0.02, 1.0, 40.0
FILLERS = [f"filler_{i}" for i in range(5)]


def orthogonal_patterns(n_patterns):
    """Unit patterns orthogonal to the context design and to each other."""
    days = (DATES - DATES[0]).days.to_numpy(dtype=float)
    home = (VENUE == "home").astype(float)
    design = np.column_stack([
        np.ones(MATCHES), days, home, 21 - np.array(OPP_POSITION, dtype=float)])
    seed = np.eye(MATCHES)[:, 4:4 + n_patterns]
    residual = seed - design @ np.linalg.lstsq(design, seed, rcond=None)[0]
    q, _ = np.linalg.qr(residual)
    return q


@pytest.fixture(scope="module")
def league():
    patterns = orthogonal_patterns(2 + len(FILLERS) + 1)
    columns = ["universal", "swing", *FILLERS, "never"]
    rows = []
    for team in TEAMS:
        swing = TINY if team in TEAMS[:3] else WILD
        amplitude = [TINY, swing, *([MID] * len(FILLERS)), WILD]
        block = pd.DataFrame({
            "team": team,
            "date": DATES,
            "venue": VENUE,
            "opp_final_position": OPP_POSITION,
        })
        for i, (name, amp) in enumerate(zip(columns, amplitude)):
            block[name] = 100 + amp * patterns[:, i]
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def test_a_feature_stable_for_every_club_is_an_anchor(league):
    features = league_anchors(league, min_teams=len(TEAMS)).features
    assert "universal" in features
    assert "swing" not in features        # stable for only three of the six
    assert "never" not in features


def test_the_threshold_selects_how_many_clubs_must_agree(league):
    """`swing` anchors for exactly the three clubs given the tiny amplitude."""
    assert "swing" in league_anchors(league, min_teams=3).features
    assert "swing" not in league_anchors(league, min_teams=4).features


def test_the_spine_is_what_every_club_agrees_on(league):
    anchors = league_anchors(league, min_teams=3)
    assert "universal" in anchors.spine
    assert "swing" not in anchors.spine


def test_counts_report_how_many_clubs_each_feature_anchors_for(league):
    counts = league_anchors(league, min_teams=1).counts
    assert counts["universal"] == 6
    assert counts["swing"] == 3
    assert counts["never"] == 0


def test_raising_the_threshold_can_only_shrink_the_tier(league):
    wide = set(league_anchors(league, min_teams=2).features)
    narrow = set(league_anchors(league, min_teams=5).features)
    assert narrow <= wide


def test_a_partial_coverage_season_is_refused(league):
    """2003/04 is Arsenal-only, so a league-wide tier is not defined for it."""
    thin = league[league["team"].isin(TEAMS[:2]) | (league.index % 10 == 0)]
    with pytest.raises(ValueError, match="coverage"):
        league_anchors(thin, min_teams=3)
