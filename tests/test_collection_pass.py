"""S9 — is `collection_pass` measured, or just asserted?

The registry tags each season with the collection that produced it, and the tag
decides something consequential: seasons sharing a pass may be compared raw,
across passes the paired contrast is mandatory. Until this file the tag was a
hypothesis written into a dict.

Possessions per team is the measure, because that is the quantity 3d caught
drifting (2003/04 logged 108.9 a match where 2015/16 logged 95.1) and it is
structural rather than stylistic.
"""

import pytest

from src.validation.collection import facing_club_effect, generic_team_density

BIG5_2015_16 = ["2015/2016", "La Liga 2015/2016",
                "Serie A 2015/2016", "Ligue 1 2015/2016"]


def density(season: str) -> float:
    try:
        return generic_team_density(season)
    except FileNotFoundError:
        pytest.skip(f"{season} not built (data/ is gitignored)")


def test_2003_04_segments_possessions_more_finely_than_any_2015_16_league():
    """The known-positive. This is the drift 3d had to correct for, and it must
    still be visible or the measure is not sensitive enough to trust elsewhere.

    The confound runs *against* this finding rather than for it: 2003/04's
    figure comes from Arsenal's opponents, measured only while facing a
    possession-dominant side, which suppresses possession counts. It reads
    higher anyway.
    """
    old = density("2003/2004")
    for season in BIG5_2015_16:
        assert old > density(season) + 5, (
            f"2003/04 reads {old:.1f} against {season} at {density(season):.1f}")


def test_facing_a_dominant_club_moves_possessions_more_than_a_pass_change_does():
    """The correction that stops a false positive, measured inside one season so
    the collection is held constant and the effect can only be football.

    Barcelona's La Liga 2015/16 opponents record about sixteen fewer possessions
    than the same league records elsewhere. Any comparison between a
    Barcelona-only season and a full league inherits that gap before a single
    question about collections is asked.
    """
    try:
        effect = facing_club_effect("La Liga 2015/2016", "Barcelona")
    except FileNotFoundError:
        pytest.skip("La Liga 2015/2016 not built")

    assert effect["n_vs_club"] == 38
    assert effect["effect"] < -10, effect


def test_the_messi_seasons_are_not_shown_to_be_a_separate_pass():
    """A negative result, recorded because it contradicts the registry.

    Barcelona 2014/15 reads about 14 possessions below La Liga 2015/16, which
    looks like a pass boundary until the facing-Barcelona effect is subtracted.
    That effect is about 16, so the raw gap is *smaller* than the confound and
    the residual points the other way. This measure gives no evidence the two
    are different collections.
    """
    try:
        gap = density("La Liga 2015/2016") - density("Barcelona 2014/2015")
        confound = -facing_club_effect("La Liga 2015/2016", "Barcelona")["effect"]
    except FileNotFoundError:
        pytest.skip("seasons not built")

    assert gap > 0                       # the raw gap looks like a boundary
    assert confound > gap, (             # and the confound more than covers it
        f"raw gap {gap:.1f} exceeds the facing-Barcelona effect {confound:.1f}, "
        f"which would make the pass boundary real after all")
