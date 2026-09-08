"""Is a `collection_pass` tag a real thing, or a label we assigned?

Phase 3d established that drift belongs to the collection rather than the year:
2003/04 segmented possessions more finely than 2015/16, so a raw cross-era
comparison measured StatsBomb's segmentation and called it football. The season
registry carries a `collection_pass` tag encoding which seasons share a
collection, and until this module that tag was an assumption.

Possession count per team is the measure, because it is what 3d caught drifting
and it is structural rather than stylistic.

The trap is that a single-club season measures its opponents *only in matches
against the focal club*, and facing a possession-dominant side suppresses the
possession count for both teams. Comparing those opponents to a full league
therefore confounds "different collection" with "played Barcelona". Within
La Liga 2015/16, a single pass, that effect is about -16 possessions, which is
larger than the cross-pass gap it would otherwise appear to prove.
"""

from __future__ import annotations

import pandas as pd

# Seasons whose coverage is one club's fixture list, and that club.
FOCAL_CLUB = {"2003/2004": "Arsenal",
              **{f"Barcelona {y - 1}/{y}": "Barcelona" for y in range(2008, 2016)}}


def _season_style(season: str, columns: list[str]) -> pd.DataFrame:
    from src.config import FEATURES_DIR
    from src.utils.constants import SEASONS

    path = FEATURES_DIR / f"match_team_style_{SEASONS[season]['suffix']}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=columns)


def generic_team_density(season: str) -> float:
    """Mean possessions per team, excluding the focal club of a single-club season.

    The focal club is dropped for the same reason ``era.py`` keeps it out of a
    drift estimate: including a possession-dominant side measures that side's
    style rather than the collection's granularity.
    """
    style = _season_style(season, ["team", "poss_possessions"])
    focal = FOCAL_CLUB.get(season)
    generic = style[style["team"] != focal] if focal else style
    return float(generic["poss_possessions"].mean())


def facing_club_effect(season: str, club: str) -> dict:
    """How much facing one club moves its opponents' possession count.

    Measured inside a single season, so the collection is held constant and
    whatever this returns is football rather than drift. It is the correction
    that has to be applied before a single-club season can be compared with a
    full league.
    """
    style = _season_style(season, ["match_id", "team", "poss_possessions"])
    their_matches = set(style.loc[style["team"] == club, "match_id"])
    if not their_matches:
        raise KeyError(f"{club!r} does not appear in {season!r}")

    others = style[style["team"] != club]
    against = others[others["match_id"].isin(their_matches)]["poss_possessions"]
    elsewhere = others[~others["match_id"].isin(their_matches)]["poss_possessions"]
    return {
        "club": club,
        "season": season,
        "vs_club": float(against.mean()),
        "vs_everyone_else": float(elsewhere.mean()),
        "effect": float(against.mean() - elsewhere.mean()),
        "n_vs_club": int(len(against)),
    }
