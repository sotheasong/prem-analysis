"""Does a built season agree with the league it claims to be?

The project's strongest validation, and the only fully independent one. FBref's
advanced statistics are StatsBomb-derived, so agreeing with them tests the
aggregation rather than the measurement. A final league table is different: it
falls out of the results, it is public, and nothing here influenced it.

One rule serves every coverage shape. A club's points deficit against the
official table may not exceed 3 per match the collection is missing for that
club, and no club may hold more points than it actually finished with. The
strictness then follows from the coverage instead of being configured: a
complete season admits no deficit at all, Ligue 1 2015/16 admits exactly its
three absent fixtures, and a single-club season binds the focal club while
saying little about opponents seen twice.
"""

from __future__ import annotations

import pandas as pd

FULL_SEASON_MATCHES = 38
POINTS_FOR_A_WIN = 3


def club_points(matches: pd.DataFrame) -> pd.DataFrame:
    """Points and matches played per club, from the results as collected."""
    rows = []
    for _, m in matches.iterrows():
        home, away = m["match_home_team"], m["match_away_team"]
        hg, ag = (int(x) for x in str(m["match_score"]).split(" - "))
        hp, ap = ((3, 0) if hg > ag else (0, 3) if ag > hg else (1, 1))
        rows.append({"club": home, "points": hp, "played": 1})
        rows.append({"club": away, "points": ap, "played": 1})
    return (pd.DataFrame(rows).groupby("club")
            .agg(points=("points", "sum"), played=("played", "sum")))


def reconcile(matches: pd.DataFrame, official: dict,
              full_season_matches: int = FULL_SEASON_MATCHES) -> pd.DataFrame:
    """Per-club comparison of collected points against the official table.

    ``official`` maps club -> (position, points), keyed by the same names the
    match list uses. A club in the official table but absent from the collection
    is reported with 0 played, not dropped.
    """
    built = club_points(matches)
    rows = []
    for club, (_position, official_points) in official.items():
        points = int(built.loc[club, "points"]) if club in built.index else 0
        played = int(built.loc[club, "played"]) if club in built.index else 0
        missing = full_season_matches - played
        deficit = official_points - points
        max_deficit = POINTS_FOR_A_WIN * missing
        rows.append({
            "club": club,
            "played": played,
            "points": points,
            "official_points": official_points,
            "deficit": deficit,
            "max_deficit": max_deficit,
            "ok": 0 <= deficit <= max_deficit,
        })
    return pd.DataFrame(rows).set_index("club").sort_values(
        "official_points", ascending=False)


def reconcile_season(season: str) -> pd.DataFrame:
    """Reconcile one registry season against its official final table.

    Raises ``FileNotFoundError`` when the season has not been built, so callers
    can skip rather than guess. ``data/`` is gitignored, so that is the normal
    state on a fresh clone.
    """
    from src.config import PROCESSED_DIR
    from src.utils.constants import OFFICIAL_TABLES, SEASONS

    if season not in SEASONS:
        raise KeyError(f"{season!r} is not in the season registry")
    if season not in OFFICIAL_TABLES:
        raise KeyError(f"{season!r} has no official table to check against")

    path = PROCESSED_DIR / f"matches_{SEASONS[season]['suffix']}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return reconcile(pd.read_csv(path), OFFICIAL_TABLES[season])
