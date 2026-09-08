"""The corpus manifest: one row per season, and the phase's frozen output.

Phase 4 reads this instead of rediscovering what was ingested. Every column is
either declared in the registry or measured by the validators in this package,
so a season cannot appear here without having passed the gate.
"""

from __future__ import annotations

import pandas as pd

from src.extraction.catalog import missing_fixtures
from src.extraction.reconcile import reconcile_season
from src.validation.collection import FOCAL_CLUB, generic_team_density


def build_manifest() -> pd.DataFrame:
    """One row per registry season, declared properties plus measured ones."""
    from src.config import FEATURES_DIR, PROCESSED_DIR
    from src.utils.constants import SEASONS

    rows = []
    for season, entry in SEASONS.items():
        suffix = entry["suffix"]
        matches_path = PROCESSED_DIR / f"matches_{suffix}.csv"
        style_path = FEATURES_DIR / f"match_team_style_{suffix}.csv"
        row = {
            "season": season,
            "competition": entry["competition"],
            "competition_id": entry["competition_id"],
            "season_id": entry["season_id"],
            "suffix": suffix,
            "coverage": entry["coverage"],
            "collection_pass": entry["collection_pass"],
            "role": entry["role"],
            "focal_club": FOCAL_CLUB.get(season, ""),
            "built": matches_path.exists() and style_path.exists(),
        }

        if row["built"]:
            matches = pd.read_csv(matches_path)
            style = pd.read_csv(style_path)
            report = reconcile_season(season)
            row.update({
                "matches": len(matches),
                "clubs": len(set(matches["match_home_team"])
                              | set(matches["match_away_team"])),
                "style_rows": len(style),
                "style_cols": style.shape[1],
                # Only meaningful for full coverage. A single-club season is
                # missing ~345 of a round robin's 380 fixtures by design, and
                # printing that reads as catastrophic incompleteness rather
                # than as the shape the season was always going to have.
                "absent_fixtures": (
                    len(missing_fixtures(matches.rename(columns={
                        "match_home_team": "home_team",
                        "match_away_team": "away_team"})))
                    if entry["coverage"] == "full" else None),
                "possessions_per_team": round(generic_team_density(season), 2),
                "reconciles": bool(report["ok"].all()),
                "clubs_exact": int((report["deficit"] == 0).sum()),
            })
        rows.append(row)

    return pd.DataFrame(rows)


def freeze_manifest() -> "pd.DataFrame":
    """Write the manifest to ``data/features/phase3e_corpus_manifest.csv``."""
    from src.config import FEATURES_DIR

    manifest = build_manifest()
    manifest.to_csv(FEATURES_DIR / "phase3e_corpus_manifest.csv", index=False)
    return manifest
