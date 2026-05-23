from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

from src.config import RAW_DIR


def get_epl_matches() -> pd.DataFrame:
    comps = sb.competitions()
    epl = comps[comps["competition_name"] == "Premier League"]
    all_matches = []
    for _, row in epl.iterrows():
        comp_id = row["competition_id"]
        season_id = row["season_id"]
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        matches["competition_id"] = comp_id
        matches["season_id"] = season_id
        all_matches.append(matches)
    return pd.concat(all_matches, ignore_index=True)


def _parts_dir(raw_dir: Path) -> Path:
    return raw_dir / "events_parts"


def _migrate_csv_to_parts(raw_dir: Path) -> None:
    """Split an existing events.csv into per-match parquet parts for resume."""
    events_path = raw_dir / "events.csv"
    parts_dir = _parts_dir(raw_dir)
    if not events_path.exists() or list(parts_dir.glob("*.parquet")):
        return
    print("Migrating existing events.csv into events_parts/ ...")
    parts_dir.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(events_path, low_memory=False)
    for match_id, grp in existing.groupby("match_id"):
        grp.to_parquet(parts_dir / f"{int(match_id)}.parquet", index=False)


def _existing_match_ids(raw_dir: Path, resume: bool) -> set[int]:
    if not resume:
        return set()
    parts = _parts_dir(raw_dir)
    if parts.exists() and list(parts.glob("*.parquet")):
        return {int(p.stem) for p in parts.glob("*.parquet")}
    return set()


def get_match_events(match_id: int) -> pd.DataFrame:
    events = sb.events(match_id=match_id)
    events["match_id"] = match_id
    return events


def extract_all_events(
    matches_df: pd.DataFrame,
    raw_dir: Path | str = RAW_DIR,
    *,
    resume: bool = True,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = _parts_dir(raw_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        _migrate_csv_to_parts(raw_dir)

    match_ids = matches_df["match_id"].unique().tolist()
    done = _existing_match_ids(raw_dir, resume)
    pending = [mid for mid in match_ids if mid not in done]

    errors: list[dict] = []
    errors_path = raw_dir / "extraction_errors.json"

    if resume and done:
        print(f"Resuming: skipping {len(done)} matches already on disk, {len(pending)} remaining.")

    for match_id in tqdm(pending, desc="Extracting events"):
        try:
            df = get_match_events(match_id)
            df.to_parquet(parts_dir / f"{match_id}.parquet", index=False)
        except Exception as e:
            errors.append({"match_id": int(match_id), "error": str(e)})
            print(f"Error processing match {match_id}: {e}")

    if errors:
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Logged {len(errors)} failures to {errors_path}")

    part_files = sorted(parts_dir.glob("*.parquet"))
    if not part_files:
        raise RuntimeError("No event files extracted.")

    return pd.concat(
        (pd.read_parquet(p) for p in part_files),
        ignore_index=True,
    )


def save_raw(matches_df: pd.DataFrame, events_df: pd.DataFrame, raw_dir: Path | str = RAW_DIR) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(raw_dir / "matches.csv", index=False)
    events_df.to_csv(raw_dir / "events.csv", index=False)


def event_coverage_report(
    matches_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    match_meta = matches_df[["match_id", "season"]].drop_duplicates()
    events_with_season = events_df[["match_id"]].merge(match_meta, on="match_id", how="left")

    matches_by_season = match_meta.groupby("season").agg(matches=("match_id", "nunique"))
    events_by_season = events_with_season.groupby("season").size().rename("events")
    events_matches = events_with_season.groupby("season")["match_id"].nunique().rename("matches_with_events")

    coverage = matches_by_season.join(events_matches, how="left").join(events_by_season, how="left")
    coverage = coverage.fillna(0).astype(
        {"matches_with_events": int, "events": int}
    )
    coverage["missing_matches"] = coverage["matches"] - coverage["matches_with_events"]
    return coverage


def run_full_extraction(raw_dir: Path | str = RAW_DIR, *, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_df = get_epl_matches()
    events_df = extract_all_events(matches_df, raw_dir, resume=resume)
    save_raw(matches_df, events_df, raw_dir)
    return matches_df, events_df
