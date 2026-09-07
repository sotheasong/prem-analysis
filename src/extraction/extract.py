from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsbombpy import public, sb
from tqdm import tqdm

from src.config import PROCESSED_DIR, RAW_DIR
from src.extraction.events_format import is_normalized, normalize_events_df
from src.utils.constants import SEASONS


def season_plan(season: str, processed_dir: Path | str = PROCESSED_DIR) -> dict:
    """What a season build fetches and where it writes, from the registry.

    Deriving this rather than passing identifiers by hand is what stops a
    season's events being written under another season's suffix, which would
    silently replace a build. Raises ``KeyError`` for a season the registry
    does not declare, before anything is fetched.
    """
    if season not in SEASONS:
        raise KeyError(
            f"{season!r} is not in the season registry. Declare it in "
            f"src/utils/constants.py first; known: {sorted(SEASONS)}")
    entry = SEASONS[season]
    processed_dir = Path(processed_dir)
    suffix = entry["suffix"]
    return {
        "season": season,
        "competition_id": entry["competition_id"],
        "season_id": entry["season_id"],
        "suffix": suffix,
        "coverage": entry["coverage"],
        "events_path": processed_dir / f"events_{suffix}.parquet",
        "matches_path": processed_dir / f"matches_{suffix}.csv",
    }


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

def get_matches_formatted(comp_id: int, season: int) -> pd.DataFrame:
    matches = sb.matches(competition_id=comp_id, season_id=season)
    all_matches = []
    for _, row in matches.iterrows():
        if row["match_status"] == "available":
            match_id = row["match_id"]
            match_week = row["match_week"]
            match_kickoff = f"{row['match_date']} {row['kick_off']}"
            match_stadium = row["stadium"]
            match_home_team = row["home_team"]
            match_home_team_manager = row.get("home_manager_name")
            match_away_team = row["away_team"]
            match_away_team_manager = row.get("away_manager_name")
            match_score = f"{row['home_score']} - {row['away_score']}"
            all_matches.append({
                "match_id": match_id,
                "match_week": match_week,
                "match_kickoff": match_kickoff,
                "match_stadium": match_stadium,
                "match_home_team": match_home_team,
                "match_home_team_manager": match_home_team_manager,
                "match_away_team": match_away_team,
                "match_away_team_manager": match_away_team_manager,
                "match_score": match_score,
            })
    return pd.DataFrame(all_matches)
def _parts_dir(raw_dir: Path) -> Path:
    return raw_dir / "events_parts"


def _json_dir(raw_dir: Path) -> Path:
    return raw_dir / "events_json"


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


def _existing_json_match_ids(raw_dir: Path) -> set[int]:
    json_dir = _json_dir(raw_dir)
    if not json_dir.exists():
        return set()
    return {int(p.stem) for p in json_dir.glob("*.json")}


def get_match_events_json(match_id: int) -> list[dict]:
    """Return one match's doc-faithful event JSON as a list."""
    events = public.events(match_id)
    return list(events.values())


def save_match_events_json(match_id: int, raw_dir: Path | str = RAW_DIR) -> Path:
    raw_dir = Path(raw_dir)
    json_dir = _json_dir(raw_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{match_id}.json"
    events = get_match_events_json(match_id)
    with open(path, "w") as f:
        json.dump(events, f)
    return path


def get_match_events(match_id: int) -> pd.DataFrame:
    events = sb.events(match_id=match_id)
    events["match_id"] = match_id
    return normalize_events_df(events)


def _load_match_parts(match_ids: list[int], raw_dir: Path) -> pd.DataFrame:
    parts_dir = _parts_dir(raw_dir)
    frames = []
    for match_id in match_ids:
        path = parts_dir / f"{int(match_id)}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing events part for match {match_id}: {path}")
        df = pd.read_parquet(path)
        if not is_normalized(df):
            df = normalize_events_df(df)
        frames.append(df)
    if not frames:
        raise RuntimeError("No event files extracted.")
    return pd.concat(frames, ignore_index=True)


def extract_all_events(
    matches_df: pd.DataFrame,
    raw_dir: Path | str = RAW_DIR,
    *,
    resume: bool = True,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = _parts_dir(raw_dir)
    json_dir = _json_dir(raw_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        _migrate_csv_to_parts(raw_dir)

    match_ids = [int(mid) for mid in matches_df["match_id"].unique()]
    done = _existing_match_ids(raw_dir, resume)
    pending = [mid for mid in match_ids if mid not in done]
    json_done = _existing_json_match_ids(raw_dir) if resume else set()

    errors: list[dict] = []
    errors_path = raw_dir / "extraction_errors.json"

    if resume and done:
        print(f"Resuming: skipping {len(done)} matches already on disk, {len(pending)} remaining.")

    for match_id in tqdm(pending, desc="Extracting events"):
        try:
            if int(match_id) not in json_done:
                save_match_events_json(int(match_id), raw_dir)
            df = get_match_events(match_id)
            df.to_parquet(parts_dir / f"{match_id}.parquet", index=False)
        except Exception as e:
            errors.append({"match_id": int(match_id), "error": str(e)})
            print(f"Error processing match {match_id}: {e}")

    if errors:
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Logged {len(errors)} failures to {errors_path}")

    missing = [mid for mid in match_ids if not (parts_dir / f"{mid}.parquet").exists()]
    if missing:
        raise RuntimeError(f"Missing event files for {len(missing)} matches.")
    return _load_match_parts(match_ids, raw_dir)


def build_season_events(
    matches_df: pd.DataFrame,
    season_label: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    *,
    save: bool = True,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    match_ids = [int(mid) for mid in matches_df["match_id"].unique()]
    events_df = _load_match_parts(match_ids, raw_dir)
    events_df = events_df.sort_values(["match_id", "index"], kind="stable").reset_index(drop=True)

    if save:
        out_path = processed_dir / f"events_{season_label}.parquet"
        events_df.to_parquet(out_path, index=False)
        print(f"Saved {len(events_df):,} events to {out_path}")

    return events_df


def backfill_events_json(
    match_ids: list[int],
    raw_dir: Path | str = RAW_DIR,
    *,
    resume: bool = True,
) -> None:
    raw_dir = Path(raw_dir)
    _json_dir(raw_dir).mkdir(parents=True, exist_ok=True)
    done = _existing_json_match_ids(raw_dir) if resume else set()
    pending = [int(mid) for mid in match_ids if int(mid) not in done]

    if resume and done:
        print(f"JSON backfill: skipping {len(done)} matches, {len(pending)} remaining.")

    for match_id in tqdm(pending, desc="Backfilling JSON"):
        try:
            save_match_events_json(match_id, raw_dir)
        except Exception as e:
            print(f"Error backfilling JSON for match {match_id}: {e}")


def normalize_existing_parts(
    match_ids: list[int] | None = None,
    raw_dir: Path | str = RAW_DIR,
) -> None:
    """Normalize saved parquet parts without re-downloading events."""
    raw_dir = Path(raw_dir)
    parts_dir = _parts_dir(raw_dir)
    paths = sorted(parts_dir.glob("*.parquet"))
    if match_ids is not None:
        match_set = {int(mid) for mid in match_ids}
        paths = [path for path in paths if int(path.stem) in match_set]

    for path in tqdm(paths, desc="Normalizing parts"):
        df = pd.read_parquet(path)
        if is_normalized(df):
            continue
        normalize_events_df(df).to_parquet(path, index=False)


def save_raw(matches_df: pd.DataFrame, events_df: pd.DataFrame, raw_dir: Path | str = RAW_DIR) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(raw_dir / "matches.csv", index=False)
    events_df.to_csv(raw_dir / "events.csv", index=False)


def event_coverage_report(
    matches_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    if "season" in matches_df.columns:
        match_meta = matches_df[["match_id", "season"]].drop_duplicates()
    else:
        match_meta = matches_df[["match_id"]].drop_duplicates()
        match_meta["season"] = "unknown"
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


def extract_season(
    season: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    *,
    resume: bool = True,
    verify_coverage: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and persist one registry season: events parquet + matches csv.

    The season-generic replacement for ``run_full_extraction``, which hard-codes
    the Premier League. Resumable, because a full league is ~380 matches and a
    dropped connection should not restart the download.

    ``verify_coverage`` re-classifies the fetched match list and refuses to
    continue if it disagrees with the registry. That check costs one request and
    is what stops an hour being spent on a season that turns out to be one
    club's fixture list, which is exactly the Bundesliga 2015/16 trap.
    """
    from src.extraction.catalog import classify_coverage

    plan = season_plan(season, processed_dir)
    matches = get_matches_formatted(plan["competition_id"], plan["season_id"])

    if verify_coverage:
        found = classify_coverage(matches.rename(columns={
            "match_home_team": "home_team", "match_away_team": "away_team"}))
        if found != plan["coverage"]:
            raise ValueError(
                f"{season}: registry declares coverage {plan['coverage']!r} but "
                f"the fetched {len(matches)} matches classify as {found!r}. "
                f"Fix the registry before building.")

    extract_all_events(matches, raw_dir, resume=resume)
    events = build_season_events(matches, plan["suffix"], raw_dir, processed_dir,
                                 save=True)
    matches.to_csv(plan["matches_path"], index=False)
    print(f"Saved {len(matches):,} matches to {plan['matches_path']}")
    return matches, events


def run_full_extraction(raw_dir: Path | str = RAW_DIR, *, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_df = get_epl_matches()
    events_df = extract_all_events(matches_df, raw_dir, resume=resume)
    save_raw(matches_df, events_df, raw_dir)
    return matches_df, events_df
