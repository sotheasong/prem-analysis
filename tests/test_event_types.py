"""S2 — coverage is not usability.

A season can hold 380 matches and still be unusable, because the feature
pipeline needs particular event types and an old collection may not carry them.
Phase 3e checks that before queuing a build, not after one fails.

The dependency is declared once in ``REQUIRED_EVENT_TYPES``. The first test
below is what keeps that declaration honest: it reads the feature source and
fails if the two ever disagree, so adding ``events["type"] == "Goal Keeper"`` to
a feature without declaring it is caught here rather than in a season build.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.extraction.catalog import (PER_MATCH_EVENT_TYPES,
                                    REQUIRED_EVENT_TYPES,
                                    check_event_types,
                                    event_type_report,
                                    missing_event_types)

# StatsBomb's event-type vocabulary, from their spec rather than from our code.
# Used as a filter so that non-type strings in the feature source ("Regular
# Play", "High Pass", "Goal") cannot be mistaken for event types.
STATSBOMB_EVENT_TYPES = [
    "50/50", "Bad Behaviour", "Ball Receipt*", "Ball Recovery", "Block",
    "Camera On", "Camera off", "Carry", "Clearance", "Dispossessed", "Dribble",
    "Dribbled Past", "Duel", "Error", "Foul Committed", "Foul Won",
    "Goal Keeper", "Half End", "Half Start", "Injury Stoppage", "Interception",
    "Miscontrol", "Offside", "Own Goal Against", "Own Goal For", "Pass",
    "Player Off", "Player On", "Pressure", "Referee Ball-Drop", "Shield",
    "Shot", "Starting XI", "Substitution", "Tactical Shift",
]

FEATURES_DIR = Path(__file__).resolve().parent.parent / "src" / "features"


def types_referenced_by_the_feature_pipeline() -> set[str]:
    source = "\n".join(p.read_text() for p in sorted(FEATURES_DIR.glob("*.py")))
    return {t for t in STATSBOMB_EVENT_TYPES if f'"{t}"' in source}


def test_the_declared_dependency_matches_what_the_features_actually_use():
    assert set(REQUIRED_EVENT_TYPES) == types_referenced_by_the_feature_pipeline()


def events_frame(types: list[str], match_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame({"type": types, "match_id": match_id})


def test_a_collection_without_carries_is_reported_as_missing_them():
    """Carry is the type most likely to be absent from an old collection, and
    the one the possession pipeline leans on hardest."""
    present = sorted(REQUIRED_EVENT_TYPES - {"Carry"})
    assert missing_event_types(events_frame(present)) == ["Carry"]


def test_a_complete_collection_is_missing_nothing():
    assert missing_event_types(events_frame(sorted(REQUIRED_EVENT_TYPES))) == []


def test_density_is_reported_per_match_not_per_collection():
    """Two matches, ten passes and six carries between them.

    Density is the number that exposes collection drift: 2003/04 logs 108.9
    possessions a match where 2015/16 logs 95.1. It is reported and never
    failed on, because that difference is drift and `era.py` handles it.
    """
    events = pd.concat([
        events_frame(["Pass"] * 6 + ["Carry"] * 4, match_id=1),
        events_frame(["Pass"] * 4 + ["Carry"] * 2, match_id=2),
    ])
    report = event_type_report(events)

    assert report["n_matches"] == 2
    assert report["density"]["Pass"] == 5.0     # 10 over 2 matches
    assert report["density"]["Carry"] == 3.0    # 6 over 2 matches
    assert "Shot" in report["missing"]          # never appeared


def test_sampling_only_speaks_for_the_types_every_match_carries():
    """Across 7 PL 2015/16 matches, fourteen required types appeared in all
    seven, `Tactical Shift` in five and `Own Goal For` in two.

    So a sampled season that lacks those two has told us nothing, and failing it
    would reject a perfectly good collection. The full dependency still knows
    they are absent; the sample just does not get to vote on it.
    """
    sample = events_frame(sorted(PER_MATCH_EVENT_TYPES))
    assert missing_event_types(sample, required=PER_MATCH_EVENT_TYPES) == []
    assert missing_event_types(sample) == ["Own Goal For", "Tactical Shift"]


# --- integration: the oldest collections Phase 3e might touch ---------------

def test_the_oldest_collections_still_carry_what_the_pipeline_needs():
    """2007/08 is the earliest season this phase builds and 2004/05 is the
    oldest that exists. If either lacked a per-match type, Track 2 would have
    to be rescoped, so this is checked before any build is queued."""
    try:
        from statsbombpy import sb
        comps = sb.competitions()
    except Exception:
        pytest.skip("StatsBomb open data unreachable")

    for season_name in ["2007/2008", "2004/2005"]:
        row = comps[(comps.competition_name == "La Liga")
                    & (comps.season_name == season_name)]
        assert len(row) == 1, f"La Liga {season_name} not in the catalogue"
        report = check_event_types(int(row.iloc[0].competition_id),
                                   int(row.iloc[0].season_id), sample_n=3)
        assert report["missing"] == [], f"La Liga {season_name}: {report}"
        assert report["density"]["Pass"] > 500     # a real match, not a stub
