"""S5 — a season build is planned from the registry, never from arguments.

`run_full_extraction` hard-codes the Premier League, which was fine while the
project held two PL seasons and is not fine now that the registry holds
thirteen across four competitions. Passing the identifiers by hand is how a
season's events end up written under another season's suffix, silently
replacing a build that took half an hour.

So the plan is derived: give it a registry key, it resolves what to fetch and
where the two processed files go.
"""

import pytest

from src.config import PROCESSED_DIR
from src.extraction.extract import season_plan


def test_a_plan_resolves_identifiers_and_paths_from_the_registry():
    plan = season_plan("Ligue 1 2015/2016")
    assert plan["competition_id"] == 7
    assert plan["season_id"] == 27
    assert plan["events_path"] == PROCESSED_DIR / "events_ligue12016.parquet"
    assert plan["matches_path"] == PROCESSED_DIR / "matches_ligue12016.csv"


def test_the_two_premier_league_seasons_keep_their_existing_files():
    """The legacy keys must still resolve to the files already on disk, or a
    rerun would rebuild 2015/16 under a new name and orphan the old one."""
    assert season_plan("2015/2016")["events_path"].name == "events_2016.parquet"
    assert season_plan("2003/2004")["events_path"].name == "events_2004.parquet"


def test_a_season_outside_the_registry_fails_before_anything_is_fetched():
    """Bundesliga 2015/16 is the season that looks buildable and is 34
    Leverkusen matches. It is deliberately absent from the registry, and the
    failure has to arrive now rather than an hour into a download."""
    with pytest.raises(KeyError):
        season_plan("Bundesliga 2015/2016")
