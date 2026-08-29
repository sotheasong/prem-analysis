"""S3 — the module reproduces the registries the notebooks froze.

``phase2_arsenal_feature_registry.csv`` and
``phase3b_leicester_feature_registry.csv`` were written by the hand-rolled
notebook engine, before this module existed. They are the acceptance gate: the
extracted engine has to land on the same 135 features, the same tiers and the
same numbers, or the extraction changed the science.
"""

import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.diagnostics.style import diagnose, team_season

# ``data/`` is gitignored, so these tests run against locally built tables. Skip
# rather than fail when the pipeline has not been run in this checkout.
pytestmark = pytest.mark.skipif(
    not all((FEATURES_DIR / f).exists() for f in ['match_team_style.csv', 'match_team_style_2016.csv', 'phase2_arsenal_feature_registry.csv', 'phase3b_leicester_feature_registry.csv']),
    reason="requires the built feature tables in data/features/",
)

SEASONS = [
    pytest.param("match_team_style.csv", "Arsenal",
                 "phase2_arsenal_feature_registry.csv", id="arsenal-2003/04"),
    pytest.param("match_team_style_2016.csv", "Leicester City",
                 "phase3b_leicester_feature_registry.csv", id="leicester-2015/16"),
]

ROUNDING = {"cv": 4, "context_r2": 4, "dom_beta": 3, "dom_p": 4}


def frozen_and_rebuilt(style_file, team, registry_file):
    style = pd.read_csv(FEATURES_DIR / style_file)
    rebuilt = diagnose(team_season(style, team)).registry
    for column, places in ROUNDING.items():
        rebuilt[column] = rebuilt[column].round(places)
    frozen = pd.read_csv(FEATURES_DIR / registry_file)
    key = "feature"
    return (frozen.set_index(key).sort_index(),
            rebuilt.set_index(key).sort_index())


@pytest.mark.parametrize("style_file, team, registry_file", SEASONS)
def test_same_feature_universe(style_file, team, registry_file):
    frozen, rebuilt = frozen_and_rebuilt(style_file, team, registry_file)
    assert list(rebuilt.index) == list(frozen.index)


@pytest.mark.parametrize("style_file, team, registry_file", SEASONS)
def test_same_tiers(style_file, team, registry_file):
    frozen, rebuilt = frozen_and_rebuilt(style_file, team, registry_file)
    disagreements = frozen.index[rebuilt["tier"] != frozen["tier"]]
    assert list(disagreements) == []


@pytest.mark.parametrize("style_file, team, registry_file", SEASONS)
def test_same_numbers(style_file, team, registry_file):
    frozen, rebuilt = frozen_and_rebuilt(style_file, team, registry_file)
    for column in ROUNDING:
        pd.testing.assert_series_equal(
            rebuilt[column], frozen[column], check_names=False, atol=1e-9)


@pytest.mark.parametrize("style_file, team, registry_file", SEASONS)
def test_same_clustering_and_retention(style_file, team, registry_file):
    frozen, rebuilt = frozen_and_rebuilt(style_file, team, registry_file)
    assert rebuilt["representative"].tolist() == frozen["representative"].tolist()
    assert rebuilt["retained"].tolist() == frozen["retained"].tolist()
    # Cluster ids are labels, so compare the partitions they induce.
    def partition(series):
        return {frozenset(g.index) for k, g in series.groupby(series) if k != -1}
    assert partition(rebuilt["cluster"]) == partition(frozen["cluster"])


@pytest.mark.parametrize("style_file, team, registry_file", SEASONS)
def test_row_order_differs_only_where_features_are_tied(
        style_file, team, registry_file):
    """The module sorts on exact CV; the notebooks sorted on 4dp-rounded CV.

    Features that tie at four decimals can therefore swap places. Anything that
    is not a tie must keep its position, and the tier blocks must line up.
    """
    style = pd.read_csv(FEATURES_DIR / style_file)
    rebuilt = diagnose(team_season(style, team)).registry
    frozen = pd.read_csv(FEATURES_DIR / registry_file)

    assert rebuilt["tier"].tolist() == frozen["tier"].tolist()

    moved = [f for f, g in zip(rebuilt["feature"], frozen["feature"]) if f != g]
    tied = rebuilt.set_index("feature").loc[moved, "cv"].round(4)
    assert tied.duplicated(keep=False).all() if moved else True
