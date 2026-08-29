"""S1 — the (match, team) style table's own column taxonomy.

``build_match_team_style`` knows exactly which columns it wrote as keys, as
exogenous context, and as outcomes. These tests pin that knowledge to the module
that creates the columns instead of to a hand-maintained ``DROP`` set in each
notebook.

Ground truth is the two registries frozen by the Phase 2 / Phase 3b notebooks
before this module existed: each lists exactly the 135 engineered features that
season's diagnostic ran on.
"""

import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.features.batch import style_columns

# ``data/`` is gitignored, so these tests run against locally built tables. Skip
# rather than fail when the pipeline has not been run in this checkout.
pytestmark = pytest.mark.skipif(
    not all((FEATURES_DIR / f).exists() for f in ['match_team_style.csv', 'match_team_style_2016.csv', 'phase2_arsenal_feature_registry.csv', 'phase3b_leicester_feature_registry.csv']),
    reason="requires the built feature tables in data/features/",
)


@pytest.fixture(scope="module")
def frozen():
    """(style table, registry feature list) for both frozen seasons."""
    def load(style_file, registry_file):
        style = pd.read_csv(FEATURES_DIR / style_file)
        registry = pd.read_csv(FEATURES_DIR / registry_file)
        return style, sorted(registry["feature"])

    return {
        "2003/2004": load("match_team_style.csv",
                          "phase2_arsenal_feature_registry.csv"),
        "2015/2016": load("match_team_style_2016.csv",
                          "phase3b_leicester_feature_registry.csv"),
    }


@pytest.mark.parametrize("season", ["2003/2004", "2015/2016"])
def test_style_columns_reproduces_the_frozen_feature_universe(frozen, season):
    style, expected = frozen[season]
    assert sorted(style_columns(style)) == expected


def test_score_state_shares_are_outcomes_not_features():
    """They are a function of goals scored, so they must never be a predictor."""
    style = pd.read_csv(FEATURES_DIR / "match_team_style_2016.csv")
    assert "pct_time_leading" in style.columns
    assert "pct_time_leading" not in style_columns(style)
