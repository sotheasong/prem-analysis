"""S8 — does the pipeline recover football that was already known?

The reconciliation gate (S7) proves the results were ingested correctly. It says
nothing about whether the *style* features measure anything real. This does, by
predicting the direction of a well-documented side's numbers before computing
them and then checking.

The directions were sealed in `.scratch/corpus-expansion/prereg-08-face-validity.md`
and committed one commit before the results existed. That ordering is the whole
point: a direction chosen after seeing the number is a description, not a test.

Not every check here is independent, and each one says which it is.
"""

import pandas as pd
import pytest

from src.config import FEATURES_DIR
from src.representation.era import team_contrast

# Sealed 2026-09-06. Barcelona's margin over the opponents it actually faced,
# required to hold in every Guardiola season.
PEP_SEASONS = {f"{y - 1}/{y}": f"barca{y}" for y in range(2009, 2013)}
PEP_PREDICTIONS = {
    "pass_completion_pct": +1,     # extreme retention
    "avg_pass_length_m": -1,       # short passing
    "pi_connectivity": +1,         # the whole side in the circulation
    "poss_mean_directness": -1,    # patient, not direct
}


def style(suffix: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"match_team_style_{suffix}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not built (data/ is gitignored)")
    return pd.read_csv(path)


def test_the_pipeline_independently_recovers_guardiolas_barcelona():
    """INDEPENDENT of StatsBomb's own derived statistics. The consensus being
    tested predates the data and nothing here was tuned toward it."""
    features = list(PEP_PREDICTIONS)
    for season, suffix in PEP_SEASONS.items():
        margin = team_contrast(style(suffix), "Barcelona", features)
        for feature, expected_sign in PEP_PREDICTIONS.items():
            got = margin[feature]
            assert got * expected_sign > 0, (
                f"{season}: {feature} came out {got:+.3f}, predicted "
                f"{'positive' if expected_sign > 0 else 'negative'}")


def test_the_same_instrument_separates_leicester_from_barcelona():
    """The control. An instrument that called every good team patient and
    possession-heavy would be measuring quality, not style.

    Phase 3b found Leicester opponent-driven and counter-attacking, so its
    directness margin must come out on the *opposite* side of zero from
    Guardiola's Barcelona, on the same feature and the same instrument.
    """
    leicester = team_contrast(style("2016"), "Leicester City",
                              ["poss_mean_directness"])["poss_mean_directness"]
    barcelona = team_contrast(style("barca2011"), "Barcelona",
                              ["poss_mean_directness"])["poss_mean_directness"]

    assert leicester > 0, f"Leicester read {leicester:+.3f}, expected direct"
    assert barcelona < 0, f"Barcelona read {barcelona:+.3f}, expected patient"


def test_xg_aggregation_reproduces_the_per_shot_values_exactly():
    """NOT INDEPENDENT. This checks the aggregation, not the measurement.

    StatsBomb ships an xG value per shot and the pipeline sums them per
    (match, team). Agreement proves the summation, the joins and the team
    attribution are right; it says nothing about whether StatsBomb's xG model
    is any good, because both sides come from the same source.

    It stands in for the FBref cross-check the ticket planned, which serves the
    same purpose and is weaker: FBref's advanced stats are themselves
    StatsBomb-derived, and comparing to a rounded published table is
    approximate where this is exact.
    """
    from src.config import PROCESSED_DIR

    for suffix in ["2016", "barca2011", "ligue12016"]:
        events_path = PROCESSED_DIR / f"events_{suffix}.parquet"
        style_path = FEATURES_DIR / f"match_team_style_{suffix}.csv"
        if not (events_path.exists() and style_path.exists()):
            pytest.skip(f"{suffix} not built")

        events = pd.read_parquet(
            events_path, columns=["match_id", "team", "shot_statsbomb_xg"])
        per_shot = (events.dropna(subset=["shot_statsbomb_xg"])
                    .groupby(["match_id", "team"])["shot_statsbomb_xg"].sum()
                    .rename("raw_xg").reset_index())

        built = pd.read_csv(style_path)[["match_id", "team", "xg"]]
        merged = built.merge(per_shot, on=["match_id", "team"],
                             how="left").fillna({"raw_xg": 0.0})
        worst = (merged["xg"] - merged["raw_xg"]).abs().max()
        assert worst < 1e-9, f"{suffix}: xG differs by up to {worst:.3e}"


# --- an outside opinion on xG ----------------------------------------------

def test_our_xg_agrees_with_an_independently_modelled_xg():
    """INDEPENDENT. Understat fits its own xG model on its own event data, so
    this is the one check in the phase where a second *model* gets a vote.

    Agreement is bounded in what it proves. Two competent xG models correlate
    around r = 0.9; that they do says neither is right. What it would catch is a
    pipeline that had quietly stopped measuring chances at all.

    The join tolerates a day, because late kickoffs land on different calendar
    dates in the two sources. `goals_agree` is what makes that safe: if the
    tolerance ever paired the wrong fixtures, the scorelines would stop matching.
    """
    pytest.importorskip("understatapi")
    from scipy import stats

    from src.validation.understat import UNDERSTAT_SEASONS, compare_xg
    from src.config import FEATURES_DIR
    from src.utils.constants import SEASONS

    checked = 0
    for season in UNDERSTAT_SEASONS:
        suffix = SEASONS[season]["suffix"]
        if not (FEATURES_DIR / f"match_team_style_{suffix}.csv").exists():
            continue
        try:
            joined = compare_xg(season)
        except Exception as exc:                      # network, or upstream shape
            pytest.skip(f"Understat unavailable: {type(exc).__name__} {exc}")
        checked += 1

        assert joined["understat_xg"].notna().all(), (
            f"{season}: {int(joined['understat_xg'].isna().sum())} matches "
            f"found no Understat counterpart")
        assert joined["goals_agree"].all(), (
            f"{season}: the date-tolerant join paired mismatched fixtures")

        r = stats.pearsonr(joined["xg"], joined["understat_xg"])[0]
        assert r >= 0.85, f"{season}: xG correlation is only {r:.3f}"

    if checked == 0:
        pytest.skip("no Understat-covered seasons built")
