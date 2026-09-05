"""Making two seasons from different decades commensurable.

`style_space` builds a field out of one season. This module handles the part it
cannot: two seasons collected years apart are not the same measurement. The
2003/04 open data segments possessions more finely than 2015/16, enough that
Arsenal 2003/04 places at +5.69 sd on `poss_possessions` in the 2015/16 field
while its own 2003/04 opponents sit at the same inflated level. That number is
about StatsBomb's collection pass, not about Arsenal.

The correction is a paired within-match contrast: a team's value minus its
opponent's value in that same match. Whatever bias a season's collection adds to
both sides of a fixture cancels in their difference, without needing a model of
what the bias is. What it does not cancel is definitional drift. If a possession
means something different in 2003 than in 2015, the contrast removes the level
and leaves the meaning, which is a limit to state rather than one to paper over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.representation.style_space import style_space


def opponent_relative(style_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Each team's per-match margin over the team it actually faced.

    Returns a match-level frame with the same shape as ``style_df``, so the
    Phase 3c field builder consumes it unchanged: a contrast field is just a
    style space over different numbers.
    """
    features = list(features)
    sizes = style_df.groupby("match_id").size()
    if (sizes != 2).any():
        bad = sizes[sizes != 2]
        raise ValueError(
            f"every match needs exactly two rows to form a contrast; "
            f"{len(bad)} do not, e.g. match {bad.index[0]} has {int(bad.iloc[0])}"
        )

    values = style_df[features]
    # Each match's two rows sum to (a + b); subtracting twice a row's own value
    # from that sum leaves (a - b) for one side and (b - a) for the other.
    totals = values.groupby(style_df["match_id"]).transform("sum")
    contrast = values - (totals - values)

    return pd.concat(
        [style_df[["match_id", "team"]].reset_index(drop=True),
         contrast.reset_index(drop=True)],
        axis=1,
    )


def _opponent_pool(style_df: pd.DataFrame, focal: str,
                   features: list[str]) -> pd.DataFrame:
    """The rows of everyone the focal team played, in the matches it played.

    Never the focal team's own rows. A drift estimate that includes them
    subtracts the team's style away as if it were a collection artifact, and on
    the real 2003/04 table, where Arsenal is half the sample, that moves the
    estimate by a median 0.73 sd across the 49 anchors.
    """
    if focal not in set(style_df["team"]):
        raise KeyError(f"{focal!r} does not appear in this season")

    played = style_df.loc[style_df["team"] == focal, "match_id"]
    in_focal_matches = style_df[style_df["match_id"].isin(played)]
    return in_focal_matches.loc[in_focal_matches["team"] != focal, list(features)]


def era_offset(df_old: pd.DataFrame, focal_old: str,
               df_new: pd.DataFrame, focal_new: str,
               features: list[str]) -> pd.Series:
    """How far the older season's measurements sit from the newer one's.

    Estimated entirely from teams other than the two being compared, so it
    carries collection drift and league-wide change rather than either club's
    style. Positive means the old season reads higher.
    """
    features = list(features)
    old = _opponent_pool(df_old, focal_old, features).mean()
    new = _opponent_pool(df_new, focal_new, features).mean()
    return (old - new).rename("era_offset")


def team_contrast(style_df: pd.DataFrame, team: str,
                  features: list[str]) -> pd.Series:
    """One team's season-mean margin over the opponents it actually faced."""
    played = style_df.loc[style_df["team"] == team, "match_id"]
    if played.empty:
        raise KeyError(f"{team!r} does not appear in this season")

    contrast = opponent_relative(style_df[style_df["match_id"].isin(played)], features)
    return contrast.loc[contrast["team"] == team, list(features)].mean()


def contrast_space(style_df: pd.DataFrame, features: list[str],
                   min_matches: int = 30):
    """The league's field of margins, built with Phase 3c's field builder."""
    return style_space(opponent_relative(style_df, features), features,
                       min_matches=min_matches)


def confound_ledger(df_old: pd.DataFrame, focal_old: str,
                    df_new: pd.DataFrame, focal_new: str,
                    features: list[str], min_matches: int = 30,
                    mixed: float = 0.25, dominated: float = 0.75,
                    no_gap: float = 0.10) -> pd.DataFrame:
    """How much of each cross-era gap is the collection rather than the team.

    ``offset_share`` is the drift divided by the raw gap, both in units of the
    new season's field. The field's spread cancels, so the share is scale-free
    and comparable across features that live on wildly different scales.

    The ledger annotates and never gates. On the real data ``poss_possessions``
    scores 0.97, and the useful thing to say about it is that Arsenal 2003/04
    and its own opponents both ran ~108 possessions while nobody in 2015/16 came
    close. Dropping the row would hide that; flagging it states it.
    """
    features = list(features)
    raw_field = style_space(df_new, features, min_matches=min_matches)
    margins = contrast_space(df_new, features, min_matches=min_matches)

    raw_z = raw_field.place(df_old.loc[df_old["team"] == focal_old, features].mean())["z"]
    contrast_z = margins.place(team_contrast(df_old, focal_old, features))["z"]

    offset = era_offset(df_old, focal_old, df_new, focal_new, features)
    offset_sd = offset / raw_field.profiles.std(ddof=0).replace(0, np.nan)

    share = (offset_sd / raw_z).abs().where(raw_z.abs() >= no_gap)
    verdict = pd.Series(
        np.select(
            [share.isna(), share >= dominated, share >= mixed],
            ["no-gap", "offset-dominated", "mixed"],
            default="clean",
        ),
        index=share.index,
    )

    return pd.DataFrame({
        "raw_z": raw_z, "offset": offset, "offset_sd": offset_sd,
        "offset_share": share, "verdict": verdict, "contrast_z": contrast_z,
    }).loc[features]


def _paired_arrays(style_df: pd.DataFrame, team: str,
                   features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One row per match: the team's values, and its opponent's, aligned."""
    played = style_df.loc[style_df["team"] == team, "match_id"]
    if played.empty:
        raise KeyError(f"{team!r} does not appear in this season")

    sub = style_df[style_df["match_id"].isin(played)]
    sizes = sub.groupby("match_id").size()
    if (sizes != 2).any():
        raise ValueError(
            f"{team!r} has {int((sizes != 2).sum())} match(es) without exactly "
            f"two rows, so no opponent can be paired against them"
        )

    focal = sub[sub["team"] == team].set_index("match_id")[features].sort_index()
    opponent = sub[sub["team"] != team].set_index("match_id")[features]
    return focal.to_numpy(float), opponent.reindex(focal.index).to_numpy(float)


def benjamini_hochberg(p: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Which p-values survive an FDR correction, as a boolean Series.

    Step-up, so a p-value that fails its own rank is still kept when a
    larger-ranked one passes. Testing 49 anchors at once at alpha=0.05 flags two
    or three by chance, which is why the headline claims go through this.
    """
    ordered = p.dropna().sort_values()
    m = len(ordered)
    if m == 0:
        return pd.Series(False, index=p.index, dtype=bool)

    thresholds = np.arange(1, m + 1) * alpha / m
    passing = np.nonzero(ordered.to_numpy() <= thresholds)[0]
    cutoff = ordered.iloc[passing[-1]] if len(passing) else -np.inf
    return (p <= cutoff).fillna(False)


def decompose(df_old: pd.DataFrame, team_old: str,
              df_new: pd.DataFrame, team_new: str,
              features: list[str], scale: pd.Series | None = None,
              n_resamples: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Split a cross-era change into what the league did and what the team did.

    ``(A_new - A_old) = (O_new - O_old) + (C_new - C_old)`` where ``C = A - O``.
    An identity rather than a model, so the residual is float noise and the two
    terms answer genuinely different questions: did football change, or did this
    team change relative to the football around it.

    Read ``team_moved`` and ``league_moved`` very differently. ``team_moved`` is
    the drift-immune term: it is a difference of within-match margins, so any
    bias a season's collection adds to both sides of a fixture cancels twice
    over. ``league_moved`` is not. It is exactly ``-era_offset``, the same number
    the confound ledger reports as drift, so real league change and collection
    artifact are inseparable inside it at n=2 seasons. A large ``league_moved``
    is a reason to distrust the raw comparison, never evidence that football
    changed.

    Matches are resampled whole, so all three terms come out of the same
    resample and the identity holds inside the bootstrap instead of being
    imposed on top of it. ``scale`` divides every estimate and interval, normally
    the new season's field spread, so the result reads in standard deviations of
    the league being compared against.
    """
    features = list(features)
    old_team, old_pool = _paired_arrays(df_old, team_old, features)
    new_team, new_pool = _paired_arrays(df_new, team_new, features)

    def terms(a_old, o_old, a_new, o_new):
        total = a_new - a_old
        league = o_new - o_old
        return total, league, total - league

    point = terms(old_team.mean(0), old_pool.mean(0),
                  new_team.mean(0), new_pool.mean(0))

    rng = np.random.default_rng(seed)
    pick_old = rng.integers(0, len(old_team), size=(n_resamples, len(old_team)))
    pick_new = rng.integers(0, len(new_team), size=(n_resamples, len(new_team)))
    draws = terms(old_team[pick_old].mean(1), old_pool[pick_old].mean(1),
                  new_team[pick_new].mean(1), new_pool[pick_new].mean(1))

    out = {}
    for name, estimate, sample in zip(("total", "league", "team"), point, draws):
        lo, hi = np.percentile(sample, [2.5, 97.5], axis=0)
        # Two-sided bootstrap p, floored at one resample so it never reads as 0.
        tail = np.minimum((sample <= 0).mean(0), (sample >= 0).mean(0))
        out[f"{name}_{'change' if name == 'total' else 'moved'}"] = estimate
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
        out[f"{name}_p"] = np.clip(2 * np.maximum(tail, 1 / n_resamples), 0, 1)

    frame = pd.DataFrame(out, index=features)
    if scale is not None:
        columns = [c for c in frame.columns if not c.endswith("_p")]
        frame[columns] = frame[columns].div(scale, axis=0)
    return frame
