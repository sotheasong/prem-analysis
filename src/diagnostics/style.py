"""Season style diagnostic: one team-season in, one feature registry out.

This is the engine the Phase 1 / 2 / 3b notebooks ran by hand, extracted so that
every later phase runs the *same* code instead of a copy of it. The notebooks
keep their narrative and their plots; the arithmetic lives here.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.features.batch import style_columns


@dataclass(frozen=True)
class TierPolicy:
    """The judgment calls in the classification, in one place.

    ``r2_hi``       context R^2 at or above which a feature is context-driven.
    ``alpha``       significance the dominant coefficient must clear.
    ``corr_hi``     |r| at or above which two features are treated as redundant.
    ``cv_quantile`` quantile of the CV distribution below which a feature counts
                    as stable. Data-relative, so it travels across seasons.
    """

    r2_hi: float = 0.30
    alpha: float = 0.05
    corr_hi: float = 0.90
    cv_quantile: float = 0.50


@dataclass(frozen=True)
class TeamSeason:
    """One team's matches for one season, in date order.

    ``frame`` carries the style table's own columns plus ``home`` (the 0/1
    encoding of ``venue``). ``features`` is the engineered feature universe, the
    columns a diagnostic is entitled to look at.
    """

    team: str
    frame: pd.DataFrame
    features: list[str]

    @property
    def n(self) -> int:
        return len(self.frame)


def team_season(style_df: pd.DataFrame, team: str) -> TeamSeason:
    """Slice a ``match_team_style`` table down to one team, ready to diagnose."""
    rows = style_df[style_df["team"] == team]
    if rows.empty:
        raise ValueError(f"no matches for team {team!r} in this style table")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["home"] = (frame["venue"] == "home").astype(int)

    return TeamSeason(team=team, frame=frame, features=style_columns(frame))


# ---------------------------------------------------------------------------
# Context design (the seam that varies by phase)
# ---------------------------------------------------------------------------
def match_context_design(ts: TeamSeason) -> pd.DataFrame:
    """The Phase 1/2/3b covariates: season drift, venue, opponent strength.

    A design is any ``TeamSeason -> DataFrame`` of predictor columns, one row per
    match in ``ts.frame`` order. Later phases swap in their own (league-relative
    covariates, the opponent's measured style) without touching the engine.
    """
    frame = ts.frame
    return pd.DataFrame({
        "date": (frame["date"] - frame["date"].min()).dt.days,
        "home": frame["home"],
        # Higher = stronger opponent, so coefficient signs read naturally.
        "opp": 21 - frame["opp_final_position"],
    })


def opponent_style_design(style_df: pd.DataFrame, feature: str,
                          min_matches: int = 10):
    """Design whose opponent covariate is *measured* rather than looked up.

    ``match_context_design`` proxies opponent strength with final league
    position, a number from outside the event data. When a season's coverage is
    the whole league, the opponent's own season average of ``feature`` can stand
    in its place: what that side actually does on the ball, not where it
    finished. Returns a design callable, so it composes like any other::

        diagnose(ts, design=opponent_style_design(style, "poss_mean_directness"))

    Refuses a partial-coverage season, where an "average" would rest on the two
    fixtures against one team rather than on an opponent's season.
    """
    if feature not in style_df.columns:
        raise KeyError(f"{feature!r} is not a column of this style table")

    played = style_df["team"].value_counts()
    season_mean = style_df.groupby("team")[feature].mean()

    def design(ts: TeamSeason) -> pd.DataFrame:
        thin = played.reindex(ts.frame["opponent"].unique())
        if (thin < min_matches).any():
            worst = int(thin.min())
            raise ValueError(
                f"season coverage is too thin for a measured opponent covariate: "
                f"one opponent has {worst} matches in this table, under the "
                f"{min_matches} required"
            )
        frame = ts.frame
        return pd.DataFrame({
            "date": (frame["date"] - frame["date"].min()).dt.days,
            "home": frame["home"],
            "opp_style": frame["opponent"].map(season_mean),
        })

    return design


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not sd:
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / sd


# ---------------------------------------------------------------------------
# One feature's fit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fit:
    """Least-squares fit of one z-scored feature on the z-scored design.

    Coefficients are comparable across features because both sides are z-scored.
    ``cooks_d`` and ``resid`` are aligned to the matches actually fitted, which
    excludes any match where the feature is NaN.
    """

    feature: str
    beta: dict[str, float]
    se: dict[str, float]
    p: dict[str, float]
    ci: dict[str, tuple[float, float]]
    r2: float
    dof: int
    fitted: np.ndarray
    resid: np.ndarray
    cooks_d: np.ndarray

    @property
    def dominant(self) -> str:
        """Predictor with the largest absolute standardized coefficient."""
        return max(self.beta, key=lambda k: abs(self.beta[k]))


class Diagnosis:
    """The diagnostic result for one team-season.

    Built by :func:`diagnose`. ``registry`` is the headline product, one row per
    feature; ``fit`` exposes the underlying regression for the features a
    notebook wants to report or plot in detail.
    """

    def __init__(self, ts: TeamSeason, design: pd.DataFrame,
                 policy: TierPolicy = TierPolicy()):
        self.ts = ts
        self.design = design
        self.policy = policy
        self.predictors = list(design.columns)

        frame = ts.frame
        self.degenerate = [c for c in ts.features if frame[c].nunique() <= 1]
        self.live = [c for c in ts.features if c not in self.degenerate]

        if len(design) != len(frame):
            raise ValueError(
                f"design has {len(design)} rows, season has {len(frame)}")

        self._pz = np.column_stack([_zscore(design[c].to_numpy()) for c in self.predictors])
        self._x = np.column_stack([np.ones(len(frame)), self._pz])
        rank = np.linalg.matrix_rank(self._x)
        if rank < self._x.shape[1]:
            raise ValueError(
                f"design is rank {rank} with {self._x.shape[1]} columns "
                "(intercept included): the covariates are collinear, so "
                "coefficients are not identified"
            )
        self._fits: dict[str, Fit] = {}

    # -- regression ---------------------------------------------------------
    def fit(self, feature: str) -> Fit:
        """Fit (and cache) one feature against the design."""
        if feature not in self._fits:
            self._fits[feature] = self._solve(feature)
        return self._fits[feature]

    def _solve(self, feature: str) -> Fit:
        y = _zscore(self.ts.frame[feature].to_numpy())
        mask = ~np.isnan(y)
        x, yy = self._x[mask], y[mask]
        n, k = x.shape

        xtx_inv = np.linalg.inv(x.T @ x)
        beta = xtx_inv @ x.T @ yy
        fitted = x @ beta
        resid = yy - fitted
        dof = n - k
        s2 = max((resid @ resid) / dof, 0.0)
        se = np.sqrt(np.diag(s2 * xtx_inv))
        tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        pval = 2 * stats.t.sf(np.abs(tstat), dof)
        tcrit = stats.t.ppf(0.975, dof)
        r2 = 1 - (resid @ resid) / ((yy - yy.mean()) ** 2).sum()

        leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            studentized = resid / np.sqrt(s2 * (1 - leverage))
        cooks_d = (studentized ** 2 / k) * (leverage / (1 - leverage))

        # index 0 is the intercept; callers only ever ask about the predictors.
        named = lambda arr: {p: float(arr[i + 1]) for i, p in enumerate(self.predictors)}
        return Fit(
            feature=feature,
            beta=named(beta),
            se=named(se),
            p=named(pval),
            ci={p: (float(beta[i + 1] - tcrit * se[i + 1]),
                    float(beta[i + 1] + tcrit * se[i + 1]))
                for i, p in enumerate(self.predictors)},
            r2=float(r2),
            dof=int(dof),
            fitted=fitted,
            resid=resid,
            cooks_d=cooks_d,
        )

    @property
    def vif(self) -> dict[str, float]:
        """Variance inflation per predictor. ~1 means an orthogonal design."""
        out = {}
        n = self._pz.shape[0]
        for j, name in enumerate(self.predictors):
            others = np.column_stack(
                [np.ones(n)] + [self._pz[:, m] for m in range(self._pz.shape[1]) if m != j]
            )
            target = self._pz[:, j]
            b = np.linalg.lstsq(others, target, rcond=None)[0]
            ss_res = ((target - others @ b) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            out[name] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
        return out

    # -- dispersion ---------------------------------------------------------
    @property
    def dispersion(self) -> pd.DataFrame:
        """Per-feature mean, standard deviation and CV, most stable first.

        CV is scale-free, which is what makes "stable" comparable across
        features measured in different units. It is undefined for a feature
        centred on zero, and NaN says so rather than pretending.
        """
        d = self.ts.frame[self.live]
        out = pd.DataFrame({"mean": d.mean(), "std": d.std(ddof=1)})
        out["cv"] = (out["std"] / out["mean"].abs()).replace(
            [np.inf, -np.inf], np.nan)
        return out.sort_values("cv")

    # -- resampling ---------------------------------------------------------
    def bootstrap(self, features, n_resamples: int = 2000,
                  seed: int = 0) -> pd.DataFrame:
        """Percentile bootstrap CIs for each feature's season mean and CV.

        Answers how much of a summary is the season and how much is 38 matches.
        Matches where the feature is missing are dropped before resampling, so a
        lone NaN narrows the sample rather than voiding the interval. ``seed``
        makes a notebook's printed numbers reproducible.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for feature in features:
            values = self.ts.frame[feature].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            draws = values[rng.integers(0, len(values), size=(n_resamples, len(values)))]
            means = draws.mean(axis=1)
            cvs = draws.std(axis=1, ddof=1) / np.abs(means)
            mean_lo, mean_hi = np.percentile(means, [2.5, 97.5])
            cv_lo, cv_hi = np.nanpercentile(cvs, [2.5, 97.5])
            rows.append({
                "feature": feature,
                "mean": values.mean(), "mean_lo": mean_lo, "mean_hi": mean_hi,
                "cv": values.std(ddof=1) / abs(values.mean()),
                "cv_lo": cv_lo, "cv_hi": cv_hi,
            })
        return pd.DataFrame(rows)

    # -- redundancy ---------------------------------------------------------
    @property
    def clusters(self) -> list[list[str]]:
        """Connected components of the "|r| above threshold" graph, largest first.

        Components rather than pairs: a and c can be redundant with each other
        only through b, and keeping one representative per component is what
        actually prunes the design matrix.
        """
        corr = self.ts.frame[self.live].corr().abs()
        adjacency: dict[str, set[str]] = collections.defaultdict(set)
        for a, b in itertools.combinations(self.live, 2):
            if corr.loc[a, b] > self.policy.corr_hi:
                adjacency[a].add(b)
                adjacency[b].add(a)

        seen, components = set(), []
        for feature in self.live:
            if feature not in adjacency or feature in seen:
                continue
            stack, component = [feature], set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                seen.add(node)
                stack += [n for n in adjacency[node] if n not in component]
            components.append(sorted(component))
        components.sort(key=len, reverse=True)
        return components

    @property
    def representatives(self) -> dict[str, str]:
        """Feature -> the lowest-CV member of its cluster. Singletons map to self."""
        cv = self.dispersion["cv"]
        out = {}
        for component in self.clusters:
            keep = min(component, key=lambda f: cv.get(f, np.inf))
            out.update({f: keep for f in component})
        return out

    # -- registry -----------------------------------------------------------
    @property
    def registry(self) -> pd.DataFrame:
        """One row per feature: its tier and the evidence behind it.

        This is the leakage-free hand-off to modeling. ``retained`` marks the
        features a design matrix may use; everything else is set aside with the
        reason recorded alongside it.
        """
        cv = self.dispersion["cv"]
        cv_cut = cv.quantile(self.policy.cv_quantile)
        representative = self.representatives
        cluster_of = {f: i for i, comp in enumerate(self.clusters) for f in comp}

        rows = []
        for feature in self.live:
            f = self.fit(feature)
            rows.append({
                "feature": feature,
                "tier": self._tier_of(feature, f, cv, cv_cut, representative),
                "cv": cv[feature],
                "context_r2": f.r2,
                "dominant": f.dominant,
                "dom_beta": f.beta[f.dominant],
                "dom_p": f.p[f.dominant],
                "cluster": cluster_of.get(feature, -1),
                "representative": representative.get(feature, feature) == feature,
            })
        rows += [
            {"feature": d, "tier": "drop", "cluster": -1, "representative": True}
            for d in self.degenerate
        ]
        out = pd.DataFrame(rows)
        out["retained"] = out["tier"].isin(("identity", "context"))
        # Stable sort: features tied on (tier, cv) keep the style table's own
        # column order, so the frozen CSV is reproducible run to run.
        return out.sort_values(["tier", "cv"], kind="stable").reset_index(drop=True)

    def _tier_of(self, feature, fit, cv, cv_cut, representative) -> str:
        if representative.get(feature, feature) != feature:
            return "redundant"
        if fit.r2 >= self.policy.r2_hi and fit.p[fit.dominant] < self.policy.alpha:
            return "context"
        if cv[feature] <= cv_cut and fit.r2 < self.policy.r2_hi:
            return "identity"
        return "experimental"


def diagnose(ts: TeamSeason, design=match_context_design,
             policy: TierPolicy = TierPolicy()) -> Diagnosis:
    """Run the season style diagnostic. ``design`` is the context seam."""
    return Diagnosis(ts, design(ts), policy)
