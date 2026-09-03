"""Era-normalized style representation: a team-season as a position in its league.

Phases 2 and 3b answer "is this team's fingerprint real?" for one team at a
time. This module answers "where does that fingerprint sit among its
contemporaries?", which is what makes two seasons from different decades
comparable at all: a raw pass-completion percentage from 2003 and one from 2015
are not the same measurement, but "80th percentile of its own league" is.

Requires a full-league season. The 2003/04 open data is Arsenal-only, so it
cannot define a field; it can only be *placed* in one built from a full season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.diagnostics.style import diagnose, team_season
from src.features.batch import style_columns


@dataclass(frozen=True)
class AnchorSet:
    """The identity-anchor tier, measured league-wide.

    ``features`` is the tier itself, in a stable order. ``counts`` records how
    many clubs each candidate feature anchors for, so the tier's margin is
    visible rather than implied: a feature kept at 15 of 20 is a weaker axis
    than one kept at 20 of 20.
    """

    features: list[str]
    counts: pd.Series
    n_teams: int
    min_teams: int

    @property
    def spine(self) -> list[str]:
        """Features that anchor for every club in the league."""
        return sorted(self.counts.index[self.counts == self.n_teams])


def _require_full_coverage(style_df: pd.DataFrame, min_matches: int) -> pd.Series:
    """Match counts per club, or an error naming the coverage problem."""
    played = style_df["team"].value_counts()
    if (played < min_matches).any():
        raise ValueError(
            f"season coverage is too thin to define a league-wide field: one club "
            f"has {int(played.min())} matches, under the {min_matches} required"
        )
    return played


def league_anchors(style_df: pd.DataFrame, min_teams: int = 15,
                   min_matches: int = 30) -> AnchorSet:
    """Features that are identity anchors for at least ``min_teams`` clubs.

    Runs the season diagnostic once per club and counts the verdicts. Refuses a
    season whose coverage is partial, where "the league" is not in the data.
    """
    played = _require_full_coverage(style_df, min_matches)

    counts = pd.Series(0, index=style_columns(style_df), dtype=int)
    for team in played.index:
        registry = diagnose(team_season(style_df, team)).registry
        anchored = registry.loc[registry["tier"] == "identity", "feature"]
        counts[anchored] += 1

    return AnchorSet(
        features=sorted(counts.index[counts >= min_teams]),
        counts=counts,
        n_teams=len(played),
        min_teams=min_teams,
    )


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StyleSpace:
    """One league season as a field of team-season positions.

    ``profiles`` is the raw aggregate, one season mean per club per feature.
    ``z`` re-expresses it against the field and is the vector distances are
    measured in; ``percentiles`` is the same position stated the way a reader
    reads it. Both are needed: with 20 clubs a percentile moves in 5-point steps
    and cannot tell a narrow leader from a runaway one, which is exactly the
    thing Leicester's directness makes interesting.

    The 20 clubs are the whole league rather than a sample from it, so the
    spread uses a population standard deviation.
    """

    profiles: pd.DataFrame
    z: pd.DataFrame
    percentiles: pd.DataFrame

    @property
    def teams(self) -> list[str]:
        return list(self.profiles.index)

    @property
    def features(self) -> list[str]:
        return list(self.profiles.columns)

    @property
    def distances(self) -> pd.DataFrame:
        """Euclidean distance between every pair of clubs, in z-space.

        z rather than percentile: ranks say who is ahead, distances need to say
        by how much. Every anchor axis is weighted equally, which is the point
        of normalizing first.
        """
        z = self.z.to_numpy()
        gaps = z[:, None, :] - z[None, :, :]
        matrix = np.sqrt(np.nansum(gaps ** 2, axis=2))
        return pd.DataFrame(matrix, index=self.teams, columns=self.teams)

    def neighbours(self, team: str, n: int = 3) -> pd.Series:
        """The ``n`` clubs closest to ``team``, nearest first, itself excluded."""
        if team not in self.profiles.index:
            raise KeyError(f"{team!r} is not a club in this field")
        others = self.distances[team].drop(index=team)
        return others.sort_values().head(n)

    def place(self, profile: pd.Series) -> pd.DataFrame:
        """Position a team-season that is not in this field.

        ``profile`` is one raw season mean per anchor feature. Returns the raw
        value alongside its z and percentile against these clubs. Feeding back a
        club that is in the field returns that club's own row.

        This is arithmetic, not equivalence: placing a 2003/04 profile in the
        2015/16 field says where those numbers fall among those clubs, and
        carries every confound between the two seasons with it.
        """
        missing = [f for f in self.features if f not in profile.index]
        if missing:
            raise KeyError(f"profile is missing anchor features: {missing}")

        values = profile.reindex(self.features).astype(float)
        centre = self.profiles.mean()
        spread = self.profiles.std(ddof=0).replace(0, np.nan)
        return pd.DataFrame({
            "value": values,
            "z": (values - centre) / spread,
            "percentile": _percentile_in(self.profiles, values),
        })


def _percentile_in(field: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Share of the field at or below each value, per feature, as a percentage.

    Used for the clubs in the field and for outsiders alike, which is what makes
    ``StyleSpace.place`` agree with ``StyleSpace.percentiles`` exactly.
    """
    return pd.Series(
        {f: (field[f] <= values[f]).mean() * 100 for f in field.columns},
        dtype=float,
    )


def style_space(style_df: pd.DataFrame, features: list[str],
                min_matches: int = 30) -> StyleSpace:
    """Build the era-normalized field for one full-league season."""
    _require_full_coverage(style_df, min_matches)

    profiles = style_df.groupby("team")[list(features)].mean()
    centre, spread = profiles.mean(), profiles.std(ddof=0)
    z = (profiles - centre) / spread.replace(0, np.nan)
    percentiles = pd.DataFrame(
        {team: _percentile_in(profiles, row) for team, row in profiles.iterrows()}
    ).T.loc[profiles.index, profiles.columns]

    return StyleSpace(profiles=profiles, z=z, percentiles=percentiles)
