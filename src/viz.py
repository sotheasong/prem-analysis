"""Playstyle visualizations for StatsBomb event data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics import ERA_A, ERA_B, parse_pitch_x

# StatsBomb open-data pitch (meters)
PITCH_LENGTH = 120
PITCH_WIDTH = 80

VIZ_EXTRA_COLS = ["location", "minute", "second", "player"]


def ensure_viz_columns(
    events: pd.DataFrame,
    events_path,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Attach spatial/tempo columns if they were omitted from the initial EVENT_COLS load."""
    extra_cols = extra_cols or VIZ_EXTRA_COLS
    missing = [c for c in extra_cols if c not in events.columns]
    if not missing:
        return events

    if "id" in events.columns:
        extra = pd.read_csv(events_path, usecols=["id", *missing], low_memory=False)
        return events.merge(extra, on="id", how="left")

    extra = pd.read_csv(events_path, usecols=missing, low_memory=False)
    if len(extra) != len(events):
        raise ValueError(
            "Cannot attach viz columns: row count differs from events.csv. "
            "Re-run the data-load cell after EVENT_COLS includes 'id', "
            "or reload with id + location/minute/player."
        )
    out = events.copy()
    for col in missing:
        out[col] = extra[col].values
    return out


def parse_pitch_y(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip("[]").strip()
    if not s:
        return np.nan
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = s.split()
    if len(parts) < 2:
        return np.nan
    try:
        return float(parts[1])
    except ValueError:
        return np.nan


def add_pitch_coords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = out["location"].map(parse_pitch_x)
    out["y"] = out["location"].map(parse_pitch_y)
    return out.dropna(subset=["x", "y"])


def plot_pass_length_histograms(
    events: pd.DataFrame,
    seasons: tuple[str, str] = (ERA_A, ERA_B),
    *,
    bins: int = 50,
    sample_per_season: int | None = 100_000,
    ax=None,
):
    passes = events[events["type"] == "Pass"].dropna(subset=["pass_length"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for season in seasons:
        s = passes[passes["season"] == season]["pass_length"]
        if sample_per_season and len(s) > sample_per_season:
            s = s.sample(sample_per_season, random_state=0)
        ax.hist(s, bins=bins, alpha=0.5, density=True, label=season)

    ax.set_xlabel("Pass length (m)")
    ax.set_ylabel("Density")
    ax.set_title("Pass length distribution by season")
    ax.legend()
    return ax


def plot_possession_duration_histograms(
    possession_chains: pd.DataFrame,
    seasons: tuple[str, str] = (ERA_A, ERA_B),
    *,
    bins: int = 50,
    max_sec: float = 60,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for season in seasons:
        d = possession_chains.loc[possession_chains["season"] == season, "duration_sec"]
        d = d[(d > 0) & (d <= max_sec)]
        ax.hist(d, bins=bins, alpha=0.5, density=True, label=season)

    ax.set_xlabel("Possession duration (sec)")
    ax.set_ylabel("Density")
    ax.set_title("Possession duration distribution by season")
    ax.legend()
    return ax


def _draw_pitch(ax):
    ax.plot([0, 0, PITCH_LENGTH, PITCH_LENGTH, 0], [0, PITCH_WIDTH, PITCH_WIDTH, 0, 0], "w-", lw=1)
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], "w-", lw=0.8, alpha=0.7)
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d5016")


def plot_team_spatial_heatmap(
    events: pd.DataFrame,
    seasons: tuple[str, str] = (ERA_A, ERA_B),
    *,
    sample_per_season: int = 80_000,
    gridsize: int = 30,
):
    loc = add_pitch_coords(events)
    fig, axes = plt.subplots(1, len(seasons), figsize=(6 * len(seasons), 5))
    if len(seasons) == 1:
        axes = [axes]

    for ax, season in zip(axes, seasons):
        s = loc[loc["season"] == season]
        if len(s) > sample_per_season:
            s = s.sample(sample_per_season, random_state=0)
        _draw_pitch(ax)
        hb = ax.hexbin(
            s["x"], s["y"], gridsize=gridsize, cmap="YlOrRd", mincnt=1, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH]
        )
        ax.set_title(f"Event locations — {season}")
        plt.colorbar(hb, ax=ax, label="Event count")

    fig.suptitle("Team spatial activity (all event types)", y=1.02)
    plt.tight_layout()
    return fig


def plot_player_location_heatmaps(
    events: pd.DataFrame,
    season: str,
    *,
    top_n: int = 6,
    min_touches: int = 200,
    sample_per_player: int = 5000,
    gridsize: int = 25,
):
    loc = add_pitch_coords(events)
    loc = loc[loc["season"] == season].dropna(subset=["player"])
    counts = loc.groupby("player").size()
    players = counts[counts >= min_touches].nlargest(top_n).index.tolist()
    if not players:
        raise ValueError(f"No players with >={min_touches} touches in {season}")

    n = len(players)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, player in zip(axes, players):
        s = loc[loc["player"] == player]
        if len(s) > sample_per_player:
            s = s.sample(sample_per_player, random_state=0)
        _draw_pitch(ax)
        ax.hexbin(s["x"], s["y"], gridsize=gridsize, cmap="Blues", mincnt=1, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH])
        ax.set_title(player[:22], fontsize=9)

    for ax in axes[len(players) :]:
        ax.axis("off")

    fig.suptitle(f"Player touch locations (top {n}) — {season}", y=1.02)
    plt.tight_layout()
    return fig


def build_match_tempo(events: pd.DataFrame) -> pd.DataFrame:
    """Events per minute of play, averaged across matches, by season."""
    e = events.dropna(subset=["minute"]).copy()
    e["minute_bin"] = e["minute"].clip(0, 95)
    per_match = (
        e.groupby(["season", "match_id", "minute_bin"])
        .size()
        .rename("events")
        .reset_index()
    )
    return (
        per_match.groupby(["season", "minute_bin"])["events"]
        .mean()
        .reset_index(name="avg_events_per_minute")
    )


def plot_match_tempo_timeseries(
    tempo: pd.DataFrame,
    seasons: tuple[str, str] = (ERA_A, ERA_B),
    *,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    for season in seasons:
        s = tempo[tempo["season"] == season].sort_values("minute_bin")
        ax.plot(s["minute_bin"], s["avg_events_per_minute"], marker="o", ms=3, label=season)

    ax.set_xlabel("Match minute")
    ax.set_ylabel("Avg events per minute (across matches)")
    ax.set_title("Match tempo over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
