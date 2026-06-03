"""Playstyle visualizations for StatsBomb event data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from src.analysis.metrics import ERA_A, ERA_B, parse_pitch_x

# StatsBomb open-data pitch (meters)
PITCH_LENGTH = 120
PITCH_WIDTH = 80

VIZ_EXTRA_COLS = ["location_x", "location_y", "minute", "second", "player"]


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
        path = Path(events_path)
        if path.suffix == ".parquet":
            extra = pd.read_parquet(path, columns=["id", *missing])
        else:
            extra = pd.read_csv(path, usecols=["id", *missing], low_memory=False)
        return events.merge(extra, on="id", how="left")

    path = Path(events_path)
    if path.suffix == ".parquet":
        extra = pd.read_parquet(path, columns=missing)
    else:
        extra = pd.read_csv(path, usecols=missing, low_memory=False)
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
    if "location_x" in out.columns and "location_y" in out.columns:
        out["x"] = out["location_x"]
        out["y"] = out["location_y"]
    else:
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


def _team_color_map(values) -> dict:
    teams = [team for team in pd.Series(values).dropna().unique()]
    cmap = plt.get_cmap("tab10")
    return {team: cmap(i % 10) for i, team in enumerate(teams)}


def _scaled_sizes(values: pd.Series, min_size: float = 35, max_size: float = 250) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    if values.empty or values.max() == values.min():
        return pd.Series(min_size, index=values.index)
    scaled = (values - values.min()) / (values.max() - values.min())
    return min_size + scaled * (max_size - min_size)


def plot_possession_duration_by_team(
    possession_summary: pd.DataFrame,
    *,
    bins: int = 20,
    max_sec: float | None = 90,
    ax=None,
):
    """Plot possession clock-duration distributions by team."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    colors = _team_color_map(possession_summary["possession_team"])
    for team, group in possession_summary.groupby("possession_team", sort=False):
        durations = pd.to_numeric(group["clock_duration"], errors="coerce").dropna()
        if max_sec is not None:
            durations = durations[durations <= max_sec]
        ax.hist(durations, bins=bins, alpha=0.5, label=team, color=colors.get(team), density=True)

    ax.set_xlabel("Possession duration (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Possession Duration by Team")
    ax.legend()
    return ax


def plot_possession_progression_scatter(possession_summary: pd.DataFrame, *, ax=None):
    """Plot possession duration against net x progression, sized by xG created."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    colors = _team_color_map(possession_summary["possession_team"])
    sizes = _scaled_sizes(possession_summary.get("xg_created", pd.Series(0, index=possession_summary.index)))
    for team, group in possession_summary.groupby("possession_team", sort=False):
        ax.scatter(
            group["clock_duration"],
            group["net_x_progression"],
            s=sizes.loc[group.index],
            alpha=0.65,
            label=team,
            color=colors.get(team),
            edgecolors="none",
        )

    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel("Possession duration (seconds)")
    ax.set_ylabel("Net x progression")
    ax.set_title("Duration vs Progression")
    ax.legend()
    return ax


def plot_pressure_directness_scatter(possession_summary: pd.DataFrame, *, ax=None):
    """Plot pressure rate against directness ratio, sized by possession duration."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    colors = _team_color_map(possession_summary["possession_team"])
    sizes = _scaled_sizes(possession_summary.get("clock_duration", pd.Series(0, index=possession_summary.index)))
    for team, group in possession_summary.groupby("possession_team", sort=False):
        ax.scatter(
            group["pressure_rate"],
            group["directness_ratio"],
            s=sizes.loc[group.index],
            alpha=0.65,
            label=team,
            color=colors.get(team),
            edgecolors="none",
        )

    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel("Pressure rate")
    ax.set_ylabel("Directness ratio")
    ax.set_title("Pressure vs Directness")
    ax.legend()
    return ax


def plot_entry_rates(possession_summary: pd.DataFrame, *, ax=None):
    """Plot final-third, box-entry, and shot-ending possession rates by team."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    rate_cols = ["final_third_entry", "box_entry", "ends_with_shot"]
    rates = possession_summary.groupby("possession_team")[rate_cols].mean()
    rates.rename(
        columns={
            "final_third_entry": "Final third entry",
            "box_entry": "Box entry",
            "ends_with_shot": "Ends with shot",
        }
    ).plot(kind="bar", ax=ax)

    ax.set_xlabel("Team")
    ax.set_ylabel("Share of possessions")
    ax.set_title("Possession Entry and Shot Rates")
    ax.set_ylim(0, 1)
    ax.legend(title="")
    return ax


def plot_possession_arrows(
    possession_summary: pd.DataFrame,
    *,
    team: str | None = None,
    min_duration: float = 0,
    ax=None,
):
    """Draw start-to-end arrows for possession movement on the StatsBomb pitch."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    data = possession_summary.copy()
    if team is not None:
        data = data[data["possession_team"] == team]
    if min_duration:
        data = data[pd.to_numeric(data["clock_duration"], errors="coerce") >= min_duration]
    data = data.dropna(subset=["start_x", "start_y", "end_x", "end_y"])

    _draw_pitch(ax)
    colors = _team_color_map(data["possession_team"])
    for _, row in data.iterrows():
        ax.arrow(
            row["start_x"],
            row["start_y"],
            row["end_x"] - row["start_x"],
            row["end_y"] - row["start_y"],
            length_includes_head=True,
            head_width=1.8,
            head_length=2.5,
            alpha=0.35,
            color=colors.get(row["possession_team"], "white"),
            linewidth=0.8,
        )

    title_team = team if team is not None else "Both Teams"
    ax.set_title(f"Possession Start-to-End Arrows - {title_team}")
    return ax



def plot_possession_start_end_heatmaps(
    possession_summary: pd.DataFrame,
    *,
    team: str | None = None,
    min_duration: float = 0,
    cmap: str = "viridis",
):
    """Plot smooth KDE heatmaps for possession start and end locations."""

    data = possession_summary.copy()

    if team is not None:
        data = data[data["possession_team"] == team]

    if min_duration:
        data = data[pd.to_numeric(data["clock_duration"], errors="coerce") >= min_duration]

    # clean coordinates
    starts = data.dropna(subset=["start_x", "start_y"])
    ends = data.dropna(subset=["end_x", "end_y"])

    pitch = Pitch(pitch_type="statsbomb", line_color="white", pitch_color="#2d5016")

    fig, axes = pitch.grid(
        nrows=1,
        ncols=2,
        figheight=5,
        title_height=0.08,
        endnote_height=0.02,
    )

    ax_start = axes["pitch"][0]
    ax_end = axes["pitch"][1]

    # -----------------------
    # START LOCATIONS KDE
    # -----------------------
    if not starts.empty:
        pitch.kdeplot(
            starts["start_x"],
            starts["start_y"],
            ax=ax_start,
            fill=True,
            levels=100,
            thresh=0.05,
            cmap=cmap,
            alpha=0.9,
        )

        pitch.scatter(
            starts["start_x"],
            starts["start_y"],
            ax=ax_start,
            s=10,
            c="white",
            alpha=0.25,
            edgecolors="none",
        )

    ax_start.set_title("Possession starts")

    # -----------------------
    # END LOCATIONS KDE
    # -----------------------
    if not ends.empty:
        pitch.kdeplot(
            ends["end_x"],
            ends["end_y"],
            ax=ax_end,
            fill=True,
            levels=100,
            thresh=0.05,
            cmap=cmap,
            alpha=0.9,
        )

        pitch.scatter(
            ends["end_x"],
            ends["end_y"],
            ax=ax_end,
            s=10,
            c="white",
            alpha=0.25,
            edgecolors="none",
        )

    ax_end.set_title("Possession ends")

    title_team = team if team is not None else "All teams"
    axes["title"].text(
        0.5,
        0.5,
        f"{title_team} possession start/end locations",
        ha="center",
        va="center",
        fontsize=14,
    )

    return fig, axes


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
