"""Playstyle visualizations for StatsBomb event data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from src.analysis.metrics import ERA_2004, ERA_2016, parse_pitch_x

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
    seasons: tuple[str, str] | None = (ERA_2004, ERA_2016),
    *,
    teams: list[str] | None = None,
    play_pattern: str | None = None,
    bins: int = 50,
    sample_per_season: int | None = 100_000,
    ax=None,
):
    passes = events[events["type"] == "Pass"].dropna(subset=["pass_length"])
    if play_pattern is not None:
        passes = passes[passes["play_pattern"] == play_pattern]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if teams is not None:
        for team in teams:
            s = passes[passes["team"] == team]["pass_length"]
            ax.hist(s, bins=bins, alpha=0.5, density=True, label=team)
        title = "Pass length distribution by team"
    else:
        for season in seasons or []:
            s = passes[passes["season"] == season]["pass_length"]
            if sample_per_season and len(s) > sample_per_season:
                s = s.sample(sample_per_season, random_state=0)
            ax.hist(s, bins=bins, alpha=0.5, density=True, label=season)
        title = "Pass length distribution by season"

    ax.set_xlabel("Pass length (m)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_possession_duration_histograms(
    possession_chains: pd.DataFrame,
    seasons: tuple[str, str] = (ERA_2004, ERA_2016),
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


def plot_team_metric_comparison(
    comparison: pd.DataFrame,
    metric: str,
    *,
    scenario: str | None = None,
    ax=None,
    ylabel: str | None = None,
):
    """Grouped bar chart for one metric across teams and optional scenario filter."""
    data = comparison.copy()
    if scenario is not None:
        data = data[data["scenario"] == scenario]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    teams = data["team"].tolist()
    values = data[metric].tolist()
    colors = _team_color_map(teams)
    ax.bar(teams, values, color=[colors[t] for t in teams], alpha=0.85)
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    title = metric.replace("_", " ")
    if scenario:
        title = f"{title} — {scenario}"
    ax.set_title(title)
    return ax


def plot_pass_height_comparison(
    events: pd.DataFrame,
    teams: list[str],
    *,
    ax=None,
):
    """Stacked bar chart of pass height mix by team."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    passes = events[events["type"] == "Pass"].copy()
    passes = passes[passes["team"].isin(teams)]
    mix = (
        passes.groupby("team")["pass_height"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )
    order = ["Ground Pass", "Low Pass", "High Pass"]
    mix = mix.reindex(columns=[c for c in order if c in mix.columns], fill_value=0)
    mix.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", alpha=0.9)
    ax.set_ylabel("Share of passes (%)")
    ax.set_xlabel("Team")
    ax.set_title("Pass height mix")
    ax.legend(title="Height", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    return ax


def plot_pass_count_distribution(
    possession_summary: pd.DataFrame,
    teams: list[str],
    *,
    play_pattern: str = "Regular Play",
    ax=None,
):
    """Side-by-side pass-count bins for regular-play possessions."""
    from src.analysis.possession.compare import pass_count_distribution

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    width = 0.35
    x = np.arange(6)
    labels = ["0", "1", "2", "3-4", "5-7", "8+"]
    colors = _team_color_map(teams)

    for i, team in enumerate(teams):
        dist = pass_count_distribution(
            possession_summary, team, play_pattern=play_pattern
        )
        counts = dist.reindex(labels, fill_value=0)
        total = counts.sum() or 1
        pct = counts / total * 100
        offset = (i - (len(teams) - 1) / 2) * width
        ax.bar(
            x + offset,
            pct,
            width=width,
            label=team,
            color=colors.get(team),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Passes per possession")
    ax.set_ylabel("Share of regular-play possessions (%)")
    ax.set_title(f"Pass-count distribution — {play_pattern}")
    ax.legend()
    return ax


def plot_pass_angle_rose(
    events: pd.DataFrame,
    teams: list[str],
    *,
    play_pattern: str = "Regular Play",
    bins: int = 16,
):
    """Polar histograms of pass-angle distributions by team."""
    fig, axes = plt.subplots(
        1,
        len(teams),
        subplot_kw={"projection": "polar"},
        figsize=(5 * len(teams), 4),
    )
    axes = np.atleast_1d(axes)
    colors = _team_color_map(teams)

    for ax, team in zip(axes, teams):
        passes = events[
            events["type"].eq("Pass")
            & events["team"].eq(team)
            & events["play_pattern"].eq(play_pattern)
        ].dropna(subset=["pass_angle"])
        angles = np.mod(passes["pass_angle"].astype(float), 2 * np.pi)
        counts, edges = np.histogram(angles, bins=bins, range=(0, 2 * np.pi))
        widths = np.diff(edges)
        ax.bar(
            edges[:-1],
            counts,
            width=widths,
            align="edge",
            alpha=0.75,
            color=colors.get(team),
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(f"{team} pass angles")

    fig.suptitle(f"Pass direction distribution — {play_pattern}", y=1.05)
    return fig, axes


def plot_pass_network_heatmap(
    events: pd.DataFrame,
    team: str,
    *,
    play_pattern: str = "Regular Play",
    ax=None,
):
    """Weighted player-to-player pass matrix for completed passes."""
    from src.analysis.passing.passing import filter_regular_play_passes, pass_network_adjacency

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    passes = filter_regular_play_passes(events, team)
    if play_pattern != "Regular Play":
        passes = events[
            events["type"].eq("Pass")
            & events["team"].eq(team)
            & events["play_pattern"].eq(play_pattern)
        ].copy()
    matrix = pass_network_adjacency(passes, team)
    if matrix.empty:
        ax.set_title(f"No completed pass network — {team}")
        return ax

    im = ax.imshow(matrix.values, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels([str(p)[:14] for p in matrix.columns], rotation=90, fontsize=8)
    ax.set_yticklabels([str(p)[:14] for p in matrix.index], fontsize=8)
    ax.set_title(f"Completed pass network — {team}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Passes")
    return ax


def plot_first_pass_direction(
    first_pass_table: pd.DataFrame,
    *,
    ax=None,
):
    """Stacked first-pass direction chart."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    pivot = first_pass_table.pivot_table(
        index="team", columns="direction", values="pct", fill_value=0
    )
    order = [col for col in ["forward", "lateral", "backward"] if col in pivot.columns]
    pivot[order].plot(kind="bar", stacked=True, ax=ax, colormap="viridis", alpha=0.9)
    ax.set_xlabel("Team")
    ax.set_ylabel("First passes (%)")
    ax.set_title("First pass behavior — Regular Play possessions")
    ax.legend(title="Direction", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    return ax


def plot_pressure_progression_split(
    passing_style: pd.DataFrame,
    *,
    metric: str = "completion",
    ax=None,
):
    """Compare pressured vs unpressured completion or x-progression."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if metric == "completion":
        cols = ["pressured_completion_pct", "unpressured_completion_pct"]
        labels = ["Pressured", "Unpressured"]
        ylabel = "Completion (%)"
        title = "Passing stability under pressure"
    else:
        cols = ["pressured_mean_dx", "unpressured_mean_dx"]
        labels = ["Pressured", "Unpressured"]
        ylabel = "Mean pass x-progression"
        title = "Pass progression under pressure"

    x = np.arange(len(passing_style))
    width = 0.35
    for i, (col, label) in enumerate(zip(cols, labels)):
        ax.bar(x + (i - 0.5) * width, passing_style[col], width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(passing_style["team"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_pass_vs_carry_progression(
    passing_style: pd.DataFrame,
    *,
    ax=None,
):
    """Stacked contribution of passes and carries to positive x progression."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    plot_data = passing_style.set_index("team")[
        ["pass_progression_share_pct", "carry_progression_share_pct"]
    ].rename(
        columns={
            "pass_progression_share_pct": "Passes",
            "carry_progression_share_pct": "Carries",
        }
    )
    plot_data.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", alpha=0.9)
    ax.set_xlabel("Team")
    ax.set_ylabel("Positive x-progression share (%)")
    ax.set_title("Progression mechanism — passes vs carries")
    ax.legend(title="")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    return ax


def plot_first_pass_recipient_roles(first_passes: pd.DataFrame, *, ax=None):
    """Stacked first-pass recipient role distribution by team."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    role_mix = (
        first_passes.groupby("team")["recipient_role"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )
    role_order = [
        "goalkeeper",
        "center_back",
        "fullback",
        "central_midfield",
        "wide_midfield",
        "winger",
        "attacking_midfield",
        "forward",
        "unknown",
    ]
    role_mix = role_mix.reindex(
        columns=[role for role in role_order if role in role_mix.columns],
        fill_value=0,
    )
    role_mix.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", alpha=0.9)
    ax.set_xlabel("Team")
    ax.set_ylabel("First passes (%)")
    ax.set_title("First pass recipient roles")
    ax.legend(title="Recipient role", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    return ax


def plot_buildup_width_depth(buildup_shape: pd.DataFrame, *, ax=None):
    """Compare early endpoint width/depth spread in Regular Play buildups."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    plot_data = buildup_shape.set_index("team")[
        ["early_width_y_std", "early_depth_x_std"]
    ].rename(
        columns={
            "early_width_y_std": "Width spread (y std)",
            "early_depth_x_std": "Depth spread (x std)",
        }
    )
    plot_data.plot(kind="bar", ax=ax, alpha=0.9)
    ax.set_xlabel("Team")
    ax.set_ylabel("Average std, first 3 pass endpoints")
    ax.set_title("Early buildup spread")
    ax.legend(title="")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    return ax


def plot_buildup_shape_arrows(
    sequence: pd.DataFrame,
    teams: list[str],
    *,
    max_pass_number: int = 1,
):
    """Draw first pass or first-n-pass buildup arrows on separate pitches."""
    fig, axes = plt.subplots(1, len(teams), figsize=(6 * len(teams), 4))
    axes = np.atleast_1d(axes)
    colors = _team_color_map(teams)

    for ax, team in zip(axes, teams):
        data = sequence[
            sequence["team"].eq(team)
            & sequence["pass_number_in_possession"].le(max_pass_number)
        ].dropna(subset=["location_x", "location_y", "pass_end_x", "pass_end_y"])
        _draw_pitch(ax)
        for _, row in data.iterrows():
            ax.arrow(
                row["location_x"],
                row["location_y"],
                row["pass_end_x"] - row["location_x"],
                row["pass_end_y"] - row["location_y"],
                length_includes_head=True,
                head_width=1.6,
                head_length=2.2,
                alpha=0.35,
                color=colors.get(team),
                linewidth=0.8,
            )
        ax.set_title(f"{team}: first {max_pass_number} buildup pass(es)")

    fig.suptitle("Early Regular Play buildup arrows", y=1.03)
    plt.tight_layout()
    return fig, axes


def plot_midfield_bypass_rates(buildup_shape: pd.DataFrame, *, ax=None):
    """Compare midfield-bypass and defensive-to-advanced first-pass rates."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    plot_data = buildup_shape.set_index("team")[
        ["first_pass_bypass_midfield_pct", "defensive_to_advanced_first_pass_pct"]
    ].rename(
        columns={
            "first_pass_bypass_midfield_pct": "Defensive third -> beyond midfield",
            "defensive_to_advanced_first_pass_pct": "Defensive role -> advanced role",
        }
    )
    plot_data.plot(kind="bar", ax=ax, alpha=0.9)
    ax.set_xlabel("Team")
    ax.set_ylabel("First passes (%)")
    ax.set_title("Midfield bypass behavior")
    ax.legend(title="")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
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
    seasons: tuple[str, str] = (ERA_2004, ERA_2016),
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


def plot_tempo_distribution(
    rhythm_summary: pd.DataFrame,
    teams: list[str],
    *,
    ax=None,
):
    """
    Violin + box plot of per-possession median inter-event gap by team.

    Lower gap = faster possession tempo.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    colors = _team_color_map(teams)
    data = [
        rhythm_summary.loc[
            rhythm_summary["possession_team"] == t, "gap_median"
        ].dropna().values
        for t in teams
    ]
    parts = ax.violinplot(data, positions=range(len(teams)), showmedians=True, widths=0.6)
    for i, (pc, team) in enumerate(zip(parts["bodies"], teams)):
        pc.set_facecolor(colors.get(team))
        pc.set_alpha(0.6)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("white")
        parts[key].set_linewidth(1.2)
    ax.set_xticks(range(len(teams)))
    ax.set_xticklabels(teams)
    ax.set_ylabel("Median inter-event gap (sec)")
    ax.set_title("Possession tempo distribution")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_rhythm_consistency(
    rhythm_summary: pd.DataFrame,
    teams: list[str],
    *,
    min_events: int = 4,
    ax=None,
):
    """
    Scatter: median gap (speed) vs gap coefficient of variation (consistency).

    Each point is one possession.  Lower gap = faster; lower CV = more metronomic.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    colors = _team_color_map(teams)
    data = rhythm_summary[rhythm_summary["action_count"] >= min_events] if "action_count" in rhythm_summary.columns else rhythm_summary

    for team in teams:
        grp = data[data["possession_team"] == team].dropna(subset=["gap_median", "gap_cv"])
        ax.scatter(
            grp["gap_median"],
            grp["gap_cv"],
            alpha=0.55,
            s=30,
            label=team,
            color=colors.get(team),
            edgecolors="none",
        )

    ax.axvline(
        rhythm_summary["gap_median"].median(),
        color="0.4", lw=0.8, ls="--", label="Overall median gap",
    )
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":", label="CV = 1 (irregular)")
    ax.set_xlabel("Median inter-event gap (sec) — lower = faster")
    ax.set_ylabel("Gap CV — lower = more consistent tempo")
    ax.set_title("Possession tempo: speed vs consistency")
    ax.legend(fontsize=8)
    return ax


def plot_intra_possession_tempo_profile(
    intra_tempo: pd.DataFrame,
    teams: list[str],
    *,
    max_position: int = 12,
    min_obs: int = 5,
    ax=None,
):
    """
    Line chart of average inter-event gap by event position within the possession.

    Position 1 = gap between own-team events 1 and 2, etc.
    Only positions with ≥ min_obs observations are plotted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    colors = _team_color_map(teams)

    for team in teams:
        grp = intra_tempo[
            (intra_tempo["possession_team"] == team)
            & (intra_tempo["event_position"] <= max_position)
        ]
        profile = (
            grp.groupby("event_position")["gap_sec"]
            .agg(["median", "count"])
            .rename(columns={"median": "gap_median", "count": "n"})
            .reset_index()
        )
        profile = profile[profile["n"] >= min_obs]
        ax.plot(
            profile["event_position"],
            profile["gap_median"],
            marker="o",
            ms=5,
            label=team,
            color=colors.get(team),
            lw=1.8,
        )

    ax.set_xlabel("Event position within possession")
    ax.set_ylabel("Median gap to next event (sec)")
    ax.set_title("How tempo changes through a possession chain")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_circulation_vs_progression(
    rhythm_summary: pd.DataFrame,
    teams: list[str],
    *,
    ax=None,
):
    """
    Scatter: passes before first progression vs total pass count.

    Points on the diagonal = entire possession was circulation (no progression).
    Points below = team progressed quickly relative to total passes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    colors = _team_color_map(teams)
    data = rhythm_summary.dropna(subset=["pre_prog_pass_count", "pass_count"])

    # diagonal reference
    max_passes = int(data["pass_count"].max()) + 1
    ax.plot([0, max_passes], [0, max_passes], color="0.5", lw=0.8, ls="--", zorder=0)

    for team in teams:
        grp = data[data["possession_team"] == team]
        ax.scatter(
            grp["pass_count"],
            grp["pre_prog_pass_count"],
            alpha=0.55,
            s=35,
            label=team,
            color=colors.get(team),
            edgecolors="none",
        )

    ax.set_xlabel("Total passes in possession")
    ax.set_ylabel("Passes before first progressive action")
    ax.set_title("Circulation before progression")
    ax.legend()
    return ax


def plot_progression_efficiency(
    rhythm_summary: pd.DataFrame,
    teams: list[str],
    *,
    ax=None,
):
    """
    Side-by-side bars: median passes per 10 m of net x-progression by team.

    Lower = more efficient at converting passes into territory gained.
    Only possessions that progressed forward (net_x_progression > 0) are included.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    colors = _team_color_map(teams)
    vals = []
    for team in teams:
        grp = rhythm_summary[
            rhythm_summary["possession_team"] == team
        ]["passes_per_10m"].dropna() if "passes_per_10m" in rhythm_summary.columns else pd.Series(dtype=float)
        vals.append(float(grp.median()) if not grp.empty else np.nan)

    ax.bar(teams, vals, color=[colors.get(t) for t in teams], alpha=0.85)
    ax.set_ylabel("Passes per 10 m of net x-progression")
    ax.set_title("Progression efficiency")
    ax.grid(axis="y", alpha=0.3)
    return ax


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
    seasons: tuple[str, str] = (ERA_2004, ERA_2016),
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
