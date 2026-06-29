"""Visualizations for passing structure analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _team_colors(teams) -> dict:
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(pd.Series(teams).dropna().unique())}

def _team_color_map(values) -> dict:
    teams = [team for team in pd.Series(values).dropna().unique()]
    cmap = plt.get_cmap("tab10")
    return {team: cmap(i % 10) for i, team in enumerate(teams)}


# ---------------------------------------------------------------------------
# Zone geometry
# ---------------------------------------------------------------------------

def plot_zone_pass_geometry(
    zone_geometry: pd.DataFrame,
    teams: list[str],
    *,
    ax=None,
):
    """
    Grouped bar chart: forward / lateral / backward pass share by zone for each team.

    One group of bars per zone (defensive, midfield, attacking).
    Each team gets a cluster within the group.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    zones = ["defensive", "midfield", "attacking"]
    directions = ["forward_pct", "lateral_pct", "backward_pct"]
    dir_labels = ["Forward", "Lateral", "Backward"]
    dir_colors = ["#2ecc71", "#3498db", "#e74c3c"]

    n_zones = len(zones)
    n_teams = len(teams)
    n_dirs = len(directions)

    group_width = 0.8
    cluster_width = group_width / n_teams
    bar_width = cluster_width / n_dirs * 0.85

    x_base = np.arange(n_zones)

    for ti, team in enumerate(teams):
        td = zone_geometry[zone_geometry["team"] == team].set_index("zone")
        t_offset = (ti - (n_teams - 1) / 2) * cluster_width
        for di, (col, label, color) in enumerate(zip(directions, dir_labels, dir_colors)):
            d_offset = (di - (n_dirs - 1) / 2) * bar_width
            vals = [float(td.loc[z, col]) if z in td.index else 0.0 for z in zones]
            bar_kwargs = dict(width=bar_width, alpha=0.85, color=color)
            if ti == 0:
                bar_kwargs["label"] = label
            xpos = x_base + t_offset + d_offset
            ax.bar(xpos, vals, **bar_kwargs)
            if n_teams > 1:
                ax.text(
                    xpos[0] + group_width / 2 * 0 - cluster_width * (n_teams - 1) / 2 + ti * cluster_width,
                    -4,
                    team[:3],
                    ha="center",
                    fontsize=7,
                    color="0.3",
                )

    ax.set_xticks(x_base)
    ax.set_xticklabels([z.capitalize() for z in zones])
    ax.set_ylabel("Share of passes in zone (%)")
    ax.set_title("Pass direction by pitch zone")
    ax.legend(title="Direction", loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_zone_direction_heatmap(
    zone_geometry: pd.DataFrame,
    teams: list[str],
    *,
    metric: str = "forward_pct",
    ax=None,
):
    """
    Simple heatmap: metric value per (team × zone).
    Good for comparing a single metric across zones and teams at a glance.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    zones = ["defensive", "midfield", "attacking"]
    matrix = (
        zone_geometry[zone_geometry["team"].isin(teams)]
        .pivot(index="team", columns="zone", values=metric)
        .reindex(index=teams, columns=zones)
        .fillna(0)
    )

    im = ax.imshow(matrix.values, cmap="YlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels([z.capitalize() for z in zones])
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams)
    for i in range(len(teams)):
        for j in range(len(zones)):
            ax.text(j, i, f"{matrix.values[i, j]:.0f}%", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(metric.replace("_", " ").capitalize())
    return ax


# ---------------------------------------------------------------------------
# Zone transitions
# ---------------------------------------------------------------------------

def plot_zone_transition_heatmap(
    zone_transitions: pd.DataFrame,
    team: str,
    *,
    ax=None,
):
    """
    3×3 heatmap of completed pass zone transitions for one team.
    Cell (row, col) = % of that team's completed passes going from row-zone to col-zone.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    zones = ["defensive", "midfield", "attacking"]
    td = zone_transitions[zone_transitions["team"] == team]
    matrix = (
        td.pivot_table(index="from_zone", columns="to_zone", values="pct", aggfunc="sum", fill_value=0)
        .reindex(index=zones, columns=zones, fill_value=0)
    )

    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Def", "Mid", "Att"])
    ax.set_yticklabels(["Def", "Mid", "Att"])
    ax.set_xlabel("To zone")
    ax.set_ylabel("From zone")
    for i in range(3):
        for j in range(3):
            val = matrix.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if val > 20 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{team} — zone transitions")
    return ax


# ---------------------------------------------------------------------------
# Zone exit timing
# ---------------------------------------------------------------------------

def plot_zone_exit_timing(
    zone_exit: pd.DataFrame,
    teams: list[str],
    *,
    ax=None,
):
    """
    Side-by-side violin + median marker of passes-before-exit distributions.
    Only possessions that successfully exited are included.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    colors = _team_colors(teams)
    data = [
        zone_exit[(zone_exit["possession_team"] == t) & zone_exit["exited"]]["passes_before_exit"].dropna().values
        for t in teams
    ]

    parts = ax.violinplot(data, positions=range(len(teams)), showmedians=True, widths=0.55)
    for pc, team in zip(parts["bodies"], teams):
        pc.set_facecolor(colors.get(team))
        pc.set_alpha(0.6)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("white")
        parts[key].set_linewidth(1.2)

    ax.set_xticks(range(len(teams)))
    ax.set_xticklabels(teams)
    ax.set_ylabel("Passes before exiting defensive third")
    ax.set_title("How many passes to escape the defensive third?")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_zone_exit_speed(
    zone_exit: pd.DataFrame,
    teams: list[str],
    *,
    ax=None,
):
    """
    Bar chart: median seconds before exit + direct exit rate side by side.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    colors = _team_colors(teams)
    x = np.arange(len(teams))
    width = 0.35

    medians = []
    direct = []
    for team in teams:
        grp = zone_exit[(zone_exit["possession_team"] == team) & zone_exit["exited"]]
        medians.append(float(grp["seconds_before_exit"].median()) if not grp.empty else 0.0)
        direct.append(
            float((grp["passes_before_exit"] == 0).mean() * 100) if not grp.empty else 0.0
        )

    b1 = ax.bar(x - width / 2, medians, width, label="Median sec to exit",
                color=[colors.get(t) for t in teams], alpha=0.85)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + width / 2, direct, width, label="Direct exit %",
                 color=[colors.get(t) for t in teams], alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(teams)
    ax.set_ylabel("Median seconds before exit")
    ax2.set_ylabel("Direct exits (0 passes first) %")
    ax.set_title("Zone exit speed out of defensive third")
    lines = [
        mpatches.Patch(color="0.5", alpha=0.85, label="Median sec"),
        mpatches.Patch(color="0.5", alpha=0.45, hatch="//", label="Direct exit %"),
    ]
    ax.legend(handles=lines, loc="upper right", fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Pitch-positioned pass network
# ---------------------------------------------------------------------------

def plot_pitch_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    team: str,
    *,
    min_edge_weight: int = 2,
    ax=None,
):
    """
    Draw the completed Regular Play pass network on a StatsBomb pitch.

    Nodes are placed at each player's average pass location.
    Node size scales with total involvements.
    Edge thickness and alpha scale with pass weight.
    """
    pitch = Pitch(pitch_type="statsbomb", line_color="white", pitch_color="#1a1a2e")

    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
    else:
        pitch.draw(ax=ax)

    if nodes.empty:
        ax.set_title(f"No network data — {team}")
        return ax

    edges_filtered = edges[edges["weight"] >= min_edge_weight].copy()
    max_weight = edges_filtered["weight"].max() if not edges_filtered.empty else 1
    max_inv = nodes["involvements"].max() if not nodes.empty else 1

    # Draw edges first (underneath nodes)
    for _, edge in edges_filtered.iterrows():
        src = nodes[nodes["player"] == edge["player"]]
        dst = nodes[nodes["player"] == edge["pass_recipient"]]
        if src.empty or dst.empty:
            continue
        x0, y0 = float(src["avg_x"].iloc[0]), float(src["avg_y"].iloc[0])
        x1, y1 = float(dst["avg_x"].iloc[0]), float(dst["avg_y"].iloc[0])
        lw = 1.0 + 4.0 * (edge["weight"] / max_weight)
        alpha = 0.25 + 0.55 * (edge["weight"] / max_weight)
        ax.plot([x0, x1], [y0, y1], color="white", lw=lw, alpha=alpha, solid_capstyle="round")

    # Draw nodes
    for _, node in nodes.iterrows():
        size = 100 + 600 * (node["involvements"] / max_inv)
        ax.scatter(
            node["avg_x"], node["avg_y"],
            s=size, color="#f39c12", edgecolors="white", linewidths=1.5, zorder=5
        )
        short = str(node["player"]).split()[-1][:10]
        ax.text(
            node["avg_x"], node["avg_y"] + 3.2,
            short, ha="center", fontsize=6.5, color="white", zorder=6,
        )

    ax.set_title(f"{team} — pass network (Regular Play, completed, ≥{min_edge_weight} passes)")
    return ax


# ---------------------------------------------------------------------------
# Betweenness centrality
# ---------------------------------------------------------------------------

def plot_betweenness_centrality(
    betweenness_tables: dict[str, pd.DataFrame],
    *,
    top_n: int = 8,
    ax=None,
):
    """
    Horizontal bar chart of betweenness centrality for each team side by side.

    betweenness_tables : {team_name: DataFrame with columns player, betweenness}
    """
    teams = list(betweenness_tables.keys())
    colors = _team_colors(teams)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(top_n)
    width = 0.8 / len(teams)

    for i, team in enumerate(teams):
        df = betweenness_tables[team].head(top_n)
        offset = (i - (len(teams) - 1) / 2) * width
        ax.bar(
            x[: len(df)] + offset,
            df["betweenness"],
            width,
            label=team,
            color=colors.get(team),
            alpha=0.85,
        )

    # x-axis tick labels: use the shorter of the two teams' top-N player names
    reference_team = teams[0]
    ref_df = betweenness_tables[reference_team].head(top_n)
    tick_labels = [str(p).split()[-1][:12] for p in ref_df["player"]]
    ax.set_xticks(x[: len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Betweenness centrality (normalised)")
    ax.set_title("Pass network hub — betweenness centrality")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Buildup shape arrows
# ---------------------------------------------------------------------------

def plot_buildup_shape_arrows(
    sequence: pd.DataFrame,
    team: str,
    teams: list[str],
    *,
    max_pass_number: int = 1,
    title: str | None = None,
    ax=None,
):
    """
    Draw the first N passes of each Regular Play possession as arrows on a pitch.

    Pass ax= to embed into an existing mplsoccer pitch axis (e.g. from
    pitch.draw(nrows=2, ncols=1)). When ax is None a new figure is created.

    sequence must contain: team, pass_number_in_possession,
    location_x, location_y, pass_end_x, pass_end_y.
    """
    pitch = Pitch(pitch_type="statsbomb", line_color="white", pitch_color="#c7d5cc")
    if ax is None:
        fig, ax = pitch.draw(figsize=(14, 9))
    else:
        fig = ax.get_figure()
        pitch.draw(ax=ax)

    arrow_color = _team_colors(teams)

    data = sequence[
        sequence["team"].eq(team)
        & sequence["pass_number_in_possession"].le(max_pass_number)
    ].dropna(subset=["location_x", "location_y", "pass_end_x", "pass_end_y"])

    if not data.empty:
        pitch.arrows(
            data["location_x"].values,
            data["location_y"].values,
            data["pass_end_x"].values,
            data["pass_end_y"].values,
            width=2,
            headwidth=3,
            headlength=5,
            color=arrow_color.get(team),
            alpha=0.55,
            ax=ax,
        )

    ax.set_title(
        title or f"{team} — first {max_pass_number} buildup pass(es) per possession",
        color="white",
        fontsize=13,
        pad=8,
    )
    return fig, ax


def plot_betweenness_vs_involvement(
    betweenness_tables: dict[str, pd.DataFrame],
    *,
    ax=None,
):
    """
    Scatter: connection share % (x) vs betweenness centrality (y) for all players.

    Connection share is degree-based: the count of distinct teammates a player
    distributes to and receives from, as a share of all such connections — it
    measures breadth of links, not raw pass volume.

    High connectivity + high betweenness = true passing hub.
    High connectivity + low betweenness = broadly linked but not structurally central.
    Low connectivity + high betweenness = connector / bridge player.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    teams = list(betweenness_tables.keys())
    colors = _team_colors(teams)

    for team in teams:
        df = betweenness_tables[team]
        if df.empty:
            continue
        ax.scatter(
            df["involvement_pct"],
            df["betweenness"],
            s=60,
            alpha=0.8,
            label=team,
            color=colors.get(team),
            edgecolors="white",
            linewidths=0.5,
        )
        for _, row in df.head(4).iterrows():
            ax.annotate(
                str(row["player"]).split()[-1][:10],
                (row["involvement_pct"], row["betweenness"]),
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
                color=colors.get(team),
            )

    ax.set_xlabel("Connection share % (distinct passing partners — distributing + receiving)")
    ax.set_ylabel("Betweenness centrality (normalised)")
    ax.set_title("Playmaker profile: connectivity vs structural centrality")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
