"""Plots for transition analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _team_colors(values) -> dict:
    teams = [t for t in pd.Series(values).dropna().unique()]
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(teams)}


def plot_transition_response(off_summary: pd.DataFrame, *, ax=None):
    """Stacked bar: share of ball wins answered with attack / secure / recycle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    order = ["attack_pct", "secure_pct", "recycle_back_pct"]
    labels = ["Attack forward", "Secure", "Recycle back"]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    teams = off_summary["team"].tolist()
    bottom = np.zeros(len(teams))
    for col, label, color in zip(order, labels, colors):
        vals = off_summary[col].to_numpy()
        ax.bar(teams, vals, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += vals
    ax.set_ylabel("% of open-play ball wins")
    ax.set_title("First action after winning the ball")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_transition_win_zones(win_zones: pd.DataFrame, *, ax=None):
    """Grouped bar: where each team wins the ball in open play."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    order = ["defensive", "middle", "attacking"]
    pivot = (
        win_zones.pivot(index="win_zone", columns="team", values="pct")
        .reindex(order)
    )
    pivot.plot(kind="bar", ax=ax, alpha=0.85, colormap="tab10")
    ax.set_ylabel("% of ball wins")
    ax.set_xlabel("Pitch third where ball was won")
    ax.set_title("Where the ball is won (open play)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title=None, fontsize=8)
    return ax


def plot_transition_aggression(off_summary: pd.DataFrame, *, ax=None):
    """Grouped bar of aggression markers per team."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    metrics = ["median_territory_10s", "reached_final_third_pct", "median_time_to_final_third"]
    labels = ["Territory in 10s (m)", "Reached final third (%)", "Time to final third (s)"]
    teams = off_summary["team"].tolist()
    colors = _team_colors(teams)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(teams), 1)
    for i, team in enumerate(teams):
        row = off_summary[off_summary["team"].eq(team)].iloc[0]
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=team, color=colors.get(team), alpha=0.85)
    ax.set_xticks(x + width * (len(teams) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("How aggressively teams transition into attack")
    ax.legend(fontsize=8)
    return ax


def plot_defensive_reaction(def_summary: pd.DataFrame, *, ax=None):
    """Grouped bar: counterpress and defensive-action rates after losing the ball."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    metrics = ["counterpress_pct", "def_action_5s_pct", "median_territory_conceded_5s"]
    labels = ["Counterpress (%)", "Any def. action ≤5s (%)", "Territory conceded 5s (m)"]
    teams = def_summary["team"].tolist()
    colors = _team_colors(teams)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(teams), 1)
    for i, team in enumerate(teams):
        row = def_summary[def_summary["team"].eq(team)].iloc[0]
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=team, color=colors.get(team), alpha=0.85)
    ax.set_xticks(x + width * (len(teams) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Reaction after losing the ball")
    ax.legend(fontsize=8)
    return ax


def plot_transition_danger(danger: pd.DataFrame, *, metric="xg_per_possession", ax=None):
    """Grouped bar comparing transition vs other possessions per team."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    pivot = danger.pivot(index="team", columns="possession_type", values=metric)
    cols = [c for c in ["transition", "other"] if c in pivot.columns]
    pivot[cols].plot(kind="bar", ax=ax, alpha=0.85, color=["#ff7f0e", "#7f7f7f"])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel(None)
    ax.set_title(f"Danger: transition vs other possessions ({metric.replace('_',' ')})")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title=None, fontsize=8)
    return ax


def plot_transition_scatter(trans_table: pd.DataFrame, teams: list[str], *, ax=None):
    """Scatter: opening directness (first_dx) vs territory gained in 10s,
    coloured by team — shows whether attacking openings gain more ground."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    colors = _team_colors(teams)
    for team in teams:
        t = trans_table[trans_table["team"].eq(team)]
        ax.scatter(
            t["first_dx"], t["territory_10s"],
            s=45, alpha=0.7, label=team, color=colors.get(team),
            edgecolors="white", linewidths=0.5,
        )
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("First-action forward distance (m)")
    ax.set_ylabel("Territory gained in first 10s (m)")
    ax.set_title("Opening directness vs territory gained")
    ax.legend(fontsize=8)
    return ax
