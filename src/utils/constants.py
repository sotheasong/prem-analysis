"""Shared constants used across feature, visualization, and aggregation modules.

Kept in a leaf module (no internal imports) so any module can import these
without creating an import cycle.
"""

from __future__ import annotations

# Pass length (metres) above which a pass is classified as "long".
LONG_PASS_M = 30

# Season labels used to tag eras when comparing across years.
ERA_2004 = "2003/2004"
ERA_2016 = "2015/2016"

# ---------------------------------------------------------------------------
# Scaling phase — per-season dataset registry & opponent-strength lookups
# ---------------------------------------------------------------------------

# Final 2003/04 Premier League table: team -> (final_position, final_points).
# The StatsBomb open data for this season is Arsenal-only, so opponent strength
# cannot be derived from the local matches; this static historical table
# supplies it. (For full-league seasons, standings are computed from the data.)
FINAL_TABLE_2004 = {
    "Arsenal":                 (1, 90),
    "Chelsea":                 (2, 79),
    "Manchester United":       (3, 75),
    "Liverpool":               (4, 60),
    "Newcastle United":        (5, 56),
    "Aston Villa":             (6, 56),
    "Charlton Athletic":       (7, 53),
    "Bolton Wanderers":        (8, 53),
    "Fulham":                  (9, 52),
    "Birmingham City":         (10, 50),
    "Middlesbrough":           (11, 48),
    "Southampton":             (12, 47),
    "Portsmouth":              (13, 45),
    "Tottenham Hotspur":       (14, 45),
    "Blackburn Rovers":        (15, 44),
    "Manchester City":         (16, 41),
    "Everton":                 (17, 39),
    "Leicester City":          (18, 33),
    "Leeds United":            (19, 33),
    "Wolverhampton Wanderers": (20, 33),
}

# Per-season metadata used by the batch feature runner (src/features/batch.py).
#   suffix   -> processed-file suffix: events_<suffix>.parquet / matches_<suffix>.csv
#   coverage -> "full" (whole league) or "partial" (subset; not league-analyzable)
#   strength -> static {team: (position, points)} when coverage is partial;
#               None means compute real standings from the season's own matches.
SEASONS = {
    "2003/2004": {"suffix": "2004", "coverage": "partial", "strength": FINAL_TABLE_2004},
    "2015/2016": {"suffix": "2016", "coverage": "full", "strength": None},
}

# Event columns carrying card info, and the values that count as a sending-off.
CARD_COLS = ("foul_committed_card", "bad_behaviour_card")
RED_CARD_VALUES = frozenset({"Red Card", "Second Yellow"})

# Columns loaded from the processed event store for feature engineering.
EVENT_COLS = [
    # Core
    "id",
    "match_id",
    "team",
    "team_id",
    "player",
    "player_id",
    "type",
    "period",
    "minute",
    "second",
    "timestamp",
    "duration",

    # Possession
    "possession",
    "possession_team",
    "possession_team_id",
    "play_pattern",

    # Spatial (normalized from StatsBomb location arrays)
    "location_x",
    "location_y",
    "pass_end_x",
    "pass_end_y",
    "carry_end_x",
    "carry_end_y",

    # Passing
    "pass_length",
    "pass_angle",
    "pass_outcome",
    "pass_complete",
    "pass_cross",
    "pass_goal_assist",
    "pass_switch",
    "pass_through_ball",
    "pass_type",
    "pass_height",

    # Pressing / defense
    "under_pressure",
    "counterpress",
    "duel_type",
    "duel_outcome",
    "interception_outcome",
    "ball_recovery_offensive",

    # Physicality
    "foul_committed_type",
    "foul_won_defensive",

    # Chance creation
    "shot_statsbomb_xg",
    "shot_outcome",
    "shot_type",
    "pass_shot_assist",

    # Dribbling
    "dribble_outcome",
]
