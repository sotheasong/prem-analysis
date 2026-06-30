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
