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


# Final La Liga tables for the Barcelona seasons Phase 3e builds: the same
# pattern as FINAL_TABLE_2004. Those seasons are Barcelona-only, so standings
# cannot be computed from their own results and opponent strength has to come
# from history.
#
# Club names are StatsBomb's spelling, not the source's. That distinction is the
# whole risk here: match_context does strength[opponent], so "Athletic Bilbao"
# instead of "Athletic Club" is a build-time KeyError, and a quieter mismatch
# would drop opponents from the covariate. tests/test_la_liga_tables.py checks
# every club StatsBomb records against these tables.
#
# Positions and points: Wikipedia season tables, mapped onto StatsBomb names.
LA_LIGA_TABLES = {
    "2007/2008": {
        "Real Madrid":                 ( 1, 85),
        "Villarreal":                  ( 2, 77),
        "Barcelona":                   ( 3, 67),
        "Atlético Madrid":             ( 4, 64),
        "Sevilla":                     ( 5, 64),
        "Racing Santander":            ( 6, 60),
        "Mallorca":                    ( 7, 59),
        "Almería":                     ( 8, 52),
        "RC Deportivo La Coruña":      ( 9, 52),
        "Valencia":                    (10, 51),
        "Athletic Club":               (11, 50),
        "Espanyol":                    (12, 48),
        "Real Betis":                  (13, 47),
        "Getafe":                      (14, 47),
        "Real Valladolid":             (15, 45),
        "Recreativo Huelva":           (16, 44),
        "Osasuna":                     (17, 43),
        "Real Zaragoza":               (18, 42),
        "Real Murcia CF":              (19, 30),
        "Levante UD":                  (20, 26),
    },
    "2008/2009": {
        "Barcelona":                   ( 1, 87),
        "Real Madrid":                 ( 2, 78),
        "Sevilla":                     ( 3, 70),
        "Atlético Madrid":             ( 4, 67),
        "Villarreal":                  ( 5, 65),
        "Valencia":                    ( 6, 62),
        "RC Deportivo La Coruña":      ( 7, 58),
        "Málaga":                      ( 8, 55),
        "Mallorca":                    ( 9, 51),
        "Espanyol":                    (10, 47),
        "Almería":                     (11, 46),
        "Racing Santander":            (12, 46),
        "Athletic Club":               (13, 44),
        "Sporting Gijón":              (14, 43),
        "Osasuna":                     (15, 43),
        "Real Valladolid":             (16, 43),
        "Getafe":                      (17, 42),
        "Real Betis":                  (18, 42),
        "CD Numancia de Soria":        (19, 35),
        "Recreativo Huelva":           (20, 33),
    },
    "2009/2010": {
        "Barcelona":                   ( 1, 99),
        "Real Madrid":                 ( 2, 96),
        "Valencia":                    ( 3, 71),
        "Sevilla":                     ( 4, 63),
        "Mallorca":                    ( 5, 62),
        "Getafe":                      ( 6, 58),
        "Villarreal":                  ( 7, 56),
        "Athletic Club":               ( 8, 54),
        "Atlético Madrid":             ( 9, 47),
        "RC Deportivo La Coruña":      (10, 47),
        "Espanyol":                    (11, 44),
        "Osasuna":                     (12, 43),
        "Almería":                     (13, 42),
        "Real Zaragoza":               (14, 41),
        "Sporting Gijón":              (15, 40),
        "Racing Santander":            (16, 39),
        "Málaga":                      (17, 37),
        "Real Valladolid":             (18, 36),
        "Tenerife":                    (19, 36),
        "Xerez":                       (20, 34),
    },
    "2010/2011": {
        "Barcelona":                   ( 1, 96),
        "Real Madrid":                 ( 2, 92),
        "Valencia":                    ( 3, 71),
        "Villarreal":                  ( 4, 62),
        "Sevilla":                     ( 5, 58),
        "Athletic Club":               ( 6, 58),
        "Atlético Madrid":             ( 7, 58),
        "Espanyol":                    ( 8, 49),
        "Osasuna":                     ( 9, 47),
        "Sporting Gijón":              (10, 47),
        "Málaga":                      (11, 46),
        "Racing Santander":            (12, 46),
        "Real Zaragoza":               (13, 45),
        "Levante UD":                  (14, 45),
        "Real Sociedad":               (15, 45),
        "Getafe":                      (16, 44),
        "Mallorca":                    (17, 44),
        "RC Deportivo La Coruña":      (18, 43),
        "Hércules":                    (19, 35),
        "Almería":                     (20, 30),
    },
    "2011/2012": {
        "Real Madrid":                 ( 1, 100),
        "Barcelona":                   ( 2, 91),
        "Valencia":                    ( 3, 61),
        "Málaga":                      ( 4, 58),
        "Atlético Madrid":             ( 5, 56),
        "Levante UD":                  ( 6, 55),
        "Osasuna":                     ( 7, 54),
        "Mallorca":                    ( 8, 52),
        "Sevilla":                     ( 9, 50),
        "Athletic Club":               (10, 49),
        "Getafe":                      (11, 47),
        "Real Sociedad":               (12, 47),
        "Real Betis":                  (13, 47),
        "Espanyol":                    (14, 46),
        "Rayo Vallecano":              (15, 43),
        "Real Zaragoza":               (16, 43),
        "Granada":                     (17, 42),
        "Villarreal":                  (18, 41),
        "Sporting Gijón":              (19, 37),
        "Racing Santander":            (20, 27),
    },
    "2012/2013": {
        "Barcelona":                   ( 1, 100),
        "Real Madrid":                 ( 2, 85),
        "Atlético Madrid":             ( 3, 76),
        "Real Sociedad":               ( 4, 66),
        "Valencia":                    ( 5, 65),
        "Málaga":                      ( 6, 57),
        "Real Betis":                  ( 7, 56),
        "Rayo Vallecano":              ( 8, 53),
        "Sevilla":                     ( 9, 50),
        "Getafe":                      (10, 47),
        "Levante UD":                  (11, 46),
        "Athletic Club":               (12, 45),
        "Espanyol":                    (13, 44),
        "Real Valladolid":             (14, 43),
        "Granada":                     (15, 42),
        "Osasuna":                     (16, 39),
        "Celta Vigo":                  (17, 37),
        "Mallorca":                    (18, 36),
        "RC Deportivo La Coruña":      (19, 35),
        "Real Zaragoza":               (20, 34),
    },
    "2013/2014": {
        "Atlético Madrid":             ( 1, 90),
        "Barcelona":                   ( 2, 87),
        "Real Madrid":                 ( 3, 87),
        "Athletic Club":               ( 4, 70),
        "Sevilla":                     ( 5, 63),
        "Villarreal":                  ( 6, 59),
        "Real Sociedad":               ( 7, 59),
        "Valencia":                    ( 8, 49),
        "Celta Vigo":                  ( 9, 49),
        "Levante UD":                  (10, 48),
        "Málaga":                      (11, 45),
        "Rayo Vallecano":              (12, 43),
        "Getafe":                      (13, 42),
        "Espanyol":                    (14, 42),
        "Granada":                     (15, 41),
        "Elche":                       (16, 40),
        "Almería":                     (17, 40),
        "Osasuna":                     (18, 39),
        "Real Valladolid":             (19, 36),
        "Real Betis":                  (20, 25),
    },
    "2014/2015": {
        "Barcelona":                   ( 1, 94),
        "Real Madrid":                 ( 2, 92),
        "Atlético Madrid":             ( 3, 78),
        "Valencia":                    ( 4, 77),
        "Sevilla":                     ( 5, 76),
        "Villarreal":                  ( 6, 60),
        "Athletic Club":               ( 7, 55),
        "Celta Vigo":                  ( 8, 51),
        "Málaga":                      ( 9, 50),
        "Espanyol":                    (10, 49),
        "Rayo Vallecano":              (11, 49),
        "Real Sociedad":               (12, 46),
        "Elche":                       (13, 41),
        "Levante UD":                  (14, 37),
        "Getafe":                      (15, 37),
        "RC Deportivo La Coruña":      (16, 35),
        "Granada":                     (17, 35),
        "Eibar":                       (18, 35),
        "Almería":                     (19, 29),
        "Córdoba CF":                  (20, 20),
    },
}

# Per-season metadata used by the batch feature runner (src/features/batch.py).
# The registry is the single place a season is declared; everything downstream
# reads it rather than hard-coding a season's properties.
#
#   competition / competition_id / season_id
#            StatsBomb's identifiers. season_id is NOT unique on its own: every
#            2015/16 league is season 27, so a season is the *pair*.
#   suffix   processed-file suffix: events_<suffix>.parquet / matches_<suffix>.csv.
#            Must be unique, since a collision silently overwrites a season.
#   coverage "full" | "single_club" | "sparse", the vocabulary
#            src.extraction.catalog.classify_coverage returns. Declared here and
#            checked against the live catalogue in tests/test_season_registry.py.
#   collection_pass
#            Which collection effort produced the season. Phase 3d established
#            that drift is a property of the collection, not the year: seasons
#            sharing a pass are comparable raw, across passes the paired contrast
#            in src/representation/era.py is mandatory. The values below are the
#            current hypothesis; ticket 09 of Phase 3e measures them.
#   role     "field" for a season contributing distinct clubs to fit on, "series"
#            for repeated measures of one club, which Phase 4 projects into a
#            fitted field and never fits on. See the Phase 3e sampling
#            constraint: Barcelona is ~10% of the corpus and would otherwise
#            manufacture its own archetype.
#   strength static {team: (position, points)} when standings cannot be computed
#            from the season's own results; None means compute them.
#
# NOTE: the two Premier League entries keep their bare season-label keys for
# backward compatibility with the existing notebooks, which call
# build_season_features("2015/2016"). New entries are qualified by competition.
SEASONS = {
    "2003/2004": {
        "competition": "Premier League",
        "competition_id": 2, "season_id": 44,
        "suffix": "2004",
        "coverage": "single_club",
        "collection_pass": "retrospective_2003_04",
        "role": "series",
        "strength": FINAL_TABLE_2004,
    },
    "2015/2016": {
        "competition": "Premier League",
        "competition_id": 2, "season_id": 27,
        "suffix": "2016",
        "coverage": "full",
        "collection_pass": "big5_2015_16",
        "role": "field",
        "strength": None,
    },
    "La Liga 2015/2016": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 27,
        "suffix": "laliga2016",
        "coverage": "full",
        "collection_pass": "big5_2015_16",
        "role": "field",
        "strength": None,
    },
    "Serie A 2015/2016": {
        "competition": "Serie A",
        "competition_id": 12, "season_id": 27,
        "suffix": "seriea2016",
        "coverage": "full",
        "collection_pass": "big5_2015_16",
        "role": "field",
        "strength": None,
    },
    "Ligue 1 2015/2016": {
        "competition": "Ligue 1",
        "competition_id": 7, "season_id": 27,
        "suffix": "ligue12016",
        "coverage": "full",
        "collection_pass": "big5_2015_16",
        "role": "field",
        "strength": None,
    },
    # Track 2, the Barcelona depth series. Single-club coverage, so each is
    # projected into a fitted field rather than fitted on, and each needs the
    # historical La Liga table its own results cannot produce. Consecutive on
    # purpose: gaps break change-point detection.
    "Barcelona 2007/2008": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 40,
        "suffix": "barca2008",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2007/2008"],
    },
    "Barcelona 2008/2009": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 41,
        "suffix": "barca2009",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2008/2009"],
    },
    "Barcelona 2009/2010": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 21,
        "suffix": "barca2010",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2009/2010"],
    },
    "Barcelona 2010/2011": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 22,
        "suffix": "barca2011",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2010/2011"],
    },
    "Barcelona 2011/2012": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 23,
        "suffix": "barca2012",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2011/2012"],
    },
    "Barcelona 2012/2013": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 24,
        "suffix": "barca2013",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2012/2013"],
    },
    "Barcelona 2013/2014": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 25,
        "suffix": "barca2014",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2013/2014"],
    },
    "Barcelona 2014/2015": {
        "competition": "La Liga",
        "competition_id": 11, "season_id": 26,
        "suffix": "barca2015",
        "coverage": "single_club",
        "collection_pass": "messi_la_liga",
        "role": "series",
        "strength": LA_LIGA_TABLES["2014/2015"],
    },
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
