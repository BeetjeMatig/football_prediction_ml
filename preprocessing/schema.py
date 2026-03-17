"""Column schema and alias normalization for football-data CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class ColumnSpec:
    """Schema metadata for one canonical column."""

    canonical_name: str
    aliases: Set[str]
    dtype: str
    group: str
    required_for_training: bool


COLUMN_SCHEMA: Dict[str, ColumnSpec] = {
    "div": ColumnSpec("div", {"Div"}, "string", "core_result", False),
    "date": ColumnSpec("date", {"Date"}, "datetime", "core_result", True),
    "time": ColumnSpec("time", {"Time"}, "string", "core_result", False),
    "home_team": ColumnSpec("home_team", {"HomeTeam"}, "string", "core_result", True),
    "away_team": ColumnSpec("away_team", {"AwayTeam"}, "string", "core_result", True),
    "full_time_home_goals": ColumnSpec(
        "full_time_home_goals",
        {"FTHG", "HG"},
        "Int64",
        "core_result",
        True,
    ),
    "full_time_away_goals": ColumnSpec(
        "full_time_away_goals",
        {"FTAG", "AG"},
        "Int64",
        "core_result",
        True,
    ),
    "target_result": ColumnSpec(
        "target_result",
        {"FTR", "Res"},
        "category",
        "core_result",
        True,
    ),
    "half_time_home_goals": ColumnSpec(
        "half_time_home_goals",
        {"HTHG"},
        "Int64",
        "match_stats",
        False,
    ),
    "half_time_away_goals": ColumnSpec(
        "half_time_away_goals",
        {"HTAG"},
        "Int64",
        "match_stats",
        False,
    ),
    "half_time_result": ColumnSpec(
        "half_time_result",
        {"HTR"},
        "category",
        "match_stats",
        False,
    ),
    "home_shots": ColumnSpec("home_shots", {"HS"}, "Int64", "match_stats", False),
    "away_shots": ColumnSpec("away_shots", {"AS"}, "Int64", "match_stats", False),
    "home_shots_on_target": ColumnSpec(
        "home_shots_on_target",
        {"HST"},
        "Int64",
        "match_stats",
        False,
    ),
    "away_shots_on_target": ColumnSpec(
        "away_shots_on_target",
        {"AST"},
        "Int64",
        "match_stats",
        False,
    ),
    "home_corners": ColumnSpec("home_corners", {"HC"}, "Int64", "match_stats", False),
    "away_corners": ColumnSpec("away_corners", {"AC"}, "Int64", "match_stats", False),
    "home_yellow_cards": ColumnSpec(
        "home_yellow_cards",
        {"HY"},
        "Int64",
        "match_stats",
        False,
    ),
    "away_yellow_cards": ColumnSpec(
        "away_yellow_cards",
        {"AY"},
        "Int64",
        "match_stats",
        False,
    ),
    "home_red_cards": ColumnSpec("home_red_cards", {"HR"}, "Int64", "match_stats", False),
    "away_red_cards": ColumnSpec("away_red_cards", {"AR"}, "Int64", "match_stats", False),
    "odds_max_home": ColumnSpec("odds_max_home", {"MaxH"}, "float64", "odds_1x2", False),
    "odds_max_draw": ColumnSpec("odds_max_draw", {"MaxD"}, "float64", "odds_1x2", False),
    "odds_max_away": ColumnSpec("odds_max_away", {"MaxA"}, "float64", "odds_1x2", False),
    "odds_avg_home": ColumnSpec("odds_avg_home", {"AvgH"}, "float64", "odds_1x2", False),
    "odds_avg_draw": ColumnSpec("odds_avg_draw", {"AvgD"}, "float64", "odds_1x2", False),
    "odds_avg_away": ColumnSpec("odds_avg_away", {"AvgA"}, "float64", "odds_1x2", False),
}


ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical_name, spec in COLUMN_SCHEMA.items():
    for alias in spec.aliases:
        key = alias.strip().lower()
        if key in ALIAS_TO_CANONICAL and ALIAS_TO_CANONICAL[key] != canonical_name:
            raise ValueError(f"Duplicate alias mapping for '{alias}'")
        ALIAS_TO_CANONICAL[key] = canonical_name


REQUIRED_COLUMNS = {
    canonical_name
    for canonical_name, spec in COLUMN_SCHEMA.items()
    if spec.required_for_training
}

ODDS_GROUPS = {"odds_1x2", "odds_ou25", "odds_ah"}


def normalize_column_name(column_name: str) -> str:
    """Map raw source column names to canonical names when known."""

    normalized = column_name.strip().lower()
    return ALIAS_TO_CANONICAL.get(normalized, normalized)


def normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    """Return old->new rename mapping for a collection of source columns."""

    rename_map: Dict[str, str] = {}
    seen_targets: Set[str] = set()

    for original in columns:
        target = normalize_column_name(original)

        # Prevent collisions where two source columns map to the same canonical name.
        if target in seen_targets and original != target:
            continue

        rename_map[original] = target
        seen_targets.add(target)

    return rename_map


def get_dtype(canonical_name: str) -> Optional[str]:
    """Look up target dtype by canonical column name."""

    spec = COLUMN_SCHEMA.get(canonical_name)
    return spec.dtype if spec else None


def get_columns_by_groups(groups: Set[str]) -> List[str]:
    """Return canonical columns in schema order for the requested groups."""

    return [
        canonical_name
        for canonical_name, spec in COLUMN_SCHEMA.items()
        if spec.group in groups
    ]


def get_output_columns(include_odds: bool) -> List[str]:
    """Return canonical output columns, optionally including betting odds groups."""

    selected_groups = {"core_result", "match_stats"}
    if include_odds:
        selected_groups.update(ODDS_GROUPS)

    return get_columns_by_groups(selected_groups)
