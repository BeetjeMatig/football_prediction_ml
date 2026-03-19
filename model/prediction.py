"""Match outcome prediction and feature engineering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from model.utils import (
    INT_TO_LABEL,
    get_models_variant_dir,
    get_variant_name,
    load_model_artifact,
)

# Label encoding
LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}

# Override aliases for odds parameters
OVERRIDE_ALIASES = {
    "odds_home_win": "odds_max_home",
    "odds_draw": "odds_max_draw",
    "odds_away_win": "odds_max_away",
}

# Stat override keys that can be manually set
STAT_OVERRIDE_KEYS = {"expected_home_goals", "expected_away_goals"}

# Baseline event stat values for scenario adjustment
EVENT_STAT_BASELINES = {
    "home_red_cards": 0.0,
    "away_red_cards": 0.0,
    "home_yellow_cards": 2.0,
    "away_yellow_cards": 2.0,
    "home_corners": 5.0,
    "away_corners": 5.0,
    "home_shots": 12.0,
    "away_shots": 12.0,
    "home_shots_on_target": 4.0,
    "away_shots_on_target": 4.0,
}

# Effect multipliers for stat adjustments -> expected goals
EVENT_STAT_EFFECTS = {
    # Positive values increase own expected goals, negative values reduce it.
    "home_red_cards": (-0.35, +0.18),
    "away_red_cards": (+0.18, -0.35),
    "home_yellow_cards": (-0.05, +0.02),
    "away_yellow_cards": (+0.02, -0.05),
    "home_corners": (+0.06, 0.00),
    "away_corners": (0.00, +0.06),
    "home_shots": (+0.03, 0.00),
    "away_shots": (0.00, +0.03),
    "home_shots_on_target": (+0.10, 0.00),
    "away_shots_on_target": (0.00, +0.10),
}

STAT_OVERRIDE_KEYS = STAT_OVERRIDE_KEYS | set(EVENT_STAT_EFFECTS.keys())


@dataclass
class PredictionSummary:
    """Summary for a one-off match outcome prediction."""

    variant_name: str
    cutoff_date: str
    division: str
    home_team: str
    away_team: str
    predicted_outcome: str
    probability_home_win: float
    probability_draw: float
    probability_away_win: float
    expected_home_goals: float
    expected_away_goals: float

    def summary(self) -> str:
        return (
            f"variant={self.variant_name}, cutoff_date={self.cutoff_date}, "
            f"match={self.home_team} vs {self.away_team} ({self.division}), "
            f"predicted={self.predicted_outcome}, "
            f"P(H)={self.probability_home_win:.3f}, "
            f"P(D)={self.probability_draw:.3f}, "
            f"P(A)={self.probability_away_win:.3f}, "
            f"E[home_goals]={self.expected_home_goals:.2f}, "
            f"E[away_goals]={self.expected_away_goals:.2f}"
        )


def normalize_split_team_state_rows(
    df: pd.DataFrame,
    team: str,
    division: str,
    as_of_date: Optional[pd.Timestamp],
    recent_form_window: int,
) -> pd.DataFrame:
    """Normalize split dataframe rows for team state extraction."""
    if as_of_date is not None:
        filtered = df.loc[df["date"] <= as_of_date].copy()
    else:
        filtered = df.copy()

    filtered = filtered.loc[filtered["div"] == division].copy()

    home = filtered.loc[filtered["home_team"] == team].copy()
    away = filtered.loc[filtered["away_team"] == team].copy()

    home = home.rename(
        columns={
            "home_matches_played_before_match": "matches_played_before_match",
            f"home_points_avg_last_{recent_form_window}": "points_avg_last_5",
            f"home_goals_for_avg_last_{recent_form_window}": "goals_for_avg_last_5",
            f"home_goals_against_avg_last_{recent_form_window}": "goals_against_avg_last_5",
            f"home_goal_diff_avg_last_{recent_form_window}": "goal_diff_avg_last_5",
        }
    )
    away = away.rename(
        columns={
            "away_matches_played_before_match": "matches_played_before_match",
            f"away_points_avg_last_{recent_form_window}": "points_avg_last_5",
            f"away_goals_for_avg_last_{recent_form_window}": "goals_for_avg_last_5",
            f"away_goals_against_avg_last_{recent_form_window}": "goals_against_avg_last_5",
            f"away_goal_diff_avg_last_{recent_form_window}": "goal_diff_avg_last_5",
        }
    )

    keep_columns = [
        "date",
        "time",
        "matches_played_before_match",
        "points_avg_last_5",
        "goals_for_avg_last_5",
        "goals_against_avg_last_5",
        "goal_diff_avg_last_5",
    ]
    combined = pd.concat([home[keep_columns], away[keep_columns]], ignore_index=True)
    return combined.sort_values(["date", "time"], ascending=True).reset_index(drop=True)


def estimate_team_state(
    split_df: pd.DataFrame,
    team: str,
    division: str,
    as_of_date: Optional[str],
    recent_form_window: int,
) -> Dict[str, float]:
    """Estimate team state (form, history) from split dataframe."""
    timestamp = pd.Timestamp(as_of_date) if as_of_date else None
    rows = normalize_split_team_state_rows(
        df=split_df,
        team=team,
        division=division,
        as_of_date=timestamp,
        recent_form_window=recent_form_window,
    )

    if rows.empty:
        return {
            "matches_played_before_match": 0.0,
            "points_avg_last_5": float("nan"),
            "goals_for_avg_last_5": float("nan"),
            "goals_against_avg_last_5": float("nan"),
            "goal_diff_avg_last_5": float("nan"),
            "sparse_history": 1.0,
        }

    latest = rows.iloc[-1]
    matches_before = float(latest["matches_played_before_match"]) + 1.0

    return {
        "matches_played_before_match": matches_before,
        "points_avg_last_5": float(latest["points_avg_last_5"]),
        "goals_for_avg_last_5": float(latest["goals_for_avg_last_5"]),
        "goals_against_avg_last_5": float(latest["goals_against_avg_last_5"]),
        "goal_diff_avg_last_5": float(latest["goal_diff_avg_last_5"]),
        "sparse_history": 1.0 if matches_before < recent_form_window else 0.0,
    }


def parse_overrides(overrides: Optional[List[str]]) -> Dict[str, float]:
    """Parse feature/stat override strings from command line format."""
    parsed: Dict[str, float] = {}
    for raw_item in overrides or []:
        if "=" not in raw_item:
            raise ValueError(
                f"Invalid override '{raw_item}'. Use the format feature_name=value."
            )
        key, value = raw_item.split("=", 1)
        key = key.strip()
        value = value.strip()
        parsed[OVERRIDE_ALIASES.get(key, key)] = float(value)
    return parsed


def split_feature_and_stat_overrides(
    overrides: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Separate model-feature overrides from predicted-stat overrides."""
    feature_overrides = {
        key: value for key, value in overrides.items() if key not in STAT_OVERRIDE_KEYS
    }
    stat_overrides = {
        key: value for key, value in overrides.items() if key in STAT_OVERRIDE_KEYS
    }
    return feature_overrides, stat_overrides


def apply_event_stat_overrides_to_expected_goals(
    expected_home_goals: float,
    expected_away_goals: float,
    stat_overrides: Dict[str, float],
) -> Tuple[float, float]:
    """Apply scenario adjustments from event-stat overrides to expected goals."""
    home_goals = expected_home_goals
    away_goals = expected_away_goals
    for key, value in stat_overrides.items():
        if key in {"expected_home_goals", "expected_away_goals"}:
            continue
        if key not in EVENT_STAT_EFFECTS:
            continue

        baseline = EVENT_STAT_BASELINES[key]
        delta = value - baseline
        home_effect, away_effect = EVENT_STAT_EFFECTS[key]
        home_goals += delta * home_effect
        away_goals += delta * away_effect

    # Keep lambdas positive for Poisson conversion.
    home_goals = max(0.05, home_goals)
    away_goals = max(0.05, away_goals)
    return home_goals, away_goals


def poisson_pmf(lmbda: float, k: int) -> float:
    """Calculate Poisson probability mass function."""
    if lmbda < 0:
        raise ValueError("Poisson mean must be non-negative.")
    return math.exp(-lmbda) * (lmbda**k) / math.factorial(k)


def outcome_probabilities_from_expected_goals(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 10,
) -> Tuple[float, float, float]:
    """Approximate P(H), P(D), P(A) via independent Poisson goal models."""
    home_probs = [poisson_pmf(expected_home_goals, k) for k in range(max_goals + 1)]
    away_probs = [poisson_pmf(expected_away_goals, k) for k in range(max_goals + 1)]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            joint = home_probs[home_goals] * away_probs[away_goals]
            if home_goals > away_goals:
                home_win += joint
            elif home_goals == away_goals:
                draw += joint
            else:
                away_win += joint

    total = home_win + draw + away_win
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return home_win / total, draw / total, away_win / total


def build_feature_row(
    feature_columns: List[str],
    home_state: Dict[str, float],
    away_state: Dict[str, float],
    feature_fill_values: Dict[str, float],
    overrides: Dict[str, float],
    kickoff_time_minutes: Optional[float],
) -> pd.DataFrame:
    """Build a single feature row from team states and overrides."""
    row: Dict[str, float] = {}
    for column in feature_columns:
        if column == "time":
            row[column] = (
                float("nan") if kickoff_time_minutes is None else kickoff_time_minutes
            )
        elif column.startswith("home_"):
            row[column] = home_state.get(column.replace("home_", ""), float("nan"))
        elif column.startswith("away_"):
            row[column] = away_state.get(column.replace("away_", ""), float("nan"))
        else:
            row[column] = float("nan")

    for key, value in overrides.items():
        if key not in row:
            available = ", ".join(sorted(row.keys()))
            raise ValueError(
                f"Override feature '{key}' does not exist in model feature set. "
                f"Available features: {available}"
            )
        row[key] = value

    feature_df = pd.DataFrame([row], columns=feature_columns)
    for column, fill_value in feature_fill_values.items():
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].fillna(float(fill_value))

    return feature_df


def parse_kickoff_time_minutes(kickoff_time: Optional[str]) -> Optional[float]:
    """Convert HH:MM kickoff time format to minutes since midnight."""
    if kickoff_time is None:
        return None
    parts = kickoff_time.strip().split(":")
    if len(parts) != 2:
        raise ValueError("Kickoff time must be in HH:MM format.")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("Kickoff time HH:MM is out of range.")
    return float((hour * 60) + minute)


def predict_match_outcome(
    splits_dir: Path,
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    division: str,
    home_team: str,
    away_team: str,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
    as_of_date: Optional[str] = None,
    kickoff_time: Optional[str] = None,
    feature_overrides: Optional[List[str]] = None,
) -> PredictionSummary:
    """Predict one match outcome with optional manual feature overrides."""
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    model_dir = get_models_variant_dir(
        models_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    artifact = load_model_artifact(model_dir / "best_model.pkl")

    split_variant_dir = (
        splits_dir
        / f"date_{cutoff_date}"
        / get_variant_name(
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
    )
    train_path = split_variant_dir / "train.csv"
    test_path = split_variant_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Split files not found in {split_variant_dir}. Run --stage split first."
        )

    split_df = pd.concat(
        [pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True
    )
    split_df["date"] = pd.to_datetime(split_df["date"], errors="coerce")

    home_state = estimate_team_state(
        split_df=split_df,
        team=home_team,
        division=division,
        as_of_date=as_of_date,
        recent_form_window=recent_form_window,
    )
    away_state = estimate_team_state(
        split_df=split_df,
        team=away_team,
        division=division,
        as_of_date=as_of_date,
        recent_form_window=recent_form_window,
    )

    kickoff_time_minutes = parse_kickoff_time_minutes(kickoff_time)
    overrides = parse_overrides(feature_overrides)
    model_feature_overrides, stat_overrides = split_feature_and_stat_overrides(
        overrides
    )

    feature_columns = [str(column) for column in artifact["feature_columns"]]
    feature_fill_values = {
        str(key): float(value)
        for key, value in dict(artifact["feature_fill_values"]).items()
    }
    model: Any = artifact["model"]
    home_goal_regressor: Optional[Any] = artifact.get("home_goal_regressor")
    away_goal_regressor: Optional[Any] = artifact.get("away_goal_regressor")

    feature_row = build_feature_row(
        feature_columns=feature_columns,
        home_state=home_state,
        away_state=away_state,
        feature_fill_values=feature_fill_values,
        overrides=model_feature_overrides,
        kickoff_time_minutes=kickoff_time_minutes,
    )

    probabilities = model.predict_proba(feature_row)[0]
    expected_home_goals = float("nan")
    expected_away_goals = float("nan")
    if home_goal_regressor is not None and away_goal_regressor is not None:
        expected_home_goals = float(home_goal_regressor.predict(feature_row)[0])
        expected_away_goals = float(away_goal_regressor.predict(feature_row)[0])

    if "expected_home_goals" in stat_overrides:
        expected_home_goals = stat_overrides["expected_home_goals"]
    if "expected_away_goals" in stat_overrides:
        expected_away_goals = stat_overrides["expected_away_goals"]

    if not math.isnan(expected_home_goals) and not math.isnan(expected_away_goals):
        expected_home_goals, expected_away_goals = (
            apply_event_stat_overrides_to_expected_goals(
                expected_home_goals=expected_home_goals,
                expected_away_goals=expected_away_goals,
                stat_overrides=stat_overrides,
            )
        )

    if not math.isnan(expected_home_goals) and not math.isnan(expected_away_goals):
        probabilities = outcome_probabilities_from_expected_goals(
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
        )

    probabilities = [float(probability) for probability in probabilities]

    predicted_int = int(probabilities.index(max(probabilities)))

    return PredictionSummary(
        variant_name=variant_name,
        cutoff_date=cutoff_date,
        division=division,
        home_team=home_team,
        away_team=away_team,
        predicted_outcome=INT_TO_LABEL[predicted_int],
        probability_home_win=float(probabilities[0]),
        probability_draw=float(probabilities[1]),
        probability_away_win=float(probabilities[2]),
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
    )
