"""Model prediction smoke testing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from model.prediction import predict_match_outcome
from model.utils import get_variant_name


@dataclass
class SmokeTestSummary:
    """Summary for model prediction smoke test."""

    variant_name: str
    passed: bool
    details: str

    def summary(self) -> str:
        status = "passed" if self.passed else "failed"
        return f"variant={self.variant_name}, status={status}, details={self.details}"


def run_prediction_smoke_test(
    splits_dir: Path,
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
    division: str = "E0",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
) -> SmokeTestSummary:
    """Run a minimal prediction smoke test and validate output sanity."""
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    try:
        prediction = predict_match_outcome(
            splits_dir=splits_dir,
            models_dir=models_dir,
            cutoff_date=cutoff_date,
            include_odds=include_odds,
            division=division,
            home_team=home_team,
            away_team=away_team,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
    except FileNotFoundError as e:
        return SmokeTestSummary(
            variant_name=variant_name,
            passed=False,
            details=f"Skipped - {e}",
        )
    probability_sum = (
        prediction.probability_home_win
        + prediction.probability_draw
        + prediction.probability_away_win
    )
    probs = [
        prediction.probability_home_win,
        prediction.probability_draw,
        prediction.probability_away_win,
    ]
    valid_probs = all(0.0 <= value <= 1.0 for value in probs)
    valid_sum = abs(probability_sum - 1.0) < 1e-6
    valid_goals = (
        not math.isnan(prediction.expected_home_goals)
        and not math.isnan(prediction.expected_away_goals)
        and prediction.expected_home_goals >= 0.0
        and prediction.expected_away_goals >= 0.0
    )
    passed = valid_probs and valid_sum and valid_goals
    details = (
        f"prob_sum={probability_sum:.6f}, valid_probs={valid_probs}, "
        f"valid_goals={valid_goals}, predicted={prediction.predicted_outcome}"
    )
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return SmokeTestSummary(variant_name=variant_name, passed=passed, details=details)
