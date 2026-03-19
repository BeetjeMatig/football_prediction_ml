"""Model training, candidate selection, and evaluation."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.utils import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    get_modeling_variant_dir,
    get_models_variant_dir,
    get_variant_name,
    load_modeling_data,
    load_split_targets,
)


@dataclass
class TrainRunSummary:
    """Summary of one model training run for a dataset variant."""

    include_odds: bool
    add_recent_form_features: bool
    recent_form_window: int
    cutoff_date: str
    variant_name: str
    rows_train: int
    rows_test: int
    feature_count: int
    best_model_name: str
    best_log_loss: float
    model_dir: Path

    def summary(self) -> str:
        return (
            f"variant={self.variant_name}, cutoff_date={self.cutoff_date}, "
            f"train_rows={self.rows_train}, test_rows={self.rows_test}, "
            f"features={self.feature_count}, best_model={self.best_model_name}, "
            f"log_loss={self.best_log_loss:.5f}, model_dir={self.model_dir}"
        )


def fit_goal_regressors(
    x_train: pd.DataFrame,
    y_home_train: pd.Series,
    y_away_train: pd.Series,
) -> Tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    """Fit goal expectation regressors for home and away goals."""
    home_regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        max_depth=6,
        random_state=42,
    )
    away_regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        max_depth=6,
        random_state=42,
    )
    home_regressor.fit(x_train, y_home_train)
    away_regressor.fit(x_train, y_away_train)
    return home_regressor, away_regressor


def build_candidate_models() -> Dict[str, Any]:
    """Build dictionary of candidate classifier models."""
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=350,
        max_depth=6,
        random_state=42,
    )
    return {
        "logistic_regression": logistic,
        "hist_gradient_boosting": hgb,
    }


def evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""
    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    return {
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }


def train_model_variant(
    modeling_dir: Path,
    splits_dir: Path,
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> TrainRunSummary:
    """Train candidate models for one prepared modeling dataset variant."""
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )

    input_dir = get_modeling_variant_dir(
        modeling_dir=modeling_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    x_train, y_train, x_test, y_test, train_metadata, test_metadata = (
        load_modeling_data(input_dir)
    )

    y_home_train, y_away_train, y_home_test, y_away_test = load_split_targets(
        splits_dir=splits_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )

    candidate_models = build_candidate_models()
    evaluations: List[Dict[str, float | str]] = []
    fitted_models: Dict[str, Any] = {}

    for model_name, model in candidate_models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model=model, x_test=x_test, y_test=y_test)
        evaluations.append({"model": model_name, **metrics})
        fitted_models[model_name] = model

    evaluation_df = (
        pd.DataFrame(evaluations).sort_values("log_loss").reset_index(drop=True)
    )
    best_model_name = str(evaluation_df.loc[0, "model"])
    best_model = fitted_models[best_model_name]

    model_output_dir = get_models_variant_dir(
        models_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    model_output_dir.mkdir(parents=True, exist_ok=True)

    best_probabilities = best_model.predict_proba(x_test)
    best_predictions = best_model.predict(x_test)
    prediction_df = test_metadata.copy()
    prediction_df["actual_result"] = y_test.map(INT_TO_LABEL)
    prediction_df["predicted_result"] = pd.Series(best_predictions).map(INT_TO_LABEL)
    prediction_df["prob_home_win"] = best_probabilities[:, 0]
    prediction_df["prob_draw"] = best_probabilities[:, 1]
    prediction_df["prob_away_win"] = best_probabilities[:, 2]

    artifact = {
        "model": best_model,
        "variant_name": variant_name,
        "cutoff_date": cutoff_date,
        "feature_columns": list(x_train.columns),
        "feature_fill_values": x_train.median(numeric_only=True).fillna(0.0).to_dict(),
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "recent_form_window": recent_form_window,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": best_model_name,
    }

    evaluation_df.to_csv(model_output_dir / "metrics.csv", index=False)
    prediction_df.to_csv(model_output_dir / "test_predictions.csv", index=False)

    home_goal_regressor, away_goal_regressor = fit_goal_regressors(
        x_train=x_train,
        y_home_train=y_home_train,
        y_away_train=y_away_train,
    )
    home_goal_test_pred = home_goal_regressor.predict(x_test)
    away_goal_test_pred = away_goal_regressor.predict(x_test)
    goal_metrics = {
        "home_goals_mae": float(mean_absolute_error(y_home_test, home_goal_test_pred)),
        "away_goals_mae": float(mean_absolute_error(y_away_test, away_goal_test_pred)),
    }
    pd.DataFrame([goal_metrics]).to_csv(
        model_output_dir / "goal_metrics.csv", index=False
    )

    artifact["home_goal_regressor"] = home_goal_regressor
    artifact["away_goal_regressor"] = away_goal_regressor
    artifact["goal_metrics"] = goal_metrics

    artifact_meta = {
        key: value
        for key, value in artifact.items()
        if key not in {"model", "home_goal_regressor", "away_goal_regressor"}
    }

    with (model_output_dir / "best_model.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)

    with (model_output_dir / "artifact_meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(artifact_meta, handle, indent=2)

    return TrainRunSummary(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
        cutoff_date=cutoff_date,
        variant_name=variant_name,
        rows_train=len(x_train),
        rows_test=len(x_test),
        feature_count=len(x_train.columns),
        best_model_name=best_model_name,
        best_log_loss=float(evaluation_df["log_loss"].iloc[0]),
        model_dir=model_output_dir,
    )


def train_model_variants(
    modeling_dir: Path,
    splits_dir: Path,
    models_dir: Path,
    cutoff_date: str,
    include_odds_variants: List[bool],
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> List[TrainRunSummary]:
    """Train candidate models for each selected variant."""
    return [
        train_model_variant(
            modeling_dir=modeling_dir,
            splits_dir=splits_dir,
            models_dir=models_dir,
            cutoff_date=cutoff_date,
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
        for include_odds in include_odds_variants
    ]
