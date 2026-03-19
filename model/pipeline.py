"""Model training and scenario prediction for football outcomes."""

from __future__ import annotations

import json
import math
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.pipeline import get_processed_variant_name

LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
OVERRIDE_ALIASES = {
    "odds_home_win": "odds_max_home",
    "odds_draw": "odds_max_draw",
    "odds_away_win": "odds_max_away",
}
STAT_OVERRIDE_KEYS = {"expected_home_goals", "expected_away_goals"}
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


@dataclass
class FreezeRunSummary:
    """Summary for freezing one trained variant artifact bundle."""

    variant_name: str
    source_dir: Path
    frozen_dir: Path
    freeze_label: str

    def summary(self) -> str:
        return (
            f"variant={self.variant_name}, freeze_label={self.freeze_label}, "
            f"source_dir={self.source_dir}, frozen_dir={self.frozen_dir}"
        )


@dataclass
class BaselineReportSummary:
    """Summary for a consolidated baseline metrics report."""

    cutoff_date: str
    output_path: Path
    variants_included: int

    def summary(self) -> str:
        return (
            f"cutoff_date={self.cutoff_date}, variants={self.variants_included}, "
            f"output={self.output_path}"
        )


@dataclass
class SmokeTestSummary:
    """Summary for model prediction smoke test."""

    variant_name: str
    passed: bool
    details: str

    def summary(self) -> str:
        status = "passed" if self.passed else "failed"
        return f"variant={self.variant_name}, status={status}, details={self.details}"


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


def _get_variant_name(
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
) -> str:
    return get_processed_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )


def _get_modeling_variant_dir(
    modeling_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
) -> Path:
    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return modeling_dir / f"date_{cutoff_date}" / variant_name


def _get_models_variant_dir(
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
) -> Path:
    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return models_dir / f"date_{cutoff_date}" / variant_name


def _get_frozen_variant_dir(
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
    freeze_label: str,
) -> Path:
    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return models_dir / "frozen" / freeze_label / f"date_{cutoff_date}" / variant_name


def _load_modeling_data(
    input_dir: Path,
) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame
]:
    x_train_path = input_dir / "X_train.csv"
    y_train_path = input_dir / "y_train.csv"
    x_test_path = input_dir / "X_test.csv"
    y_test_path = input_dir / "y_test.csv"
    train_meta_path = input_dir / "train_metadata.csv"
    test_meta_path = input_dir / "test_metadata.csv"

    required = [
        x_train_path,
        y_train_path,
        x_test_path,
        y_test_path,
        train_meta_path,
        test_meta_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing modeling dataset files. Run modeldata stage first. Missing: "
            + ", ".join(missing)
        )

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)["target_result"]
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["target_result"]
    train_meta = pd.read_csv(train_meta_path)
    test_meta = pd.read_csv(test_meta_path)

    y_train_encoded = y_train.map(LABEL_TO_INT)
    y_test_encoded = y_test.map(LABEL_TO_INT)
    if y_train_encoded.isna().any() or y_test_encoded.isna().any():
        raise ValueError("Unexpected target labels found. Expected only H, D, A.")

    return (
        x_train,
        y_train_encoded.astype("int64"),
        x_test,
        y_test_encoded.astype("int64"),
        train_meta,
        test_meta,
    )


def _load_split_targets(
    splits_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load and align home/away goal targets to modeling train/test row order."""

    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    split_variant_dir = splits_dir / f"date_{cutoff_date}" / variant_name
    train_split_path = split_variant_dir / "train.csv"
    test_split_path = split_variant_dir / "test.csv"
    if not train_split_path.exists() or not test_split_path.exists():
        raise FileNotFoundError(
            f"Split files not found in {split_variant_dir}. Run --stage split first."
        )

    train_split = pd.read_csv(train_split_path)
    test_split = pd.read_csv(test_split_path)

    key_columns = ["date", "div", "home_team", "away_team"]
    target_columns = ["full_time_home_goals", "full_time_away_goals"]

    merged_train = train_metadata.merge(
        train_split[key_columns + target_columns], on=key_columns, how="left"
    )
    merged_test = test_metadata.merge(
        test_split[key_columns + target_columns], on=key_columns, how="left"
    )

    if (
        merged_train[target_columns].isna().any().any()
        or merged_test[target_columns].isna().any().any()
    ):
        raise ValueError(
            "Could not align split targets with modeling metadata for goal regression."
        )

    return (
        merged_train["full_time_home_goals"],
        merged_train["full_time_away_goals"],
        merged_test["full_time_home_goals"],
        merged_test["full_time_away_goals"],
    )


def _fit_goal_regressors(
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


def _build_candidate_models() -> Dict[str, Any]:
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


def _evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
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

    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    input_dir = _get_modeling_variant_dir(
        modeling_dir=modeling_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    x_train, y_train, x_test, y_test, train_metadata, test_metadata = (
        _load_modeling_data(input_dir)
    )

    y_home_train, y_away_train, y_home_test, y_away_test = _load_split_targets(
        splits_dir=splits_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )

    candidate_models = _build_candidate_models()
    evaluations: List[Dict[str, float | str]] = []
    fitted_models: Dict[str, Any] = {}

    for model_name, model in candidate_models.items():
        model.fit(x_train, y_train)
        metrics = _evaluate_model(model=model, x_test=x_test, y_test=y_test)
        evaluations.append({"model": model_name, **metrics})
        fitted_models[model_name] = model

    evaluation_df = (
        pd.DataFrame(evaluations).sort_values("log_loss").reset_index(drop=True)
    )
    best_model_name = str(evaluation_df.loc[0, "model"])
    best_model = fitted_models[best_model_name]

    model_output_dir = _get_models_variant_dir(
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
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "model_name": best_model_name,
    }

    evaluation_df.to_csv(model_output_dir / "metrics.csv", index=False)
    prediction_df.to_csv(model_output_dir / "test_predictions.csv", index=False)

    home_goal_regressor, away_goal_regressor = _fit_goal_regressors(
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


def freeze_model_variant(
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
    freeze_label: str = "official",
) -> FreezeRunSummary:
    """Copy one trained variant artifact bundle to a frozen release directory."""

    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    source_dir = _get_models_variant_dir(
        models_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    if not (source_dir / "best_model.pkl").exists():
        raise FileNotFoundError(
            f"No trained model found in {source_dir}. Run --stage train first."
        )

    frozen_dir = _get_frozen_variant_dir(
        models_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
        freeze_label=freeze_label,
    )
    frozen_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "best_model.pkl",
        "metrics.csv",
        "goal_metrics.csv",
        "test_predictions.csv",
        "artifact_meta.json",
    ]:
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, frozen_dir / filename)

    manifest = {
        "freeze_label": freeze_label,
        "cutoff_date": cutoff_date,
        "variant_name": variant_name,
        "source_dir": str(source_dir),
        "frozen_dir": str(frozen_dir),
        "frozen_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }
    with (frozen_dir / "freeze_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return FreezeRunSummary(
        variant_name=variant_name,
        source_dir=source_dir,
        frozen_dir=frozen_dir,
        freeze_label=freeze_label,
    )


def freeze_model_variants(
    models_dir: Path,
    cutoff_date: str,
    include_odds_variants: List[bool],
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
    freeze_label: str = "official",
) -> List[FreezeRunSummary]:
    """Freeze model artifacts for multiple selected variants."""

    return [
        freeze_model_variant(
            models_dir=models_dir,
            cutoff_date=cutoff_date,
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
            freeze_label=freeze_label,
        )
        for include_odds in include_odds_variants
    ]


def build_baseline_metrics_report(
    models_dir: Path,
    cutoff_date: str,
    include_odds_variants: List[bool],
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> BaselineReportSummary:
    """Create a consolidated baseline metrics report across selected variants."""

    report_rows: List[Dict[str, Any]] = []
    for include_odds in include_odds_variants:
        variant_name = _get_variant_name(
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
        variant_dir = _get_models_variant_dir(
            models_dir=models_dir,
            cutoff_date=cutoff_date,
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
        metrics_path = variant_dir / "metrics.csv"
        goals_path = variant_dir / "goal_metrics.csv"
        meta_path = variant_dir / "artifact_meta.json"
        if not metrics_path.exists() or not goals_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing metrics files for {variant_name} in {variant_dir}. Run --stage train first."
            )

        metrics_df = pd.read_csv(metrics_path)
        goal_df = pd.read_csv(goals_path)
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        best_row = metrics_df.sort_values("log_loss").iloc[0]
        home_goals_mae = pd.to_numeric(goal_df["home_goals_mae"], errors="coerce").iloc[0]
        away_goals_mae = pd.to_numeric(goal_df["away_goals_mae"], errors="coerce").iloc[0]
        report_rows.append(
            {
                "variant_name": variant_name,
                "best_model": meta.get("model_name", str(best_row["model"])),
                "log_loss": float(best_row["log_loss"]),
                "accuracy": float(best_row["accuracy"]),
                "macro_f1": float(best_row["macro_f1"]),
                "home_goals_mae": float(home_goals_mae),
                "away_goals_mae": float(away_goals_mae),
                "trained_at_utc": meta.get("trained_at_utc"),
            }
        )

    output_path = models_dir / f"date_{cutoff_date}" / "baseline_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cutoff_date": cutoff_date,
                "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
                "rows": report_rows,
            },
            handle,
            indent=2,
        )

    return BaselineReportSummary(
        cutoff_date=cutoff_date,
        output_path=output_path,
        variants_included=len(report_rows),
    )


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
        prediction.expected_home_goals == prediction.expected_home_goals
        and prediction.expected_away_goals == prediction.expected_away_goals
        and prediction.expected_home_goals >= 0.0
        and prediction.expected_away_goals >= 0.0
    )
    passed = valid_probs and valid_sum and valid_goals
    details = (
        f"prob_sum={probability_sum:.6f}, valid_probs={valid_probs}, "
        f"valid_goals={valid_goals}, predicted={prediction.predicted_outcome}"
    )
    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return SmokeTestSummary(variant_name=variant_name, passed=passed, details=details)


def _load_model_artifact(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run --stage train first."
        )
    with model_path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Model artifact has unexpected format.")
    return loaded


def _normalize_split_team_state_rows(
    df: pd.DataFrame,
    team: str,
    division: str,
    as_of_date: Optional[pd.Timestamp],
    recent_form_window: int,
) -> pd.DataFrame:
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


def _estimate_team_state(
    split_df: pd.DataFrame,
    team: str,
    division: str,
    as_of_date: Optional[str],
    recent_form_window: int,
) -> Dict[str, float]:
    timestamp = pd.Timestamp(as_of_date) if as_of_date else None
    rows = _normalize_split_team_state_rows(
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


def _parse_overrides(overrides: Optional[List[str]]) -> Dict[str, float]:
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


def _split_feature_and_stat_overrides(
    overrides: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Split overrides into model-feature overrides and predicted-stat overrides."""

    feature_overrides = {
        key: value for key, value in overrides.items() if key not in STAT_OVERRIDE_KEYS
    }
    stat_overrides = {
        key: value for key, value in overrides.items() if key in STAT_OVERRIDE_KEYS
    }
    return feature_overrides, stat_overrides


def _apply_event_stat_overrides_to_expected_goals(
    expected_home_goals: float,
    expected_away_goals: float,
    stat_overrides: Dict[str, float],
) -> Tuple[float, float]:
    """Apply fun scenario adjustments from event-stat overrides to expected goals."""

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


def _poisson_pmf(lmbda: float, k: int) -> float:
    if lmbda < 0:
        raise ValueError("Poisson mean must be non-negative.")
    return math.exp(-lmbda) * (lmbda**k) / math.factorial(k)


def _outcome_probabilities_from_expected_goals(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 10,
) -> Tuple[float, float, float]:
    """Approximate P(H), P(D), P(A) via independent Poisson goal models."""

    home_probs = [_poisson_pmf(expected_home_goals, k) for k in range(max_goals + 1)]
    away_probs = [_poisson_pmf(expected_away_goals, k) for k in range(max_goals + 1)]

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


def _build_feature_row(
    feature_columns: List[str],
    home_state: Dict[str, float],
    away_state: Dict[str, float],
    feature_fill_values: Dict[str, float],
    overrides: Dict[str, float],
    kickoff_time_minutes: Optional[float],
) -> pd.DataFrame:
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


def _parse_kickoff_time_minutes(kickoff_time: Optional[str]) -> Optional[float]:
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

    variant_name = _get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    model_dir = _get_models_variant_dir(
        models_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    artifact = _load_model_artifact(model_dir / "best_model.pkl")

    split_variant_dir = (
        splits_dir
        / f"date_{cutoff_date}"
        / _get_variant_name(
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

    home_state = _estimate_team_state(
        split_df=split_df,
        team=home_team,
        division=division,
        as_of_date=as_of_date,
        recent_form_window=recent_form_window,
    )
    away_state = _estimate_team_state(
        split_df=split_df,
        team=away_team,
        division=division,
        as_of_date=as_of_date,
        recent_form_window=recent_form_window,
    )

    kickoff_time_minutes = _parse_kickoff_time_minutes(kickoff_time)
    overrides = _parse_overrides(feature_overrides)
    model_feature_overrides, stat_overrides = _split_feature_and_stat_overrides(
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

    feature_row = _build_feature_row(
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
            _apply_event_stat_overrides_to_expected_goals(
                expected_home_goals=expected_home_goals,
                expected_away_goals=expected_away_goals,
                stat_overrides=stat_overrides,
            )
        )

    if not math.isnan(expected_home_goals) and not math.isnan(expected_away_goals):
        probabilities = _outcome_probabilities_from_expected_goals(
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


def print_train_summary(summary: TrainRunSummary) -> None:
    """Print a compact model training summary."""

    print(summary.summary())


def print_freeze_summary(summary: FreezeRunSummary) -> None:
    """Print a compact freeze summary."""

    print(summary.summary())


def print_baseline_report_summary(summary: BaselineReportSummary) -> None:
    """Print a compact baseline-report summary."""

    print(summary.summary())


def print_smoke_test_summary(summary: SmokeTestSummary) -> None:
    """Print a compact smoke-test summary."""

    print(summary.summary())


def print_prediction_summary(summary: PredictionSummary) -> None:
    """Print a compact prediction summary."""

    print(summary.summary())
