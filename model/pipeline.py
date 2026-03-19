"""Model training and prediction orchestrator (re-exports from submodules)."""

from __future__ import annotations

# Import all public APIs from submodules for backward compatibility
from model.artifact import FreezeRunSummary, freeze_model_variant, freeze_model_variants
from model.prediction import (
    EVENT_STAT_BASELINES,
    EVENT_STAT_EFFECTS,
    LABEL_TO_INT,
    OVERRIDE_ALIASES,
    PredictionSummary,
    STAT_OVERRIDE_KEYS,
    predict_match_outcome,
)
from model.printing import (
    print_baseline_report_summary,
    print_freeze_summary,
    print_prediction_summary,
    print_smoke_test_summary,
    print_train_summary,
)
from model.reporting import (
    BaselineReportSummary,
    build_baseline_metrics_report,
)
from model.testing import SmokeTestSummary, run_prediction_smoke_test
from model.training import (
    TrainRunSummary,
    build_candidate_models,
    evaluate_model,
    fit_goal_regressors,
    train_model_variant,
    train_model_variants,
)
from model.utils import (
    INT_TO_LABEL,
    get_frozen_variant_dir,
    get_modeling_variant_dir,
    get_models_variant_dir,
    get_variant_name,
    load_model_artifact,
    load_modeling_data,
    load_split_targets,
)

__all__ = [
    # Constants
    "LABEL_TO_INT",
    "INT_TO_LABEL",
    "OVERRIDE_ALIASES",
    "STAT_OVERRIDE_KEYS",
    "EVENT_STAT_BASELINES",
    "EVENT_STAT_EFFECTS",
    # Dataclasses
    "TrainRunSummary",
    "FreezeRunSummary",
    "BaselineReportSummary",
    "SmokeTestSummary",
    "PredictionSummary",
    # Utilities
    "get_variant_name",
    "get_modeling_variant_dir",
    "get_models_variant_dir",
    "get_frozen_variant_dir",
    "load_modeling_data",
    "load_split_targets",
    "load_model_artifact",
    # Training
    "fit_goal_regressors",
    "build_candidate_models",
    "evaluate_model",
    "train_model_variant",
    "train_model_variants",
    # Artifact
    "freeze_model_variant",
    "freeze_model_variants",
    # Reporting
    "build_baseline_metrics_report",
    # Prediction
    "predict_match_outcome",
    # Testing
    "run_prediction_smoke_test",
    # Printing
    "print_train_summary",
    "print_freeze_summary",
    "print_baseline_report_summary",
    "print_smoke_test_summary",
    "print_prediction_summary",
]
