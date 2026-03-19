"""Output formatting and printing utilities."""

from __future__ import annotations

from model.artifact import FreezeRunSummary
from model.prediction import PredictionSummary
from model.reporting import BaselineReportSummary
from model.testing import SmokeTestSummary
from model.training import TrainRunSummary


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
