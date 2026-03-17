"""Modeling-dataset preparation from date-split football data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .pipeline import get_processed_variant_name


TARGET_COLUMN = "target_result"
IDENTIFIER_COLUMNS = ["date", "div", "home_team", "away_team"]

# These columns contain information from the match being predicted and must not be
# used as direct model inputs for a pre-match outcome model.
LEAKAGE_COLUMNS = {
    "full_time_home_goals",
    "full_time_away_goals",
    "half_time_home_goals",
    "half_time_away_goals",
    "half_time_result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
}


@dataclass
class ModelingDatasetSummary:
    """Aggregate summary for one modeling-dataset preparation run."""

    include_odds: bool
    add_recent_form_features: bool
    recent_form_window: int
    cutoff_date: str
    input_dir: Path
    output_dir: Path
    train_rows: int
    test_rows: int
    feature_count: int

    def summary(self) -> str:
        variant = "extended" if self.include_odds else "base"
        feature_suffix = (
            f", recent_form_window={self.recent_form_window}"
            if self.add_recent_form_features
            else ""
        )
        return (
            f"variant={variant}, cutoff_date={self.cutoff_date}, "
            f"train={self.train_rows}, test={self.test_rows}, "
            f"features={self.feature_count}, "
            f"recent_form={self.add_recent_form_features}{feature_suffix}, "
            f"output_dir={self.output_dir}"
        )


def get_split_variant_dir(
    splits_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
) -> Path:
    """Return the split directory for a selected processed-data variant."""

    variant_name = get_processed_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return splits_dir / f"date_{cutoff_date}" / variant_name


def get_modeling_feature_columns(columns: Iterable[str]) -> List[str]:
    """Return leakage-safe feature columns based on available dataset columns."""

    available_columns = list(columns)
    excluded_columns = set(IDENTIFIER_COLUMNS)
    excluded_columns.add(TARGET_COLUMN)
    excluded_columns.update(LEAKAGE_COLUMNS)

    return [column for column in available_columns if column not in excluded_columns]


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split a dataset into features, target, and identifier metadata."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    feature_columns = get_modeling_feature_columns(df.columns)
    metadata_columns = [column for column in IDENTIFIER_COLUMNS if column in df.columns]

    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    metadata = df[metadata_columns].copy()
    return x, y, metadata


def build_modeling_dataset(
    splits_dir: Path,
    modeling_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> ModelingDatasetSummary:
    """Create leakage-safe X/y train-test datasets from a date-based split."""

    input_dir = get_split_variant_dir(
        splits_dir=splits_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Split files not found. Run the split stage before building modeling datasets."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train, y_train, train_metadata = split_features_and_target(train_df)
    x_test, y_test, test_metadata = split_features_and_target(test_df)

    variant_name = get_processed_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    output_dir = modeling_dir / f"date_{cutoff_date}" / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(output_dir / "X_train.csv", index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_csv(output_dir / "y_train.csv", index=False)
    x_test.to_csv(output_dir / "X_test.csv", index=False)
    y_test.to_frame(name=TARGET_COLUMN).to_csv(output_dir / "y_test.csv", index=False)
    train_metadata.to_csv(output_dir / "train_metadata.csv", index=False)
    test_metadata.to_csv(output_dir / "test_metadata.csv", index=False)

    return ModelingDatasetSummary(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
        cutoff_date=cutoff_date,
        input_dir=input_dir,
        output_dir=output_dir,
        train_rows=len(x_train),
        test_rows=len(x_test),
        feature_count=len(x_train.columns),
    )


def build_modeling_dataset_variants(
    splits_dir: Path,
    modeling_dir: Path,
    cutoff_date: str,
    include_odds_variants: List[bool],
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> List[ModelingDatasetSummary]:
    """Build modeling datasets for multiple processed-data variants."""

    return [
        build_modeling_dataset(
            splits_dir=splits_dir,
            modeling_dir=modeling_dir,
            cutoff_date=cutoff_date,
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
        for include_odds in include_odds_variants
    ]


def print_modeling_summary(summary: ModelingDatasetSummary) -> None:
    """Print a compact modeling-dataset summary."""

    print(summary.summary())