"""Modeling-dataset preparation from date-split football data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .pipeline import get_processed_variant_name

TARGET_COLUMN = "target_result"
IDENTIFIER_COLUMNS = ["date", "div", "home_team", "away_team"]
ALLOWED_CONTEXT_FEATURE_COLUMNS = {"time"}
ALLOWED_FEATURE_PREFIXES = ("odds_", "home_", "away_")

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


def is_modeling_feature_column(column: str) -> bool:
    """Return whether a column is allowed as a leakage-safe model feature."""

    if column in IDENTIFIER_COLUMNS:
        return False
    if column == TARGET_COLUMN:
        return False
    if column in LEAKAGE_COLUMNS:
        return False
    if column in ALLOWED_CONTEXT_FEATURE_COLUMNS:
        return True
    return column.startswith(ALLOWED_FEATURE_PREFIXES)


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
    return [
        column for column in available_columns if is_modeling_feature_column(column)
    ]


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


def _encode_kickoff_time_to_minutes(series: pd.Series) -> pd.Series:
    """Convert HH:MM kickoff times into minutes since midnight."""

    as_text = series.astype("string").str.strip()
    extracted = as_text.str.extract(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})")

    hours = pd.to_numeric(extracted["hour"], errors="coerce")
    minutes = pd.to_numeric(extracted["minute"], errors="coerce")
    return (hours * 60.0) + minutes


def preprocess_modeling_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    recent_form_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply train-fitted feature preprocessing for modeling datasets."""

    train_processed = x_train.copy()
    test_processed = x_test.copy()

    if "time" in train_processed.columns:
        train_processed["time"] = _encode_kickoff_time_to_minutes(
            train_processed["time"]
        )
        test_processed["time"] = _encode_kickoff_time_to_minutes(test_processed["time"])

    for side in ("home", "away"):
        matches_col = f"{side}_matches_played_before_match"
        sparse_col = f"{side}_sparse_history"
        if matches_col in train_processed.columns:
            train_processed[sparse_col] = (
                train_processed[matches_col] < recent_form_window
            ).astype("int64")
            test_processed[sparse_col] = (
                test_processed[matches_col] < recent_form_window
            ).astype("int64")

    numeric_columns = train_processed.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        fill_values = (
            train_processed[numeric_columns].median(numeric_only=True).fillna(0.0)
        )
        train_processed[numeric_columns] = train_processed[numeric_columns].fillna(
            fill_values
        )
        test_processed[numeric_columns] = test_processed[numeric_columns].fillna(
            fill_values
        )

    return train_processed, test_processed


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
    x_train, x_test = preprocess_modeling_features(
        x_train=x_train,
        x_test=x_test,
        recent_form_window=recent_form_window,
    )

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
