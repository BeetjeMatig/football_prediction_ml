"""Path management, variant naming, and data loading utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from preprocessing.pipeline import get_processed_variant_name


def _find_flexible_variant_dir(
    base_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    marker_file: str,
) -> Path:
    """Find variant directory allowing for suffixes like base_recent_form_w5.

    First tries exact match (e.g., base/ or extended/), then looks for
    directories starting with the prefix + "_" (e.g., base_recent_form_w5/)."""
    prefix = "extended" if include_odds else "base"
    date_dir = base_dir / f"date_{cutoff_date}"

    if not date_dir.exists():
        raise FileNotFoundError(f"Date directory not found: {date_dir}")

    # First try exact match
    exact_dir = date_dir / prefix
    if (exact_dir / marker_file).exists():
        return exact_dir

    # Then look for directories starting with prefix + "_"
    for candidate in date_dir.iterdir():
        if candidate.is_dir() and candidate.name.startswith(f"{prefix}_"):
            if (candidate / marker_file).exists():
                return candidate

    raise FileNotFoundError(
        f"No directory found with prefix '{prefix}' containing '{marker_file}' in {date_dir}"
    )


def find_models_variant_dir(
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
) -> Path:
    """Find trained model directory allowing for suffixes like base_recent_form_w5."""
    return _find_flexible_variant_dir(
        base_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        marker_file="best_model.pkl",
    )


def find_splits_variant_dir(
    splits_dir: Path,
    cutoff_date: str,
    include_odds: bool,
) -> Path:
    """Find splits directory allowing for suffixes like base_recent_form_w5."""
    return _find_flexible_variant_dir(
        base_dir=splits_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        marker_file="train.csv",
    )

LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}


def get_variant_name(
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
) -> str:
    """Generate standardized variant name from configuration parameters."""
    return get_processed_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )


def get_modeling_variant_dir(
    modeling_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
) -> Path:
    """Get path to modeling dataset variant directory."""
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return modeling_dir / f"date_{cutoff_date}" / variant_name


def get_models_variant_dir(
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
) -> Path:
    """Get path to trained models variant directory."""
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return models_dir / f"date_{cutoff_date}" / variant_name


def get_frozen_variant_dir(
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
    freeze_label: str,
) -> Path:
    """Get path to frozen variant artifact directory."""
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    return models_dir / "frozen" / freeze_label / f"date_{cutoff_date}" / variant_name


def load_modeling_data(
    input_dir: Path,
) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame
]:
    """Load training and test data with features and targets."""
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


def load_split_targets(
    splits_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool,
    recent_form_window: int,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load and align home/away goal targets to modeling train/test row order."""
    variant_name = get_variant_name(
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


def load_model_artifact(model_path: Path) -> Dict[str, Any]:
    """Load trained model artifact from pickle file.

    Security note: pickle can execute arbitrary code during deserialization.
    Only load artifacts from trusted sources. For production use, consider
    signed artifacts or safer serialization formats.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run --stage train first."
        )
    with model_path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Model artifact has unexpected format.")
    return loaded
