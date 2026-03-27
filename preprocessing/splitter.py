"""Date-based train/test splitting for processed football datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .features import add_cross_season_recent_form_features
from .pipeline import get_processed_variant_name


@dataclass
class SplitRunSummary:
    """Aggregate summary for one date-based split run."""

    include_odds: bool
    add_recent_form_features: bool
    recent_form_window: int
    cutoff_date: str
    input_dir: Path
    output_dir: Path
    files_read: int
    total_rows: int
    train_rows: int
    test_rows: int

    def summary(self) -> str:
        variant = "extended" if self.include_odds else "base"
        feature_suffix = (
            f", recent_form_window={self.recent_form_window}"
            if self.add_recent_form_features
            else ""
        )
        return (
            f"variant={variant}, cutoff_date={self.cutoff_date}, "
            f"files={self.files_read}, rows={self.total_rows}, "
            f"train={self.train_rows}, test={self.test_rows}, "
            f"features={self.add_recent_form_features}{feature_suffix}, "
            f"output_dir={self.output_dir}"
        )


def _find_input_dir(processed_dir: Path, include_odds: bool) -> Path:
    """Find input directory allowing for suffixes like base_recent_form_w5."""
    prefix = "extended" if include_odds else "base"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    # First try exact match
    exact_dir = processed_dir / prefix
    if exact_dir.exists():
        return exact_dir

    # Then look for directories starting with prefix + "_"
    for candidate in processed_dir.iterdir():
        if candidate.is_dir() and candidate.name.startswith(f"{prefix}_"):
            return candidate

    raise FileNotFoundError(
        f"No processed directory found with prefix '{prefix}' in {processed_dir}"
    )


def load_processed_dataset(input_dir: Path) -> Tuple[pd.DataFrame, int]:
    """Load and combine all processed CSV files under a variant directory."""

    csv_paths = sorted(input_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No processed CSV files found in {input_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined, len(csv_paths)


def split_dataset_by_date(
    df: pd.DataFrame,
    cutoff_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a combined dataset into train and test sets by cutoff date."""

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column for date-based splitting.")

    cutoff = pd.Timestamp(cutoff_date)
    ordered = df.sort_values(["date", "div", "home_team", "away_team"]).reset_index(
        drop=True
    )

    train_df = ordered.loc[ordered["date"] < cutoff].reset_index(drop=True)
    test_df = ordered.loc[ordered["date"] >= cutoff].reset_index(drop=True)
    return train_df, test_df


def run_date_split(
    processed_dir: Path,
    splits_dir: Path,
    include_odds: bool,
    cutoff_date: str,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> SplitRunSummary:
    """Combine processed files for one variant and write date-based train/test CSVs."""

    input_dir = _find_input_dir(processed_dir=processed_dir, include_odds=include_odds)
    variant_name = get_processed_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )

    combined_df, files_read = load_processed_dataset(input_dir)
    if add_recent_form_features:
        combined_df = add_cross_season_recent_form_features(
            combined_df,
            window=recent_form_window,
        )
    train_df, test_df = split_dataset_by_date(combined_df, cutoff_date=cutoff_date)

    output_dir = splits_dir / f"date_{cutoff_date}" / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    return SplitRunSummary(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
        cutoff_date=cutoff_date,
        input_dir=input_dir,
        output_dir=output_dir,
        files_read=files_read,
        total_rows=len(combined_df),
        train_rows=len(train_df),
        test_rows=len(test_df),
    )


def run_date_split_variants(
    processed_dir: Path,
    splits_dir: Path,
    include_odds_variants: List[bool],
    cutoff_date: str,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> List[SplitRunSummary]:
    """Run date-based splitting for multiple processed data variants."""

    return [
        run_date_split(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            include_odds=include_odds,
            cutoff_date=cutoff_date,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
        for include_odds in include_odds_variants
    ]


def print_split_summary(summary: SplitRunSummary) -> None:
    """Print a compact split summary."""

    print(summary.summary())
