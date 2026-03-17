"""End-to-end preprocessing pipeline for raw football CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .cleaner import CleaningResult, clean_file


@dataclass
class PipelineRunSummary:
    """Aggregate summary for one preprocessing run."""

    include_odds: bool
    output_dir: Path
    files_written: int
    rows_in: int
    rows_out: int
    rows_dropped: int

    def summary(self) -> str:
        variant = "extended" if self.include_odds else "base"
        return (
            f"variant={variant}, files={self.files_written}, "
            f"rows {self.rows_in}->{self.rows_out}, dropped={self.rows_dropped}, "
            f"output_dir={self.output_dir}"
        )


def _variant_dir(processed_dir: Path, include_odds: bool) -> Path:
    return processed_dir / ("extended" if include_odds else "base")


def run_preprocessing(
    raw_dir: Path,
    processed_dir: Path,
    include_odds: bool,
) -> PipelineRunSummary:
    """Clean all raw CSV files and write processed outputs preserving folder layout."""

    variant_dir = _variant_dir(processed_dir, include_odds)

    rows_in = 0
    rows_out = 0
    rows_dropped = 0
    files_written = 0

    for csv_path in sorted(raw_dir.rglob("*.csv")):
        cleaned_df, result = clean_file(csv_path=csv_path, include_odds=include_odds)

        relative_path = csv_path.relative_to(raw_dir)
        output_path = variant_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)

        rows_in += result.rows_in
        rows_out += result.rows_out
        rows_dropped += result.dropped_rows
        files_written += 1

    return PipelineRunSummary(
        include_odds=include_odds,
        output_dir=variant_dir,
        files_written=files_written,
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped=rows_dropped,
    )


def run_preprocessing_variants(
    raw_dir: Path,
    processed_dir: Path,
    include_odds_variants: List[bool],
) -> List[PipelineRunSummary]:
    """Run preprocessing for multiple output variants."""

    return [
        run_preprocessing(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            include_odds=include_odds,
        )
        for include_odds in include_odds_variants
    ]


def print_pipeline_summary(summary: PipelineRunSummary) -> None:
    """Print a compact run summary."""

    print(summary.summary())
