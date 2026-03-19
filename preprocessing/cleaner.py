"""Data cleaning utilities for schema-driven preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .schema import COLUMN_SCHEMA, REQUIRED_COLUMNS, normalize_columns
from .selection import select_output_columns


@dataclass
class CleaningResult:
    """Cleaning outcome for one CSV file."""

    file_path: Path
    rows_in: int
    rows_out: int
    dropped_rows: int
    include_odds: bool
    output_columns: List[str]

    def summary(self) -> str:
        return (
            f"{self.file_path.name}: rows {self.rows_in} -> {self.rows_out} "
            f"(dropped {self.dropped_rows}), cols={len(self.output_columns)}, "
            f"include_odds={self.include_odds}"
        )


def _coerce_known_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply schema dtypes for columns that exist in the dataframe."""

    for canonical_name, spec in COLUMN_SCHEMA.items():
        if canonical_name not in df.columns:
            continue

        if spec.dtype == "datetime":
            df[canonical_name] = _parse_date_series(df[canonical_name])
        elif spec.dtype == "Int64":
            df[canonical_name] = pd.to_numeric(
                df[canonical_name], errors="coerce"
            ).astype("Int64")
        elif spec.dtype == "float64":
            df[canonical_name] = pd.to_numeric(df[canonical_name], errors="coerce")
        elif spec.dtype == "string":
            df[canonical_name] = df[canonical_name].astype("string")
        elif spec.dtype == "category":
            series = df[canonical_name].astype("string").str.upper().str.strip()
            df[canonical_name] = series.astype("category")

    return df


def _parse_date_series(series: pd.Series) -> pd.Series:
    """Parse football-data date strings with explicit formats.

    Typical source formats are dd/mm/yy and dd/mm/YYYY. We parse explicitly to
    avoid format-inference warnings and keep parsing behavior stable.
    """

    values = series.astype("string").str.strip()

    parsed = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()

    if missing.any():
        parsed_short = pd.to_datetime(
            values[missing], format="%d/%m/%y", errors="coerce"
        )
        parsed.loc[missing] = parsed_short
        missing = parsed.isna()

    # Fallback for rare mixed formats while keeping day-first semantics explicit.
    if missing.any():
        parsed_mixed = pd.to_datetime(
            values[missing], format="mixed", dayfirst=True, errors="coerce"
        )
        parsed.loc[missing] = parsed_mixed

    return parsed


def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    """Read CSV with encoding fallbacks for legacy football-data files."""

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue

    if last_error is not None:
        raise last_error

    # Defensive fallback; loop above should always return or raise.
    return pd.read_csv(csv_path)


def clean_dataframe(df: pd.DataFrame, include_odds: bool) -> pd.DataFrame:
    """Normalize column names, coerce dtypes, and select output columns."""

    rename_map = normalize_columns(df.columns)
    cleaned = df.rename(columns=rename_map).copy()
    cleaned = _coerce_known_dtypes(cleaned)

    required_present = [
        column for column in REQUIRED_COLUMNS if column in cleaned.columns
    ]
    if required_present:
        cleaned = cleaned.dropna(subset=required_present)

    output_columns = select_output_columns(
        available_columns=list(cleaned.columns),
        include_odds=include_odds,
    )

    return cleaned[output_columns]


def clean_file(
    csv_path: Path, include_odds: bool
) -> Tuple[pd.DataFrame, CleaningResult]:
    """Read and clean a single CSV file."""

    df = _read_csv_with_fallback(csv_path)
    rows_in = len(df)

    cleaned = clean_dataframe(df, include_odds=include_odds)
    rows_out = len(cleaned)

    result = CleaningResult(
        file_path=csv_path,
        rows_in=rows_in,
        rows_out=rows_out,
        dropped_rows=rows_in - rows_out,
        include_odds=include_odds,
        output_columns=list(cleaned.columns),
    )

    return cleaned, result


def clean_all(raw_dir: Path, include_odds: bool) -> List[CleaningResult]:
    """Run cleaning for all CSV files under raw_dir."""

    results: List[CleaningResult] = []
    for csv_path in sorted(raw_dir.rglob("*.csv")):
        _, result = clean_file(csv_path=csv_path, include_odds=include_odds)
        results.append(result)

    return results


def print_cleaning_report(results: List[CleaningResult]) -> None:
    """Print aggregate cleaning statistics."""

    total_in = sum(r.rows_in for r in results)
    total_out = sum(r.rows_out for r in results)
    total_dropped = sum(r.dropped_rows for r in results)

    print(f"=== Cleaning Report: {len(results)} files ===")
    print(f"Rows in      : {total_in}")
    print(f"Rows out     : {total_out}")
    print(f"Rows dropped : {total_dropped}")
    print()

    print("--- Largest row drops (top 5) ---")
    top_drops = sorted(results, key=lambda r: r.dropped_rows, reverse=True)[:5]
    for result in top_drops:
        print(result.summary())
