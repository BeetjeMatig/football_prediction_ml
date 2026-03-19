"""Schema validation for raw football-data CSV files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from .schema import COLUMN_SCHEMA, ODDS_GROUPS, REQUIRED_COLUMNS, normalize_columns


@dataclass
class ValidationResult:
    """Validation outcome for a single raw CSV file."""

    file_path: Path
    missing_required: List[str] = field(default_factory=list)
    available_groups: Dict[str, bool] = field(default_factory=dict)
    unknown_columns: List[str] = field(default_factory=list)
    is_valid: bool = True

    def summary(self) -> str:
        lines = [f"File: {self.file_path.name}"]

        if self.is_valid:
            lines.append("  Status  : OK")
        else:
            lines.append("  Status  : INVALID")
            for col in self.missing_required:
                lines.append(f"  Missing : {col}")

        for group, present in self.available_groups.items():
            mark = "yes" if present else "no"
            lines.append(f"  Group '{group}': {mark}")

        if self.unknown_columns:
            lines.append(
                f"  Unknown columns ({len(self.unknown_columns)}): {', '.join(self.unknown_columns[:10])}"
            )

        return "\n".join(lines)


def validate_file(csv_path: Path) -> ValidationResult:
    """Validate a single raw CSV file against the schema."""

    try:
        raw_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        OSError,
    ) as exc:
        result = ValidationResult(file_path=csv_path, is_valid=False)
        result.missing_required = [f"(could not read file: {exc})"]
        return result

    rename_map = normalize_columns(raw_columns)
    normalized = {rename_map.get(c, c.strip().lower()) for c in raw_columns}

    missing_required = sorted(REQUIRED_COLUMNS - normalized)

    known_canonical = set(COLUMN_SCHEMA.keys())
    unknown_columns = sorted(
        col
        for col in raw_columns
        if rename_map.get(col, col.strip().lower()) not in known_canonical
    )

    available_groups: Dict[str, bool] = {}
    all_groups = {spec.group for spec in COLUMN_SCHEMA.values()}
    for group in sorted(all_groups):
        group_cols = {
            name for name, spec in COLUMN_SCHEMA.items() if spec.group == group
        }
        available_groups[group] = bool(group_cols & normalized)

    return ValidationResult(
        file_path=csv_path,
        missing_required=missing_required,
        available_groups=available_groups,
        unknown_columns=unknown_columns,
        is_valid=len(missing_required) == 0,
    )


def validate_all(raw_dir: Path) -> List[ValidationResult]:
    """Validate every CSV under raw_dir, returning one result per file."""

    results = []
    for csv_path in sorted(raw_dir.rglob("*.csv")):
        results.append(validate_file(csv_path))
    return results


def print_validation_report(results: List[ValidationResult]) -> None:
    """Print a human-readable validation report to stdout."""

    invalid = [r for r in results if not r.is_valid]
    odds_missing = [
        r
        for r in results
        if r.is_valid and not any(r.available_groups.get(g, False) for g in ODDS_GROUPS)
    ]

    print(f"=== Validation Report: {len(results)} files ===")
    print(f"Valid        : {len(results) - len(invalid)}")
    print(f"Invalid      : {len(invalid)}")
    print(f"No odds data : {len(odds_missing)}")
    print()

    if invalid:
        print("--- Invalid files ---")
        for r in invalid:
            print(r.summary())
            print()

    print("--- Group availability (first 5 files) ---")
    for r in results[:5]:
        print(r.summary())
        print()
