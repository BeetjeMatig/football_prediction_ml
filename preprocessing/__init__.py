"""Preprocessing package for schema-driven data preparation."""

from .schema import (
    COLUMN_SCHEMA,
    ODDS_GROUPS,
    REQUIRED_COLUMNS,
    get_output_columns,
    normalize_column_name,
    normalize_columns,
)
from .selection import select_output_columns
from .validator import ValidationResult, print_validation_report, validate_all, validate_file

__all__ = [
    "COLUMN_SCHEMA",
    "ODDS_GROUPS",
    "REQUIRED_COLUMNS",
    "get_output_columns",
    "normalize_column_name",
    "normalize_columns",
    "select_output_columns",
    "ValidationResult",
    "validate_file",
    "validate_all",
    "print_validation_report",
]
