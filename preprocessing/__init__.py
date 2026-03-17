"""Preprocessing package for schema-driven data preparation."""

from .cleaner import (
    CleaningResult,
    clean_all,
    clean_dataframe,
    clean_file,
    print_cleaning_report,
)
from .pipeline import (
    PipelineRunSummary,
    print_pipeline_summary,
    run_preprocessing,
    run_preprocessing_variants,
)
from .schema import (
    COLUMN_SCHEMA,
    ODDS_GROUPS,
    REQUIRED_COLUMNS,
    get_output_columns,
    normalize_column_name,
    normalize_columns,
)
from .selection import select_output_columns
from .validator import (
    ValidationResult,
    print_validation_report,
    validate_all,
    validate_file,
)

__all__ = [
    "COLUMN_SCHEMA",
    "ODDS_GROUPS",
    "REQUIRED_COLUMNS",
    "get_output_columns",
    "normalize_column_name",
    "normalize_columns",
    "CleaningResult",
    "clean_dataframe",
    "clean_file",
    "clean_all",
    "print_cleaning_report",
    "PipelineRunSummary",
    "run_preprocessing",
    "run_preprocessing_variants",
    "print_pipeline_summary",
    "select_output_columns",
    "ValidationResult",
    "validate_file",
    "validate_all",
    "print_validation_report",
]
