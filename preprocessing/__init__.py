"""Preprocessing package for schema-driven data preparation."""

from .cleaner import (
    CleaningResult,
    clean_all,
    clean_dataframe,
    clean_file,
    print_cleaning_report,
)
from .features import (
    RECENT_FORM_REQUIRED_COLUMNS,
    add_recent_form_features,
    get_recent_form_feature_columns,
)
from .pipeline import (
    PipelineRunSummary,
    get_processed_variant_dir,
    get_processed_variant_name,
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
from .splitter import (
    SplitRunSummary,
    load_processed_dataset,
    print_split_summary,
    run_date_split,
    run_date_split_variants,
    split_dataset_by_date,
)
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
    "RECENT_FORM_REQUIRED_COLUMNS",
    "add_recent_form_features",
    "get_recent_form_feature_columns",
    "PipelineRunSummary",
    "get_processed_variant_name",
    "get_processed_variant_dir",
    "run_preprocessing",
    "run_preprocessing_variants",
    "print_pipeline_summary",
    "select_output_columns",
    "SplitRunSummary",
    "load_processed_dataset",
    "split_dataset_by_date",
    "run_date_split",
    "run_date_split_variants",
    "print_split_summary",
    "ValidationResult",
    "validate_file",
    "validate_all",
    "print_validation_report",
]
