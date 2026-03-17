"""Feature/column selection helpers for preprocessing outputs."""

from __future__ import annotations

from typing import List

from .schema import get_output_columns


def select_output_columns(
    available_columns: List[str], include_odds: bool
) -> List[str]:
    """Return ordered columns to keep in final output based on odds toggle."""

    requested = get_output_columns(include_odds=include_odds)
    available_set = set(available_columns)
    return [column for column in requested if column in available_set]
