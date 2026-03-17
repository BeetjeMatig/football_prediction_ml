"""Utility helpers for football-data season and country parsing."""

from __future__ import annotations

from pathlib import Path


def page_to_country_slug(page_name: str) -> str:
    """Convert a country page filename into a clean country slug."""
    slug = Path(page_name).stem.lower()
    if slug.endswith("m"):
        slug = slug[:-1]
    return slug


def normalize_country_page(page_name: str) -> str:
    """Normalize country page filenames from the site into a lowercase form."""
    normalized = Path(page_name).name.lower()
    if normalized == "argentia.php":
        return "argentina.php"
    return normalized


def season_code_to_label(season_code: str) -> str:
    """Convert a football-data season code like ``2324`` into ``2023-24``."""
    if len(season_code) != 4 or not season_code.isdigit():
        raise ValueError(f"Invalid season code '{season_code}'.")

    start_two_digits = int(season_code[:2])
    end_two_digits = int(season_code[2:])

    start_year = (
        2000 + start_two_digits if start_two_digits < 90 else 1900 + start_two_digits
    )
    return f"{start_year}-{end_two_digits:02d}"


def season_label_to_code(season_label: str) -> str:
    """Convert a season label like ``2023-24`` into football-data code ``2324``."""
    try:
        start_year, end_year = season_label.split("-")
    except ValueError as error:
        raise ValueError(
            f"Invalid season label '{season_label}'. Expected format like '2023-24'."
        ) from error

    if (
        len(start_year) != 4
        or len(end_year) != 2
        or not start_year.isdigit()
        or not end_year.isdigit()
    ):
        raise ValueError(
            f"Invalid season label '{season_label}'. Expected format like '2023-24'."
        )

    return f"{start_year[2:]}{end_year}"


def season_start_year_from_label(season_label: str) -> int:
    """Return the first year of a season label like ``2010-11``."""
    return int(season_label.split("-")[0])
