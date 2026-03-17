"""Scrape top-flight league CSVs from football-data.co.uk and save season files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

from .config import MIN_START_YEAR, REQUEST_DELAY_SECONDS
from .discovery import discover_country_pages, make_session, scrape_top_league_links
from .downloader import download_csv_file
from .utils import (
    page_to_country_slug,
    season_code_to_label,
    season_label_to_code,
    season_start_year_from_label,
)


def scrape_top_flight_leagues(
    min_start_year: int = MIN_START_YEAR,
    countries: Iterable[str] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Scrape top-flight league CSVs from the site's country pages.

    Args:
        min_start_year: Download only seasons whose starting year is at least this value.
        countries: Optional iterable of country slugs to include, such as ``["england", "germany"]``.
                   Defaults to all country pages discovered from the main download page.
        output_dir: Destination directory. Defaults to ``data/raw``.

    Returns:
        List of ``Path`` objects for each written CSV file.
    """
    target_dir = output_dir or Path("data") / "raw"
    session = make_session()

    print("Discovering country download pages from football-data.co.uk...")
    country_pages = discover_country_pages(session)
    if countries is not None:
        requested = {country.lower() for country in countries}
        country_pages = [
            page for page in country_pages if page_to_country_slug(page) in requested
        ]

    time.sleep(REQUEST_DELAY_SECONDS)

    written: list[Path] = []
    for country_index, country_page in enumerate(country_pages):
        try:
            country_slug, league_code, available_links = scrape_top_league_links(
                session, country_page
            )
        except RuntimeError as error:
            print(f"Skipping {country_page}: {error}")
            continue

        season_labels = sorted(
            (season_code_to_label(code) for code in available_links),
            key=season_start_year_from_label,
        )

        season_labels = [
            label
            for label in season_labels
            if season_start_year_from_label(label) >= min_start_year
        ]
        if not season_labels:
            continue

        print(
            f"{country_slug}: top league {league_code}, {len(season_labels)} seasons from {min_start_year} onward"
        )

        for season_index, season_label in enumerate(season_labels):
            season_code = season_label_to_code(season_label)
            url = available_links[season_code]
            output_path = (
                target_dir
                / country_slug
                / f"{country_slug}_{league_code}_{season_label}.csv"
            )
            print(f"  Downloading {season_label} from {url}...")
            path, row_count = download_csv_file(
                url=url, output_path=output_path, session=session
            )
            written.append(path)
            print(f"    Saved {row_count} rows → {path}")

            if not (
                country_index == len(country_pages) - 1
                and season_index == len(season_labels) - 1
            ):
                time.sleep(REQUEST_DELAY_SECONDS)

    return written


def scrape_premier_league_seasons(
    seasons: Iterable[str] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Backward-compatible wrapper for scraping only England's top flight."""
    target_dir = output_dir or Path("data") / "raw"
    session = make_session()
    country_slug, league_code, available_links = scrape_top_league_links(
        session, "englandm.php"
    )

    target_seasons = sorted(
        (season_code_to_label(code) for code in available_links),
        key=season_start_year_from_label,
    )
    if seasons is not None:
        target_seasons = list(seasons)

    written: list[Path] = []
    for index, season_label in enumerate(target_seasons):
        season_code = season_label_to_code(season_label)
        if season_code not in available_links:
            raise RuntimeError(
                f"Season '{season_label}' was not found for England top flight."
            )
        output_path = (
            target_dir
            / country_slug
            / f"{country_slug}_{league_code}_{season_label}.csv"
        )
        path, _ = download_csv_file(available_links[season_code], output_path, session)
        written.append(path)
        if index < len(target_seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    return written


def main() -> None:
    """Run the scraper for top-flight league data across all available countries."""
    parser = argparse.ArgumentParser(
        description="Download top-flight CSVs from football-data.co.uk."
    )
    parser.add_argument(
        "--min-start-year",
        type=int,
        default=MIN_START_YEAR,
        help="Only download seasons whose starting year is at least this value.",
    )
    args = parser.parse_args()

    paths = scrape_top_flight_leagues(min_start_year=args.min_start_year)
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
