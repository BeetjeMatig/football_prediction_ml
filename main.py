"""Entry point for scraping and preprocessing pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing.pipeline import print_pipeline_summary, run_preprocessing_variants
from scraper.config import MIN_START_YEAR
from scraper.football_data_scraper import scrape_top_flight_leagues


def main() -> None:
    """Run scraping, preprocessing, or both stages."""

    parser = argparse.ArgumentParser(
        description="Run football data scraping and preprocessing stages."
    )
    parser.add_argument(
        "--stage",
        choices=["scrape", "preprocess", "all"],
        default="scrape",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--min-start-year",
        type=int,
        default=MIN_START_YEAR,
        help="For scraping: download seasons whose starting year is at least this value.",
    )
    parser.add_argument(
        "--include-odds",
        action="store_true",
        help="For preprocessing: include odds columns in output.",
    )
    parser.add_argument(
        "--write-both-variants",
        action="store_true",
        help="For preprocessing: write both base (no odds) and extended (with odds).",
    )
    parser.add_argument(
        "--add-recent-form-features",
        action="store_true",
        help="For preprocessing: add rolling recent-form features based only on prior matches.",
    )
    parser.add_argument(
        "--recent-form-window",
        type=int,
        default=5,
        help="For preprocessing with recent-form features: rolling window size.",
    )
    args = parser.parse_args()

    if args.stage in {"scrape", "all"}:
        written = scrape_top_flight_leagues(min_start_year=args.min_start_year)
        print(f"Scraping complete. Files written: {len(written)}")

    if args.stage in {"preprocess", "all"}:
        variants = [False, True] if args.write_both_variants else [args.include_odds]
        summaries = run_preprocessing_variants(
            raw_dir=Path("data") / "raw",
            processed_dir=Path("data") / "processed",
            include_odds_variants=variants,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
        )
        for summary in summaries:
            print_pipeline_summary(summary)


if __name__ == "__main__":
    main()
