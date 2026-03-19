"""Entry point for scraping and preprocessing pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from model.pipeline import (
    build_baseline_metrics_report,
    freeze_model_variants,
    predict_match_outcome,
    print_baseline_report_summary,
    print_freeze_summary,
    print_prediction_summary,
    print_smoke_test_summary,
    print_train_summary,
    run_prediction_smoke_test,
    train_model_variants,
)
from preprocessing.modeling import (
    build_modeling_dataset_variants,
    print_modeling_summary,
)
from preprocessing.pipeline import print_pipeline_summary, run_preprocessing_variants
from preprocessing.splitter import print_split_summary, run_date_split_variants
from scraper.config import MIN_START_YEAR
from scraper.football_data_scraper import scrape_top_flight_leagues


def main() -> None:
    """Run scraping, preprocessing, or both stages."""

    parser = argparse.ArgumentParser(
        description="Run football data scraping and preprocessing stages."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "scrape",
            "preprocess",
            "split",
            "modeldata",
            "train",
            "freeze",
            "report",
            "smoke",
            "predict",
            "all",
        ],
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
    parser.add_argument(
        "--split-cutoff-date",
        type=str,
        help="For date-based splitting: first date included in the test set, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--division",
        type=str,
        help="For predict stage: division code (for example E0, SP1, I1).",
    )
    parser.add_argument(
        "--home-team",
        type=str,
        help="For predict stage: home team name.",
    )
    parser.add_argument(
        "--away-team",
        type=str,
        help="For predict stage: away team name.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="For predict stage: use team history up to this YYYY-MM-DD date.",
    )
    parser.add_argument(
        "--kickoff-time",
        type=str,
        help="For predict stage: kickoff time in HH:MM format (optional).",
    )
    parser.add_argument(
        "--feature-override",
        action="append",
        default=[],
        help=(
            "For predict stage: override one feature as feature=value. "
            "Can be repeated, e.g. --feature-override odds_home_win=1.9"
        ),
    )
    parser.add_argument(
        "--freeze-label",
        type=str,
        default="official",
        help="For freeze stage: label used under data/models/frozen/<label>/...",
    )
    args = parser.parse_args()

    if (
        args.stage
        in {
            "split",
            "modeldata",
            "train",
            "freeze",
            "report",
            "smoke",
            "predict",
            "all",
        }
        and not args.split_cutoff_date
    ):
        parser.error(
            "--split-cutoff-date is required when --stage is 'split', 'modeldata', 'train', 'freeze', 'report', 'smoke', 'predict', or 'all'."
        )

    if args.stage == "predict":
        missing_predict_args = [
            name
            for name, value in [
                ("--division", args.division),
                ("--home-team", args.home_team),
                ("--away-team", args.away_team),
            ]
            if not value
        ]
        if missing_predict_args:
            parser.error(
                "Missing required predict arguments: " + ", ".join(missing_predict_args)
            )

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

    if args.stage in {"split", "all"}:
        variants = [False, True] if args.write_both_variants else [args.include_odds]
        summaries = run_date_split_variants(
            processed_dir=Path("data") / "processed",
            splits_dir=Path("data") / "splits",
            include_odds_variants=variants,
            cutoff_date=args.split_cutoff_date,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
        )
        for summary in summaries:
            print_split_summary(summary)

    if args.stage in {"modeldata", "all"}:
        variants = [False, True] if args.write_both_variants else [args.include_odds]
        summaries = build_modeling_dataset_variants(
            splits_dir=Path("data") / "splits",
            modeling_dir=Path("data") / "modeling",
            cutoff_date=args.split_cutoff_date,
            include_odds_variants=variants,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
        )
        for summary in summaries:
            print_modeling_summary(summary)

    if args.stage in {"train", "all"}:
        variants = [False, True] if args.write_both_variants else [args.include_odds]
        summaries = train_model_variants(
            modeling_dir=Path("data") / "modeling",
            splits_dir=Path("data") / "splits",
            models_dir=Path("data") / "models",
            cutoff_date=args.split_cutoff_date,
            include_odds_variants=variants,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
        )
        for summary in summaries:
            print_train_summary(summary)

    if args.stage in {"freeze", "all"}:
        variants = [False, True] if args.write_both_variants else [args.include_odds]
        summaries = freeze_model_variants(
            models_dir=Path("data") / "models",
            cutoff_date=args.split_cutoff_date,
            include_odds_variants=variants,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
            freeze_label=args.freeze_label,
        )
        for summary in summaries:
            print_freeze_summary(summary)

    if args.stage in {"report", "all"}:
        variants = [False, True] if args.write_both_variants else [args.include_odds]
        summary = build_baseline_metrics_report(
            models_dir=Path("data") / "models",
            cutoff_date=args.split_cutoff_date,
            include_odds_variants=variants,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
        )
        print_baseline_report_summary(summary)

    if args.stage in {"smoke", "all"}:
        summary = run_prediction_smoke_test(
            splits_dir=Path("data") / "splits",
            models_dir=Path("data") / "models",
            cutoff_date=args.split_cutoff_date,
            include_odds=args.include_odds,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
            division=args.division or "E0",
            home_team=args.home_team or "Arsenal",
            away_team=args.away_team or "Chelsea",
        )
        print_smoke_test_summary(summary)

    if args.stage == "predict":
        summary = predict_match_outcome(
            splits_dir=Path("data") / "splits",
            models_dir=Path("data") / "models",
            cutoff_date=args.split_cutoff_date,
            include_odds=args.include_odds,
            division=args.division,
            home_team=args.home_team,
            away_team=args.away_team,
            add_recent_form_features=args.add_recent_form_features,
            recent_form_window=args.recent_form_window,
            as_of_date=args.as_of_date,
            kickoff_time=args.kickoff_time,
            feature_overrides=args.feature_override,
        )
        print_prediction_summary(summary)


if __name__ == "__main__":
    main()
