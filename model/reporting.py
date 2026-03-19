"""Baseline metrics report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from model.utils import get_models_variant_dir, get_variant_name


@dataclass
class BaselineReportSummary:
    """Summary for a consolidated baseline metrics report."""

    cutoff_date: str
    output_path: Path
    variants_included: int

    def summary(self) -> str:
        return (
            f"cutoff_date={self.cutoff_date}, variants={self.variants_included}, "
            f"output={self.output_path}"
        )


def build_baseline_metrics_report(
    models_dir: Path,
    cutoff_date: str,
    include_odds_variants: List[bool],
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
) -> BaselineReportSummary:
    """Create a consolidated baseline metrics report across selected variants."""
    report_rows: List[Dict[str, Any]] = []
    for include_odds in include_odds_variants:
        variant_name = get_variant_name(
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
        variant_dir = get_models_variant_dir(
            models_dir=models_dir,
            cutoff_date=cutoff_date,
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
        )
        metrics_path = variant_dir / "metrics.csv"
        goals_path = variant_dir / "goal_metrics.csv"
        meta_path = variant_dir / "artifact_meta.json"
        if (
            not metrics_path.exists()
            or not goals_path.exists()
            or not meta_path.exists()
        ):
            raise FileNotFoundError(
                f"Missing metrics files for {variant_name} in {variant_dir}. Run --stage train first."
            )

        metrics_df = pd.read_csv(metrics_path)
        goal_df = pd.read_csv(goals_path)
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        best_row = metrics_df.sort_values("log_loss").iloc[0]
        home_goals_mae = pd.to_numeric(goal_df["home_goals_mae"], errors="coerce").iloc[
            0
        ]
        away_goals_mae = pd.to_numeric(goal_df["away_goals_mae"], errors="coerce").iloc[
            0
        ]
        report_rows.append(
            {
                "variant_name": variant_name,
                "best_model": meta.get("model_name", str(best_row["model"])),
                "log_loss": float(best_row["log_loss"]),
                "accuracy": float(best_row["accuracy"]),
                "macro_f1": float(best_row["macro_f1"]),
                "home_goals_mae": float(home_goals_mae),
                "away_goals_mae": float(away_goals_mae),
                "trained_at_utc": meta.get("trained_at_utc"),
            }
        )

    output_path = models_dir / f"date_{cutoff_date}" / "baseline_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cutoff_date": cutoff_date,
                "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
                "rows": report_rows,
            },
            handle,
            indent=2,
        )

    return BaselineReportSummary(
        cutoff_date=cutoff_date,
        output_path=output_path,
        variants_included=len(report_rows),
    )
