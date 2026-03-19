"""Model artifact freezing and management."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from model.utils import get_frozen_variant_dir, get_models_variant_dir, get_variant_name


@dataclass
class FreezeRunSummary:
    """Summary for freezing one trained variant artifact bundle."""

    variant_name: str
    source_dir: Path
    frozen_dir: Path
    freeze_label: str

    def summary(self) -> str:
        return (
            f"variant={self.variant_name}, freeze_label={self.freeze_label}, "
            f"source_dir={self.source_dir}, frozen_dir={self.frozen_dir}"
        )


def freeze_model_variant(
    models_dir: Path,
    cutoff_date: str,
    include_odds: bool,
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
    freeze_label: str = "official",
) -> FreezeRunSummary:
    """Copy one trained variant artifact bundle to a frozen release directory."""
    variant_name = get_variant_name(
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    source_dir = get_models_variant_dir(
        models_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
    )
    if not (source_dir / "best_model.pkl").exists():
        raise FileNotFoundError(
            f"No trained model found in {source_dir}. Run --stage train first."
        )

    frozen_dir = get_frozen_variant_dir(
        models_dir=models_dir,
        cutoff_date=cutoff_date,
        include_odds=include_odds,
        add_recent_form_features=add_recent_form_features,
        recent_form_window=recent_form_window,
        freeze_label=freeze_label,
    )
    frozen_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "best_model.pkl",
        "metrics.csv",
        "goal_metrics.csv",
        "test_predictions.csv",
        "artifact_meta.json",
    ]:
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, frozen_dir / filename)

    manifest = {
        "freeze_label": freeze_label,
        "cutoff_date": cutoff_date,
        "variant_name": variant_name,
        "source_dir": str(source_dir),
        "frozen_dir": str(frozen_dir),
        "frozen_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }
    with (frozen_dir / "freeze_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return FreezeRunSummary(
        variant_name=variant_name,
        source_dir=source_dir,
        frozen_dir=frozen_dir,
        freeze_label=freeze_label,
    )


def freeze_model_variants(
    models_dir: Path,
    cutoff_date: str,
    include_odds_variants: List[bool],
    add_recent_form_features: bool = False,
    recent_form_window: int = 5,
    freeze_label: str = "official",
) -> List[FreezeRunSummary]:
    """Freeze model artifacts for multiple selected variants."""
    return [
        freeze_model_variant(
            models_dir=models_dir,
            cutoff_date=cutoff_date,
            include_odds=include_odds,
            add_recent_form_features=add_recent_form_features,
            recent_form_window=recent_form_window,
            freeze_label=freeze_label,
        )
        for include_odds in include_odds_variants
    ]
