#!/usr/bin/env python
"""Quick test of data loading"""
from pathlib import Path
import pickle
import pandas as pd

model_dir = Path("data/models/frozen/official/date_2024-08-01/extended_recent_form_w5")
predictions_dir = Path("data/models/date_2024-08-01/extended_recent_form_w5")

print("=" * 50)
print("DATA LOADING TEST")
print("=" * 50)

# Load model
print(f"\n1. Loading model from {model_dir / 'best_model.pkl'}")
with open(model_dir / "best_model.pkl", "rb") as f:
    model_data = pickle.load(f)
print(f"   ✓ Model loaded. Keys: {list(model_data.keys())}")

# Load predictions
print(f"\n2. Loading predictions from {predictions_dir / 'test_predictions.csv'}")
test_pred = pd.read_csv(predictions_dir / "test_predictions.csv")
print(f"   ✓ Predictions loaded. Shape: {test_pred.shape}")
print(f"   Columns: {list(test_pred.columns)}")
print(f"   Sample:\n{test_pred.head()}")

# Load processed
print(f"\n3. Loading processed data")
processed_data_dir = Path("data/processed/extended_recent_form_w5")
countries = sorted([d.name for d in processed_data_dir.iterdir() if d.is_dir()])
print(f"   ✓ Found countries: {countries}")
print(f"   Sample divisions in england:")
england_files = list((processed_data_dir / "england").glob("*.csv"))
print(f"   Files: {[f.name for f in england_files[:2]]}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
