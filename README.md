# Football Match Outcome Predictor

Scrapes top-flight league CSVs from [football-data.co.uk](https://www.football-data.co.uk/englandm.php), preprocesses them into model-ready datasets, trains classifiers, and predicts match outcomes with probability estimates.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Pipeline

The pipeline is driven by `main.py` with a `--stage` flag. Stages run sequentially — each one depends on the output of the previous.

### Full pipeline (recommended first run)

```bash
python main.py --stage all --write-both-variants --add-recent-form-features --split-cutoff-date 2024-08-01
```

### Stage by stage

**1. Scrape** — download raw CSVs from football-data.co.uk

```bash
python main.py --stage scrape --min-start-year 2010
```

**2. Preprocess** — clean and normalize raw data

```bash
# Base output only
python main.py --stage preprocess

# Both base and odds-extended outputs
python main.py --stage preprocess --write-both-variants

# With rolling recent-form features
python main.py --stage preprocess --write-both-variants --add-recent-form-features
```

**3. Split** — create a date-based train/test split

```bash
python main.py --stage split --add-recent-form-features --split-cutoff-date 2024-08-01
```

**4. Model data** — build leakage-safe feature/target datasets

```bash
python main.py --stage modeldata --add-recent-form-features --split-cutoff-date 2024-08-01
```

**5. Train** — fit candidate models and save the best

```bash
python main.py --stage train --add-recent-form-features --split-cutoff-date 2024-08-01
```

**6. Freeze** — bundle trained artifacts under a named label

```bash
python main.py --stage freeze --write-both-variants --add-recent-form-features --split-cutoff-date 2024-08-01 --freeze-label official
```

**7. Report** — generate a consolidated baseline metrics report

```bash
python main.py --stage report --write-both-variants --add-recent-form-features --split-cutoff-date 2024-08-01
```

**8. Smoke test** — confirm stable prediction behavior

```bash
python main.py --stage smoke --include-odds --add-recent-form-features --split-cutoff-date 2024-08-01
```

**9. Predict** — predict a specific matchup from team history

```bash
python main.py --stage predict --add-recent-form-features --split-cutoff-date 2024-08-01 \
  --division E0 --home-team "Arsenal" --away-team "Chelsea"
```

With scenario overrides:

```bash
python main.py --stage predict --include-odds --add-recent-form-features --split-cutoff-date 2024-08-01 \
  --division E0 --home-team "Arsenal" --away-team "Chelsea" \
  --kickoff-time 20:00 \
  --feature-override odds_home_win=1.85 \
  --feature-override home_points_avg_last_5=2.4
```

---

## Output Structure

```
data/
├── raw/                                          # Scraped CSVs
├── processed/
│   ├── base/                                     # Cleaned, no odds
│   ├── extended/                                 # Cleaned, with odds
│   ├── base_recent_form_w5/                      # + rolling form features
│   └── extended_recent_form_w5/
├── splits/
│   └── date_YYYY-MM-DD/<variant>/
│       ├── train.csv
│       └── test.csv
├── modeling/
│   └── date_YYYY-MM-DD/<variant>/
│       ├── X_train.csv / y_train.csv
│       ├── X_test.csv  / y_test.csv
│       └── train_metadata.csv / test_metadata.csv
└── models/
    ├── date_YYYY-MM-DD/<variant>/
    │   ├── best_model.pkl
    │   ├── metrics.csv
    │   ├── goal_metrics.csv
    │   ├── test_predictions.csv
    │   └── artifact_meta.json
    ├── date_YYYY-MM-DD/
    │   └── baseline_metrics.json
    └── frozen/<freeze_label>/date_YYYY-MM-DD/<variant>/
```

---

## Considerations

### Train/test split

Splits are always date-based — never random. The model trains on older matches and is evaluated on newer ones. A random split would leak future data into training and produce overly optimistic scores.

Example cutoff: train on seasons up to 2023-24, test on 2024-25 (`--split-cutoff-date 2024-08-01`).

### Leakage prevention

The modeling stage removes any columns that reveal information about the match being predicted (full-time goals, half-time result, in-match shots/corners/cards). Retained features are strictly pre-match: kickoff time, division, team identifiers, rolling form, sparse-history indicators, and optionally market odds.

Missing numeric values are imputed using training-set medians before being applied to both train and test sets.

### Recent-form features

When `--add-recent-form-features` is used at the split stage, form features are recomputed on the full combined dataset so that team history carries across season boundaries correctly.

### Prediction overrides

Any model feature can be overridden at prediction time with `--feature-override feature=value` to run what-if scenarios. In-game event stats (red cards, corners, shots, expected goals) adjust the goals regression and recompute outcome probabilities. Examples:

- Stronger home form: `--feature-override home_points_avg_last_5=2.3`
- Away red card: `--feature-override away_red_cards=1`
- High-corner game: `--feature-override home_corners=9 --feature-override away_corners=7`

### Scraper behaviour

The scraper uses plain HTTP requests (no browser automation) and waits 6 seconds between requests with a polite User-Agent header. football-data.co.uk may change its page structure over time, which could break discovery.

### Known limitations

- Only a subset of betting odds columns are standardized into the current schema.
- Team name matching is string-based; historical name changes or alternate spellings may cause gaps in form history.

---

## Project Structure

```
football_prediction_ml/
├── model/
│   ├── __init__.py
│   └── pipeline.py
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py
│   ├── features.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── schema.py
│   ├── selection.py
│   ├── splitter.py
│   └── validator.py
├── scraper/
│   ├── __init__.py
│   ├── config.py
│   ├── discovery.py
│   ├── downloader.py
│   ├── football_data_scraper.py
│   └── utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── legend.txt
├── main.py
├── requirements.txt
└── README.md
```

---

## Disclaimer

This project is for educational purposes only. Data is sourced from football-data.co.uk — always check their terms of use before scraping. Do not use this project for betting or any commercial application without permission.
