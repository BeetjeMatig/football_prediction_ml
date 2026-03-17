# Football Match Outcome Predictor

Scrapes top-flight league CSVs from football-data.co.uk and preprocesses them into model-ready datasets for football match outcome prediction.

## Data Source

- [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — Premier League season CSV files
- The scraper first hits the main download page, discovers country league pages, then downloads the top-flight CSVs for each country
- Data is collected through web scraping (no API)

## Current Scope

The project currently supports six stages:

- Scraping raw top-flight league CSV data from football-data.co.uk.
- Preprocessing raw CSVs into cleaned datasets, with optional betting odds columns and optional recent-form features.
- Combining processed datasets and splitting them into train/test sets by date.
- Building leakage-safe modeling datasets from those train/test splits.
- Training candidate classification models and selecting the best by test log loss.
- Predicting a future matchup from team histories, with optional manual feature overrides for scenario testing.

Current preprocessing capabilities:

- Legend-driven schema and column alias normalization.
- Schema validation across raw files.
- Data cleaning and dtype coercion.
- Two output variants: without odds and with odds.
- Optional cross-season recent-form features built from prior matches only.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

No browser automation required — the scraper uses plain HTTP requests.

Example with a minimum season start year of 2010:

```bash
python main.py --min-start-year 2010
```

Preprocess existing raw data without betting odds:

```bash
python main.py --stage preprocess
```

Preprocess existing raw data and write both base and extended outputs:

```bash
python main.py --stage preprocess --write-both-variants
```

Preprocess with recent-form features added:

```bash
python main.py --stage preprocess --write-both-variants --add-recent-form-features
```

Run scraping and preprocessing in one command:

```bash
python main.py --stage all --write-both-variants
```

Create a date-based split from processed data:

```bash
python main.py --stage split --add-recent-form-features --split-cutoff-date 2024-08-01
```

When recent-form is enabled in the split stage, features are recomputed on the
combined multi-file dataset so team history continues across seasons.

Create leakage-safe modeling datasets from an existing split:

```bash
python main.py --stage modeldata --add-recent-form-features --split-cutoff-date 2024-08-01
```

Train candidate models and save the best artifact:

```bash
python main.py --stage train --add-recent-form-features --split-cutoff-date 2024-08-01
```

Predict one matchup from historical team form:

```bash
python main.py --stage predict --add-recent-form-features --split-cutoff-date 2024-08-01 --division E0 --home-team "Arsenal" --away-team "Chelsea"
```

Scenario analysis with manual overrides:

```bash
python main.py --stage predict --include-odds --add-recent-form-features --split-cutoff-date 2024-08-01 --division E0 --home-team "Arsenal" --away-team "Chelsea" --kickoff-time 20:00 --feature-override odds_home_win=1.85 --feature-override home_points_avg_last_5=2.4
```

Processed output directories currently follow this pattern:

- `data/processed/base`
- `data/processed/extended`
- `data/processed/base_recent_form_w5`
- `data/processed/extended_recent_form_w5`

The recent-form variants are written separately so cleaned-only outputs are not overwritten.

Date-based split outputs are written under:

- `data/splits/date_YYYY-MM-DD/<variant>/train.csv`
- `data/splits/date_YYYY-MM-DD/<variant>/test.csv`

Modeling datasets are written under:

- `data/modeling/date_YYYY-MM-DD/<variant>/X_train.csv`
- `data/modeling/date_YYYY-MM-DD/<variant>/y_train.csv`
- `data/modeling/date_YYYY-MM-DD/<variant>/X_test.csv`
- `data/modeling/date_YYYY-MM-DD/<variant>/y_test.csv`
- `data/modeling/date_YYYY-MM-DD/<variant>/train_metadata.csv`
- `data/modeling/date_YYYY-MM-DD/<variant>/test_metadata.csv`

Trained model outputs are written under:

- `data/models/date_YYYY-MM-DD/<variant>/best_model.pkl`
- `data/models/date_YYYY-MM-DD/<variant>/metrics.csv`
- `data/models/date_YYYY-MM-DD/<variant>/test_predictions.csv`
- `data/models/date_YYYY-MM-DD/<variant>/artifact_meta.json`

The modeling stage is independent of how many years or seasons were included upstream. It works from whatever rows exist in the selected split and filters columns by leakage rules rather than by hard-coded seasons.

## Train/Test Split Strategy

When the project moves to model training, the train/test split should be date-based rather than random.

Why this matters:

- This is a forecasting problem, so the model should train on older matches and be evaluated on newer matches.
- A random split would mix future matches into the training set and give an unrealistically optimistic score.
- The existing recent-form features are already time-aware, so the evaluation strategy should follow the same logic.

Recommended approach:

- Train on earlier seasons or matches before a chosen cutoff date.
- Test on later seasons or matches on or after that cutoff date.

Example:

- Train: seasons `2020-21` through `2023-24`
- Test: season `2024-25`

This is now the default split strategy for the repository's split stage and should remain the default for any future model training pipeline.

## Modeling Dataset Strategy

The modeling dataset builder removes columns that would leak information from the match being predicted.

Examples of excluded columns:

- full-time goals
- half-time goals and half-time result
- current-match shots, corners, and cards

Examples of retained columns:

- kickoff time (`time`) as a valid pre-match context feature
- kickoff time encoded to minutes since midnight during modeling dataset prep
- division, teams, and date in separate metadata files
- rolling recent-form features
- sparse-history indicators (`home_sparse_history`, `away_sparse_history`)
- optional market odds columns when that variant is selected

Additional modeling-prep transformations:

- Missing numeric feature values are imputed using training-set medians and then
	applied to both train and test.
- Sparse-history indicators mark teams with fewer than the selected recent-form
	window of prior matches.

This stage does not assume a fixed set of seasons. It uses whatever train/test rows are present in the selected split directory.

## Model Training Strategy

The training stage currently fits two multiclass classifiers on each selected variant:

- Multinomial logistic regression (interpretable baseline)
- Histogram-based gradient boosting (strong tabular model)

Model selection currently uses test-set log loss as the primary metric. Accuracy and macro F1 are also exported for comparison.

## Prediction and Simulation

The prediction stage builds a pre-match feature row from each team's most recent known history in the chosen division and cutoff dataset.

You can manually override any model feature using one or more `--feature-override feature=value` arguments to test what-if scenarios.

Examples:

- Stronger home form: `--feature-override home_points_avg_last_5=2.3`
- Weaker away defense: `--feature-override away_goals_against_avg_last_5=1.8`
- Better home market price: `--feature-override odds_home_win=1.75`

## Rate Limiting

The scraper waits `6` seconds between requests and sends a polite User-Agent header.

## Known Limitations

- football-data.co.uk may change its page structure or remove country/season links over time.
- Only a subset of market-level betting odds columns are standardized into the current schema.
- The current train/test split combines processed files and splits by cutoff date, but model training itself is not implemented yet.

## Current Structure

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
├── .gitignore
└── README.md
```

## Disclaimer
This project is for educational purposes only. The data is sourced from football-data.co.uk, and the scraper is designed to be respectful of their site. Always check the site's terms of use before scraping.

## Free use policy
This project is open-source and free to use for educational and non-commercial purposes. You may modify and distribute the code as needed, but please give credit to the original author and do not use it for commercial applications without permission. Please also respect the data source's terms of use when accessing and using the scraped data. Do not use this code for betting or any activities that may violate the data source's policies.
