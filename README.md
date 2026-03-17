# Football Match Outcome Predictor

Scrapes top-flight league CSVs from football-data.co.uk and preprocesses them into model-ready datasets for football match outcome prediction.

## Data Source

- [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — Premier League season CSV files
- The scraper first hits the main download page, discovers country league pages, then downloads the top-flight CSVs for each country
- Data is collected through web scraping (no API)

## Current Scope

The project currently supports two stages:

- Scraping raw top-flight league CSV data from football-data.co.uk.
- Preprocessing raw CSVs into cleaned datasets, with optional betting odds columns and optional recent-form features.

Current preprocessing capabilities:

- Legend-driven schema and column alias normalization.
- Schema validation across raw files.
- Data cleaning and dtype coercion.
- Two output variants: without odds and with odds.
- Optional recent-form features built from prior matches only.

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

Processed output directories currently follow this pattern:

- `data/processed/base`
- `data/processed/extended`
- `data/processed/base_recent_form_w5`
- `data/processed/extended_recent_form_w5`

The recent-form variants are written separately so cleaned-only outputs are not overwritten.

## Rate Limiting

The scraper waits `6` seconds between requests and sends a polite User-Agent header.

## Known Limitations

- football-data.co.uk may change its page structure or remove country/season links over time.
- Recent-form features are currently computed within each CSV file separately rather than across a full multi-season team history.
- Only a subset of market-level betting odds columns are standardized into the current schema.

## Current Structure

```
football_prediction_ml/
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py
│   ├── features.py
│   ├── pipeline.py
│   ├── schema.py
│   ├── selection.py
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
