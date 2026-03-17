# Football Match Outcome Predictor

Scrapes top-flight league CSVs from football-data.co.uk across the available country league pages.

## Data Source

- [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — Premier League season CSV files
- The scraper first hits the main download page, discovers country league pages, then downloads the top-flight CSVs for each country
- Data is collected through web scraping (no API)

## V1 Scope

This version only handles scraping and saving raw top-flight league CSV data.

- It discovers the available country league pages from the main download page.
- For each country page, it identifies the first league code shown, treats it as that country's top division, and downloads those CSVs.
- Files are saved as `data/raw/<country>/<country>_<leaguecode>_YYYY-YY.csv`.
- A minimum start year can be set with `--min-start-year`.

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

## Rate Limiting

The scraper waits `6` seconds between requests and sends a polite User-Agent header.

## Known Limitations

- football-data.co.uk may change its page structure or remove country/season links over time.
- The downloaded files are stored raw, so they include all columns provided by the source site.

## Current Structure

```
football_prediction_ml/
├── scraper/
│   ├── __init__.py
│   └── football_data_scraper.py
├── data/
│   └── raw/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Disclaimer
This project is for educational purposes only. The data is sourced from football-data.co.uk, and the scraper is designed to be respectful of their site. Always check the site's terms of use before scraping.

## Free use policy
This project is open-source and free to use for educational and non-commercial purposes. You may modify and distribute the code as needed, but please give credit to the original author and do not use it for commercial applications without permission. Please also respect the data source's terms of use when accessing and using the scraped data. Do not use this code for betting or any activities that may violate the data source's policies.
