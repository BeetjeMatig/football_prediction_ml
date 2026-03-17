"""Configuration values for the football-data scraper."""

BASE_URL = "https://www.football-data.co.uk"
DOWNLOAD_INDEX_URL = f"{BASE_URL}/data.php"
REQUEST_DELAY_SECONDS = 6  #
REQUEST_TIMEOUT_SECONDS = 60
MIN_START_YEAR = 2020
USER_AGENT = (
    "football_prediction_ml/1.0 "
    "(educational project; github.com/justinmulder/football_prediction_ml)"
)

# Exclude pages that are not relevant for scraping league links or match data.
EXCLUDED_DOWNLOAD_PAGES = {
    "contact.php",
    "data.php",
    "downloadm.php",
    "help_footballdata.php",
    "link.php",
    "matches.php",
    "matches_new_leagues.php",
    "all_new_data.php",
    "books.php",
    "disclaimer.php",
}

# Allowlist of country pages to scrape for league links.
# This is necessary because some country pages do not contain
# scrapable league links, and we want to avoid unnecessary requests.
COUNTRY_PAGE_ALLOWLIST = {
    "englandm.php",
    "scotlandm.php",
    "germanym.php",
    "italym.php",
    "spainm.php",
    "francem.php",
    "netherlandsm.php",
    "belgiumm.php",
    "portugalm.php",
    "turkeym.php",
    "greecem.php",
}
