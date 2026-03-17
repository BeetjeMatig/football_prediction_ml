"""HTTP session and page discovery logic for football-data.co.uk scraping."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import (
    BASE_URL,
    COUNTRY_PAGE_ALLOWLIST,
    DOWNLOAD_INDEX_URL,
    EXCLUDED_DOWNLOAD_PAGES,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)
from .utils import normalize_country_page, page_to_country_slug


def make_session() -> requests.Session:
    """Return a requests Session with a polite User-Agent header."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def discover_country_pages(session: requests.Session) -> list[str]:
    """Scrape the main download page and return available country page filenames."""
    response = session.get(DOWNLOAD_INDEX_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    country_pages: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href = str(tag["href"])
        parsed = urlparse(urljoin(BASE_URL, href))

        if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
            continue

        page_name = normalize_country_page(Path(parsed.path).name)
        if not page_name.endswith(".php"):
            continue
        if page_name in EXCLUDED_DOWNLOAD_PAGES:
            continue
        if page_name not in COUNTRY_PAGE_ALLOWLIST:
            continue

        country_pages.add(page_name)

    return sorted(country_pages)


def scrape_top_league_links(
    session: requests.Session, country_page: str
) -> tuple[str, str, dict[str, str]]:
    """Return country slug, top-league code, and season-link mapping for one country page."""
    response = session.get(
        urljoin(BASE_URL + "/", country_page), timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    csv_hrefs: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"])
        if "mmz4281/" in href and href.endswith(".csv"):
            csv_hrefs.append(href)

    if not csv_hrefs:
        raise RuntimeError(f"No CSV links found on country page '{country_page}'.")

    top_league_code = Path(csv_hrefs[0]).stem
    top_links: dict[str, str] = {}
    for href in csv_hrefs:
        season_code = href.split("/")[-2]
        league_code = Path(href).stem
        if league_code != top_league_code:
            continue
        top_links[season_code] = (
            href if href.startswith("http") else f"{BASE_URL}/{href}"
        )

    return page_to_country_slug(country_page), top_league_code, top_links
