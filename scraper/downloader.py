"""CSV download and persistence helpers for football-data scraper."""

from __future__ import annotations

from pathlib import Path

import requests

from .config import REQUEST_TIMEOUT_SECONDS


def download_csv_file(
    url: str,
    output_path: Path,
    session: requests.Session,
) -> tuple[Path, int]:
    """Download one season CSV and write it to disk exactly as received."""
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding=response.encoding or "utf-8")

    line_count = sum(1 for line in response.text.splitlines()[1:] if line.strip())
    return output_path, line_count
