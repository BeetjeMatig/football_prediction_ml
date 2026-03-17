"""Scraper package for football match data collection."""

from .football_data_scraper import (
    scrape_premier_league_seasons,
    scrape_top_flight_leagues,
)

__all__ = ["scrape_premier_league_seasons", "scrape_top_flight_leagues"]
