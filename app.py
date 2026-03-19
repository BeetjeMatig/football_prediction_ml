"""
Football Match Prediction Streamlit App
Displays team stats, predicts match outcomes, and shows historical results.
"""

import json
import pickle
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from scraper.config import COUNTRY_PAGE_ALLOWLIST, REQUEST_DELAY_SECONDS
from scraper.discovery import (
    discover_country_pages,
    make_session,
    scrape_top_league_links,
)
from scraper.downloader import download_csv_file
from scraper.utils import (
    page_to_country_slug,
    season_code_to_label,
    season_label_to_code,
    season_start_year_from_label,
)

# Configure page
st.set_page_config(page_title="Football Prediction", layout="wide")


# Background scrape job store for this Streamlit server process.
SCRAPE_JOBS = {}
SCRAPE_JOBS_LOCK = threading.Lock()


# ===== HELPERS =====
@st.cache_resource
def load_model_and_data():
    """Load trained model and processed data."""
    model_dir = Path(
        "data/models/frozen/official/date_2024-08-01/extended_recent_form_w5"
    )
    predictions_dir = Path("data/models/date_2024-08-01/extended_recent_form_w5")

    # Load model
    with open(model_dir / "best_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Load test predictions
    test_pred = pd.read_csv(predictions_dir / "test_predictions.csv")

    # Convert date column to datetime
    test_pred["date"] = pd.to_datetime(test_pred["date"])

    # Load processed data for all countries
    processed_data_dir = Path("data/processed/extended_recent_form_w5")
    raw_data = {}

    for country_dir in processed_data_dir.iterdir():
        if country_dir.is_dir():
            country = country_dir.name
            raw_data[country] = {}
            for csv_file in country_dir.glob("*.csv"):
                df = pd.read_csv(csv_file)
                raw_data[country][csv_file.stem] = df

    return model_data, test_pred, raw_data


def get_unique_divisions():
    """Get all unique divisions from test predictions."""
    _, test_pred, _ = load_model_and_data()
    return sorted(test_pred["div"].unique())


def get_teams_by_division(div):
    """Get unique teams in a division."""
    _, test_pred, _ = load_model_and_data()
    div_data = test_pred[test_pred["div"] == div]
    teams = set(div_data["home_team"].unique()) | set(div_data["away_team"].unique())
    return sorted(teams)


def get_recent_stats(team, country, processed_data):
    """Get recent match stats for a team."""
    recent_stats = {
        "Matches Played": 0,
        "Avg Points": 0,
        "Avg Goals For": 0,
        "Avg Goals Against": 0,
        "Avg Goal Diff": 0,
    }

    if country not in processed_data:
        return recent_stats

    for df in processed_data[country].values():
        # Home matches
        home_matches = df[df["home_team"] == team]
        away_matches = df[df["away_team"] == team]

        if not home_matches.empty:
            valid_home = home_matches.dropna(subset=["home_points_avg_last_5"])
            if not valid_home.empty:
                recent_stats["Matches Played"] += len(valid_home)
                recent_stats["Avg Points"] += valid_home["home_points_avg_last_5"].sum()
                recent_stats["Avg Goals For"] += valid_home[
                    "home_goals_for_avg_last_5"
                ].sum()
                recent_stats["Avg Goals Against"] += valid_home[
                    "home_goals_against_avg_last_5"
                ].sum()
                recent_stats["Avg Goal Diff"] += valid_home[
                    "home_goal_diff_avg_last_5"
                ].sum()

        if not away_matches.empty:
            valid_away = away_matches.dropna(subset=["away_points_avg_last_5"])
            if not valid_away.empty:
                recent_stats["Matches Played"] += len(valid_away)
                recent_stats["Avg Points"] += valid_away["away_points_avg_last_5"].sum()
                recent_stats["Avg Goals For"] += valid_away[
                    "away_goals_for_avg_last_5"
                ].sum()
                recent_stats["Avg Goals Against"] += valid_away[
                    "away_goals_against_avg_last_5"
                ].sum()
                recent_stats["Avg Goal Diff"] += valid_away[
                    "away_goal_diff_avg_last_5"
                ].sum()

    if recent_stats["Matches Played"] > 0:
        n = recent_stats["Matches Played"]
        recent_stats["Avg Points"] /= n
        recent_stats["Avg Goals For"] /= n
        recent_stats["Avg Goals Against"] /= n
        recent_stats["Avg Goal Diff"] /= n

    return recent_stats


def predict_match(home_team, away_team, model_data, processed_data):
    """Get prediction for a match between home and away team."""
    # This is simplified - use historical averages and test data for similar matchups
    match_data = {
        "home_team": home_team,
        "away_team": away_team,
    }
    return match_data


def get_team_raw_stats(team, country, processed_data):
    """Get baseline raw match stats for a team across available seasons."""
    baseline = {
        "shots": 0.0,
        "shots_on_target": 0.0,
        "corners": 0.0,
        "yellow_cards": 0.0,
        "red_cards": 0.0,
    }

    if country not in processed_data:
        return baseline

    rows = []
    for df in processed_data[country].values():
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]

        for _, r in home.iterrows():
            rows.append(
                {
                    "shots": r.get("home_shots", np.nan),
                    "shots_on_target": r.get("home_shots_on_target", np.nan),
                    "corners": r.get("home_corners", np.nan),
                    "yellow_cards": r.get("home_yellow_cards", np.nan),
                    "red_cards": r.get("home_red_cards", np.nan),
                }
            )
        for _, r in away.iterrows():
            rows.append(
                {
                    "shots": r.get("away_shots", np.nan),
                    "shots_on_target": r.get("away_shots_on_target", np.nan),
                    "corners": r.get("away_corners", np.nan),
                    "yellow_cards": r.get("away_yellow_cards", np.nan),
                    "red_cards": r.get("away_red_cards", np.nan),
                }
            )

    if not rows:
        return baseline

    stats_df = pd.DataFrame(rows)
    means = stats_df.mean(numeric_only=True).fillna(0.0)
    return {
        "shots": float(means.get("shots", 0.0)),
        "shots_on_target": float(means.get("shots_on_target", 0.0)),
        "corners": float(means.get("corners", 0.0)),
        "yellow_cards": float(means.get("yellow_cards", 0.0)),
        "red_cards": float(means.get("red_cards", 0.0)),
    }


def compute_adjusted_xg(base_xg, selected_stats, baseline_stats):
    """Convert stat changes into an adjusted expected goals value."""
    effects = {
        "shots": 0.05,
        "shots_on_target": 0.12,
        "corners": 0.03,
        "yellow_cards": -0.04,
        "red_cards": -0.35,
    }

    adjusted = base_xg
    for name, weight in effects.items():
        adjusted += weight * (selected_stats[name] - baseline_stats[name])

    return float(np.clip(adjusted, 0.0, 5.0))


def run_main_stage(stage, options=None):
    """Run one main.py stage and return success flag + combined output."""
    cmd = [sys.executable, "main.py", "--stage", stage]
    if options:
        cmd.extend(options)

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
    )

    output = []
    output.append("$ " + " ".join(cmd))
    if result.stdout:
        output.append(result.stdout.strip())
    if result.stderr:
        output.append(result.stderr.strip())
    output_text = "\n\n".join([part for part in output if part])
    return result.returncode == 0, output_text


def run_scrape_with_progress(
    min_start_year,
    selected_countries,
    request_delay_seconds=REQUEST_DELAY_SECONDS,
    progress_callback=None,
    should_cancel_callback=None,
):
    """Scrape selected countries with callback updates for UI progress."""
    target_dir = Path("data") / "raw"
    session = make_session()

    country_pages = discover_country_pages(session)
    requested = {c.lower() for c in selected_countries}
    country_pages = [p for p in country_pages if page_to_country_slug(p) in requested]

    written = []
    logs = []

    total_countries = len(country_pages)
    if total_countries == 0:
        return (
            written,
            ["No matching country pages found for selected countries."],
            False,
        )

    for country_index, country_page in enumerate(country_pages, start=1):
        if should_cancel_callback and should_cancel_callback():
            if progress_callback:
                progress_callback(
                    event="canceled",
                    country="",
                    country_index=country_index,
                    total_countries=total_countries,
                    season_index=0,
                    total_seasons=0,
                    message="Scrape canceled by user.",
                )
            logs.append("Scrape canceled by user.")
            return written, logs, True

        country_slug = page_to_country_slug(country_page)
        if progress_callback:
            progress_callback(
                event="country_start",
                country=country_slug,
                country_index=country_index,
                total_countries=total_countries,
                season_index=0,
                total_seasons=0,
            )

        try:
            country_slug, league_code, available_links = scrape_top_league_links(
                session, country_page
            )
        except RuntimeError as error:
            msg = f"Skipping {country_page}: {error}"
            logs.append(msg)
            if progress_callback:
                progress_callback(
                    event="country_error",
                    country=country_slug,
                    country_index=country_index,
                    total_countries=total_countries,
                    season_index=0,
                    total_seasons=0,
                    message=msg,
                )
            continue

        season_labels = sorted(
            (season_code_to_label(code) for code in available_links),
            key=season_start_year_from_label,
        )
        season_labels = [
            label
            for label in season_labels
            if season_start_year_from_label(label) >= min_start_year
        ]

        if not season_labels:
            msg = f"{country_slug}: no seasons from {min_start_year} onward"
            logs.append(msg)
            if progress_callback:
                progress_callback(
                    event="country_done",
                    country=country_slug,
                    country_index=country_index,
                    total_countries=total_countries,
                    season_index=0,
                    total_seasons=0,
                    message=msg,
                )
            continue

        logs.append(
            f"{country_slug}: top league {league_code}, {len(season_labels)} seasons from {min_start_year} onward"
        )

        for season_index, season_label in enumerate(season_labels, start=1):
            if should_cancel_callback and should_cancel_callback():
                if progress_callback:
                    progress_callback(
                        event="canceled",
                        country=country_slug,
                        country_index=country_index,
                        total_countries=total_countries,
                        season_index=season_index,
                        total_seasons=len(season_labels),
                        season=season_label,
                        message="Scrape canceled by user.",
                    )
                logs.append("Scrape canceled by user.")
                return written, logs, True

            if progress_callback:
                progress_callback(
                    event="season_start",
                    country=country_slug,
                    country_index=country_index,
                    total_countries=total_countries,
                    season_index=season_index,
                    total_seasons=len(season_labels),
                    season=season_label,
                )

            season_code = season_label_to_code(season_label)
            url = available_links[season_code]
            output_path = (
                target_dir
                / country_slug
                / f"{country_slug}_{league_code}_{season_label}.csv"
            )

            path, row_count = download_csv_file(
                url=url, output_path=output_path, session=session
            )
            written.append(path)
            logs.append(f"  Saved {season_label}: {row_count} rows -> {path}")

            if progress_callback:
                progress_callback(
                    event="season_done",
                    country=country_slug,
                    country_index=country_index,
                    total_countries=total_countries,
                    season_index=season_index,
                    total_seasons=len(season_labels),
                    season=season_label,
                )

            if not (
                country_index == total_countries and season_index == len(season_labels)
            ):
                time.sleep(float(max(0.0, request_delay_seconds)))

        if progress_callback:
            progress_callback(
                event="country_done",
                country=country_slug,
                country_index=country_index,
                total_countries=total_countries,
                season_index=len(season_labels),
                total_seasons=len(season_labels),
            )

    return written, logs, False


def _update_scrape_job(job_id, **updates):
    with SCRAPE_JOBS_LOCK:
        if job_id in SCRAPE_JOBS:
            SCRAPE_JOBS[job_id].update(updates)


def _append_scrape_log(job_id, line):
    with SCRAPE_JOBS_LOCK:
        if job_id in SCRAPE_JOBS:
            SCRAPE_JOBS[job_id]["logs"].append(line)


def _compute_progress_fraction(info):
    country_idx = info.get("country_index", 0)
    total = max(1, info.get("total_countries", 1))
    season_idx = info.get("season_index", 0)
    season_total = info.get("total_seasons", 0)
    event = info.get("event", "")

    if event == "country_done":
        frac = country_idx / total
    elif event == "season_start" and season_total > 0:
        frac = ((country_idx - 1) + (season_idx - 1) / season_total) / total
    elif event == "season_done" and season_total > 0:
        frac = ((country_idx - 1) + season_idx / season_total) / total
    else:
        frac = (country_idx - 1) / total if country_idx > 0 else 0.0

    return float(max(0.0, min(1.0, frac)))


def _start_scrape_job(min_start_year, selected_countries, request_delay_seconds):
    job_id = str(uuid.uuid4())
    with SCRAPE_JOBS_LOCK:
        SCRAPE_JOBS[job_id] = {
            "status": "running",
            "progress": 0.0,
            "status_line": "Starting scrape...",
            "detail_line": "",
            "logs": [],
            "written_files": [],
            "cancel_requested": False,
        }

    def on_progress(**info):
        country_idx = info.get("country_index", 0)
        total = max(1, info.get("total_countries", 1))
        country = info.get("country", "")
        season_idx = info.get("season_index", 0)
        season_total = info.get("total_seasons", 0)
        season = info.get("season", "")
        message = info.get("message", "")

        _update_scrape_job(
            job_id,
            progress=_compute_progress_fraction(info),
            status_line=(
                f"Country {country_idx}/{total}: {country}" if country else "Running..."
            ),
            detail_line=(
                f"Season {season_idx}/{season_total}: {season}"
                if season
                else (message or "")
            ),
        )
        if message:
            _append_scrape_log(job_id, message)

    def should_cancel():
        with SCRAPE_JOBS_LOCK:
            return SCRAPE_JOBS.get(job_id, {}).get("cancel_requested", False)

    def worker():
        try:
            written, logs, canceled = run_scrape_with_progress(
                min_start_year=min_start_year,
                selected_countries=selected_countries,
                request_delay_seconds=request_delay_seconds,
                progress_callback=on_progress,
                should_cancel_callback=should_cancel,
            )
            with SCRAPE_JOBS_LOCK:
                if job_id in SCRAPE_JOBS:
                    SCRAPE_JOBS[job_id]["written_files"] = [str(p) for p in written]
                    SCRAPE_JOBS[job_id]["logs"].extend(logs)
                    SCRAPE_JOBS[job_id]["progress"] = (
                        1.0 if not canceled else SCRAPE_JOBS[job_id]["progress"]
                    )
                    SCRAPE_JOBS[job_id]["status"] = (
                        "canceled" if canceled else "completed"
                    )
                    SCRAPE_JOBS[job_id]["status_line"] = (
                        "Scrape canceled" if canceled else "Scrape completed"
                    )
        except Exception as exc:  # pragma: no cover - defensive UI path
            _append_scrape_log(job_id, f"Error: {exc}")
            _update_scrape_job(job_id, status="failed", status_line="Scrape failed")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return job_id


def get_season_from_date(date_str):
    """Extract season from date. E.g., 2024-08-15 -> '2024-25'"""
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month
    # Season starts in August
    if month >= 8:
        return f"{year}-{year+1:02d}"[-2:]
    else:
        return f"{year-1}-{year:02d}"[-2:]


def get_unique_seasons(test_pred):
    """Get all unique seasons from test predictions."""
    test_pred_copy = test_pred.copy()
    test_pred_copy["season"] = test_pred_copy["date"].apply(get_season_from_date)
    return sorted(test_pred_copy["season"].unique(), reverse=True)


def get_league_standings(div, test_pred, season=None):
    """Calculate league standings from test predictions for a given season."""
    div_data = test_pred[test_pred["div"] == div].copy()

    # Filter by season if specified
    if season:
        div_data["season"] = div_data["date"].apply(get_season_from_date)
        div_data = div_data[div_data["season"] == season]

    if div_data.empty:
        return pd.DataFrame()

    teams = set(div_data["home_team"]) | set(div_data["away_team"])
    standings = []

    for team in sorted(teams):
        home_matches = div_data[div_data["home_team"] == team]
        away_matches = div_data[div_data["away_team"] == team]

        # Count results
        home_wins = (home_matches["actual_result"] == "H").sum()
        home_draws = (home_matches["actual_result"] == "D").sum()
        home_losses = (home_matches["actual_result"] == "A").sum()

        away_wins = (away_matches["actual_result"] == "A").sum()
        away_draws = (away_matches["actual_result"] == "D").sum()
        away_losses = (away_matches["actual_result"] == "H").sum()

        # Aggregate
        total_matches = len(home_matches) + len(away_matches)
        wins = home_wins + away_wins
        draws = home_draws + away_draws
        losses = home_losses + away_losses
        points = wins * 3 + draws

        standings.append(
            {
                "Team": team,
                "Matches": total_matches,
                "Wins": wins,
                "Draws": draws,
                "Losses": losses,
                "Points": points if total_matches > 0 else 0,
            }
        )

    standings_df = pd.DataFrame(standings)
    if not standings_df.empty:
        standings_df = standings_df.sort_values(
            ["Points"], ascending=False
        ).reset_index(drop=True)
        standings_df.index = standings_df.index + 1

    return standings_df


# ===== MAIN APP =====

st.title("⚽ Football Match Prediction")

# Load data
model_data, test_pred, processed_data = load_model_and_data()

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Team Stats",
        "🎯 Match Prediction",
        "📋 League Table",
        "📈 Results Table",
        "🕸️ Scrape",
        "🏋️ Train",
    ]
)

# ===== TAB 1: TEAM STATS =====
with tab1:
    st.header("Recent Team Statistics")

    col1, col2 = st.columns(2)

    with col1:
        divisions = get_unique_divisions()
        selected_div = st.selectbox("Select Division", divisions)

    with col2:
        teams = get_teams_by_division(selected_div)
        selected_team = st.selectbox("Select Team", teams)

    if selected_team:
        # Map division to country
        div_to_country = {
            "E0": "england",
            "B1": "belgium",
            "F1": "france",
            "D1": "germany",
            "G1": "greece",
            "I1": "italy",
            "N1": "netherlands",
            "P1": "portugal",
            "SC0": "scotland",
            "SP1": "spain",
            "T1": "turkey",
        }

        country = div_to_country.get(selected_div, "")
        stats = get_recent_stats(selected_team, country, processed_data)

        st.subheader(f"📈 {selected_team} - Recent Form (Last 5 Matches)")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Matches", f"{int(stats['Matches Played'])}")
        with col2:
            st.metric("Avg Points", f"{stats['Avg Points']:.2f}")
        with col3:
            st.metric("Avg Goals For", f"{stats['Avg Goals For']:.2f}")
        with col4:
            st.metric("Avg Goals Against", f"{stats['Avg Goals Against']:.2f}")
        with col5:
            st.metric("Avg Goal Diff", f"{stats['Avg Goal Diff']:.2f}")

# ===== TAB 2: MATCH PREDICTION =====
with tab2:
    st.header("Match Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pred_div = st.selectbox(
            "Select Division", get_unique_divisions(), key="pred_div_select"
        )
        pred_teams = get_teams_by_division(pred_div)
        home_team = st.selectbox("Home Team", pred_teams, key="home_team_select")

    with col2:
        away_team = st.selectbox(
            "Away Team",
            [t for t in pred_teams if t != home_team],
            key="away_team_select",
        )

    if st.button("🔮 Predict Match"):
        st.session_state.pred_show = True

    # Show prediction if button was clicked
    if st.session_state.get("pred_show", False):
        div_to_country = {
            "E0": "england",
            "B1": "belgium",
            "F1": "france",
            "D1": "germany",
            "G1": "greece",
            "I1": "italy",
            "N1": "netherlands",
            "P1": "portugal",
            "SC0": "scotland",
            "SP1": "spain",
            "T1": "turkey",
        }

        country = div_to_country.get(pred_div, "")

        # Get stats
        home_stats = get_recent_stats(home_team, country, processed_data)
        away_stats = get_recent_stats(away_team, country, processed_data)
        home_raw_baseline = get_team_raw_stats(home_team, country, processed_data)
        away_raw_baseline = get_team_raw_stats(away_team, country, processed_data)

        # Keep prediction block above controls by creating a top placeholder.
        prediction_block = st.container()

        # Editable controls are shown underneath prediction and trigger instant recalculation.
        st.markdown("---")
        st.subheader("Predicted Stats (Editable)")
        st.caption(
            "Change any value below. The prediction above recalculates automatically."
        )

        matchup_key = f"{pred_div}_{home_team}_{away_team}".replace(" ", "_")

        home_ctrl_col, away_ctrl_col = st.columns(2)
        with home_ctrl_col:
            st.markdown(f"#### {home_team}")
            home_selected = {
                "shots": st.number_input(
                    "Shots",
                    min_value=0.0,
                    max_value=40.0,
                    value=round(home_raw_baseline["shots"], 1),
                    step=0.5,
                    key=f"home_shots_{matchup_key}",
                ),
                "shots_on_target": st.number_input(
                    "Shots On Target",
                    min_value=0.0,
                    max_value=20.0,
                    value=round(home_raw_baseline["shots_on_target"], 1),
                    step=0.5,
                    key=f"home_sot_{matchup_key}",
                ),
                "corners": st.number_input(
                    "Corners",
                    min_value=0.0,
                    max_value=20.0,
                    value=round(home_raw_baseline["corners"], 1),
                    step=0.5,
                    key=f"home_corners_{matchup_key}",
                ),
                "yellow_cards": st.number_input(
                    "Yellow Cards",
                    min_value=0.0,
                    max_value=10.0,
                    value=round(home_raw_baseline["yellow_cards"], 1),
                    step=0.5,
                    key=f"home_yellow_{matchup_key}",
                ),
                "red_cards": st.number_input(
                    "Red Cards",
                    min_value=0.0,
                    max_value=5.0,
                    value=round(home_raw_baseline["red_cards"], 1),
                    step=0.5,
                    key=f"home_red_{matchup_key}",
                ),
            }

        with away_ctrl_col:
            st.markdown(f"#### {away_team}")
            away_selected = {
                "shots": st.number_input(
                    "Shots ",
                    min_value=0.0,
                    max_value=40.0,
                    value=round(away_raw_baseline["shots"], 1),
                    step=0.5,
                    key=f"away_shots_{matchup_key}",
                ),
                "shots_on_target": st.number_input(
                    "Shots On Target ",
                    min_value=0.0,
                    max_value=20.0,
                    value=round(away_raw_baseline["shots_on_target"], 1),
                    step=0.5,
                    key=f"away_sot_{matchup_key}",
                ),
                "corners": st.number_input(
                    "Corners ",
                    min_value=0.0,
                    max_value=20.0,
                    value=round(away_raw_baseline["corners"], 1),
                    step=0.5,
                    key=f"away_corners_{matchup_key}",
                ),
                "yellow_cards": st.number_input(
                    "Yellow Cards ",
                    min_value=0.0,
                    max_value=10.0,
                    value=round(away_raw_baseline["yellow_cards"], 1),
                    step=0.5,
                    key=f"away_yellow_{matchup_key}",
                ),
                "red_cards": st.number_input(
                    "Red Cards ",
                    min_value=0.0,
                    max_value=5.0,
                    value=round(away_raw_baseline["red_cards"], 1),
                    step=0.5,
                    key=f"away_red_{matchup_key}",
                ),
            }

        # xG derived from current controls.
        home_xg = compute_adjusted_xg(
            base_xg=float(home_stats["Avg Goals For"]),
            selected_stats=home_selected,
            baseline_stats=home_raw_baseline,
        )
        away_xg = compute_adjusted_xg(
            base_xg=float(away_stats["Avg Goals For"]),
            selected_stats=away_selected,
            baseline_stats=away_raw_baseline,
        )

        # Model reference probabilities.
        similar = test_pred[
            (test_pred["home_team"] == home_team)
            | (test_pred["away_team"] == away_team)
        ]

        from scipy.stats import poisson

        max_goals = 10
        home_probs = poisson.pmf(np.arange(max_goals), home_xg)
        away_probs = poisson.pmf(np.arange(max_goals), away_xg)

        prob_home = sum(
            home_probs[h] * away_probs[a]
            for h in range(max_goals)
            for a in range(max_goals)
            if h > a
        )
        prob_draw = sum(home_probs[g] * away_probs[g] for g in range(max_goals))
        prob_away = sum(
            home_probs[h] * away_probs[a]
            for h in range(max_goals)
            for a in range(max_goals)
            if h < a
        )

        predicted_home_score = round(home_xg)
        predicted_away_score = round(away_xg)
        if predicted_home_score > predicted_away_score:
            predicted_outcome = "Home Win"
        elif predicted_home_score < predicted_away_score:
            predicted_outcome = "Away Win"
        else:
            predicted_outcome = "Draw"

        poisson_probs = {
            "Home Win": prob_home,
            "Draw": prob_draw,
            "Away Win": prob_away,
        }
        max_prob = poisson_probs[predicted_outcome]

        with prediction_block:
            st.subheader(f"Match: {home_team} vs {away_team}")
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"### {home_team} 🏠")
                st.metric("Recent Points Avg", f"{home_stats['Avg Points']:.2f}")
                st.metric("Avg Goals For", f"{home_stats['Avg Goals For']:.2f}")
                st.metric("Avg Goals Against", f"{home_stats['Avg Goals Against']:.2f}")

            with col2:
                st.markdown("### Prediction")
                st.markdown("---")
                outcome_emoji = {
                    "Home Win": "🏟️ HOME WIN",
                    "Draw": "⚖️ DRAW",
                    "Away Win": "🔗 AWAY WIN",
                }
                st.markdown(f"### {outcome_emoji[predicted_outcome]}")
                st.markdown(f"**Confidence: {max_prob:.1%}**")
                st.markdown("**Predicted Score:**")
                st.markdown(f"# **{predicted_home_score} - {predicted_away_score}**")
                st.markdown("**Match Probabilities:**")
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("Home Win", f"{prob_home:.1%}")
                with p2:
                    st.metric("Draw", f"{prob_draw:.1%}")
                with p3:
                    st.metric("Away Win", f"{prob_away:.1%}")
                st.markdown("**xG Used For Prediction:**")
                st.markdown(f"**{home_xg:.2f} - {away_xg:.2f}**")

                if not similar.empty:
                    avg_pred = similar[
                        ["prob_home_win", "prob_draw", "prob_away_win"]
                    ].mean()
                    with st.expander("Show model classifier probabilities (reference)"):
                        ref = pd.DataFrame(
                            {
                                "Outcome": ["Home Win", "Draw", "Away Win"],
                                "Classifier Probability": [
                                    avg_pred["prob_home_win"],
                                    avg_pred["prob_draw"],
                                    avg_pred["prob_away_win"],
                                ],
                            }
                        )
                        ref["Classifier Probability"] = ref[
                            "Classifier Probability"
                        ].map(lambda x: f"{x:.1%}")
                        st.dataframe(ref, width="stretch")

            with col3:
                st.markdown(f"### {away_team} 🔗")
                st.metric("Recent Points Avg", f"{away_stats['Avg Points']:.2f}")
                st.metric("Avg Goals For", f"{away_stats['Avg Goals For']:.2f}")
                st.metric("Avg Goals Against", f"{away_stats['Avg Goals Against']:.2f}")

        st.markdown("---")
        with st.expander("Show raw baseline stats used for controls"):
            raw_view = pd.DataFrame(
                {
                    "Stat": [
                        "Shots",
                        "Shots On Target",
                        "Corners",
                        "Yellow Cards",
                        "Red Cards",
                        "Avg Points Last 5",
                        "Avg Goals For Last 5",
                        "Avg Goals Against Last 5",
                    ],
                    home_team: [
                        round(home_raw_baseline["shots"], 2),
                        round(home_raw_baseline["shots_on_target"], 2),
                        round(home_raw_baseline["corners"], 2),
                        round(home_raw_baseline["yellow_cards"], 2),
                        round(home_raw_baseline["red_cards"], 2),
                        round(home_stats["Avg Points"], 2),
                        round(home_stats["Avg Goals For"], 2),
                        round(home_stats["Avg Goals Against"], 2),
                    ],
                    away_team: [
                        round(away_raw_baseline["shots"], 2),
                        round(away_raw_baseline["shots_on_target"], 2),
                        round(away_raw_baseline["corners"], 2),
                        round(away_raw_baseline["yellow_cards"], 2),
                        round(away_raw_baseline["red_cards"], 2),
                        round(away_stats["Avg Points"], 2),
                        round(away_stats["Avg Goals For"], 2),
                        round(away_stats["Avg Goals Against"], 2),
                    ],
                }
            )
            st.dataframe(raw_view, width="stretch")


# ===== TAB 3: LEAGUE TABLE =====
with tab3:
    st.header("League Standings")

    col1, col2 = st.columns(2)

    with col1:
        league_div = st.selectbox(
            "Select Division", get_unique_divisions(), key="league_div"
        )

    with col2:
        seasons = get_unique_seasons(test_pred)
        selected_season = st.selectbox("Select Season", seasons, key="league_season")

    standings = get_league_standings(league_div, test_pred, season=selected_season)

    if not standings.empty:
        st.subheader(f"Division {league_div} — Season {selected_season}")

        # Display as table
        st.dataframe(
            standings,
            column_config={
                "Team": st.column_config.TextColumn("Team"),
                "Matches": st.column_config.NumberColumn("P"),
                "Wins": st.column_config.NumberColumn("W"),
                "Draws": st.column_config.NumberColumn("D"),
                "Losses": st.column_config.NumberColumn("L"),
                "Points": st.column_config.NumberColumn("Pts", format="%d"),
            },
            width="stretch",
        )

        # Summary stats
        st.markdown("---")
        st.subheader("📊 Division Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Teams", len(standings))
        with col2:
            st.metric("Total Matches", int(standings["Matches"].sum() / 2))
        with col3:
            st.metric("Total Matches Played", int(standings["Matches"].sum()))
    else:
        st.warning("No data available for this division in selected season")

# ===== TAB 4: RESULTS TABLE =====
with tab4:
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        results_div = st.selectbox(
            "Filter by Division", ["All"] + get_unique_divisions(), key="results_div"
        )

    with col2:
        accuracy_option = st.selectbox("Show", ["Recent 20", "Last 50", "All Results"])

    # Filter data
    display_data = test_pred.copy()

    if results_div != "All":
        display_data = display_data[display_data["div"] == results_div]

    if accuracy_option == "Recent 20":
        display_data = display_data.iloc[-20:]
    elif accuracy_option == "Last 50":
        display_data = display_data.iloc[-50:]

    # Sort by date descending
    display_data = display_data.sort_values("date", ascending=False)

    # Display metrics
    correct = (display_data["actual_result"] == display_data["predicted_result"]).sum()
    total = len(display_data)
    accuracy = correct / total if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Correct", f"{correct}/{total}")
    with col3:
        st.metric("Predictions", total)

    st.markdown("---")

    # Format display
    display_data_formatted = display_data[
        [
            "date",
            "div",
            "home_team",
            "away_team",
            "actual_result",
            "predicted_result",
            "prob_home_win",
            "prob_draw",
            "prob_away_win",
        ]
    ].copy()

    display_data_formatted["Correct"] = (
        display_data_formatted["actual_result"]
        == display_data_formatted["predicted_result"]
    )
    display_data_formatted["prob_home_win"] = display_data_formatted[
        "prob_home_win"
    ].apply(lambda x: f"{x:.1%}")
    display_data_formatted["prob_draw"] = display_data_formatted["prob_draw"].apply(
        lambda x: f"{x:.1%}"
    )
    display_data_formatted["prob_away_win"] = display_data_formatted[
        "prob_away_win"
    ].apply(lambda x: f"{x:.1%}")

    display_data_formatted.columns = [
        "Date",
        "Division",
        "Home",
        "Away",
        "Actual",
        "Predicted",
        "P(H)",
        "P(D)",
        "P(A)",
        "✓",
    ]

    st.dataframe(display_data_formatted, width="stretch")

# ===== TAB 5: SCRAPING =====
with tab5:
    st.header("Scraping")
    st.write(
        "Run selected-country scraping from inside the dashboard with live progress."
    )

    scrape_col1, scrape_col2 = st.columns(2)
    with scrape_col1:
        min_start_year = st.number_input(
            "Minimum start year",
            min_value=1990,
            max_value=2030,
            value=2020,
            step=1,
            key="scrape_min_start_year",
        )
        scrape_delay_seconds = st.number_input(
            "Request delay (seconds)",
            min_value=0.0,
            max_value=15.0,
            value=float(REQUEST_DELAY_SECONDS),
            step=0.5,
            key="scrape_delay_seconds",
            help="Delay between downloads. Lower values are faster but increase load/risk of temporary blocking.",
        )
    with scrape_col2:
        all_countries = sorted(
            page_to_country_slug(page) for page in COUNTRY_PAGE_ALLOWLIST
        )
        selected_countries = st.multiselect(
            "Countries",
            options=all_countries,
            default=all_countries,
            key="scrape_countries",
            help="Select one or more country leagues to scrape.",
        )
        st.caption("Example: 2020 fetches seasons from 2020-21 onward.")

    active_job_id = st.session_state.get("active_scrape_job_id")
    active_job = SCRAPE_JOBS.get(active_job_id) if active_job_id else None
    is_running = bool(active_job and active_job.get("status") == "running")

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        run_clicked = st.button(
            "Run Scrape",
            key="run_scrape_button",
            disabled=is_running,
            help="Start scraping in the background. You can keep using the dashboard.",
        )
    with action_col2:
        cancel_clicked = st.button(
            "Cancel Scrape",
            key="cancel_scrape_button",
            disabled=not is_running,
            help="Request cancel. The current download finishes first, then stops.",
        )
    with action_col3:
        refresh_clicked = st.button(
            "Refresh Progress",
            key="refresh_scrape_progress",
            help="Refresh current scrape progress and logs.",
        )

    auto_col1, auto_col2 = st.columns(2)
    with auto_col1:
        auto_refresh_enabled = st.toggle(
            "Auto-refresh progress",
            value=True,
            key="scrape_auto_refresh",
            help="When enabled, progress updates automatically while scraping is running.",
        )
    with auto_col2:
        auto_refresh_seconds = st.selectbox(
            "Refresh interval",
            options=[1, 2, 3, 5],
            index=1,
            key="scrape_auto_refresh_seconds",
            help="How often the progress panel refreshes while a scrape job is active.",
        )

    if run_clicked:
        if not selected_countries:
            st.warning("Select at least one country.")
        else:
            job_id = _start_scrape_job(
                int(min_start_year),
                selected_countries,
                float(scrape_delay_seconds),
            )
            st.session_state["active_scrape_job_id"] = job_id
            st.info(
                "Scrape started in background. Use Refresh Progress to update view."
            )

    if cancel_clicked and active_job_id:
        _update_scrape_job(
            active_job_id, cancel_requested=True, detail_line="Cancel requested..."
        )
        st.warning("Cancel requested. Scrape will stop after the current step.")

    if refresh_clicked:
        st.rerun()

    # Re-read active job after any button state changes.
    active_job_id = st.session_state.get("active_scrape_job_id")
    active_job = SCRAPE_JOBS.get(active_job_id) if active_job_id else None

    if active_job:
        st.progress(float(active_job.get("progress", 0.0)))
        st.markdown(f"**{active_job.get('status_line', 'Running...')}**")
        if active_job.get("detail_line"):
            st.write(active_job["detail_line"])

        status = active_job.get("status")
        written_files = active_job.get("written_files", [])
        logs = active_job.get("logs", [])

        if status == "completed":
            st.success(f"Scrape completed. Files written: {len(written_files)}")
        elif status == "canceled":
            st.warning(f"Scrape canceled. Partial files written: {len(written_files)}")
        elif status == "failed":
            st.error("Scrape failed. Check logs below.")
        else:
            st.info("Scrape is running...")

        if logs:
            st.code("\n".join(logs[-200:]), language="bash")

        if written_files:
            output_df = pd.DataFrame({"file": written_files})
            st.dataframe(output_df, width="stretch")

        if status == "running" and auto_refresh_enabled:
            st.caption(f"Auto-refresh is on. Updating every {auto_refresh_seconds}s...")
            time.sleep(int(auto_refresh_seconds))
            st.rerun()
    else:
        st.caption("No active scrape job.")


# ===== TAB 6: TRAINING =====
with tab6:
    st.header("Training Pipeline")
    st.write(
        "Run preprocessing, split, modeling dataset build, training, freeze/report/smoke directly from the dashboard."
    )
    st.info(
        "Tip: hover the info icons (i) beside inputs and buttons for what each setting does. "
        "Typical flow: Preprocess+Split+Modeldata -> Train -> Freeze+Report+Smoke."
    )

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        train_cutoff = st.text_input(
            "Split cutoff date (YYYY-MM-DD)",
            value="2024-08-01",
            key="train_cutoff_date",
            help="Date-based train/test split boundary. Matches on/after this date are test data.",
        )
        train_recent_window = st.number_input(
            "Recent form window",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="train_recent_window",
            help="Number of prior matches used to compute recent-form rolling features.",
        )
    with t_col2:
        train_add_recent = st.checkbox(
            "Add recent form features",
            value=True,
            key="train_add_recent",
            help="Enable leakage-safe rolling features like points/goals in recent matches.",
        )
        train_write_both = st.checkbox(
            "Write both variants (base + extended)",
            value=True,
            key="train_write_both",
            help="Train both variants: base (no odds) and extended (with odds).",
        )
        train_include_odds = st.checkbox(
            "Include odds (single-variant mode)",
            value=True,
            key="train_include_odds",
            help="Only used if 'Write both variants' is off; includes betting odds columns.",
        )
    with t_col3:
        train_freeze_label = st.text_input(
            "Freeze label",
            value="official",
            key="train_freeze_label",
            help="Name for frozen model snapshot folder under data/models/frozen/<label>/...",
        )
        smoke_division = st.text_input(
            "Smoke division",
            value="E0",
            key="smoke_division",
            help="Division used for prediction smoke test after training.",
        )
        smoke_home = st.text_input(
            "Smoke home team",
            value="Arsenal",
            key="smoke_home",
            help="Home team used in smoke prediction check.",
        )
        smoke_away = st.text_input(
            "Smoke away team",
            value="Chelsea",
            key="smoke_away",
            help="Away team used in smoke prediction check.",
        )

    common_opts = [
        "--split-cutoff-date",
        train_cutoff,
        "--recent-form-window",
        str(train_recent_window),
    ]
    if train_add_recent:
        common_opts.append("--add-recent-form-features")
    if train_write_both:
        common_opts.append("--write-both-variants")
    elif train_include_odds:
        common_opts.append("--include-odds")

    smoke_opts = list(common_opts)
    smoke_opts.extend(
        [
            "--division",
            smoke_division,
            "--home-team",
            smoke_home,
            "--away-team",
            smoke_away,
        ]
    )

    st.markdown("### Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)

    if qa1.button(
        "Run Preprocess+Split+Modeldata",
        key="run_prep_stack",
        help="Creates cleaned/featured data, date splits, and leakage-safe modeling dataset exports.",
    ):
        logs = []
        ok_all = True
        with st.spinner("Running preprocess, split, and modeldata..."):
            for stage in ["preprocess", "split", "modeldata"]:
                ok, out = run_main_stage(stage, common_opts)
                logs.append(out)
                ok_all = ok_all and ok
                if not ok:
                    break
        if ok_all:
            st.success("Preprocess + Split + Modeldata completed.")
        else:
            st.error("One of preprocess/split/modeldata failed.")
        st.code("\n\n".join(logs), language="bash")

    if qa2.button(
        "Run Train",
        key="run_train_only",
        help="Trains classifier and goal regressors using prepared modeling datasets.",
    ):
        with st.spinner("Training model variants..."):
            ok, out = run_main_stage("train", common_opts)
        if ok:
            st.success("Train stage completed.")
        else:
            st.error("Train stage failed.")
        st.code(out, language="bash")

    if qa3.button(
        "Run Freeze+Report+Smoke",
        key="run_post_train",
        help="Freezes artifacts, writes baseline metrics report, and runs prediction sanity smoke test.",
    ):
        logs = []
        ok_all = True
        with st.spinner("Running freeze, report, and smoke..."):
            freeze_opts = list(common_opts) + ["--freeze-label", train_freeze_label]
            for stage, opts in [
                ("freeze", freeze_opts),
                ("report", common_opts),
                ("smoke", smoke_opts),
            ]:
                ok, out = run_main_stage(stage, opts)
                logs.append(out)
                ok_all = ok_all and ok
                if not ok:
                    break
        if ok_all:
            st.success("Freeze + Report + Smoke completed.")
        else:
            st.error("One of freeze/report/smoke failed.")
        st.code("\n\n".join(logs), language="bash")

    if qa4.button(
        "Run Full Train Pipeline",
        key="run_full_pipeline",
        help="Runs full sequence from preprocess through smoke test in one click.",
    ):
        logs = []
        ok_all = True
        with st.spinner(
            "Running full pipeline (preprocess -> split -> modeldata -> train -> freeze -> report -> smoke)..."
        ):
            full_steps = [
                ("preprocess", common_opts),
                ("split", common_opts),
                ("modeldata", common_opts),
                ("train", common_opts),
                ("freeze", list(common_opts) + ["--freeze-label", train_freeze_label]),
                ("report", common_opts),
                ("smoke", smoke_opts),
            ]
            for stage, opts in full_steps:
                ok, out = run_main_stage(stage, opts)
                logs.append(out)
                ok_all = ok_all and ok
                if not ok:
                    break
        if ok_all:
            st.success("Full pipeline completed.")
        else:
            st.error("Full pipeline failed. Check logs below.")
        st.code("\n\n".join(logs), language="bash")

st.markdown("---")
st.caption("Football Prediction Model — Data as of 2024-08-01")
