"""
Football Match Prediction Streamlit App
Displays team stats, predicts match outcomes, and shows historical results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

# Configure page
st.set_page_config(page_title="Football Prediction", layout="wide")

# ===== HELPERS =====
@st.cache_resource
def load_model_and_data():
    """Load trained model and processed data."""
    model_dir = Path("data/models/frozen/official/date_2024-08-01/extended_recent_form_w5")
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
                recent_stats["Avg Goals For"] += valid_home["home_goals_for_avg_last_5"].sum()
                recent_stats["Avg Goals Against"] += valid_home["home_goals_against_avg_last_5"].sum()
                recent_stats["Avg Goal Diff"] += valid_home["home_goal_diff_avg_last_5"].sum()
        
        if not away_matches.empty:
            valid_away = away_matches.dropna(subset=["away_points_avg_last_5"])
            if not valid_away.empty:
                recent_stats["Matches Played"] += len(valid_away)
                recent_stats["Avg Points"] += valid_away["away_points_avg_last_5"].sum()
                recent_stats["Avg Goals For"] += valid_away["away_goals_for_avg_last_5"].sum()
                recent_stats["Avg Goals Against"] += valid_away["away_goals_against_avg_last_5"].sum()
                recent_stats["Avg Goal Diff"] += valid_away["away_goal_diff_avg_last_5"].sum()
    
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


# ===== MAIN APP =====

st.title("⚽ Football Match Prediction")

# Load data
model_data, test_pred, processed_data = load_model_and_data()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Team Stats", "🎯 Match Prediction", "📈 Results Table"])

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
        pred_div = st.selectbox("Select Division", get_unique_divisions(), key="pred_div")
        pred_teams = get_teams_by_division(pred_div)
        home_team = st.selectbox("Home Team", pred_teams, key="home_team")
    
    with col2:
        away_team = st.selectbox("Away Team", [t for t in pred_teams if t != home_team], key="away_team")
    
    if st.button("🔮 Predict Match"):
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
        
        # Get match prediction data
        match_data = predict_match(home_team, away_team, model_data, processed_data)
        
        # Get stats
        home_stats = get_recent_stats(home_team, country, processed_data)
        away_stats = get_recent_stats(away_team, country, processed_data)
        
        st.subheader(f"Match: {home_team} vs {away_team}")
        
        # Display match probabilities
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"### {home_team} 🏠")
            st.metric("Recent Points Avg", f"{home_stats['Avg Points']:.2f}")
            st.metric("Avg Goals For", f"{home_stats['Avg Goals For']:.2f}")
            st.metric("Avg Goals Against", f"{home_stats['Avg Goals Against']:.2f}")
        
        with col2:
            st.markdown("### Prediction")
            # Get prediction from recent test data for similar matchup
            similar = test_pred[(test_pred["home_team"] == home_team) | (test_pred["away_team"] == away_team)]
            if not similar.empty:
                avg_pred = similar[["prob_home_win", "prob_draw", "prob_away_win"]].mean()
                
                col_h, col_d, col_a = st.columns(3)
                with col_h:
                    st.metric("Home Win", f"{avg_pred['prob_home_win']:.1%}")
                with col_d:
                    st.metric("Draw", f"{avg_pred['prob_draw']:.1%}")
                with col_a:
                    st.metric("Away Win", f"{avg_pred['prob_away_win']:.1%}")
                
                # Expected goals (mock)
                st.metric("Expected Home Goals", f"{home_stats['Avg Goals For']:.2f}")
                st.metric("Expected Away Goals", f"{away_stats['Avg Goals For']:.2f}")
            else:
                st.warning("Not enough historical data for detailed prediction")
        
        with col3:
            st.markdown(f"### {away_team} 🔗")
            st.metric("Recent Points Avg", f"{away_stats['Avg Points']:.2f}")
            st.metric("Avg Goals For", f"{away_stats['Avg Goals For']:.2f}")
            st.metric("Avg Goals Against", f"{away_stats['Avg Goals Against']:.2f}")

# ===== TAB 3: RESULTS TABLE =====
with tab3:
    st.header("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        results_div = st.selectbox("Filter by Division", ["All"] + get_unique_divisions(), key="results_div")
    
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
    display_data_formatted = display_data[[
        "date", "div", "home_team", "away_team", 
        "actual_result", "predicted_result",
        "prob_home_win", "prob_draw", "prob_away_win"
    ]].copy()
    
    display_data_formatted["Correct"] = display_data_formatted["actual_result"] == display_data_formatted["predicted_result"]
    display_data_formatted["prob_home_win"] = display_data_formatted["prob_home_win"].apply(lambda x: f"{x:.1%}")
    display_data_formatted["prob_draw"] = display_data_formatted["prob_draw"].apply(lambda x: f"{x:.1%}")
    display_data_formatted["prob_away_win"] = display_data_formatted["prob_away_win"].apply(lambda x: f"{x:.1%}")
    
    display_data_formatted.columns = [
        "Date", "Division", "Home", "Away", 
        "Actual", "Predicted", "P(H)", "P(D)", "P(A)", "✓"
    ]
    
    st.dataframe(display_data_formatted, width='stretch')

st.markdown("---")
st.caption("Football Prediction Model — Data as of 2024-08-01")
