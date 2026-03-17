"""Feature engineering helpers for model-ready football datasets."""

from __future__ import annotations

from typing import List

import pandas as pd

RECENT_FORM_REQUIRED_COLUMNS = {
    "div",
    "date",
    "home_team",
    "away_team",
    "full_time_home_goals",
    "full_time_away_goals",
    "target_result",
}


def get_recent_form_feature_columns(window: int = 5) -> List[str]:
    """Return engineered recent-form feature names for both teams."""

    metric_names = [
        f"matches_played_before_match",
        f"points_avg_last_{window}",
        f"goals_for_avg_last_{window}",
        f"goals_against_avg_last_{window}",
        f"goal_diff_avg_last_{window}",
    ]

    columns: List[str] = []
    for side in ("home", "away"):
        for metric_name in metric_names:
            columns.append(f"{side}_{metric_name}")
    return columns


def _build_recent_form_features(
    df: pd.DataFrame,
    window: int,
    group_keys: List[str],
) -> pd.DataFrame:
    """Build no-leak recent-form features for the provided grouping keys."""

    working = df.copy().reset_index(drop=True)
    working["_row_id"] = working.index

    home_points = working["target_result"].map({"H": 3, "D": 1, "A": 0})
    away_points = working["target_result"].map({"H": 0, "D": 1, "A": 3})

    home_history = pd.DataFrame(
        {
            "_row_id": working["_row_id"],
            "div": working["div"],
            "date": working["date"],
            "team": working["home_team"],
            "side": "home",
            "points": home_points,
            "goals_for": working["full_time_home_goals"],
            "goals_against": working["full_time_away_goals"],
        }
    )
    away_history = pd.DataFrame(
        {
            "_row_id": working["_row_id"],
            "div": working["div"],
            "date": working["date"],
            "team": working["away_team"],
            "side": "away",
            "points": away_points,
            "goals_for": working["full_time_away_goals"],
            "goals_against": working["full_time_home_goals"],
        }
    )

    team_history = pd.concat([home_history, away_history], ignore_index=True)
    team_history["goal_diff"] = (
        team_history["goals_for"] - team_history["goals_against"]
    )
    team_history = team_history.sort_values([*group_keys, "date", "_row_id"])

    team_history["matches_played_before_match"] = team_history.groupby(
        group_keys
    ).cumcount()

    rolling_metric_names = ["points", "goals_for", "goals_against", "goal_diff"]
    for metric_name in rolling_metric_names:
        team_history[f"{metric_name}_avg_last_{window}"] = team_history.groupby(
            group_keys
        )[metric_name].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
        )

    engineered_columns = [
        "matches_played_before_match",
        f"points_avg_last_{window}",
        f"goals_for_avg_last_{window}",
        f"goals_against_avg_last_{window}",
        f"goal_diff_avg_last_{window}",
    ]

    home_features = team_history.loc[
        team_history["side"] == "home", ["_row_id", *engineered_columns]
    ].rename(columns={column: f"home_{column}" for column in engineered_columns})
    away_features = team_history.loc[
        team_history["side"] == "away", ["_row_id", *engineered_columns]
    ].rename(columns={column: f"away_{column}" for column in engineered_columns})

    featured = working.merge(home_features, on="_row_id", how="left")
    featured = featured.merge(away_features, on="_row_id", how="left")
    featured = (
        featured.sort_values("_row_id").drop(columns="_row_id").reset_index(drop=True)
    )

    for column in [
        "home_matches_played_before_match",
        "away_matches_played_before_match",
    ]:
        featured[column] = featured[column].astype("Int64")

    return featured


def add_recent_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add no-leak recent-form features built from each team's prior matches.

    Features are computed within the current dataframe only, grouped by division and
    team, and use only matches that occurred earlier in the file's date order.
    """

    missing_columns = sorted(RECENT_FORM_REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot build recent-form features; missing columns: "
            + ", ".join(missing_columns)
        )

    return _build_recent_form_features(df=df, window=window, group_keys=["div", "team"])


def add_cross_season_recent_form_features(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Add no-leak recent-form features across all seasons for each division/team.

    This function is intended for combined multi-file datasets and preserves team
    history across season boundaries. Newly promoted teams naturally start with no
    prior top-flight history in the dataset.
    """

    missing_columns = sorted(RECENT_FORM_REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot build cross-season recent-form features; missing columns: "
            + ", ".join(missing_columns)
        )

    return _build_recent_form_features(df=df, window=window, group_keys=["div", "team"])
