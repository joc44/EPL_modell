"""Exponential time weighting followed by smoothing towards the league average."""

from math import isfinite

import pandas as pd

from analyze_match import analyze_fixture, history_before
from goal_model import calculate_expected_goals


def time_decay_weight(age_days: float, half_life_days: float | None) -> float:
    """Assign a weight between zero and one from age relative to the target match.

None disables time weighting: every match receives weight one.
"""
    if not isfinite(age_days) or age_days < 0:
        raise ValueError("Match age must be a finite, non-negative number of days.")
    if half_life_days is None:
        return 1.0
    if isinstance(half_life_days, bool) or not isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("Half-life must be a positive, finite number of days or None.")
    return 2.0 ** (-age_days / half_life_days)


def smooth_weighted_mean(
    weighted_goals: float,
    weight_sum: float,
    league_mean: float,
    prior_matches: float,
) -> float:
    """Use weighted goal totals and total weight for smoothing.

These can be fractional, so this is separate from the Step 6 function
that works with actual integer match and goal counts.
"""
    for value in [weighted_goals, weight_sum, prior_matches]:
        if isinstance(value, bool) or not isfinite(value) or value < 0:
            raise ValueError("The weighted total and prior weight must be finite and non-negative.")
    if not isfinite(league_mean) or league_mean <= 0:
        raise ValueError("The weighted league average must be finite and positive.")
    if weight_sum == 0:
        if weighted_goals != 0:
            raise ValueError("Zero total weight can only have zero weighted total goals.")
        return float(league_mean)

    mean = (weighted_goals + prior_matches * league_mean) / (weight_sum + prior_matches)
    if not isfinite(mean):
        raise ValueError("The weighted, smoothed average is not finite.")
    return mean


def weighted_team_stats(history: pd.DataFrame, team: str, venue: str) -> dict:
    """Calculate on date-filtered matches containing a Weight column."""
    venue_columns = {
        "home": ("HomeTeam", "FTHG", "FTAG"),
        "away": ("AwayTeam", "FTAG", "FTHG"),
    }
    if venue not in venue_columns:
        raise ValueError("The venue must be home or away.")
    team_column, goals_for_column, goals_against_column = venue_columns[venue]
    selected = history.loc[history[team_column] == team]
    weight_sum = float(selected["Weight"].sum())
    goals_for = float((selected[goals_for_column] * selected["Weight"]).sum())
    goals_against = float((selected[goals_against_column] * selected["Weight"]).sum())

    return {
        "matches": len(selected),
        "weight_sum": weight_sum,
        "weighted_goals_for": goals_for,
        "weighted_goals_against": goals_against,
        "goals_for_mean": goals_for / weight_sum if weight_sum > 0 else None,
        "goals_against_mean": goals_against / weight_sum if weight_sum > 0 else None,
    }


def calculate_time_weighted_expected_goals(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: str,
    lookback_days: int,
    half_life_days: float | None = 180.0,
    prior_matches: float = 5.0,
) -> dict:
    """Calculate weighted, smoothed expected goals from pre-match data.

Measure weights relative to the target matchday. The league and four team
averages use the same weighting rule. Do not modify the input table.
"""
    # The previous analyser validates the target match and available league history.
    snapshot = analyze_fixture(matches, home_team, away_team, match_date, lookback_days)
    target_date = snapshot["match_date"].normalize()
    history = history_before(matches, target_date, lookback_days)
    ages = (target_date - history["Date"]).dt.days
    history["Weight"] = ages.map(lambda age: time_decay_weight(age, half_life_days))

    weight_sum = float(history["Weight"].sum())
    if not isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("The league has zero total time weight or invalid weights. Check the half-life.")
    mu_home = float((history["FTHG"] * history["Weight"]).sum() / weight_sum)
    mu_away = float((history["FTAG"] * history["Weight"]).sum() / weight_sum)

    home = weighted_team_stats(history, home_team, "home")
    away = weighted_team_stats(history, away_team, "away")
    league = {"matches": len(history), "weight_sum": weight_sum,
              "home_goal_mean": mu_home, "away_goal_mean": mu_away}

    smoothed_home = {
        "goals_for_mean": smooth_weighted_mean(home["weighted_goals_for"], home["weight_sum"], mu_home, prior_matches),
        "goals_against_mean": smooth_weighted_mean(home["weighted_goals_against"], home["weight_sum"], mu_away, prior_matches),
    }
    smoothed_away = {
        "goals_for_mean": smooth_weighted_mean(away["weighted_goals_for"], away["weight_sum"], mu_away, prior_matches),
        "goals_against_mean": smooth_weighted_mean(away["weighted_goals_against"], away["weight_sum"], mu_home, prior_matches),
    }
    prediction = calculate_expected_goals({"league": league, "home": smoothed_home, "away": smoothed_away})

    data_weights = {}
    for venue, stats in [("home", home), ("away", away)]:
        denominator = stats["weight_sum"] + prior_matches
        data_weights[venue] = stats["weight_sum"] / denominator if denominator > 0 else 0.0

    weighted_analysis = {
        "match_id": snapshot["match_id"],
        "match_date": target_date,
        "start_date": snapshot["start_date"],
        "last_history_date": snapshot["last_history_date"],
        "league": league,
        "home": home,
        "away": away,
    }
    return {
        **prediction,
        "half_life_days": half_life_days,
        "prior_matches": prior_matches,
        "weighted_analysis": weighted_analysis,
        "smoothed_means": {"home": smoothed_home, "away": smoothed_away},
        "data_weights": data_weights,
    }
