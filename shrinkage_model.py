"""Shrink raw team goal averages towards the corresponding league averages."""

from math import isfinite

from goal_model import calculate_expected_goals


def smooth_goal_mean(
    total_goals: int,
    matches: int,
    league_mean: float,
    prior_matches: float,
) -> float:
    """Use the league average with prior_matches matches of prior weight.

Return the league average without history and the raw average with zero prior weight.
Do not modify previous matches or the observed total goals.
"""
    if isinstance(total_goals, bool) or not isinstance(total_goals, int) or total_goals < 0:
        raise ValueError("The total goal count must be a non-negative integer.")
    if isinstance(matches, bool) or not isinstance(matches, int) or matches < 0:
        raise ValueError("The match count must be a non-negative integer.")
    if not isfinite(league_mean) or league_mean <= 0:
        raise ValueError("The league average must be finite and positive.")
    if isinstance(prior_matches, bool) or not isfinite(prior_matches) or prior_matches < 0:
        raise ValueError("The prior weight must be finite and non-negative.")
    if matches == 0:
        if total_goals != 0:
            raise ValueError("Zero matches can only have zero observed goals.")
        return float(league_mean)

    mean = (total_goals + prior_matches * league_mean) / (matches + prior_matches)
    if not isfinite(mean):
        raise ValueError("The smoothed goal average is not finite.")
    return mean


def calculate_smoothed_expected_goals(analysis: dict, prior_matches: float = 5.0) -> dict:
    """Pass the four smoothed goal averages to the previous ratio model."""
    league = analysis["league"]
    home = analysis["home"]
    away = analysis["away"]
    mu_home = league["home_goal_mean"]
    mu_away = league["away_goal_mean"]

    smoothed_home = {
        "goals_for_mean": smooth_goal_mean(home["goals_for_total"], home["matches"], mu_home, prior_matches),
        "goals_against_mean": smooth_goal_mean(home["goals_against_total"], home["matches"], mu_away, prior_matches),
    }
    smoothed_away = {
        "goals_for_mean": smooth_goal_mean(away["goals_for_total"], away["matches"], mu_away, prior_matches),
        "goals_against_mean": smooth_goal_mean(away["goals_against_total"], away["matches"], mu_home, prior_matches),
    }

    adjusted_analysis = {"league": league, "home": smoothed_home, "away": smoothed_away}
    prediction = calculate_expected_goals(adjusted_analysis)

    data_weights = {}
    for venue, stats in [("home", home), ("away", away)]:
        denominator = stats["matches"] + prior_matches
        data_weights[venue] = stats["matches"] / denominator if denominator > 0 else 0.0

    return {
        **prediction,
        "prior_matches": prior_matches,
        "smoothed_means": {"home": smoothed_home, "away": smoothed_away},
        "data_weights": data_weights,
    }
