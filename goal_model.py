"""Simple expected-goals model based on final-score data.

Input is the pre-match statistics produced by analyze_fixture().
This module only calculates: it performs no file I/O or console output.
"""

from math import isfinite


def relative_factor(goal_mean: float | None, league_mean: float) -> float:
    """Compare a goal average with the corresponding league average.

Without history (None), use the neutral multiplier one.
An observed average of 0.0 remains zero.
"""
    if not isfinite(league_mean) or league_mean <= 0:
        raise ValueError("The reference league average must be finite and positive.")
    if goal_mean is None:
        return 1.0
    if not isfinite(goal_mean) or goal_mean < 0:
        raise ValueError("The team's goal average must be finite and non-negative.")
    return goal_mean / league_mean


def calculate_expected_goals(analysis: dict) -> dict:
    """Calculate two expected goal counts from attack and concede multipliers.

A higher concede multiplier indicates more goals conceded and weaker
 defence in this sample. The two separate league averages represent
 home advantage; no additional home multiplier is applied.
"""
    league = analysis["league"]
    home = analysis["home"]
    away = analysis["away"]

    mu_home = league["home_goal_mean"]
    mu_away = league["away_goal_mean"]

    home_attack = relative_factor(home["goals_for_mean"], mu_home)
    home_concede = relative_factor(home["goals_against_mean"], mu_away)
    away_attack = relative_factor(away["goals_for_mean"], mu_away)
    away_concede = relative_factor(away["goals_against_mean"], mu_home)

    lambda_home = mu_home * home_attack * away_concede
    lambda_away = mu_away * away_attack * home_concede
    lambda_total = lambda_home + lambda_away
    if not all(isfinite(value) for value in [lambda_home, lambda_away, lambda_total]):
        raise ValueError("The calculated expected goal count is not finite.")

    return {
        "home_attack_factor": home_attack,
        "home_concede_factor": home_concede,
        "away_attack_factor": away_attack,
        "away_concede_factor": away_concede,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "lambda_total": lambda_total,
    }
