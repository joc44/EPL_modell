"""Step 6: compare the raw and smoothed models."""

import pandas as pd

from analyze_match import analyze_fixture, load_clean_matches
from goal_model import calculate_expected_goals
from predict_match import AWAY_TEAM, DATA_PATH, HOME_TEAM, LOOKBACK_DAYS, MATCH_DATE
from score_probabilities import build_score_matrix, market_probabilities
from shrinkage_model import calculate_smoothed_expected_goals


# Prior weight equivalent to five league-average matches; select it later by validation.
PRIOR_MATCHES = 5.0


def main() -> None:
    matches = load_clean_matches(DATA_PATH)
    analysis = analyze_fixture(matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS)
    raw = calculate_expected_goals(analysis)
    smoothed = calculate_smoothed_expected_goals(analysis, PRIOR_MATCHES)

    raw_distribution = build_score_matrix(raw["lambda_home"], raw["lambda_away"])
    smoothed_distribution = build_score_matrix(smoothed["lambda_home"], smoothed["lambda_away"])
    raw_markets = market_probabilities(raw_distribution["matrix"])
    smoothed_markets = market_probabilities(smoothed_distribution["matrix"])

    print("Step 6: compare the raw and smoothed models")
    print(f"Match: {HOME_TEAM} – {AWAY_TEAM}, {MATCH_DATE}")
    print(f"Match identifier: {analysis['match_id']}")
    print(f"History-window length: {LOOKBACK_DAYS} days")
    print(f"Previous matches: {analysis['league']['matches']}")
    print(f"Smoothing strength: {PRIOR_MATCHES:g} matches of league-average prior weight")

    print("\nWeights of team data and league average:")
    for team, venue in [(HOME_TEAM, "home"), (AWAY_TEAM, "away")]:
        weight = smoothed["data_weights"][venue]
        print(f"{team}: {analysis[venue]['matches']} matches | team data {weight:.2%} | league average {1 - weight:.2%}")

    print("\nGoal averages (raw | smoothed):")
    for team, venue, description in [(HOME_TEAM, "home", "home"), (AWAY_TEAM, "away", "away")]:
        stats = analysis[venue]
        adjusted = smoothed["smoothed_means"][venue]
        for label, key in [("scored", "goals_for_mean"), ("conceded", "goals_against_mean")]:
            original = f"{stats[key]:.3f}" if stats[key] is not None else "no history"
            print(f"{team} {description} {label}: {original} | {adjusted[key]:.3f}")

    print("\nExpected goals (raw | smoothed):")
    for label, key in [("Home", "lambda_home"), ("Away", "lambda_away"), ("Total", "lambda_total")]:
        print(f"{label}: {raw[key]:.3f} | {smoothed[key]:.3f}")

    print("\n1X2 (raw | smoothed):")
    for label, key in [("Home win", "home"), ("Draw", "draw"), ("Away win", "away")]:
        print(f"{label}: {raw_markets['one_x_two'][key]:.2%} | {smoothed_markets['one_x_two'][key]:.2%}")

    print("\nOver / under (raw | smoothed):")
    for line, probabilities in smoothed_markets["over_under"].items():
        for outcome in ["under", "over"]:
            print(f"{line:.1f} {outcome}: {raw_markets['over_under'][line][outcome]:.2%} | {probabilities[outcome]:.2%}")

    print("\nBoth teams to score (raw | smoothed):")
    for label, key in [("Yes", "yes"), ("No", "no")]:
        print(f"{label}: {raw_markets['btts'][key]:.2%} | {smoothed_markets['btts'][key]:.2%}")

    print("\nThe smoothed model's five most probable scores:")
    for score in smoothed_markets["top_scores"]:
        print(f"{score['home_goals']}–{score['away_goals']}: {score['probability']:.2%}")

    print("\nThis is a learning setting. Model improvement will be evaluated chronologically.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Smoothing error: {error}")
