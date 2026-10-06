"""Step 7: compare smoothing and time weighting."""

import pandas as pd

from analyze_match import analyze_fixture, load_clean_matches
from predict_match import AWAY_TEAM, DATA_PATH, HOME_TEAM, LOOKBACK_DAYS, MATCH_DATE
from predict_smoothed import PRIOR_MATCHES
from score_probabilities import build_score_matrix, market_probabilities
from shrinkage_model import calculate_smoothed_expected_goals
from time_weighted_model import calculate_time_weighted_expected_goals


# Half weight after 180 days. None restores the Step 6 model.
HALF_LIFE_DAYS = 180.0


def main() -> None:
    matches = load_clean_matches(DATA_PATH)
    analysis = analyze_fixture(matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS)
    baseline = calculate_smoothed_expected_goals(analysis, PRIOR_MATCHES)
    weighted = calculate_time_weighted_expected_goals(
        matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS, HALF_LIFE_DAYS, PRIOR_MATCHES
    )
    weighted_analysis = weighted["weighted_analysis"]
    baseline_markets = market_probabilities(build_score_matrix(baseline["lambda_home"], baseline["lambda_away"])["matrix"])
    weighted_markets = market_probabilities(build_score_matrix(weighted["lambda_home"], weighted["lambda_away"])["matrix"])

    print("Step 7: time-weighted and smoothed model")
    print(f"Match: {HOME_TEAM} – {AWAY_TEAM}, {MATCH_DATE}")
    print(f"Match identifier: {analysis['match_id']}")
    print(f"History-window length: {LOOKBACK_DAYS} days")
    half_life_text = f"{HALF_LIFE_DAYS:g} days" if HALF_LIFE_DAYS is not None else "disabled"
    print(f"Half-life: {half_life_text}")
    print(f"Prior weight: {PRIOR_MATCHES:g}")
    print("Comparison: smoothed | time-weighted and smoothed")

    print("\nActual match counts and total time weights:")
    for label, key in [("League", "league"), (HOME_TEAM, "home"), (AWAY_TEAM, "away")]:
        stats = weighted_analysis[key]
        print(f"{label}: {stats['matches']} matches | total weight {stats['weight_sum']:.3f}")

    print("\nTeam-data weight in smoothing (smoothed | time-weighted and smoothed):")
    for team, venue in [(HOME_TEAM, "home"), (AWAY_TEAM, "away")]:
        print(f"{team}: {baseline['data_weights'][venue]:.2%} | {weighted['data_weights'][venue]:.2%}")

    print("\nLeague averages (equal weight | time weight):")
    for label, key in [("Home", "home_goal_mean"), ("Away", "away_goal_mean")]:
        print(f"{label}: {analysis['league'][key]:.3f} | {weighted_analysis['league'][key]:.3f}")

    print("\nSmoothed team averages (Step 6 | Step 7):")
    for team, venue in [(HOME_TEAM, "home"), (AWAY_TEAM, "away")]:
        for label, key in [("scored", "goals_for_mean"), ("conceded", "goals_against_mean")]:
            print(f"{team} {label}: {baseline['smoothed_means'][venue][key]:.3f} | {weighted['smoothed_means'][venue][key]:.3f}")

    print("\nExpected goals (Step 6 | Step 7):")
    for label, key in [("Home", "lambda_home"), ("Away", "lambda_away"), ("Total", "lambda_total")]:
        print(f"{label}: {baseline[key]:.3f} | {weighted[key]:.3f}")

    print("\n1X2 (Step 6 | Step 7):")
    for label, key in [("Home win", "home"), ("Draw", "draw"), ("Away win", "away")]:
        print(f"{label}: {baseline_markets['one_x_two'][key]:.2%} | {weighted_markets['one_x_two'][key]:.2%}")

    print("\nOver / under (Step 6 | Step 7):")
    for line, probabilities in weighted_markets["over_under"].items():
        for outcome in ["under", "over"]:
            print(f"{line:.1f} {outcome}: {baseline_markets['over_under'][line][outcome]:.2%} | {probabilities[outcome]:.2%}")

    print("\nBoth teams to score (Step 6 | Step 7):")
    for label, key in [("Yes", "yes"), ("No", "no")]:
        print(f"{label}: {baseline_markets['btts'][key]:.2%} | {weighted_markets['btts'][key]:.2%}")

    print("\nThe five most probable scores from Step 7:")
    for score in weighted_markets["top_scores"]:
        print(f"{score['home_goals']}–{score['away_goals']}: {score['probability']:.2%}")

    print("\nHalf-life is a learning setting. The next step evaluates the models on multiple matches.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Time-weighting error: {error}")
