"""Step 5 executable program: score and event probabilities."""

from math import fsum

import pandas as pd

from analyze_match import analyze_fixture, load_clean_matches
from goal_model import calculate_expected_goals
from predict_match import AWAY_TEAM, DATA_PATH, HOME_TEAM, LOOKBACK_DAYS, MATCH_DATE
from score_probabilities import build_score_matrix, market_probabilities


def main() -> None:
    # Fixture and history-window settings still come from predict_match.py.
    matches = load_clean_matches(DATA_PATH)
    analysis = analyze_fixture(matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS)
    prediction = calculate_expected_goals(analysis)
    distribution = build_score_matrix(prediction["lambda_home"], prediction["lambda_away"])
    markets = market_probabilities(distribution["matrix"])

    print("Step 5: Poisson-based match probabilities")
    print(f"Match: {HOME_TEAM} – {AWAY_TEAM}, {MATCH_DATE}")
    print(f"Match identifier: {analysis['match_id']}")
    print(f"History-window length: {LOOKBACK_DAYS} days")
    print(f"Previous matches: {analysis['league']['matches']}")
    print(f"Expected goals: {prediction['lambda_home']:.3f} – {prediction['lambda_away']:.3f}")
    for team, stats in [(HOME_TEAM, analysis["home"]), (AWAY_TEAM, analysis["away"])]:
        if stats["matches"] == 0:
            print(f"{team}: no matching history; neutral team multipliers are used.")

    print("\nScore-table coverage before normalisation:")
    print(f"Home goals: 0–{distribution['max_home_goals']}")
    print(f"Away goals: 0–{distribution['max_away_goals']}")
    print(f"Covered probability: {distribution['captured_probability']:.10%}")
    print(f"Omitted probability: {distribution['omitted_probability']:.3e}")

    one_x_two = markets["one_x_two"]
    print("\n1X2:")
    print(f"Home win: {one_x_two['home']:.2%}")
    print(f"Draw: {one_x_two['draw']:.2%}")
    print(f"Away win: {one_x_two['away']:.2%}")
    print(f"Sum: {fsum(one_x_two.values()):.2%}")

    print("\nOver / under:")
    for line, probabilities in markets["over_under"].items():
        print(f"{line:.1f} goals: under {probabilities['under']:.2%} | over {probabilities['over']:.2%}")

    print("\nBoth teams to score (BTTS):")
    print(f"Yes: {markets['btts']['yes']:.2%}")
    print(f"No: {markets['btts']['no']:.2%}")

    print("\nThe five most probable exact scores:")
    for score in markets["top_scores"]:
        print(f"{score['home_goals']}–{score['away_goals']}: {score['probability']:.2%}")

    print("\nThese are raw-model estimates. Predictive quality will be evaluated by backtesting.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Probability error: {error}")
