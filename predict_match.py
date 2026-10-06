"""Step 4 executable program: two expected goal counts."""

from pathlib import Path

import pandas as pd

from analyze_match import analyze_fixture, load_clean_matches
from goal_model import calculate_expected_goals


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"

# Set the historical fixture to analyse here for this step.
MATCH_DATE = "2023-08-12"
HOME_TEAM = "Arsenal"
AWAY_TEAM = "Nottingham Forest"
LOOKBACK_DAYS = 730


def main() -> None:
    matches = load_clean_matches(DATA_PATH)
    analysis = analyze_fixture(matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS)
    prediction = calculate_expected_goals(analysis)

    print("Step 4: expected goals before the match")
    print(f"Match: {HOME_TEAM} – {AWAY_TEAM}, {MATCH_DATE}")
    print(f"Match identifier: {analysis['match_id']}")
    print(f"History-window length: {LOOKBACK_DAYS} days")
    print(f"Previous matches: {analysis['league']['matches']}")
    print(f"Home team's home matches: {analysis['home']['matches']}")
    print(f"Away team's away matches: {analysis['away']['matches']}")

    for team, stats in [(HOME_TEAM, analysis["home"]), (AWAY_TEAM, analysis["away"])]:
        if stats["matches"] == 0:
            print(f"{team}: no matching history; both multipliers are neutral (1.000).")

    print("\nLeague averages:")
    print(f"Home: {analysis['league']['home_goal_mean']:.3f}")
    print(f"Away: {analysis['league']['away_goal_mean']:.3f}")

    print("\nRelative multipliers (1.000 = league average):")
    print(f"{HOME_TEAM} attack multiplier: {prediction['home_attack_factor']:.3f}")
    print(f"{HOME_TEAM} concede multiplier: {prediction['home_concede_factor']:.3f}")
    print(f"{AWAY_TEAM} attack multiplier: {prediction['away_attack_factor']:.3f}")
    print(f"{AWAY_TEAM} concede multiplier: {prediction['away_concede_factor']:.3f}")

    print("\nExpected goals:")
    print(f"{HOME_TEAM}: {prediction['lambda_home']:.3f}")
    print(f"{AWAY_TEAM}: {prediction['lambda_away']:.3f}")
    print(f"Total: {prediction['lambda_total']:.3f}")
    print("\nThe estimate uses previous final scores. Probabilities follow in Step 5.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Prediction error: {error}")
