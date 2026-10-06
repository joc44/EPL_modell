import copy
import unittest

import pandas as pd

from analyze_match import COLUMNS, analyze_fixture
from goal_model import calculate_expected_goals, relative_factor


def example_analysis() -> dict:
    """Asymmetric league and teams: an incorrect denominator is easy to detect."""
    return {
        "league": {"home_goal_mean": 2.0, "away_goal_mean": 1.0},
        "home": {"matches": 10, "goals_for_mean": 3.0, "goals_against_mean": 0.5},
        "away": {"matches": 8, "goals_for_mean": 0.75, "goals_against_mean": 4.0},
    }


def example_matches() -> pd.DataFrame:
    matches = pd.DataFrame([
        ["old_home", "2023/24", "2023-08-09", "Arsenal", "Team A", 3, 1],
        ["old_away", "2023/24", "2023-08-11", "Team B", "Nottingham Forest", 2, 2],
        ["target", "2023/24", "2023-08-12", "Arsenal", "Nottingham Forest", 99, 88],
        ["same_day", "2023/24", "2023-08-12", "Team C", "Team D", 77, 66],
        ["future", "2023/24", "2023-08-13", "Arsenal", "Nottingham Forest", 55, 44],
    ], columns=COLUMNS)
    matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d")
    return matches


def predict_example(matches: pd.DataFrame) -> dict:
    analysis = analyze_fixture(matches, "Arsenal", "Nottingham Forest", "2023-08-12", 730)
    return calculate_expected_goals(analysis)


class GoalModelTests(unittest.TestCase):
    def test_average_teams_reproduce_league_means(self):
        analysis = example_analysis()
        analysis["home"].update(goals_for_mean=2.0, goals_against_mean=1.0)
        analysis["away"].update(goals_for_mean=1.0, goals_against_mean=2.0)
        result = calculate_expected_goals(analysis)
        self.assertEqual(result["lambda_home"], 2.0)
        self.assertEqual(result["lambda_away"], 1.0)
        self.assertEqual(result["lambda_total"], 3.0)

    def test_asymmetric_profiles_use_correct_role_baselines(self):
        result = calculate_expected_goals(example_analysis())
        self.assertEqual(result["home_attack_factor"], 1.5)
        self.assertEqual(result["home_concede_factor"], 0.5)
        self.assertEqual(result["away_attack_factor"], 0.75)
        self.assertEqual(result["away_concede_factor"], 2.0)
        self.assertEqual(result["lambda_home"], 6.0)
        self.assertEqual(result["lambda_away"], 0.375)

    def test_missing_home_history_keeps_known_opponent_profile(self):
        analysis = example_analysis()
        analysis["home"].update(matches=0, goals_for_mean=None, goals_against_mean=None)
        result = calculate_expected_goals(analysis)
        self.assertEqual(result["home_attack_factor"], 1.0)
        self.assertEqual(result["home_concede_factor"], 1.0)
        self.assertEqual(result["lambda_home"], 4.0)
        self.assertEqual(result["lambda_away"], 0.75)

    def test_missing_both_histories_gives_league_baseline(self):
        analysis = example_analysis()
        for venue in ["home", "away"]:
            analysis[venue].update(matches=0, goals_for_mean=None, goals_against_mean=None)
        result = calculate_expected_goals(analysis)
        self.assertEqual(result["lambda_home"], 2.0)
        self.assertEqual(result["lambda_away"], 1.0)

    def test_observed_zero_is_not_replaced_with_neutral_factor(self):
        analysis = example_analysis()
        analysis["home"]["goals_for_mean"] = 0.0
        result = calculate_expected_goals(analysis)
        self.assertEqual(result["home_attack_factor"], 0.0)
        self.assertEqual(result["lambda_home"], 0.0)

    def test_invalid_league_baselines_are_rejected(self):
        for invalid in [0.0, -1.0, float("nan"), float("inf")]:
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "league average"):
                    relative_factor(1.0, invalid)

    def test_invalid_team_averages_are_rejected(self):
        for invalid in [-1.0, float("nan"), float("inf")]:
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "team's goal average"):
                    relative_factor(invalid, 1.0)

    def test_model_does_not_modify_input_statistics(self):
        analysis = example_analysis()
        original = copy.deepcopy(analysis)
        calculate_expected_goals(analysis)
        self.assertEqual(analysis, original)

    def test_target_same_day_and_future_results_do_not_change_prediction(self):
        matches = example_matches()
        original = predict_example(matches)
        for match_id in ["target", "same_day", "future"]:
            with self.subTest(match_id=match_id):
                changed = matches.copy()
                changed.loc[changed["match_id"] == match_id, ["FTHG", "FTAG"]] = [0, 0]
                self.assertEqual(original, predict_example(changed))


if __name__ == "__main__":
    unittest.main()
