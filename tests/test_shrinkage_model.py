import copy
import unittest

import pandas as pd

from analyze_match import COLUMNS, analyze_fixture
from goal_model import calculate_expected_goals
from score_probabilities import build_score_matrix, market_probabilities
from shrinkage_model import calculate_smoothed_expected_goals, smooth_goal_mean


def example_analysis() -> dict:
    return {
        "league": {"matches": 20, "home_goal_mean": 2.0, "away_goal_mean": 1.0},
        "home": {"matches": 10, "goals_for_total": 30, "goals_against_total": 5,
                 "goals_for_mean": 3.0, "goals_against_mean": 0.5},
        "away": {"matches": 8, "goals_for_total": 6, "goals_against_total": 32,
                 "goals_for_mean": 0.75, "goals_against_mean": 4.0},
    }


class ShrinkageTests(unittest.TestCase):
    def test_known_weighted_average(self):
        self.assertEqual(smooth_goal_mean(0, 1, 1.2, 5.0), 1.0)
        self.assertEqual(smooth_goal_mean(30, 10, 2.0, 5.0), 40 / 15)

    def test_smoothed_mean_stays_between_raw_mean_and_league_mean(self):
        for goals, matches, baseline in [(0, 2, 1.5), (12, 4, 1.5), (6, 4, 1.5)]:
            with self.subTest(goals=goals, matches=matches):
                mean = smooth_goal_mean(goals, matches, baseline, 5.0)
                raw = goals / matches
                self.assertGreaterEqual(mean, min(raw, baseline))
                self.assertLessEqual(mean, max(raw, baseline))

    def test_smaller_sample_is_pulled_more_towards_league_mean(self):
        small = smooth_goal_mean(6, 2, 1.5, 5.0)
        large = smooth_goal_mean(60, 20, 1.5, 5.0)
        self.assertLess(abs(small - 1.5), abs(large - 1.5))

    def test_larger_prior_weight_pulls_mean_more_towards_league_mean(self):
        weak = smooth_goal_mean(12, 4, 1.5, 2.0)
        strong = smooth_goal_mean(12, 4, 1.5, 10.0)
        self.assertLess(abs(strong - 1.5), abs(weak - 1.5))

    def test_zero_prior_reproduces_raw_model_and_all_probabilities(self):
        analysis = example_analysis()
        raw = calculate_expected_goals(analysis)
        smoothed = calculate_smoothed_expected_goals(analysis, 0.0)
        for key, value in raw.items():
            self.assertEqual(smoothed[key], value)
        raw_markets = market_probabilities(build_score_matrix(raw["lambda_home"], raw["lambda_away"])["matrix"])
        smoothed_markets = market_probabilities(build_score_matrix(smoothed["lambda_home"], smoothed["lambda_away"])["matrix"])
        self.assertEqual(raw_markets, smoothed_markets)

    def test_no_history_gives_league_mean_including_zero_prior(self):
        for prior in [0.0, 5.0]:
            with self.subTest(prior=prior):
                self.assertEqual(smooth_goal_mean(0, 0, 1.5, prior), 1.5)
                analysis = example_analysis()
                for venue in ["home", "away"]:
                    analysis[venue].update(matches=0, goals_for_total=0, goals_against_total=0,
                                           goals_for_mean=None, goals_against_mean=None)
                result = calculate_smoothed_expected_goals(analysis, prior)
                self.assertEqual(result["lambda_home"], 2.0)
                self.assertEqual(result["lambda_away"], 1.0)
                self.assertEqual(result["data_weights"], {"home": 0.0, "away": 0.0})

    def test_observed_zero_gets_positive_mean_with_positive_prior(self):
        self.assertEqual(smooth_goal_mean(0, 1, 1.2, 0.0), 0.0)
        self.assertGreater(smooth_goal_mean(0, 1, 1.2, 5.0), 0.0)

    def test_correct_home_away_baselines_are_used_for_all_four_means(self):
        result = calculate_smoothed_expected_goals(example_analysis(), 5.0)
        means = result["smoothed_means"]
        self.assertEqual(means["home"]["goals_for_mean"], 40 / 15)
        self.assertEqual(means["home"]["goals_against_mean"], 10 / 15)
        self.assertEqual(means["away"]["goals_for_mean"], 11 / 13)
        self.assertEqual(means["away"]["goals_against_mean"], 42 / 13)
        self.assertEqual(result["data_weights"]["home"], 10 / 15)
        self.assertEqual(result["data_weights"]["away"], 8 / 13)

    def test_model_preserves_raw_statistics_and_match_counts(self):
        analysis = example_analysis()
        original = copy.deepcopy(analysis)
        calculate_smoothed_expected_goals(analysis, 5.0)
        self.assertEqual(analysis, original)

    def test_large_prior_approaches_league_baseline(self):
        result = calculate_smoothed_expected_goals(example_analysis(), 1e12)
        self.assertAlmostEqual(result["lambda_home"], 2.0, delta=1e-9)
        self.assertAlmostEqual(result["lambda_away"], 1.0, delta=1e-9)

    def test_invalid_inputs_are_rejected(self):
        invalid_arguments = [
            (-1, 1, 1.5, 5), (1.5, 1, 1.5, 5), (True, 1, 1.5, 5),
            (0, -1, 1.5, 5), (0, 1.5, 1.5, 5), (0, True, 1.5, 5),
            (0, 1, 0.0, 5), (0, 1, float("nan"), 5), (0, 1, float("inf"), 5),
            (0, 1, 1.5, -1), (0, 1, 1.5, True), (0, 1, 1.5, float("inf")),
            (0, 1, 1.5, float("nan")), (1, 0, 1.5, 5),
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    smooth_goal_mean(*arguments)

    def test_target_same_day_and_future_results_cannot_change_smoothed_forecast(self):
        matches = pd.DataFrame([
            ["before_home", "2023/24", "2023-08-09", "Arsenal", "Team A", 3, 1],
            ["before_away", "2023/24", "2023-08-11", "Team B", "Nottingham Forest", 2, 2],
            ["target", "2023/24", "2023-08-12", "Arsenal", "Nottingham Forest", 99, 88],
            ["same_day", "2023/24", "2023-08-12", "Team C", "Team D", 77, 66],
            ["future", "2023/24", "2023-08-13", "Arsenal", "Nottingham Forest", 55, 44],
        ], columns=COLUMNS)
        matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d")

        def forecast(data):
            analysis = analyze_fixture(data, "Arsenal", "Nottingham Forest", "2023-08-12", 730)
            expected = calculate_smoothed_expected_goals(analysis, 5.0)
            markets = market_probabilities(build_score_matrix(expected["lambda_home"], expected["lambda_away"])["matrix"])
            return expected, markets

        original = forecast(matches)
        for match_id in ["target", "same_day", "future"]:
            with self.subTest(match_id=match_id):
                changed = matches.copy()
                changed.loc[changed["match_id"] == match_id, ["FTHG", "FTAG"]] = [0, 0]
                self.assertEqual(original, forecast(changed))


if __name__ == "__main__":
    unittest.main()
