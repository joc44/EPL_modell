import unittest

import pandas as pd

from analyze_match import COLUMNS, analyze_fixture
from goal_model import calculate_expected_goals
from score_probabilities import build_score_matrix, market_probabilities
from shrinkage_model import calculate_smoothed_expected_goals
from time_weighted_model import (
    calculate_time_weighted_expected_goals,
    smooth_weighted_mean,
    time_decay_weight,
    weighted_team_stats,
)


def example_matches() -> pd.DataFrame:
    matches = pd.DataFrame([
        ["older", "2022/23", "2023-02-12", "Arsenal", "Nottingham Forest", 0, 2],
        ["recent", "2023/24", "2023-08-11", "Arsenal", "Nottingham Forest", 3, 1],
        ["target", "2023/24", "2023-08-12", "Arsenal", "Nottingham Forest", 99, 88],
        ["same_day", "2023/24", "2023-08-12", "Team A", "Team B", 77, 66],
        ["future", "2023/24", "2023-08-13", "Arsenal", "Nottingham Forest", 55, 44],
    ], columns=COLUMNS)
    matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d")
    return matches


def forecast(matches: pd.DataFrame, half_life=180.0, prior=5.0) -> dict:
    return calculate_time_weighted_expected_goals(
        matches, "Arsenal", "Nottingham Forest", "2023-08-12", 730, half_life, prior
    )


class TimeWeightTests(unittest.TestCase):
    def test_half_life_and_monotonic_decay(self):
        self.assertEqual(time_decay_weight(0, 180), 1.0)
        self.assertEqual(time_decay_weight(180, 180), 0.5)
        self.assertEqual(time_decay_weight(360, 180), 0.25)
        self.assertEqual(time_decay_weight(720, 180), 0.0625)
        self.assertGreater(time_decay_weight(90, 180), time_decay_weight(180, 180))

    def test_disabled_decay_gives_unit_weights(self):
        for age in [0, 180, 1000]:
            self.assertEqual(time_decay_weight(age, None), 1.0)

    def test_league_weighted_mean_uses_target_date_as_anchor(self):
        # The ages are 181 and 1 days: the recent match has exactly twice the weight.
        result = forecast(example_matches())
        league = result["weighted_analysis"]["league"]
        self.assertEqual(league["matches"], 2)
        self.assertAlmostEqual(league["weight_sum"], 1.5 * time_decay_weight(1, 180))
        self.assertAlmostEqual(league["home_goal_mean"], 2.0)
        self.assertAlmostEqual(league["away_goal_mean"], 4 / 3)

    def test_team_roles_and_fractional_goal_totals(self):
        history = example_matches().iloc[:2].copy()
        history["Weight"] = [0.5, 1.0]
        home = weighted_team_stats(history, "Arsenal", "home")
        away = weighted_team_stats(history, "Nottingham Forest", "away")
        self.assertEqual(home["matches"], 2)
        self.assertEqual(home["weight_sum"], 1.5)
        self.assertEqual(home["weighted_goals_for"], 3.0)
        self.assertEqual(home["weighted_goals_against"], 2.0)
        self.assertEqual(away["weighted_goals_for"], 2.0)
        self.assertEqual(away["weighted_goals_against"], 3.0)
        self.assertEqual(away["goals_for_mean"], 4 / 3)
        self.assertEqual(away["goals_against_mean"], 2.0)

    def test_smoothing_preserves_fractional_exposure(self):
        self.assertEqual(smooth_weighted_mean(1.0, 0.5, 1.5, 5.0), 8.5 / 5.5)
        self.assertEqual(smooth_weighted_mean(1.5, 0.75, 1.5, 0.0), 2.0)

    def test_disabled_decay_reproduces_previous_smoothed_model(self):
        matches = example_matches()
        analysis = analyze_fixture(matches, "Arsenal", "Nottingham Forest", "2023-08-12", 730)
        previous = calculate_smoothed_expected_goals(analysis, 5.0)
        current = forecast(matches, None, 5.0)
        for key in ["lambda_home", "lambda_away", "lambda_total", "home_attack_factor",
                    "home_concede_factor", "away_attack_factor", "away_concede_factor"]:
            self.assertAlmostEqual(current[key], previous[key], places=12)
        self.assertEqual(current["smoothed_means"], previous["smoothed_means"])
        self.assertEqual(current["data_weights"], previous["data_weights"])

    def test_disabled_decay_and_zero_prior_reproduce_original_raw_model(self):
        matches = example_matches()
        analysis = analyze_fixture(matches, "Arsenal", "Nottingham Forest", "2023-08-12", 730)
        previous = calculate_expected_goals(analysis)
        current = forecast(matches, None, 0.0)
        for key, value in previous.items():
            self.assertAlmostEqual(current[key], value, places=12)

    def test_missing_team_history_gives_weighted_league_baseline(self):
        matches = example_matches()
        matches.loc[matches["match_id"].isin(["older", "recent"]), ["HomeTeam", "AwayTeam"]] = ["Other H", "Other A"]
        result = forecast(matches)
        league = result["weighted_analysis"]["league"]
        self.assertEqual(result["weighted_analysis"]["home"]["matches"], 0)
        self.assertEqual(result["data_weights"]["home"], 0.0)
        self.assertAlmostEqual(result["lambda_home"], league["home_goal_mean"])
        self.assertAlmostEqual(result["lambda_away"], league["away_goal_mean"])

    def test_zero_weight_sum_has_explicit_fallback(self):
        self.assertEqual(smooth_weighted_mean(0.0, 0.0, 1.5, 5.0), 1.5)
        self.assertEqual(smooth_weighted_mean(0.0, 0.0, 1.5, 0.0), 1.5)
        with self.assertRaises(ValueError):
            smooth_weighted_mean(1.0, 0.0, 1.5, 5.0)

    def test_input_dataframe_is_not_modified(self):
        matches = example_matches()
        original = matches.copy(deep=True)
        forecast(matches)
        pd.testing.assert_frame_equal(matches, original)
        self.assertNotIn("Weight", matches.columns)

    def test_target_same_day_future_results_and_added_future_rows_do_not_change_forecast(self):
        matches = example_matches()
        original = forecast(matches)
        original_markets = market_probabilities(build_score_matrix(original["lambda_home"], original["lambda_away"])["matrix"])
        for match_id in ["target", "same_day", "future"]:
            with self.subTest(match_id=match_id):
                changed = matches.copy()
                changed.loc[changed["match_id"] == match_id, ["FTHG", "FTAG"]] = [0, 0]
                result = forecast(changed)
                self.assertEqual(original, result)
                markets = market_probabilities(build_score_matrix(result["lambda_home"], result["lambda_away"])["matrix"])
                self.assertEqual(original_markets, markets)
        added = matches.iloc[[-1]].copy()
        added["match_id"] = "another_future"
        added["Date"] = pd.Timestamp("2023-09-01")
        self.assertEqual(original, forecast(pd.concat([matches, added], ignore_index=True)))

    def test_no_league_history_and_all_underflowed_weights_raise(self):
        with self.assertRaisesRegex(ValueError, "No previous matches"):
            forecast(example_matches().iloc[2:])
        with self.assertRaisesRegex(ValueError, "zero total time weight"):
            forecast(example_matches(), 1e-12)

    def test_invalid_ages_and_half_lives_are_rejected(self):
        for age in [-1.0, float("inf"), float("nan")]:
            with self.subTest(age=age):
                with self.assertRaises(ValueError):
                    time_decay_weight(age, 180.0)
        for half_life in [0.0, -1.0, float("inf"), float("nan"), True]:
            with self.subTest(half_life=half_life):
                with self.assertRaises(ValueError):
                    time_decay_weight(1.0, half_life)

    def test_invalid_weighted_smoothing_arguments_are_rejected(self):
        for arguments in [(-1, 1, 1.5, 5), (1, -1, 1.5, 5), (1, 1, 0, 5),
                          (1, 1, 1.5, -1), (float("nan"), 1, 1.5, 5),
                          (1, float("inf"), 1.5, 5), (1, 1, 1.5, True)]:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    smooth_weighted_mean(*arguments)


if __name__ == "__main__":
    unittest.main()
