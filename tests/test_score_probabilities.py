import math
import unittest

import pandas as pd

from analyze_match import COLUMNS, analyze_fixture
from goal_model import calculate_expected_goals
from score_probabilities import build_score_matrix, market_probabilities, poisson_goal_probabilities


def markets_for(lambda_home: float, lambda_away: float) -> dict:
    return market_probabilities(build_score_matrix(lambda_home, lambda_away)["matrix"])


class ScoreProbabilityTests(unittest.TestCase):
    def test_known_poisson_values(self):
        probabilities = poisson_goal_probabilities(2.0)
        self.assertAlmostEqual(probabilities[0], 0.1353352832366127)
        self.assertAlmostEqual(probabilities[3], 0.1804470443154836)

    def test_matrix_preserves_mass_and_declared_tail_tolerance(self):
        for lambda_home, lambda_away in [(0.0, 0.0), (3.4123347446242787, 0.5061909341687736), (50.0, 1.0)]:
            with self.subTest(home=lambda_home, away=lambda_away):
                distribution = build_score_matrix(lambda_home, lambda_away)
                matrix = distribution["matrix"]
                self.assertAlmostEqual(math.fsum(p for row in matrix for p in row), 1.0, places=12)
                self.assertLessEqual(distribution["omitted_probability"], 2e-10 + 1e-14)
                self.assertTrue(all(0 <= p <= 1 for row in matrix for p in row))

    def test_all_partitions_sum_to_one(self):
        result = markets_for(2.2, 1.1)
        self.assertAlmostEqual(math.fsum(result["one_x_two"].values()), 1.0)
        self.assertAlmostEqual(math.fsum(result["btts"].values()), 1.0)
        for line, probabilities in result["over_under"].items():
            with self.subTest(line=line):
                self.assertAlmostEqual(math.fsum(probabilities.values()), 1.0)

    def test_equal_rates_give_equal_win_probabilities(self):
        result = markets_for(1.7, 1.7)
        self.assertAlmostEqual(result["one_x_two"]["home"], result["one_x_two"]["away"])
        self.assertGreater(result["one_x_two"]["draw"], 0)

    def test_swapping_teams_swaps_wins_and_preserves_totals_and_btts(self):
        original = markets_for(2.7, 0.8)
        swapped = markets_for(0.8, 2.7)
        self.assertAlmostEqual(original["one_x_two"]["home"], swapped["one_x_two"]["away"])
        self.assertAlmostEqual(original["one_x_two"]["draw"], swapped["one_x_two"]["draw"])
        self.assertAlmostEqual(original["btts"]["yes"], swapped["btts"]["yes"])
        for line in original["over_under"]:
            self.assertAlmostEqual(original["over_under"][line]["over"], swapped["over_under"][line]["over"])

    def test_btts_matches_independent_closed_form(self):
        home, away = 2.2, 1.1
        expected = (1 - math.exp(-home)) * (1 - math.exp(-away))
        self.assertAlmostEqual(markets_for(home, away)["btts"]["yes"], expected, delta=3e-10)

    def test_totals_match_poisson_with_sum_of_means(self):
        home, away = 2.2, 1.1
        result = markets_for(home, away)
        total = home + away
        for line, probabilities in result["over_under"].items():
            expected_under = math.fsum(math.exp(-total) * total**k / math.factorial(k) for k in range(int(line) + 1))
            with self.subTest(line=line):
                self.assertAlmostEqual(probabilities["under"], expected_under, delta=3e-10)
        over = [result["over_under"][line]["over"] for line in [1.5, 2.5, 3.5]]
        self.assertGreater(over[0], over[1])
        self.assertGreater(over[1], over[2])

    def test_zero_rates_give_certain_zero_zero_and_no_btts(self):
        result = markets_for(0.0, 0.0)
        self.assertEqual(result["one_x_two"], {"home": 0.0, "draw": 1.0, "away": 0.0})
        self.assertEqual(result["btts"], {"yes": 0.0, "no": 1.0})
        self.assertEqual(result["top_scores"], [{"home_goals": 0, "away_goals": 0, "probability": 1.0}])
        for probabilities in result["over_under"].values():
            self.assertEqual(probabilities, {"under": 1.0, "over": 0.0})

    def test_zero_rate_for_one_team_gives_no_win_and_no_btts_for_it(self):
        result = markets_for(0.0, 1.5)
        self.assertEqual(result["one_x_two"]["home"], 0.0)
        self.assertEqual(result["btts"]["yes"], 0.0)
        self.assertAlmostEqual(result["one_x_two"]["draw"], math.exp(-1.5), delta=2e-10)

    def test_top_scores_are_sorted_and_have_unique_score_pairs(self):
        result = markets_for(3.4123347446242787, 0.5061909341687736)
        top = result["top_scores"]
        self.assertEqual((top[0]["home_goals"], top[0]["away_goals"]), (3, 0))
        self.assertAlmostEqual(top[0]["probability"], 0.13158596042839257, delta=3e-10)
        self.assertEqual([s["probability"] for s in top], sorted([s["probability"] for s in top], reverse=True))
        self.assertEqual(len({(s["home_goals"], s["away_goals"]) for s in top}), len(top))

    def test_insufficient_goal_cap_raises_instead_of_hiding_missing_mass(self):
        with self.assertRaisesRegex(ValueError, "coverage is insufficient"):
            build_score_matrix(3.4, 0.5, max_goals=2)
        with self.assertRaisesRegex(ValueError, "coverage is insufficient"):
            poisson_goal_probabilities(1000.0)

    def test_invalid_distribution_parameters_are_rejected(self):
        for invalid in [-1.0, float("nan"), float("inf")]:
            with self.subTest(mean=invalid):
                with self.assertRaises(ValueError):
                    poisson_goal_probabilities(invalid)
        for invalid in [0.0, 1.0, -0.1, float("nan")]:
            with self.subTest(tolerance=invalid):
                with self.assertRaises(ValueError):
                    poisson_goal_probabilities(1.0, tail_tolerance=invalid)
        for invalid in [-1, 2.5, True]:
            with self.subTest(cap=invalid):
                with self.assertRaises(ValueError):
                    poisson_goal_probabilities(1.0, max_goals=invalid)

    def test_invalid_matrices_and_integer_goal_lines_are_rejected(self):
        for invalid in [[], [[]], [[0.5], [0.2, 0.3]], [[-0.1, 1.1]], [[float("nan")]], [[0.5]]]:
            with self.subTest(matrix=invalid):
                with self.assertRaises(ValueError):
                    market_probabilities(invalid)
        with self.assertRaisesRegex(ValueError, "goal lines"):
            market_probabilities([[1.0]], lines=(2.0,))

    def test_target_same_day_and_future_results_do_not_change_any_probability(self):
        matches = pd.DataFrame([
            ["before_home", "2023/24", "2023-08-09", "Arsenal", "Team A", 3, 1],
            ["before_away", "2023/24", "2023-08-11", "Team B", "Nottingham Forest", 2, 2],
            ["target", "2023/24", "2023-08-12", "Arsenal", "Nottingham Forest", 99, 88],
            ["same_day", "2023/24", "2023-08-12", "Team C", "Team D", 77, 66],
            ["future", "2023/24", "2023-08-13", "Arsenal", "Nottingham Forest", 55, 44],
        ], columns=COLUMNS)
        matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d")

        def complete_prediction(data):
            analysis = analyze_fixture(data, "Arsenal", "Nottingham Forest", "2023-08-12", 730)
            expected = calculate_expected_goals(analysis)
            return markets_for(expected["lambda_home"], expected["lambda_away"])

        original = complete_prediction(matches)
        for match_id in ["target", "same_day", "future"]:
            with self.subTest(match_id=match_id):
                changed = matches.copy()
                changed.loc[changed["match_id"] == match_id, ["FTHG", "FTAG"]] = [0, 0]
                self.assertEqual(original, complete_prediction(changed))


if __name__ == "__main__":
    unittest.main()
