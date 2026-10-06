"""Tests for the low-score correction's mass, marginals and event effects."""

from math import exp, fsum
import unittest

import numpy as np

from dixon_coles_probabilities import (
    LOW_SCORES, build_dixon_coles_score_matrix, dixon_coles_score_probability, rho_bounds, tau_factors,
)
from score_probabilities import build_score_matrix, market_probabilities


class CorrectionTests(unittest.TestCase):
    def test_four_factors_match_hand_calculation(self) -> None:
        self.assertEqual(tau_factors(2.0, 1.0, -0.1),
                         {(0, 0): 1.2, (0, 1): 0.8, (1, 0): 0.9, (1, 1): 1.1})

    def test_rho_zero_reproduces_independent_poisson(self) -> None:
        original = build_score_matrix(1.8, 1.1)
        corrected = build_dixon_coles_score_matrix(1.8, 1.1, 0)
        np.testing.assert_allclose(corrected["matrix"], original["matrix"], rtol=0, atol=2e-16)

    def test_only_four_cells_change_and_mass_is_preserved(self) -> None:
        original = np.array(build_score_matrix(1.8, 1.1)["matrix"])
        grid = build_dixon_coles_score_matrix(1.8, 1.1, -0.1)
        corrected = np.array(grid["matrix"])
        for home in range(original.shape[0]):
            for away in range(original.shape[1]):
                if (home, away) not in LOW_SCORES:
                    self.assertAlmostEqual(corrected[home, away], original[home, away], places=15)
        self.assertAlmostEqual(float(corrected.sum()), 1.0, places=14)
        self.assertAlmostEqual(grid["mass_change_from_correction"], 0.0, places=14)

    def test_positive_and_negative_rho_have_opposite_directions(self) -> None:
        original = np.array(build_score_matrix(1.8, 1.1)["matrix"])
        for rho in (-0.1, 0.1):
            corrected = np.array(build_dixon_coles_score_matrix(1.8, 1.1, rho)["matrix"])
            difference = corrected - original
            with self.subTest(rho=rho):
                self.assertGreater(-rho * difference[0, 0], 0)
                self.assertGreater(-rho * difference[1, 1], 0)
                self.assertGreater(rho * difference[0, 1], 0)
                self.assertGreater(rho * difference[1, 0], 0)

    def test_equal_absolute_cell_changes_and_derived_market_changes(self) -> None:
        home, away, rho = 1.8, 1.1, -0.1
        original_grid = build_score_matrix(home, away)
        corrected_grid = build_dixon_coles_score_matrix(home, away, rho)
        original = market_probabilities(original_grid["matrix"])
        corrected = market_probabilities(corrected_grid["matrix"])
        delta = -rho * home * away * exp(-home - away) / original_grid["captured_probability"]
        for score, sign in (((0, 0), 1), ((1, 1), 1), ((0, 1), -1), ((1, 0), -1)):
            h, a = score
            self.assertAlmostEqual(corrected_grid["matrix"][h][a] - original_grid["matrix"][h][a], sign * delta, places=14)
        self.assertAlmostEqual(corrected["one_x_two"]["draw"] - original["one_x_two"]["draw"], 2 * delta, places=14)
        for outcome in ("home", "away"):
            self.assertAlmostEqual(corrected["one_x_two"][outcome] - original["one_x_two"][outcome], -delta, places=14)
        self.assertAlmostEqual(corrected["btts"]["yes"] - original["btts"]["yes"], delta, places=14)
        self.assertAlmostEqual(corrected["over_under"][1.5]["over"] - original["over_under"][1.5]["over"], delta, places=14)

    def test_over_2_5_and_3_5_are_unchanged_at_fixed_rates(self) -> None:
        original = market_probabilities(build_score_matrix(1.8, 1.1)["matrix"])
        corrected = market_probabilities(build_dixon_coles_score_matrix(1.8, 1.1, -0.1)["matrix"])
        for line in (2.5, 3.5):
            for outcome in ("over", "under"):
                self.assertAlmostEqual(corrected["over_under"][line][outcome], original["over_under"][line][outcome], places=14)

    def test_marginal_distributions_and_goal_expectations_are_unchanged(self) -> None:
        original = np.array(build_score_matrix(1.8, 1.1)["matrix"])
        corrected = np.array(build_dixon_coles_score_matrix(1.8, 1.1, -0.1)["matrix"])
        np.testing.assert_allclose(corrected.sum(axis=0), original.sum(axis=0), atol=1e-14)
        np.testing.assert_allclose(corrected.sum(axis=1), original.sum(axis=1), atol=1e-14)
        self.assertAlmostEqual(float(np.arange(len(corrected)) @ corrected.sum(axis=1)), 1.8, delta=1e-8)

    def test_swap_teams_transposes_score_distribution(self) -> None:
        first = np.array(build_dixon_coles_score_matrix(1.8, 1.1, -0.1)["matrix"])
        swapped = np.array(build_dixon_coles_score_matrix(1.1, 1.8, -0.1)["matrix"])
        np.testing.assert_allclose(first.T, swapped, rtol=0, atol=2e-16)

    def test_invalid_rho_and_boundary_are_rejected_without_clipping(self) -> None:
        lower, upper = rho_bounds(2.0, 1.0)
        self.assertEqual((lower, upper), (-0.5, 0.5))
        for rho in (-0.6, 0.6, lower, upper, np.nan, np.inf, True):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                build_dixon_coles_score_matrix(2.0, 1.0, rho)

    def test_score_probability_agrees_with_corrected_grid_and_handles_tail(self) -> None:
        grid = build_dixon_coles_score_matrix(1.8, 1.1, -0.1)
        for home, away in (*LOW_SCORES, (4, 2)):
            probability = dixon_coles_score_probability(1.8, 1.1, -0.1, home, away)
            self.assertAlmostEqual(probability / grid["captured_probability"], grid["matrix"][home][away], places=14)
        self.assertGreater(dixon_coles_score_probability(1.8, 1.1, -0.1, 30, 20), 0)

    def test_zero_and_tiny_rates_keep_all_four_cells_available(self) -> None:
        zero = build_dixon_coles_score_matrix(0.0, 0.0, -0.1)
        self.assertEqual(zero["matrix"], [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(dixon_coles_score_probability(0, 0, 0, 1, 0), 0)
        tiny = build_dixon_coles_score_matrix(1e-14, 1e-14, 0.1)
        self.assertEqual((tiny["max_home_goals"], tiny["max_away_goals"]), (1, 1))
        self.assertAlmostEqual(fsum(value for row in tiny["matrix"] for value in row), 1.0, places=14)

    def test_invalid_rates_goal_counts_or_grid_limit_are_rejected(self) -> None:
        for rate in (-1, np.inf, np.nan, True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                build_dixon_coles_score_matrix(rate, 1, 0)
        for goals in (-1, 1.2, True):
            with self.subTest(goals=goals), self.assertRaises(ValueError):
                dixon_coles_score_probability(1, 1, 0, goals, 0)
        with self.assertRaises(ValueError):
            build_dixon_coles_score_matrix(1, 1, 0, max_goals=0)


if __name__ == "__main__":
    unittest.main()
