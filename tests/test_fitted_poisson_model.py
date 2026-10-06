"""Tests of opponent strength, chronology, identifiability and fitting."""

import json
from math import isfinite
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from fitted_poisson_model import (
    _poisson_objective, fit_poisson_model, predict_goals_from_model,
)
from score_probabilities import build_score_matrix, market_probabilities


def opponent_history() -> pd.DataFrame:
    # A and B both score two goals per match, but face different opponents.
    # A plays C, which regularly concedes more goals against D.
    # B plays D, which regularly concedes fewer goals against C.
    patterns = [("A", "C", 2, 1), ("C", "A", 1, 2),
                ("B", "D", 2, 1), ("D", "B", 1, 2),
                ("C", "D", 1, 3), ("D", "C", 3, 1)]
    return pd.DataFrame([
        {"Date": pd.Timestamp("2023-09-01") + pd.Timedelta(days=day),
         "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag}
        for day in range(15) for home, away, hg, ag in patterns
    ])


def equal_team_history() -> pd.DataFrame:
    return pd.DataFrame([
        {"Date": pd.Timestamp("2023-12-01") + pd.Timedelta(days=day),
         "HomeTeam": home, "AwayTeam": away, "FTHG": 3, "FTAG": 2}
        for day in range(3) for home in ("A", "B", "C", "D")
        for away in ("A", "B", "C", "D") if home != away
    ])


class ObjectiveTests(unittest.TestCase):
    def test_analytic_gradient_matches_independent_numerical_gradient(self) -> None:
        rng = np.random.default_rng(42)
        design = rng.normal(size=(11, 8))
        goals = rng.integers(0, 5, size=11).astype(float)
        weights = rng.uniform(0.05, 1.0, size=11)
        parameters = rng.normal(scale=0.2, size=8)
        difference = check_grad(
            lambda x: _poisson_objective(x, design, goals, weights, 1.7)[0],
            lambda x: _poisson_objective(x, design, goals, weights, 1.7)[1],
            parameters,
        )
        self.assertLess(difference, 2e-6)


class FittedPoissonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = opponent_history()
        self.model = fit_poisson_model(self.history, "2024-01-01", 730, None, 1.0)

    def test_equal_goal_average_can_have_different_opponent_adjusted_attack(self) -> None:
        means = {}
        for team in ("A", "B"):
            scored = pd.concat([
                self.history.loc[self.history.HomeTeam == team, "FTHG"],
                self.history.loc[self.history.AwayTeam == team, "FTAG"],
            ])
            means[team] = scored.mean()
        self.assertEqual(means["A"], means["B"])
        self.assertGreater(self.model["teams"]["B"]["attack_factor"],
                           self.model["teams"]["A"]["attack_factor"])
        self.assertGreater(self.model["teams"]["C"]["concede_factor"],
                           self.model["teams"]["D"]["concede_factor"])

    def test_both_log_parameter_groups_sum_to_zero(self) -> None:
        self.assertAlmostEqual(sum(team["attack_log"] for team in self.model["teams"].values()), 0.0, places=12)
        self.assertAlmostEqual(sum(team["concede_log"] for team in self.model["teams"].values()), 0.0, places=12)
        self.assertEqual(self.model["free_parameters"], 2 + 2 * (4 - 1))

    def test_fit_improves_training_objective_without_claiming_predictive_quality(self) -> None:
        optimizer = self.model["optimizer"]
        self.assertTrue(optimizer["success"])
        self.assertLess(optimizer["final_penalized_nll"], optimizer["initial_penalized_nll"])
        self.assertLessEqual(optimizer["gradient_max_abs"], optimizer["gradient_acceptance_tolerance"])

    def test_fit_does_not_modify_input(self) -> None:
        original = self.history.copy(deep=True)
        fit_poisson_model(self.history, "2024-01-01")
        pd.testing.assert_frame_equal(self.history, original)

    def test_current_day_and_future_results_and_team_names_have_no_effect(self) -> None:
        excluded = pd.DataFrame([
            {"Date": pd.Timestamp("2024-01-01"), "HomeTeam": "Future A", "AwayTeam": "Future B", "FTHG": 99, "FTAG": 0},
            {"Date": pd.Timestamp("2024-02-01"), "HomeTeam": "Future C", "AwayTeam": "Future D", "FTHG": 0, "FTAG": 99},
        ])
        changed = pd.concat([self.history, excluded], ignore_index=True)
        refit = fit_poisson_model(changed, "2024-01-01", 730, None, 1.0)
        self.assertEqual(refit, self.model)
        self.assertEqual(set(refit["teams"]), {"A", "B", "C", "D"})

    def test_records_before_window_are_excluded(self) -> None:
        too_old = self.history.iloc[:1].copy()
        too_old["Date"] = pd.Timestamp("2021-12-31")
        too_old["FTHG"] = 99
        refit = fit_poisson_model(pd.concat([self.history, too_old]), "2024-01-01", 730, None, 1.0)
        self.assertEqual(refit, self.model)

    def test_renaming_teams_does_not_change_predicted_goals(self) -> None:
        names = {"A": "Zulu", "B": "Yankee", "C": "Bravo", "D": "Alpha"}
        renamed = self.history.copy()
        for column in ("HomeTeam", "AwayTeam"):
            renamed[column] = renamed[column].replace(names)
        renamed_model = fit_poisson_model(renamed, "2024-01-01", 730, None, 1.0)
        first = predict_goals_from_model(self.model, "A", "B")
        second = predict_goals_from_model(renamed_model, "Zulu", "Yankee")
        for key in ("lambda_home", "lambda_away"):
            self.assertAlmostEqual(first[key], second[key], delta=2e-6)

    def test_larger_l2_penalty_reduces_total_team_effect(self) -> None:
        regularized = fit_poisson_model(self.history, "2024-01-01", 730, None, 1000.0)
        norm = lambda model: sum(team["attack_log"] ** 2 + team["concede_log"] ** 2
                                 for team in model["teams"].values())
        self.assertLess(norm(regularized), norm(self.model) / 10)

    def test_a_team_without_scored_goals_still_has_positive_finite_prediction(self) -> None:
        changed = self.history.copy()
        changed.loc[changed.HomeTeam == "A", "FTHG"] = 0
        changed.loc[changed.AwayTeam == "A", "FTAG"] = 0
        model = fit_poisson_model(changed, "2024-01-01", 730, None, 1.0)
        predicted = predict_goals_from_model(model, "A", "B")
        self.assertTrue(isfinite(predicted["lambda_home"]) and predicted["lambda_home"] > 0)

    def test_unknown_team_is_neutral_and_explicitly_flagged(self) -> None:
        prediction = predict_goals_from_model(self.model, "New Team", "B")
        self.assertTrue(prediction["home_unknown"])
        self.assertFalse(prediction["away_unknown"])
        self.assertEqual(prediction["home_attack_factor"], 1.0)
        self.assertEqual(prediction["home_concede_factor"], 1.0)
        self.assertEqual(prediction["home_history_matches"], 0)
        self.assertNotIn("New Team", self.model["teams"])

    def test_json_round_trip_preserves_prediction(self) -> None:
        restored = json.loads(json.dumps(self.model, allow_nan=False))
        self.assertEqual(predict_goals_from_model(restored, "A", "B"),
                         predict_goals_from_model(self.model, "A", "B"))

    def test_existing_probability_engine_accepts_new_model(self) -> None:
        prediction = predict_goals_from_model(self.model, "A", "B")
        grid = build_score_matrix(prediction["lambda_home"], prediction["lambda_away"])
        markets = market_probabilities(grid["matrix"])
        self.assertAlmostEqual(sum(markets["one_x_two"].values()), 1.0, places=12)
        self.assertAlmostEqual(sum(markets["btts"].values()), 1.0, places=12)
        self.assertLessEqual(grid["omitted_probability"], 2e-10)

    def test_failed_optimizer_is_rejected(self) -> None:
        failure = SimpleNamespace(success=False, message="test failure")
        with patch("fitted_poisson_model.minimize", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                fit_poisson_model(self.history, "2024-01-01")


class BaselineAndValidationTests(unittest.TestCase):
    def test_equal_teams_recover_home_and_away_goal_baselines(self) -> None:
        model = fit_poisson_model(equal_team_history(), "2024-01-01", 730, None, 1.0)
        prediction = predict_goals_from_model(model, "A", "B")
        self.assertAlmostEqual(prediction["lambda_home"], 3.0, places=10)
        self.assertAlmostEqual(prediction["lambda_away"], 2.0, places=10)
        self.assertAlmostEqual(model["home_advantage_factor"], 1.5, places=10)

    def test_recent_high_scoring_games_have_more_effect_with_time_decay(self) -> None:
        old = equal_team_history()
        old["Date"] -= pd.Timedelta(days=365)
        old["FTHG"] = old["FTAG"] = 1
        recent = equal_team_history()
        recent["FTHG"] = recent["FTAG"] = 3
        history = pd.concat([old, recent], ignore_index=True)
        equal = fit_poisson_model(history, "2024-01-01", 730, None)
        weighted = fit_poisson_model(history, "2024-01-01", 730, 30.0)
        first = predict_goals_from_model(equal, "A", "B")
        second = predict_goals_from_model(weighted, "A", "B")
        self.assertAlmostEqual(first["lambda_home"], 2.0, places=10)
        self.assertGreater(second["lambda_home"], first["lambda_home"])

    def test_empty_history_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No previous"):
            fit_poisson_model(equal_team_history(), "2023-01-01")

    def test_invalid_ridge_strength_is_rejected(self) -> None:
        for value in (0.0, -1.0, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit_poisson_model(equal_team_history(), "2024-01-01", ridge_strength=value)

    def test_negative_and_fractional_goals_are_rejected(self) -> None:
        for goal in (-1, 0.5):
            changed = equal_team_history()
            changed["FTHG"] = changed["FTHG"].astype(float)
            changed.loc[0, "FTHG"] = goal
            with self.subTest(goal=goal), self.assertRaisesRegex(ValueError, "non-negative integers"):
                fit_poisson_model(changed, "2024-01-01")

    def test_all_zero_home_goals_are_rejected_as_degenerate_fit(self) -> None:
        changed = equal_team_history()
        changed["FTHG"] = 0
        with self.assertRaisesRegex(ValueError, "positive league mean home and away"):
            fit_poisson_model(changed, "2024-01-01")

    def test_repeated_history_fixture_is_rejected(self) -> None:
        history = equal_team_history()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fit_poisson_model(pd.concat([history, history.iloc[:1]]), "2024-01-01")


if __name__ == "__main__":
    unittest.main()
