"""Joint fitting, derivatives, chronology and the complete prediction domain."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from dixon_coles_model import (
    MINIMUM_TAU, _dixon_coles_objective, _stationarity_residual, _tau_and_jacobian,
    fit_dixon_coles_model, predict_dixon_coles_goals,
)
from dixon_coles_probabilities import build_dixon_coles_score_matrix
from fitted_poisson_model import _poisson_objective


def sample_history() -> pd.DataFrame:
    patterns = ((0, 0), (1, 1), (1, 0), (0, 1), (2, 1), (1, 2), (3, 2), (2, 3), (2, 2))
    return pd.DataFrame([
        {"Date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day), "HomeTeam": home,
         "AwayTeam": away, "FTHG": hg, "FTAG": ag}
        for day, (hg, ag) in enumerate(patterns)
        for home in ("A", "B", "C", "D") for away in ("A", "B", "C", "D") if home != away
    ])


class DerivativeTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.home = rng.normal(scale=0.3, size=(8, 6))
        self.away = rng.normal(scale=0.3, size=(8, 6))
        self.hg = np.array([0, 0, 1, 1, 2, 3, 0, 1], dtype=float)
        self.ag = np.array([0, 1, 0, 1, 1, 2, 2, 0], dtype=float)
        self.weights = rng.uniform(0.1, 1.0, 8)
        self.parameters = np.r_[rng.normal(scale=0.1, size=6), -0.08]

    def test_joint_analytic_gradient_matches_numerical_gradient(self) -> None:
        args = (self.home, self.away, self.hg, self.ag, self.weights, 1.3)
        error = check_grad(lambda x: _dixon_coles_objective(x, *args)[0],
                           lambda x: _dixon_coles_objective(x, *args)[1], self.parameters)
        self.assertLess(error, 3e-6)

    def test_constraint_jacobian_matches_central_differences(self) -> None:
        _, jacobian = _tau_and_jacobian(self.parameters, self.home, self.away)
        epsilon = 1e-6
        for column in range(len(self.parameters)):
            plus, minus = self.parameters.copy(), self.parameters.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            numerical = (_tau_and_jacobian(plus, self.home, self.away)[0]
                         - _tau_and_jacobian(minus, self.home, self.away)[0]) / (2 * epsilon)
            np.testing.assert_allclose(jacobian[:, :, column], numerical, rtol=1e-7, atol=2e-9)

    def test_rho_zero_has_same_objective_and_team_gradient_as_poisson(self) -> None:
        parameters = self.parameters.copy()
        parameters[-1] = 0
        dc_value, dc_gradient = _dixon_coles_objective(parameters, self.home, self.away, self.hg, self.ag, self.weights, 1.3)
        old_value, old_gradient = _poisson_objective(parameters[:-1], np.vstack((self.home, self.away)),
                                                     np.r_[self.hg, self.ag], np.tile(self.weights, 2), 1.3)
        self.assertAlmostEqual(dc_value, old_value, places=14)
        np.testing.assert_array_equal(dc_gradient[:-1], old_gradient)

    def test_ridge_penalizes_team_effects_and_leaves_rho_unpenalized(self) -> None:
        args = (self.home, self.away, self.hg, self.ag, self.weights)
        weak, weak_gradient = _dixon_coles_objective(self.parameters, *args, 1)
        strong, strong_gradient = _dixon_coles_objective(self.parameters, *args, 3)
        self.assertAlmostEqual(strong - weak, np.dot(self.parameters[2:-1], self.parameters[2:-1]), places=12)
        np.testing.assert_allclose(strong_gradient[2:-1] - weak_gradient[2:-1], 2 * self.parameters[2:-1], atol=1e-14)
        self.assertEqual(strong_gradient[-1], weak_gradient[-1])

    def test_active_constraint_can_have_nonzero_raw_gradient(self) -> None:
        residual, active = _stationarity_residual(np.array([2.0, 0.0]), np.array([0.0, 1.0]), np.eye(2))
        self.assertEqual(residual, 0.0)
        self.assertEqual(active, 1)


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = sample_history()
        cls.model = fit_dixon_coles_model(cls.history, "2023-08-12", 730, None, 1.0)

    def test_successful_joint_fit_has_one_extra_parameter_and_improves_training_goal(self) -> None:
        model = self.model
        self.assertEqual(model["free_parameters"], model["poisson_start_model"]["free_parameters"] + 1)
        self.assertTrue(model["optimizer"]["success"])
        self.assertLessEqual(model["optimizer"]["final_penalized_nll"], model["optimizer"]["initial_penalized_nll"])
        self.assertLessEqual(model["optimizer"]["stationarity_max_abs"], 1e-4)
        self.assertAlmostEqual(model["attack_log_sum"], 0, places=12)
        self.assertAlmostEqual(model["concede_log_sum"], 0, places=12)

    def test_every_known_and_neutral_pair_has_valid_probability_distribution(self) -> None:
        names = [*self.model["teams"], "Unknown A", "Unknown B"]
        for home in names:
            for away in names:
                if home != away:
                    prediction = predict_dixon_coles_goals(self.model, home, away)
                    grid = build_dixon_coles_score_matrix(prediction["lambda_home"], prediction["lambda_away"], prediction["rho"])
                    self.assertGreater(grid["minimum_tau"], 0)
        self.assertGreaterEqual(self.model["optimizer"]["minimum_tau_all_pairs"], MINIMUM_TAU - 1e-9)

    def test_unknown_team_is_neutral_and_explicitly_flagged(self) -> None:
        prediction = predict_dixon_coles_goals(self.model, "Unknown A", "B")
        self.assertTrue(prediction["home_unknown"])
        self.assertEqual(prediction["home_attack_factor"], 1)
        self.assertEqual(prediction["home_history_matches"], 0)
        self.assertNotIn("Unknown A", self.model["teams"])

    def test_current_day_and_future_goals_and_team_names_are_excluded(self) -> None:
        future = pd.DataFrame([
            {"Date": pd.Timestamp(date), "HomeTeam": "Future A", "AwayTeam": "Future B", "FTHG": 99, "FTAG": 99}
            for date in ("2023-08-12", "2023-08-13")
        ])
        changed = fit_dixon_coles_model(pd.concat([self.history, future]), "2023-08-12", 730, None, 1.0)
        self.assertEqual(set(changed["teams"]), set(self.model["teams"]))
        for key in ("rho", "intercept_log", "home_advantage_log"):
            self.assertAlmostEqual(changed[key], self.model[key], places=12)

    def test_records_before_window_are_excluded(self) -> None:
        old = self.history.iloc[:1].copy()
        old["Date"] = pd.Timestamp("2020-01-01")
        old["FTHG"] = 99
        changed = fit_dixon_coles_model(pd.concat([old, self.history]), "2023-08-12", 730, None, 1.0)
        self.assertAlmostEqual(changed["rho"], self.model["rho"], places=12)

    def test_input_is_not_modified_and_json_roundtrip_preserves_prediction(self) -> None:
        original = self.history.copy(deep=True)
        model = fit_dixon_coles_model(self.history, "2023-08-12", 730, None, 1.0)
        pd.testing.assert_frame_equal(self.history, original)
        restored = json.loads(json.dumps(model, allow_nan=False))
        self.assertEqual(predict_dixon_coles_goals(model, "A", "B"), predict_dixon_coles_goals(restored, "A", "B"))

    def test_failed_optimizer_is_rejected(self) -> None:
        failed = SimpleNamespace(success=False, message="test failure")
        with patch("dixon_coles_model.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "fit failed"):
                fit_dixon_coles_model(self.history, "2023-08-12")

    def test_history_without_low_scores_cannot_identify_rho(self) -> None:
        changed = self.history.copy()
        changed[["FTHG", "FTAG"]] = 2
        with self.assertRaisesRegex(ValueError, "No.*history"):
            fit_dixon_coles_model(changed, "2023-08-12")

    def test_real_boundary_fit_satisfies_kkt_with_nonzero_raw_gradient(self) -> None:
        rows = pd.DataFrame([
            {"Date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day), "HomeTeam": home,
             "AwayTeam": away, "FTHG": 1, "FTAG": 1}
            for day in range(4) for home, away in (("A", "B"), ("B", "A"))
        ])
        model = fit_dixon_coles_model(rows, "2023-08-12", 730, None, 1.0)
        self.assertGreater(model["optimizer"]["active_tau_constraints"], 0)
        self.assertGreater(model["optimizer"]["gradient_max_abs"], 1)
        self.assertLess(model["optimizer"]["stationarity_max_abs"], 1e-4)
        # The symmetric sample containing only 1–1 scores has a known boundary solution.
        self.assertAlmostEqual(model["neutral_away_goal_base"], 1 / np.sqrt(2), delta=2e-6)
        self.assertAlmostEqual(model["rho"], -np.sqrt(2), delta=2e-6)
        self.assertGreater(model["optimizer"]["minimum_tau_all_pairs"], 0)

    def test_zero_effective_low_score_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero total time weight"):
            fit_dixon_coles_model(self.history, "2023-01-10", 730, 0.001, 1.0)


if __name__ == "__main__":
    unittest.main()
