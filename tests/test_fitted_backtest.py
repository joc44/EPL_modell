"""Chronological fitting, comparable reports and paired evaluation."""

from math import log
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from backtest_models import MODEL_NAMES, run_backtest, summarize_predictions
from fitted_backtest import (
    COMPARISON_MODELS, NEW_MODEL, calibration_models, evaluate_forecast,
    paired_loss_differences, run_fitted_backtest, summarize_models, validate_comparison,
)
from fitted_poisson_model import _poisson_hessian, _poisson_objective, _refine_poisson_fit, fit_poisson_model
from score_probabilities import build_score_matrix, market_probabilities


def small_matches() -> pd.DataFrame:
    rows = []
    for home in ("A", "B", "C", "D"):
        for away in ("A", "B", "C", "D"):
            if home != away:
                rows.append({"Season": "2022/23", "Date": pd.Timestamp("2023-05-20"),
                             "HomeTeam": home, "AwayTeam": away, "FTHG": 2, "FTAG": 1})
    for date, home, away, hg, ag in (("2023-08-12", "A", "B", 2, 1),
                                    ("2023-08-12", "C", "D", 0, 0),
                                    ("2023-08-13", "B", "C", 3, 2)):
        rows.append({"Season": "2023/24", "Date": pd.Timestamp(date),
                     "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag})
    frame = pd.DataFrame(rows)
    frame["match_id"] = "EPL|" + frame.Season + "|" + frame.Date.dt.strftime("%Y-%m-%d") + "|" + frame.HomeTeam + "|" + frame.AwayTeam
    return frame


class FittedBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matches = small_matches()
        cls.fixtures = cls.matches.loc[cls.matches.Season == "2023/24"]
        cls.old = run_backtest(cls.matches, "2023/24", 730, 5.0, 180.0)
        cls.new, cls.diagnostics = run_fitted_backtest(cls.matches, "2023/24")
        cls.combined = validate_comparison(pd.concat([cls.old, cls.new], ignore_index=True), cls.fixtures)

    def test_every_fixture_is_predicted_and_each_day_is_fitted_once(self) -> None:
        with patch("fitted_backtest.fit_poisson_model", wraps=fit_poisson_model) as mocked:
            predictions, diagnostics = run_fitted_backtest(self.matches, "2023/24")
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(diagnostics.fixtures.tolist(), [2, 1])

    def test_same_day_score_changes_do_not_change_same_day_forecasts(self) -> None:
        changed = self.matches.copy(deep=True)
        mask = (changed.Date == pd.Timestamp("2023-08-12")) & (changed.HomeTeam == "A")
        changed.loc[mask, "FTHG"] = 9
        refit, _ = run_fitted_backtest(changed, "2023/24")
        columns = ["match_id", "lambda_home", "lambda_away", "p_home", "p_draw", "p_away", "p_btts"]
        first = self.new.loc[self.new.Date == "2023-08-12", columns].reset_index(drop=True)
        second = refit.loc[refit.Date == "2023-08-12", columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)
        next_original = self.new.loc[self.new.Date == "2023-08-13", "lambda_home"].iloc[0]
        next_changed = refit.loc[refit.Date == "2023-08-13", "lambda_home"].iloc[0]
        self.assertNotAlmostEqual(next_original, next_changed, places=5)

    def test_future_results_do_not_change_earlier_forecasts(self) -> None:
        changed = self.matches.copy()
        changed.loc[changed.Date == pd.Timestamp("2023-08-13"), "FTAG"] = 8
        new, _ = run_fitted_backtest(changed, "2023/24")
        original_first = self.new.loc[self.new.Date == "2023-08-12"].reset_index(drop=True)
        changed_first = new.loc[new.Date == "2023-08-12"].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_first, changed_first)

    def test_complete_history_and_venue_history_counts_are_distinct(self) -> None:
        first = self.new.iloc[0]
        self.assertEqual(first.home_history_matches, 3)
        self.assertEqual(first.home_all_history_matches, 6)
        self.assertEqual(first.away_history_matches, 3)
        self.assertEqual(first.away_all_history_matches, 6)

    def test_unknown_team_is_recorded_without_dropping_fixture(self) -> None:
        changed = self.matches.copy()
        mask = (changed.Date == pd.Timestamp("2023-08-12")) & (changed.HomeTeam == "A")
        changed.loc[mask, "HomeTeam"] = "New Team"
        changed.loc[mask, "match_id"] = "EPL|2023/24|2023-08-12|New Team|B"
        predicted, _ = run_fitted_backtest(changed, "2023/24")
        row = predicted.loc[predicted.HomeTeam == "New Team"].iloc[0]
        self.assertEqual(row.home_unknown, 1)
        self.assertEqual(row.home_all_history_matches, 0)
        self.assertEqual(len(predicted), 3)

    def test_progress_reaches_all_fixtures(self) -> None:
        calls = []
        run_fitted_backtest(self.matches, "2023/24", progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])

    def test_failed_day_fit_stops_instead_of_skipping_matches(self) -> None:
        with patch("fitted_backtest.fit_poisson_model", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(ValueError, "2023-08-12"):
                run_fitted_backtest(self.matches, "2023/24")

    def test_input_table_is_not_modified(self) -> None:
        original = self.matches.copy(deep=True)
        run_fitted_backtest(self.matches, "2023/24")
        pd.testing.assert_frame_equal(original, self.matches)

    def test_missing_model_fixture_is_rejected(self) -> None:
        incomplete = self.combined.drop(self.combined.loc[self.combined.model == NEW_MODEL].index[0])
        with self.assertRaisesRegex(ValueError, "Different match list"):
            validate_comparison(incomplete, self.fixtures)

    def test_duplicate_model_fixture_is_rejected(self) -> None:
        doubled = pd.concat([self.combined, self.combined.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_comparison(doubled, self.fixtures)

    def test_forecast_score_must_match_fixture_score(self) -> None:
        changed = self.combined.copy()
        changed.loc[0, "FTHG"] += 1
        with self.assertRaisesRegex(ValueError, "result"):
            validate_comparison(changed, self.fixtures)

    def test_tampered_saved_loss_and_brier_are_rejected(self) -> None:
        for column in ("one_x_two_log_loss", "btts_brier", "exact_score_log_loss"):
            changed = self.combined.copy()
            changed.loc[0, column] += 0.1
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "metric differs"):
                validate_comparison(changed, self.fixtures)

    def test_same_day_history_boundary_is_rejected(self) -> None:
        changed = self.combined.copy()
        changed.loc[0, "last_history_date"] = changed.loc[0, "Date"]
        with self.assertRaisesRegex(ValueError, "history boundary"):
            validate_comparison(changed, self.fixtures)

    def test_old_summary_is_unchanged_under_new_summarizer(self) -> None:
        old_summary = summarize_predictions(self.old)
        new_summary = summarize_models(self.old, MODEL_NAMES)
        pd.testing.assert_frame_equal(old_summary, new_summary)

    def test_each_match_has_equal_weight_in_log_loss_summary(self) -> None:
        summary = summarize_models(self.combined).set_index("model")
        manual_sum = 0.0
        for row in self.new.to_dict("records"):
            key = "p_" + {"home": "home", "draw": "draw", "away": "away"}[row["actual_one_x_two"]]
            manual_sum += -log(row[key])
        self.assertAlmostEqual(summary.loc[NEW_MODEL, "one_x_two_log_loss"], manual_sum / 3, places=12)

    def test_calibration_bin_counts_cover_each_event_for_all_five_models(self) -> None:
        bins, summary = calibration_models(self.combined)
        self.assertEqual(len(bins), 350)
        self.assertEqual(len(summary), 35)
        self.assertTrue(bins.groupby(["model", "event"]).matches.sum().eq(3).all())

    def test_paired_comparison_joins_by_id_even_if_rows_are_shuffled(self) -> None:
        original = paired_loss_differences(self.combined)
        shuffled = paired_loss_differences(self.combined.sample(frac=1, random_state=42))
        columns = ["reference_model", "market"]
        pd.testing.assert_frame_equal(original.sort_values(columns).reset_index(drop=True),
                                      shuffled.sort_values(columns).reset_index(drop=True), atol=1e-12)

    def test_paired_mean_equals_new_season_mean_minus_reference_mean(self) -> None:
        summary = summarize_models(self.combined).set_index("model")
        paired = paired_loss_differences(self.combined)
        for row in paired.to_dict("records"):
            key = row["market"] + "_log_loss"
            expected = summary.loc[NEW_MODEL, key] - summary.loc[row["reference_model"], key]
            self.assertAlmostEqual(row["mean_loss_difference"], expected, places=12)

    def test_actual_exact_score_outside_grid_is_still_evaluated(self) -> None:
        grid = build_score_matrix(2.0, 1.0)
        forecast = {"lambda_home": 2.0, "lambda_away": 1.0,
                    "markets": market_probabilities(grid["matrix"], top_n=1),
                    "omitted_probability": grid["omitted_probability"]}
        self.assertLess(grid["max_home_goals"], 40)
        record = evaluate_forecast(forecast, 40, 0)
        self.assertGreater(record["p_actual_score"], 0)
        self.assertEqual(record["exact_score_log_loss_clipped"], 1)
        self.assertAlmostEqual(record["exact_score_log_loss"], -log(1e-15), places=12)


class NumericalRefinementTests(unittest.TestCase):
    def test_hessian_matches_independent_gradient_differences(self) -> None:
        rng = np.random.default_rng(37)
        design = rng.normal(size=(20, 6))
        goals = rng.integers(0, 4, size=20)
        weights = rng.uniform(0.1, 1.0, size=20)
        parameters = rng.normal(scale=0.1, size=6)
        hessian = _poisson_hessian(parameters, design, weights, 1.0)
        epsilon = 1e-5
        numeric = np.empty_like(hessian)
        for column in range(len(parameters)):
            delta = np.zeros(len(parameters))
            delta[column] = epsilon
            plus = _poisson_objective(parameters + delta, design, goals, weights, 1.0)[1]
            minus = _poisson_objective(parameters - delta, design, goals, weights, 1.0)[1]
            numeric[:, column] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(hessian, numeric, rtol=1e-8, atol=1e-8)

    def test_refinement_reduces_remaining_gradient_without_changing_objective(self) -> None:
        rng = np.random.default_rng(9)
        design = rng.normal(size=(30, 8))
        design[:, 0] = 1.0
        design[:, 1] = np.arange(30) % 2
        goals = rng.integers(0, 5, size=30)
        weights = rng.uniform(0.1, 1.0, size=30)
        args = (design, goals, weights, 1.0)
        result = minimize(_poisson_objective, np.zeros(8), args=args, method="L-BFGS-B", jac=True)
        near_solution = result.x.copy()
        near_solution[2] += 0.001
        old_value, old_gradient = _poisson_objective(near_solution, *args)
        refined, steps = _refine_poisson_fit(near_solution, *args)
        new_value, new_gradient = _poisson_objective(refined, *args)
        self.assertGreater(steps, 0)
        self.assertGreater(np.max(np.abs(old_gradient)), 1e-4)
        self.assertLessEqual(np.max(np.abs(new_gradient)), 1e-4)
        self.assertLess(new_value, old_value)


if __name__ == "__main__":
    unittest.main()
