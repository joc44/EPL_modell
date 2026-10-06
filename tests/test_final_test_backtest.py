"""Check three-model chronology, fitting, neutral teams and markets."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import pandas as pd

from final_test_backtest import FINAL_MODELS, run_final_backtest, validate_final_diagnostics, validate_final_predictions
from final_test_protocol import RUNTIME_SOURCES, SELECTION_SOURCES, current_runtime, fixed_policy
from fitted_poisson_model import fit_poisson_model
from time_weighted_model import calculate_time_weighted_expected_goals


def test_payload() -> dict:
    return {**fixed_policy(), "runtime": current_runtime(), "frozen_context": {
        "input_sha256": {"data.csv": "a" * 64},
        "source_sha256": {name: "b" * 64 for name in (*RUNTIME_SOURCES, *SELECTION_SOURCES)}}}


def small_final_matches() -> pd.DataFrame:
    rows = []
    for home in ("A", "B", "C", "D"):
        for away in ("A", "B", "C", "D"):
            if home != away:
                rows.append({"Season": "2024/25", "Date": pd.Timestamp("2025-05-20"),
                             "HomeTeam": home, "AwayTeam": away, "FTHG": 2, "FTAG": 1})
    for day in range(12):
        for home, away in (("A", "B"), ("C", "D")) if day % 2 == 0 else (("B", "C"), ("D", "A")):
            rows.append({"Season": "2025/26", "Date": pd.Timestamp("2025-08-12") + pd.Timedelta(days=day),
                         "HomeTeam": home, "AwayTeam": away, "FTHG": 1 + day % 4, "FTAG": day % 3})
    frame = pd.DataFrame(rows)
    frame["match_id"] = "EPL|" + frame.Season + "|" + frame.Date.dt.strftime("%Y-%m-%d") + "|" + frame.HomeTeam + "|" + frame.AwayTeam
    return frame


class FinalBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matches, cls.payload = small_final_matches(), test_payload()
        cls.predictions, cls.diagnostics = run_final_backtest(cls.matches, cls.payload)
        cls.fixtures = cls.matches.loc[cls.matches.Season == "2025/26"]

    def test_exactly_three_models_all_fixtures_and_one_fit_per_day(self) -> None:
        with patch("fitted_backtest.fit_poisson_model", wraps=fit_poisson_model) as fit:
            predictions, diagnostics = run_final_backtest(self.matches, self.payload)
        self.assertEqual(set(predictions.model), set(FINAL_MODELS))
        self.assertEqual(len(predictions), 72)
        self.assertEqual((fit.call_count, len(diagnostics)), (12, 12))
        self.assertTrue((predictions.last_history_date < predictions.Date).all())

    def test_same_day_and_future_results_do_not_change_earlier_forecasts(self) -> None:
        changed = self.matches.copy()
        first_date = self.fixtures.Date.min()
        changed.loc[changed.Date >= first_date, "FTHG"] = 8
        repeated, _ = run_final_backtest(changed, self.payload)
        columns = ["match_id", "model", "lambda_home", "lambda_away", "p_home", "p_draw", "p_away", "p_btts"]
        first = self.predictions.loc[self.predictions.Date == first_date, columns].sort_values(["match_id", "model"]).reset_index(drop=True)
        second = repeated.loc[repeated.Date == first_date, columns].sort_values(["match_id", "model"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)
        later = first_date + pd.Timedelta(days=1)
        self.assertNotAlmostEqual(self.predictions.loc[(self.predictions.Date == later) & (self.predictions.model == "fitted_poisson"), "lambda_home"].iloc[0],
                                  repeated.loc[(repeated.Date == later) & (repeated.model == "fitted_poisson"), "lambda_home"].iloc[0])

    def test_reference_lambdas_match_the_original_models(self) -> None:
        first = self.predictions.loc[self.predictions.Date == self.fixtures.Date.min()]
        league = first.loc[first.model == "league_average"].iloc[0]
        self.assertEqual((league.lambda_home, league.lambda_away), (2.0, 1.0))
        weighted = first.loc[first.model == "time_weighted"].iloc[0]
        original = calculate_time_weighted_expected_goals(self.matches, weighted.HomeTeam, weighted.AwayTeam, "2025-08-12", 730, 180.0, 5.0)
        self.assertAlmostEqual(weighted.lambda_home, original["lambda_home"], places=12)
        self.assertAlmostEqual(weighted.lambda_away, original["lambda_away"], places=12)

    def test_failed_fit_stops_without_substitution(self) -> None:
        with patch("fitted_backtest.fit_poisson_model", side_effect=RuntimeError("failed day")), self.assertRaisesRegex(ValueError, "Fit failed"):
            run_final_backtest(self.matches, self.payload)

    def test_unknown_team_is_retained_and_flagged(self) -> None:
        changed = self.matches.copy()
        mask = changed.Season.eq("2025/26") & changed.Date.eq(self.fixtures.Date.min()) & changed.HomeTeam.eq("A")
        changed.loc[mask, "HomeTeam"] = "New Team"
        changed.loc[mask, "match_id"] = "EPL|2025/26|2025-08-12|New Team|B"
        predictions, _ = run_final_backtest(changed, self.payload)
        row = predictions.loc[(predictions.model == "fitted_poisson") & predictions.HomeTeam.eq("New Team")].iloc[0]
        self.assertEqual((row.home_unknown, row.home_all_history_matches), (1, 0))
        self.assertEqual(len(predictions), 72)

    def test_changed_protocol_parameter_is_rejected(self) -> None:
        bad = deepcopy(self.payload)
        bad["selected_parameters"]["half_life_days"] = 90.0
        with self.assertRaisesRegex(ValueError, "test rules"):
            run_final_backtest(self.matches, bad)

    def test_duplicate_or_missing_prediction_is_rejected(self) -> None:
        for bad in (self.predictions.iloc[1:], pd.concat([self.predictions, self.predictions.iloc[:1]])):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                validate_final_predictions(bad, self.fixtures)

    def test_changed_probability_or_loss_is_rejected(self) -> None:
        for column in ("p_btts", "one_x_two_log_loss", "exact_score_log_loss"):
            bad = self.predictions.copy()
            bad.loc[0, column] += 0.01
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_final_predictions(bad, self.fixtures)

    def test_wrong_day_count_or_gradient_is_rejected(self) -> None:
        for column, value in (("fixtures", 1), ("gradient_max_abs", 0.01), ("last_history_date", "2026-06-01")):
            bad = self.diagnostics.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_final_diagnostics(bad, self.matches, self.payload)

    def test_input_matches_are_not_changed(self) -> None:
        original = self.matches.copy(deep=True)
        run_final_backtest(self.matches, self.payload)
        pd.testing.assert_frame_equal(original, self.matches)


if __name__ == "__main__":
    unittest.main()
