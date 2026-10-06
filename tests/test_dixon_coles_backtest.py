"""DC-probability evaluation, matchday chronology and comparable reports."""

from math import log
import unittest
from unittest.mock import patch

import pandas as pd

from backtest_models import LOSS_MARKETS, run_backtest
from dixon_coles_backtest import (
    COMPARISON_MODELS, NEW_MODEL, PREVIOUS_MODELS, calibration_models, correction_effects,
    evaluate_dixon_coles_forecast, forecast_from_rates, paired_loss_differences,
    run_dixon_coles_backtest, validate_dixon_coles_comparison,
)
from dixon_coles_model import fit_dixon_coles_model
from dixon_coles_probabilities import dixon_coles_score_probability, tau_factors
from evaluation_metrics import one_x_two_metrics, poisson_score_probability
from fitted_backtest import calibration_models as previous_calibration, run_fitted_backtest, summarize_models


def small_matches() -> pd.DataFrame:
    patterns = ((0, 0), (1, 1), (1, 0), (0, 1), (2, 1), (1, 2), (3, 2), (2, 3), (2, 2))
    rows = [
        {"Season": "2022/23", "Date": pd.Timestamp("2023-05-01") + pd.Timedelta(days=day),
         "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag}
        for day, (hg, ag) in enumerate(patterns)
        for home in ("A", "B", "C", "D") for away in ("A", "B", "C", "D") if home != away
    ]
    for date, home, away, hg, ag in (("2023-08-12", "A", "B", 0, 0),
                                    ("2023-08-12", "C", "D", 1, 0),
                                    ("2023-08-13", "B", "C", 3, 2)):
        rows.append({"Season": "2023/24", "Date": pd.Timestamp(date),
                     "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag})
    frame = pd.DataFrame(rows)
    frame["match_id"] = "EPL|" + frame.Season + "|" + frame.Date.dt.strftime("%Y-%m-%d") + "|" + frame.HomeTeam + "|" + frame.AwayTeam
    return frame


class ExactScoreTests(unittest.TestCase):
    def test_all_four_low_scores_use_tau_in_exact_score_loss(self) -> None:
        forecast = forecast_from_rates(1.7, 0.9, -0.12)
        for score, tau in tau_factors(1.7, 0.9, -0.12).items():
            with self.subTest(score=score):
                record = evaluate_dixon_coles_forecast(forecast, *score)
                probability = poisson_score_probability(1.7, 0.9, *score) * tau
                self.assertAlmostEqual(record["p_actual_score"], probability, places=14)
                self.assertAlmostEqual(record["exact_score_log_loss"], -log(probability), places=14)
                self.assertAlmostEqual(record["exact_score_log_loss"] - record["uncorrected_exact_score_log_loss"], -log(tau), places=14)

    def test_higher_scores_keep_independent_probability_at_fixed_lambdas(self) -> None:
        forecast = forecast_from_rates(1.7, 0.9, -0.12)
        record = evaluate_dixon_coles_forecast(forecast, 2, 1)
        self.assertAlmostEqual(record["p_actual_score"], poisson_score_probability(1.7, 0.9, 2, 1), places=14)
        self.assertEqual(record["exact_score_log_loss"], record["uncorrected_exact_score_log_loss"])

    def test_score_outside_display_grid_is_not_lost(self) -> None:
        record = evaluate_dixon_coles_forecast(forecast_from_rates(1.7, 0.9, -0.12), 40, 0)
        self.assertGreater(record["p_actual_score"], 0)
        self.assertEqual(record["exact_score_log_loss_clipped"], 1)
        self.assertAlmostEqual(record["exact_score_log_loss"], -log(1e-15), places=12)

    def test_rho_zero_matches_independent_losses_on_every_market(self) -> None:
        record = evaluate_dixon_coles_forecast(forecast_from_rates(1.7, 0.9, 0.0), 1, 1)
        for market in LOSS_MARKETS:
            self.assertAlmostEqual(record[f"{market}_log_loss"], record[f"uncorrected_{market}_log_loss"], places=12)

    def test_over_two_and_three_point_five_correction_is_zero_at_fixed_rates(self) -> None:
        for score in ((0, 0), (1, 1), (3, 2)):
            record = evaluate_dixon_coles_forecast(forecast_from_rates(1.7, 0.9, -0.12), *score)
            for market in ("over_2_5", "over_3_5"):
                self.assertAlmostEqual(record[f"{market}_log_loss"], record[f"uncorrected_{market}_log_loss"], places=12)


class ChronologicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matches = small_matches()
        cls.new, cls.diagnostics = run_dixon_coles_backtest(cls.matches, "2023/24")

    def test_each_day_is_fitted_once_and_all_matches_are_predicted(self) -> None:
        calls = []
        with patch("dixon_coles_backtest.fit_dixon_coles_model", wraps=fit_dixon_coles_model) as mocked:
            forecasts, diagnostics = run_dixon_coles_backtest(self.matches, "2023/24", progress=lambda n, total: calls.append((n, total)))
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(len(forecasts), 3)
        self.assertEqual(diagnostics.fixtures.tolist(), [2, 1])
        self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])
        self.assertTrue(forecasts.groupby("Date").rho.nunique().eq(1).all())

    def test_same_day_results_do_not_change_that_days_forecasts(self) -> None:
        changed = self.matches.copy()
        mask = (changed.Date == pd.Timestamp("2023-08-12")) & changed.HomeTeam.eq("A")
        changed.loc[mask, "FTHG"] = 9
        refit, _ = run_dixon_coles_backtest(changed, "2023/24")
        columns = ["match_id", "lambda_home", "lambda_away", "rho", "p_home", "p_draw", "p_away", "p_btts"]
        pd.testing.assert_frame_equal(self.new.loc[self.new.Date == "2023-08-12", columns].reset_index(drop=True),
                                      refit.loc[refit.Date == "2023-08-12", columns].reset_index(drop=True))
        self.assertNotAlmostEqual(self.new.iloc[-1].lambda_home, refit.iloc[-1].lambda_home, places=5)

    def test_future_results_do_not_change_earlier_forecasts(self) -> None:
        changed = self.matches.copy()
        changed.loc[changed.Date == pd.Timestamp("2023-08-13"), "FTAG"] = 8
        refit, _ = run_dixon_coles_backtest(changed, "2023/24")
        pd.testing.assert_frame_equal(self.new.loc[self.new.Date == "2023-08-12"].reset_index(drop=True),
                                      refit.loc[refit.Date == "2023-08-12"].reset_index(drop=True))

    def test_unknown_team_is_flagged_without_dropping_fixture(self) -> None:
        changed = self.matches.copy()
        mask = (changed.Date == pd.Timestamp("2023-08-12")) & changed.HomeTeam.eq("A")
        changed.loc[mask, "HomeTeam"] = "New Team"
        changed.loc[mask, "match_id"] = "EPL|2023/24|2023-08-12|New Team|B"
        forecasts, _ = run_dixon_coles_backtest(changed, "2023/24")
        row = forecasts.loc[forecasts.HomeTeam == "New Team"].iloc[0]
        self.assertEqual(len(forecasts), 3)
        self.assertEqual(row.home_unknown, 1)
        self.assertEqual(row.home_all_history_matches, 0)

    def test_failed_fit_stops_instead_of_skipping_a_day(self) -> None:
        with patch("dixon_coles_backtest.fit_dixon_coles_model", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(ValueError, "2023-08-12"):
                run_dixon_coles_backtest(self.matches, "2023/24")

    def test_input_and_history_count_semantics_are_preserved(self) -> None:
        original = self.matches.copy(deep=True)
        forecasts, _ = run_dixon_coles_backtest(self.matches, "2023/24")
        pd.testing.assert_frame_equal(original, self.matches)
        self.assertEqual(forecasts.iloc[0].home_history_matches, 27)
        self.assertEqual(forecasts.iloc[0].home_all_history_matches, 54)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matches = small_matches()
        cls.fixtures = cls.matches.loc[cls.matches.Season == "2023/24"]
        raw = run_backtest(cls.matches, "2023/24", 730, 5.0, 180.0)
        fitted, _ = run_fitted_backtest(cls.matches, "2023/24")
        cls.old = pd.concat([raw, fitted], ignore_index=True)
        cls.new, _ = run_dixon_coles_backtest(cls.matches, "2023/24")
        cls.combined = validate_dixon_coles_comparison(pd.concat([cls.old, cls.new], ignore_index=True), cls.fixtures)

    def test_all_old_rows_and_summaries_are_unchanged(self) -> None:
        preserved = self.combined.loc[self.combined.model != NEW_MODEL, self.old.columns].copy()
        for column in ("Date", "history_start", "last_history_date"):
            preserved[column] = preserved[column].dt.strftime("%Y-%m-%d")
        pd.testing.assert_frame_equal(preserved.reset_index(drop=True), self.old.reset_index(drop=True), check_dtype=False)
        before = summarize_models(self.old)
        after = summarize_models(self.combined, COMPARISON_MODELS)
        pd.testing.assert_frame_equal(before, after.loc[after.model != NEW_MODEL].reset_index(drop=True))

    def test_missing_and_duplicate_model_fixture_are_rejected(self) -> None:
        index = self.combined.loc[self.combined.model == NEW_MODEL].index[0]
        for bad in (self.combined.drop(index), pd.concat([self.combined, self.combined.iloc[[index]]], ignore_index=True)):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                validate_dixon_coles_comparison(bad, self.fixtures)

    def test_independent_exact_probability_cannot_replace_dc_low_score(self) -> None:
        bad = self.combined.copy()
        index = bad.loc[bad.model == NEW_MODEL].index[0]
        row = bad.loc[index]
        bad.loc[index, "p_actual_score"] = poisson_score_probability(row.lambda_home, row.lambda_away, int(row.FTHG), int(row.FTAG))
        with self.assertRaisesRegex(ValueError, "p_actual_score"):
            validate_dixon_coles_comparison(bad, self.fixtures)

    def test_tampered_loss_rho_and_uncorrected_loss_are_rejected(self) -> None:
        index = self.combined.loc[self.combined.model == NEW_MODEL].index[0]
        for column in ("exact_score_log_loss", "rho", "uncorrected_btts_log_loss"):
            bad = self.combined.copy()
            bad.loc[index, column] += 0.01
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_dixon_coles_comparison(bad, self.fixtures)

    def test_changed_probabilities_with_matching_metrics_still_fail_rate_check(self) -> None:
        bad = self.combined.copy()
        index = bad.loc[bad.model == NEW_MODEL].index[0]
        bad.loc[index, "p_home"] += 0.01
        bad.loc[index, "p_draw"] -= 0.01
        row = bad.loc[index]
        metrics = one_x_two_metrics({"home": row.p_home, "draw": row.p_draw, "away": row.p_away}, row.actual_one_x_two)
        for field, key in (("one_x_two_log_loss", "log_loss"), ("one_x_two_brier", "brier"),
                           ("one_x_two_predicted", "predicted"), ("one_x_two_hit", "hit"),
                           ("one_x_two_log_loss_clipped", "clipped")):
            bad.loc[index, field] = metrics[key] if isinstance(metrics[key], str) else int(metrics[key]) if key in ("hit", "clipped") else metrics[key]
        with self.assertRaisesRegex(ValueError, "DC probability"):
            validate_dixon_coles_comparison(bad, self.fixtures)

    def test_same_day_history_is_rejected(self) -> None:
        bad = self.combined.copy()
        index = bad.loc[bad.model == NEW_MODEL].index[0]
        bad.loc[index, "last_history_date"] = bad.loc[index, "Date"]
        with self.assertRaisesRegex(ValueError, "history boundary"):
            validate_dixon_coles_comparison(bad, self.fixtures)

    def test_calibration_covers_all_six_models_and_preserves_old_calibration(self) -> None:
        bins, summary = calibration_models(self.combined)
        self.assertEqual(len(bins), 420)
        self.assertEqual(len(summary), 42)
        self.assertTrue(bins.groupby(["model", "event"]).matches.sum().eq(3).all())
        old_bins, old_summary = previous_calibration(self.old)
        pd.testing.assert_frame_equal(old_bins, bins.loc[bins.model != NEW_MODEL].reset_index(drop=True))
        pd.testing.assert_frame_equal(old_summary, summary.loc[summary.model != NEW_MODEL].reset_index(drop=True))

    def test_pairing_uses_identifiers_and_mean_equals_summary_difference(self) -> None:
        paired = paired_loss_differences(self.combined.sample(frac=1, random_state=42))
        summary = summarize_models(self.combined, COMPARISON_MODELS).set_index("model")
        self.assertEqual(len(paired), len(PREVIOUS_MODELS) * len(LOSS_MARKETS))
        for row in paired.to_dict("records"):
            column = row["market"] + "_log_loss"
            expected = summary.loc[NEW_MODEL, column] - summary.loc[row["reference_model"], column]
            self.assertAlmostEqual(row["mean_loss_difference"], expected, places=12)

    def test_correction_effects_add_up_and_goal_total_correction_is_zero(self) -> None:
        effects = correction_effects(self.combined)
        for row in effects.to_dict("records"):
            self.assertAlmostEqual(row["lambda_refit_loss_difference"] + row["tau_correction_loss_difference"], row["total_loss_difference"], places=12)
            if row["market"] in ("over_2_5", "over_3_5"):
                self.assertAlmostEqual(row["tau_correction_loss_difference"], 0, places=12)


if __name__ == "__main__":
    unittest.main()
