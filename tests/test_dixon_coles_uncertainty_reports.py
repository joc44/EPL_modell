"""Validate the Step 15 report chain and detect modified inputs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backtest_models import run_backtest
from dixon_coles_backtest import (
    COMPARISON_MODELS, calibration_models, correction_effects, paired_loss_differences,
    run_dixon_coles_backtest, validate_dixon_coles_comparison,
)
from dixon_coles_uncertainty_reports import load_dixon_coles_uncertainty_inputs, validate_step15_settings
from fitted_backtest import run_fitted_backtest, summarize_models, validate_comparison
from test_dixon_coles_backtest import small_matches
from validation_data import file_sha256


def valid_settings(context: dict) -> dict:
    return {
        "step": 15, "evaluation_role": "development_comparison", "evaluation_seasons": ["2023/24", "2024/25"],
        "models": list(COMPARISON_MODELS), "evaluated_matches": 760, "prediction_rows": 4560,
        "lookback_days": 730, "half_life_days": 180.0, "ridge_strength": 1.0, "reference_prior_matches": 5.0,
        "log_loss_probability_floor": 1e-15, "evaluation_weighting": "equal_weight_per_match",
        "one_x_two_brier_scale": "0_to_2", "binary_brier_scale": "0_to_1",
        "history_rule": "match_date - lookback_days <= Date < match_date", "same_day_results_used": False,
        "fitting_frequency": "once_per_target_matchday", "failed_matches_skipped": 0,
        "same_fixtures_for_all_models": True, "final_test_season_reserved": "2025/26",
        "dc_exact_score_rule": "analytic_untruncated_poisson_product_times_tau", "dc_joint_fit": True,
        "rho_penalized": False, "tau_constraint_floor": 1e-8, "stationarity_acceptance_tolerance": 1e-4,
        "correction_effects_role": "diagnostic_decomposition_not_separate_fitted_model",
        "calibration_bins": 10, "min_bin_matches": 20, "hyperparameters_tuned": False,
        "uncertainty_intervals_computed": False, "inputs": deepcopy(context),
        "source_sha256": deepcopy(context["source_sha256"]), "fits_by_season": deepcopy(context["matchdays_by_season"]),
    }


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = {"input_sha256": {"input.csv": "data-hash"}, "source_sha256": {"model.py": "code-hash"},
                        "matchdays_by_season": {"2023/24": 120, "2024/25": 109}}

    def test_matching_settings_pass_and_changed_evaluation_rules_fail(self) -> None:
        valid = valid_settings(self.context)
        validate_step15_settings(valid, self.context)
        for column, value in (("half_life_days", 90), ("same_day_results_used", True),
                               ("dc_exact_score_rule", "independent_poisson"), ("final_test_season_reserved", "2024/25"),
                               ("failed_matches_skipped", 1)):
            bad = deepcopy(valid)
            bad[column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, column):
                validate_step15_settings(bad, self.context)

    def test_changed_data_code_or_fit_counts_fail(self) -> None:
        for change in ("data", "code", "nested_code", "fits"):
            bad = valid_settings(self.context)
            if change == "data":
                bad["inputs"]["input_sha256"]["input.csv"] = "changed"
            elif change == "code":
                bad["source_sha256"]["model.py"] = "changed"
            elif change == "nested_code":
                bad["inputs"]["source_sha256"]["model.py"] = "changed"
            else:
                bad["fits_by_season"]["2023/24"] -= 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_step15_settings(bad, self.context)

    def test_non_object_settings_are_rejected(self) -> None:
        for bad in (None, [], "settings"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate_step15_settings(bad, self.context)


class SavedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        first = small_matches()
        second = first.loc[first.Season == "2023/24"].copy()
        second["Season"] = "2024/25"
        second["Date"] += pd.DateOffset(years=1)
        second["match_id"] = "EPL|" + second.Season + "|" + second.Date.dt.strftime("%Y-%m-%d") + "|" + second.HomeTeam + "|" + second.AwayTeam
        cls.matches = pd.concat([first, second], ignore_index=True)
        cls.references, cls.tables = {}, {}
        for season in ("2023/24", "2024/25"):
            fixtures = cls.matches.loc[cls.matches.Season == season]
            raw = run_backtest(cls.matches, season, 730, 5.0, 180.0)
            fitted, _ = run_fitted_backtest(cls.matches, season)
            old = validate_comparison(pd.concat([raw, fitted], ignore_index=True), fixtures)
            dc, diagnostics = run_dixon_coles_backtest(cls.matches, season)
            combined = validate_dixon_coles_comparison(pd.concat([old, dc], ignore_index=True), fixtures)
            bins, calibration = calibration_models(combined)
            cls.references[season] = old
            cls.tables[season] = {
                "comparison_predictions": combined, "comparison_summary": summarize_models(combined, COMPARISON_MODELS),
                "calibration_bins": bins, "calibration_summary": calibration,
                "paired_loss_differences": paired_loss_differences(combined),
                "correction_effects": correction_effects(combined), "fitting_diagnostics": diagnostics,
            }

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / "data" / "reports" / "dixon_coles_backtest"
        self.root.mkdir(parents=True)
        (self.project / "input.csv").write_text("checked-prefix-data\n", encoding="utf-8")
        (self.project / "model.py").write_text("# checked-prefix-source\n", encoding="utf-8")
        self.context = {
            "input_sha256": {"input.csv": file_sha256(self.project / "input.csv")},
            "source_sha256": {"model.py": file_sha256(self.project / "model.py")},
            "matchdays_by_season": {season: len(tables["fitting_diagnostics"]) for season, tables in self.tables.items()},
        }
        # Replace the complete 380-match history-validation chain here with a small,
        # already validated fixture; the new report validation performs real calculations.
        for season, tables in self.tables.items():
            slug = season.replace("/", "_")
            for name, table in tables.items():
                table.to_csv(self.root / f"{name}_{slug}.csv", index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")
        pd.concat([tables["comparison_summary"].assign(Season=season) for season, tables in self.tables.items()], ignore_index=True).to_csv(
            self.root / "two_seasons_summary_2023_24_2024_25.csv", index=False, encoding="utf-8-sig"
        )
        self.settings_path = self.root / "dixon_coles_backtest_settings_2023_24_2024_25.json"
        self.settings_path.write_text(json.dumps(valid_settings(self.context)), encoding="utf-8")

    def load(self) -> tuple:
        with patch("dixon_coles_uncertainty_reports.load_dixon_coles_inputs",
                   return_value=(self.matches, self.references, self.context)):
            return load_dixon_coles_uncertainty_inputs(self.project)

    def test_complete_saved_tables_pass_without_any_model_refit(self) -> None:
        with patch("dixon_coles_model.fit_poisson_model", side_effect=AssertionError("No refit")), \
                patch("dixon_coles_backtest.fit_dixon_coles_model", side_effect=AssertionError("No refit")):
            bundles, context = self.load()
        self.assertEqual(set(bundles), {"2023/24", "2024/25"})
        self.assertEqual(len(context["input_sha256"]), 17)
        self.assertEqual(len(bundles["2023/24"]), 18)

    def test_missing_report_has_an_actionable_error(self) -> None:
        self.settings_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run_dixon_coles_backtest.py"):
            self.load()

    def test_changed_summary_calibration_or_correction_table_is_rejected(self) -> None:
        for name, column in (("comparison_summary", "one_x_two_log_loss"),
                             ("calibration_summary", "mean_probability"), ("correction_effects", "total_loss_difference")):
            path = self.root / f"{name}_2023_24.csv"
            original = path.read_bytes()
            table = pd.read_csv(path, encoding="utf-8-sig")
            table.loc[0, column] += 0.1
            table.to_csv(path, index=False, encoding="utf-8-sig")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "saved report differ"):
                self.load()
            path.write_bytes(original)

    def test_changed_dc_exact_score_probability_is_rejected(self) -> None:
        path = self.root / "comparison_predictions_2023_24.csv"
        table = pd.read_csv(path, encoding="utf-8-sig")
        index = table.loc[table.model == "dixon_coles"].index[0]
        table.loc[index, "p_actual_score"] += 0.01
        table.to_csv(path, index=False, encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "p_actual_score"):
            self.load()

    def test_changed_daily_rho_is_rejected(self) -> None:
        path = self.root / "fitting_diagnostics_2023_24.csv"
        table = pd.read_csv(path, encoding="utf-8-sig")
        table.loc[0, "rho"] += 0.01
        table.to_csv(path, index=False, encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "rho differ"):
            self.load()


if __name__ == "__main__":
    unittest.main()
