"""Validate the provenance and completeness of the previous run."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from uncertainty_reports import STEP12_SOURCES, validate_fit_diagnostics, validate_step12_settings


def valid_settings() -> dict:
    return {
        "step": 12, "evaluation_role": "development_comparison",
        "evaluation_seasons": ["2023/24", "2024/25"],
        "models": ["league_average", "raw", "smoothed", "time_weighted", "fitted_poisson"],
        "evaluated_matches": 760, "prediction_rows": 3800,
        "lookback_days": 730, "half_life_days": 180.0, "ridge_strength": 1.0,
        "reference_prior_matches": 5.0, "log_loss_probability_floor": 1e-15,
        "evaluation_weighting": "equal_weight_per_match", "one_x_two_brier_scale": "0_to_2",
        "binary_brier_scale": "0_to_1", "history_rule": "match_date - lookback_days <= Date < match_date",
        "same_day_results_used": False, "failed_matches_skipped": 0,
        "same_fixtures_for_all_models": True, "fitting_frequency": "once_per_target_matchday",
        "final_test_season_reserved": "2025/26", "inputs": {"input_sha256": {"data.csv": "old-hash"}},
        "source_sha256": {name: "code-hash" for name in STEP12_SOURCES},
    }


class CheckpointTests(unittest.TestCase):
    def test_matching_settings_pass_and_changed_parameters_are_rejected(self) -> None:
        settings = valid_settings()
        context = {"input_sha256": {"data.csv": "old-hash"}}
        with patch("uncertainty_reports.file_sha256", return_value="code-hash"):
            validate_step12_settings(settings, context, Path("example-project"))
            for key, value in (("ridge_strength", 2), ("same_day_results_used", True),
                               ("evaluation_role", "final_test"), ("evaluated_matches", 759)):
                modified = deepcopy(settings)
                modified[key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    validate_step12_settings(modified, context, Path("example-project"))

    def test_changed_data_or_code_is_rejected(self) -> None:
        settings = valid_settings()
        with self.assertRaisesRegex(ValueError, "inputs"):
            validate_step12_settings(settings, {"input_sha256": {"data.csv": "new-hash"}}, Path("example-project"))
        with patch("uncertainty_reports.file_sha256", return_value="changed-code"):
            with self.assertRaisesRegex(ValueError, "source code changed"):
                validate_step12_settings(settings, settings["inputs"], Path("example-project"))

    def test_missing_code_identity_is_rejected(self) -> None:
        settings = valid_settings()
        settings["source_sha256"].pop(STEP12_SOURCES[0])
        with self.assertRaisesRegex(ValueError, "source-code fingerprint"):
            validate_step12_settings(settings, settings["inputs"], Path("example-project"))

    def test_failed_fit_wrong_gradient_or_day_counts_are_rejected(self) -> None:
        fixtures = pd.DataFrame({"Date": pd.to_datetime(["2023-08-12", "2023-08-12", "2023-08-13"])})
        diagnostics = pd.DataFrame({
            "Season": ["2023/24"] * 2, "fit_date": ["2023-08-12", "2023-08-13"], "fixtures": [2, 1],
            "history_start": ["2021-08-12", "2021-08-13"],
            "last_history_date": ["2023-08-11", "2023-08-12"],
            "success": [True, True], "gradient_max_abs": [1e-6, 5e-5],
        })
        validate_fit_diagnostics(diagnostics, fixtures, "2023/24")
        for column, value in (("success", False), ("gradient_max_abs", 0.0002), ("fixtures", 1),
                              ("last_history_date", "2023-08-12")):
            modified = diagnostics.copy()
            modified.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_fit_diagnostics(modified, fixtures, "2023/24")


if __name__ == "__main__":
    unittest.main()
