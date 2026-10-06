"""Validate development evidence, reports and the selection rationale."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backtest_models import LOSS_MARKETS
from dixon_coles_backtest import COMPARISON_MODELS, PREVIOUS_MODELS
from dixon_coles_uncertainty import bootstrap_loss_intervals, build_paired_losses, summarize_sensitivity
from model_selection_reports import (
    BLOCK_LENGTHS, STEP16_SOURCES, load_selection_inputs, selection_evidence, validate_step16_settings,
)
from test_dixon_coles_uncertainty import sample_predictions
from validation_data import file_sha256


def valid_settings(project: Path, context: dict) -> dict:
    sources = {**context["source_sha256"], **{name: file_sha256(project / name) for name in STEP16_SOURCES}}
    return {
        "step": 16, "evaluation_role": "exploratory_development_diagnostics",
        "evaluation_seasons": ["2023/24", "2024/25"], "new_model": "dixon_coles",
        "reference_models": list(PREVIOUS_MODELS), "focus_reference": "fitted_poisson", "markets": list(LOSS_MARKETS),
        "evaluated_matches": 760, "paired_match_rows": 3800, "interval_rows": 180,
        "block_lengths_matchdays": [3, 5, 10], "primary_block_length_matchdays": 5,
        "n_resamples_per_season_and_length": 5000, "confidence_level": 0.95, "base_seed": 42,
        "seed_rule": "base_seed + 1000 * sorted_season_index + block_length",
        "bootstrap_method": "non_circular_overlapping_moving_blocks_of_matchdays", "interval_method": "percentile",
        "quantile_method": "linear", "pairing": "match_id", "difference_sign": "dixon_coles_minus_reference",
        "resampling_unit": "complete_matchday", "statistic_weighting": "equal_weight_per_resampled_match",
        "resampled_matchday_count": "same_as_original_season",
        "resampled_match_count": "varies_with_number_of_matches_on_selected_days", "last_sampled_block_truncated": True,
        "blocks_cross_season_boundary": False, "same_resample_indices_for_all_comparisons": True,
        "intervals_are_simultaneous": False, "multiple_comparison_adjustment": False,
        "bootstrap_negative_fraction_is_reported_as_probability": False,
        "models_refitted_in_bootstrap": False, "model_parameters_changed": False,
        "conditioning": "fixed_saved_forecasts; excludes model_selection_and_refitting_uncertainty",
        "assumption": "approximately stable loss-difference process with short-range dependence",
        "final_test_season_reserved": "2025/26", "source_sha256": sources,
        "source_context": {**deepcopy(context), "source_sha256": sources},
    }


class SavedSelectionReportsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        first = sample_predictions()
        second = first.copy()
        second["Season"] = "2024/25"
        second["Date"] = (pd.to_datetime(second.Date) + pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d")
        second["match_id"] = "next-" + second.match_id
        cls.predictions = {"2023/24": first, "2024/25": second}
        cls.tables = {}
        for index, (season, predictions) in enumerate(sorted(cls.predictions.items())):
            paired = build_paired_losses(predictions)
            intervals = pd.concat([bootstrap_loss_intervals(paired, length, 5000, 0.95, 42 + 1000 * index + length)
                                   for length in BLOCK_LENGTHS], ignore_index=True)
            cls.tables[season] = {"paired_match_losses": paired, "bootstrap_intervals": intervals,
                                  "block_sensitivity": summarize_sensitivity(intervals, BLOCK_LENGTHS)}

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        for name in (*STEP16_SOURCES, "prefix_source.py", "prefix_input.csv"):
            (self.project / name).write_text(f"unchanged test fixture: {name}\n", encoding="utf-8")
        self.context = {"input_sha256": {"prefix_input.csv": file_sha256(self.project / "prefix_input.csv")},
                        "source_sha256": {"prefix_source.py": file_sha256(self.project / "prefix_source.py")}}
        self.root = self.project / "data" / "reports" / "dixon_coles_uncertainty"
        self.root.mkdir(parents=True)
        # Replace the already validated complete Step 15 history chain with a small
        # fixture. Recalculation of the new reports and file validation remain real.
        for season, tables in self.tables.items():
            for name, table in tables.items():
                table.to_csv(self.root / f"{name}_{season.replace('/', '_')}.csv", index=False,
                             date_format="%Y-%m-%d", encoding="utf-8-sig")
        pd.concat([tables["bootstrap_intervals"] for tables in self.tables.values()], ignore_index=True).to_csv(
            self.root / "bootstrap_intervals_2023_24_2024_25.csv", index=False, encoding="utf-8-sig"
        )
        (self.root / "uncertainty_dixon_coles_vs_poisson_2023_24_2024_25.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        self.settings = valid_settings(self.project, self.context)
        self.settings_path = self.root / "dixon_coles_uncertainty_settings_2023_24_2024_25.json"
        self.settings_path.write_text(json.dumps(self.settings), encoding="utf-8")

    def load(self):
        with patch("model_selection_reports.load_dixon_coles_uncertainty_inputs",
                   return_value=(self.predictions, self.context)):
            return load_selection_inputs(self.project)

    def test_complete_report_chain_is_recomputed_without_refitting(self) -> None:
        with patch("dixon_coles_backtest.fit_dixon_coles_model", side_effect=AssertionError("No refit")), \
                patch("fitted_backtest.fit_poisson_model", side_effect=AssertionError("No refit")):
            predictions, intervals, context = self.load()
        self.assertEqual(set(predictions), {"2023/24", "2024/25"})
        self.assertEqual(len(intervals), 180)
        self.assertEqual(len(context["input_sha256"]), 10)
        self.assertEqual(len(context["source_sha256"]), 5)

    def test_changed_settings_rules_and_boolean_type_are_rejected(self) -> None:
        for key, value in (("base_seed", 43), ("primary_block_length_matchdays", 3), ("pairing", "row_order"),
                           ("models_refitted_in_bootstrap", 0), ("confidence_level", 0.90)):
            bad = deepcopy(self.settings)
            bad[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_step16_settings(bad, self.context, self.project)

    def test_changed_source_or_input_hash_metadata_is_rejected(self) -> None:
        for change in ("input", "source", "actual_source"):
            bad = deepcopy(self.settings)
            if change == "input":
                bad["source_context"]["input_sha256"]["prefix_input.csv"] = "f" * 64
            elif change == "source":
                bad["source_sha256"]["prefix_source.py"] = "f" * 64
            else:
                (self.project / STEP16_SOURCES[0]).write_text("changed source")
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "fingerprints"):
                validate_step16_settings(bad, self.context, self.project)

    def test_missing_settings_gives_the_previous_runner_name(self) -> None:
        self.settings_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "analyze_dixon_coles_uncertainty.py"):
            self.load()

    def test_modified_paired_loss_is_rejected(self) -> None:
        path = self.root / "paired_match_losses_2023_24.csv"
        table = pd.read_csv(path)
        table.loc[0, "one_x_two_loss_difference"] += 0.1
        table.to_csv(path, index=False, encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "saved report differ"):
            self.load()

    def test_modified_bootstrap_bound_is_rejected(self) -> None:
        path = self.root / "bootstrap_intervals_2023_24.csv"
        table = pd.read_csv(path)
        table.loc[0, "ci_upper"] += 0.1
        table.to_csv(path, index=False, encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "saved report differ"):
            self.load()

    def test_modified_sensitivity_or_combined_table_is_rejected(self) -> None:
        for filename, column in (("block_sensitivity_2023_24.csv", "mean_loss_difference"),
                                  ("bootstrap_intervals_2023_24_2024_25.csv", "ci_lower")):
            path = self.root / filename
            before = path.read_bytes()
            table = pd.read_csv(path)
            table.loc[0, column] += 0.1
            table.to_csv(path, index=False, encoding="utf-8-sig")
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "saved report differ"):
                self.load()
            path.write_bytes(before)

    def test_invalid_png_and_changed_prefix_input_are_rejected(self) -> None:
        figure = self.root / "uncertainty_dixon_coles_vs_poisson_2023_24_2024_25.png"
        original = figure.read_bytes()
        figure.write_bytes(b"not a PNG")
        with self.assertRaisesRegex(ValueError, "PNG"):
            self.load()
        figure.write_bytes(original)
        (self.project / "prefix_input.csv").write_text("changed input")
        with self.assertRaisesRegex(ValueError, "prefix_input.csv"):
            self.load()


class SelectionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = {"2023/24": pd.DataFrame(), "2024/25": pd.DataFrame()}
        self.intervals = pd.DataFrame([
            {"Season": season, "new_model": "dixon_coles", "reference_model": "fitted_poisson",
             "market": market, "block_length_matchdays": length, "ci_lower": -0.005, "ci_upper": 0.003}
            for season in self.predictions for market in LOSS_MARKETS for length in BLOCK_LENGTHS
        ])
        # Selection-decision fixture: DC ranks first and Poisson second
        # by point estimate. With uncertain small DC gains, Poisson stays selected.
        losses = {"league_average": 1.08, "raw": 0.99, "smoothed": 0.994,
                  "time_weighted": 0.997, "fitted_poisson": 0.976, "dixon_coles": 0.975}
        self.summary = pd.DataFrame([{"model": model, "one_x_two_log_loss": losses[model],
                                      **{f"{market}_log_loss": 0.7 for market in LOSS_MARKETS if market != "one_x_two"}}
                                     for model in COMPARISON_MODELS])

    def evidence(self, predictions=None, intervals=None):
        with patch("model_selection_reports.summarize_models", return_value=self.summary.copy()):
            return selection_evidence(self.predictions if predictions is None else predictions,
                                      self.intervals if intervals is None else intervals)

    def test_simpler_model_can_be_chosen_despite_second_development_point_rank(self) -> None:
        summary, focus = self.evidence()
        self.assertEqual((len(summary), len(focus)), (12, 36))
        self.assertEqual(summary.loc[summary.selected_for_final_test, "model"].tolist(), ["fitted_poisson"] * 2)
        self.assertTrue(summary.loc[summary.model == "fitted_poisson", "development_one_x_two_rank"].eq(2).all())
        self.assertFalse(summary.loc[summary.model == "dixon_coles", "selected_for_final_test"].any())
        self.assertEqual(set(summary.loc[summary.final_reference, "model"]), {"league_average", "time_weighted"})
        self.assertEqual(set(focus.market), set(LOSS_MARKETS))

    def test_final_season_cannot_be_added_to_model_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "Both development seasons"):
            self.evidence({**self.predictions, "2025/26": pd.DataFrame()})

    def test_missing_or_duplicate_uncertainty_cell_is_rejected(self) -> None:
        for bad in (self.intervals.iloc[1:], pd.concat([self.intervals, self.intervals.iloc[:1]])):
            with self.subTest(rows=len(bad)), self.assertRaisesRegex(ValueError, "Incomplete"):
                self.evidence(intervals=bad)

    def test_changed_primary_uncertainty_requires_review_of_selection_reason(self) -> None:
        bad = self.intervals.copy()
        bad.loc[bad.market == "one_x_two", "ci_upper"] = -0.001
        with self.assertRaisesRegex(ValueError, "review the selection rationale"):
            self.evidence(intervals=bad)

    def test_changed_primary_performance_requires_review_of_selection_reason(self) -> None:
        self.summary.loc[self.summary.model == "fitted_poisson", "one_x_two_log_loss"] = 1.1
        with self.assertRaisesRegex(ValueError, "review the selection rationale"):
            self.evidence()


if __name__ == "__main__":
    unittest.main()
