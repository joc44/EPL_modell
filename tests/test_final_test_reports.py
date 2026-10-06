"""Match weights, shared samples, complete reporting and validated reruns."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest_models import LOSS_MARKETS
from final_test_backtest import REFERENCES, run_final_backtest
from final_test_reports import loss_intervals, paired_losses, primary_assessment, report_bundle
from run_final_test import SETTINGS_FILENAME, load_cached_run, save_new_run
from test_final_test_backtest import small_final_matches, test_payload
from uncertainty_analysis import moving_block_indices


class FinalReportsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matches, cls.payload = small_final_matches(), test_payload()
        cls.protocol = {"protocol_id": "test-protocol", "payload": cls.payload}
        cls.predictions, cls.diagnostics = run_final_backtest(cls.matches, cls.payload)
        cls.tables, cls.assessment, cls.note = report_bundle(cls.predictions, cls.diagnostics, cls.protocol)

    def test_all_markets_references_and_calibration_events_are_reported(self) -> None:
        self.assertEqual(len(self.tables["final_predictions"]), 72)
        self.assertEqual(len(self.tables["paired_match_losses"]), 48)
        self.assertEqual(len(self.tables["bootstrap_intervals"]), 36)
        self.assertEqual(len(self.tables["block_sensitivity"]), 12)
        self.assertEqual((len(self.tables["calibration_bins"]), len(self.tables["calibration_summary"])), (210, 21))
        self.assertEqual(set(self.tables["bootstrap_intervals"].market), set(LOSS_MARKETS))
        self.assertEqual(set(self.tables["bootstrap_intervals"].reference_model), set(REFERENCES))
        self.assertIn("historical season", self.note)

    def test_pairing_is_independent_of_csv_row_order(self) -> None:
        first = paired_losses(self.predictions)
        shuffled = paired_losses(self.predictions.sample(frac=1, random_state=4))
        pd.testing.assert_frame_equal(first, shuffled)

    def test_missing_model_and_changed_fixture_result_are_rejected(self) -> None:
        bad = self.predictions.copy()
        bad.loc[0, "FTHG"] += 1
        for invalid in (self.predictions.loc[self.predictions.model != "league_average"], bad):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                paired_losses(invalid)

    def test_every_bootstrap_bound_matches_manual_same_block_calculation(self) -> None:
        paired = self.tables["paired_match_losses"]
        base = paired.loc[paired.reference_model == "league_average"]
        dates = sorted(base.Date.unique())
        counts = base.groupby("Date").size().reindex(dates).to_numpy()
        keys, sums = [], []
        for reference in REFERENCES:
            rows = paired.loc[paired.reference_model == reference]
            for market in LOSS_MARKETS:
                keys.append((reference, market))
                sums.append(rows.groupby("Date")[f"{market}_loss_difference"].sum().reindex(dates).to_numpy())
        sums = np.column_stack(sums)
        for length in (3, 5, 10):
            indices = moving_block_indices(len(dates), length, 5000, np.random.default_rng(42 + length))
            draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
            bounds = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
            rows = self.tables["bootstrap_intervals"].loc[self.tables["bootstrap_intervals"].block_length_matchdays == length]
            self.assertEqual(list(zip(rows.reference_model, rows.market)), keys)
            np.testing.assert_allclose(rows.ci_lower, bounds[0], atol=1e-14)
            np.testing.assert_allclose(rows.ci_upper, bounds[1], atol=1e-14)
            np.testing.assert_allclose(rows.mean_loss_difference, sums.sum(axis=0) / counts.sum(), atol=1e-14)

    def test_favorable_and_uncertain_primary_outcomes_follow_the_fixed_rules(self) -> None:
        intervals = self.tables["bootstrap_intervals"].copy()
        mask = intervals.reference_model.eq("league_average") & intervals.market.eq("one_x_two")
        intervals.loc[mask, "mean_loss_difference"] = -0.02
        intervals.loc[mask, "ci_lower"] = -0.03
        intervals.loc[mask, "ci_upper"] = -0.01
        assessment = primary_assessment(intervals, self.payload)
        self.assertTrue(assessment["observed_point_improvement"])
        self.assertTrue(assessment["primary_interval_support"])
        self.assertTrue(assessment["all_three_block_lengths_support"])
        intervals.loc[mask & intervals.block_length_matchdays.eq(10), "ci_upper"] = 0.01
        assessment = primary_assessment(intervals, self.payload)
        self.assertTrue(assessment["primary_interval_support"])
        self.assertFalse(assessment["all_three_block_lengths_support"])
        intervals.loc[mask & intervals.block_length_matchdays.eq(5), "ci_upper"] = 0.0
        self.assertFalse(primary_assessment(intervals, self.payload)["primary_interval_support"])

    def test_invalid_primary_bounds_or_missing_length_are_rejected(self) -> None:
        bad = self.tables["bootstrap_intervals"].copy()
        mask = bad.reference_model.eq("league_average") & bad.market.eq("one_x_two")
        bad.loc[mask, "ci_upper"] = np.nan
        for invalid in (bad, self.tables["bootstrap_intervals"].loc[self.tables["bootstrap_intervals"].block_length_matchdays != 10]):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                primary_assessment(invalid, self.payload)

    def test_cached_run_recomputes_reports_without_model_refit_or_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            context = {"checked-test-input": "hash"}
            save_new_run(output, self.tables, self.assessment, self.note, context)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with patch("fitted_backtest.fit_poisson_model", side_effect=AssertionError("No refit")):
                tables, assessment = load_cached_run(output, context, self.matches, self.protocol)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
            self.assertEqual(len(tables["final_predictions"]), 72)
            self.assertAlmostEqual(assessment["mean_loss_difference"], self.assessment["mean_loss_difference"], places=12)

    def test_changed_run_context_is_rejected_before_model_refit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_new_run(output, self.tables, self.assessment, self.note, {"context": "original"})
            with self.assertRaisesRegex(ValueError, "different data"):
                load_cached_run(output, {"context": "changed"}, self.matches, self.protocol)

    def test_modified_saved_report_or_missing_file_is_rejected(self) -> None:
        for change in ("changed", "missing"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                save_new_run(output, self.tables, self.assessment, self.note, {})
                path = output / "final_summary_2025_26.csv"
                if change == "changed":
                    path.write_bytes(path.read_bytes() + b" ")
                else:
                    path.unlink()
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "final_summary"):
                    load_cached_run(output, {}, self.matches, self.protocol)

    def test_existing_result_cannot_be_replaced_with_different_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_new_run(output, self.tables, self.assessment, self.note, {})
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "will not be overwritten"):
                save_new_run(output, self.tables, self.assessment, self.note + "changed", {})
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
