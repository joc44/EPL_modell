"""Validate pairing, match weights, block boundaries and interval calculation."""

import unittest

import numpy as np
import pandas as pd

from backtest_models import LOSS_MARKETS, MODEL_NAMES
from uncertainty_analysis import (
    ALL_MODELS, NEW_MODEL, aggregate_matchdays, bootstrap_loss_intervals,
    build_paired_losses, moving_block_indices, resample_daily_means, summarize_sensitivity,
)


def sample_predictions(constant_difference: float | None = None) -> pd.DataFrame:
    records = []
    fixture_index = 0
    for day in range(12):
        for game in range(1 + day % 3):
            common = {"match_id": f"match-{fixture_index:03d}", "Season": "2023/24",
                      "Date": (pd.Timestamp("2023-08-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                      "HomeTeam": f"Home-{game}", "AwayTeam": f"Away-{game}", "FTHG": 2, "FTAG": 1}
            for model_index, model in enumerate(ALL_MODELS):
                row = {**common, "model": model}
                for market_index, market in enumerate(LOSS_MARKETS):
                    new_loss = 1.2 + fixture_index / 100 + market_index / 20
                    difference = (constant_difference if constant_difference is not None
                                  else np.sin(fixture_index) / 10 - 0.04 + model_index / 100 + market_index / 200)
                    row[f"{market}_log_loss"] = new_loss if model == NEW_MODEL else new_loss - difference
                records.append(row)
            fixture_index += 1
    return pd.DataFrame(records)


class PairingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = sample_predictions()
        self.paired = build_paired_losses(self.predictions)

    def test_identifier_pairing_is_independent_of_csv_order(self) -> None:
        shuffled = build_paired_losses(self.predictions.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(self.paired, shuffled)

    def test_manual_difference_and_equal_match_weight(self) -> None:
        new = self.predictions.loc[self.predictions.model == NEW_MODEL].set_index("match_id")
        old = self.predictions.loc[self.predictions.model == "time_weighted"].set_index("match_id")
        rows = self.paired.loc[self.paired.reference_model == "time_weighted"].set_index("match_id")
        manual = new.one_x_two_log_loss - old.loc[new.index, "one_x_two_log_loss"]
        np.testing.assert_allclose(rows.loc[manual.index, "one_x_two_loss_difference"], manual)
        sums, counts, keys = aggregate_matchdays(self.paired)
        column = keys.index(("time_weighted", "one_x_two"))
        self.assertAlmostEqual(sums[:, column].sum() / counts.sum(), manual.mean(), places=12)
        self.assertNotAlmostEqual((sums[:, column] / counts).mean(), manual.mean(), places=6)
        self.assertEqual(counts.tolist(), [1, 2, 3] * 4)

    def test_input_is_not_modified(self) -> None:
        original = self.predictions.copy(deep=True)
        build_paired_losses(self.predictions)
        pd.testing.assert_frame_equal(self.predictions, original)

    def test_missing_fixture_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "match lists"):
            build_paired_losses(self.predictions.drop(index=0))

    def test_duplicate_fixture_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_paired_losses(pd.concat([self.predictions, self.predictions.iloc[:1]]))

    def test_changed_score_or_date_is_rejected(self) -> None:
        for column, value in (("FTHG", 3), ("Date", "2023-08-02")):
            modified = self.predictions.copy()
            modified.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Match data differ"):
                build_paired_losses(modified)

    def test_two_seasons_cannot_be_pooled(self) -> None:
        modified = self.predictions.copy()
        modified.loc[0, "Season"] = "2024/25"
        with self.assertRaisesRegex(ValueError, "one season"):
            build_paired_losses(modified)

    def test_nonfinite_or_negative_loss_is_rejected(self) -> None:
        for value in (np.nan, np.inf, -0.2):
            modified = self.predictions.copy()
            modified.loc[0, "one_x_two_log_loss"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_paired_losses(modified)

    def test_tampered_difference_and_missing_reference_are_rejected(self) -> None:
        modified = self.paired.copy()
        modified.loc[0, "one_x_two_loss_difference"] += 0.1
        with self.assertRaisesRegex(ValueError, "difference"):
            aggregate_matchdays(modified)
        with self.assertRaisesRegex(ValueError, "model set"):
            aggregate_matchdays(self.paired.loc[self.paired.reference_model != MODEL_NAMES[0]])


class BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paired = build_paired_losses(sample_predictions())

    def test_blocks_stay_contiguous_and_inside_season(self) -> None:
        indices = moving_block_indices(11, 3, 30, np.random.default_rng(42))
        self.assertEqual(indices.shape, (30, 11))
        self.assertGreaterEqual(int(indices.min()), 0)
        self.assertLess(int(indices.max()), 11)
        for begin in (0, 3, 6, 9):
            np.testing.assert_array_equal(np.diff(indices[:, begin:min(begin + 3, 11)], axis=1), 1)

    def test_invalid_blocks_or_seeds_are_rejected(self) -> None:
        for length in (0, 12, 13, 1.5, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, length, 20)
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, 3, 20, seed=seed)

    def test_same_seed_is_reproducible_and_changed_seed_changes_interval(self) -> None:
        first = bootstrap_loss_intervals(self.paired, 3, 300, seed=7)
        same = bootstrap_loss_intervals(self.paired, 3, 300, seed=7)
        other = bootstrap_loss_intervals(self.paired, 3, 300, seed=8)
        pd.testing.assert_frame_equal(first, same)
        self.assertFalse(np.allclose(first.ci_lower, other.ci_lower, atol=1e-10))

    def test_resampling_uses_match_counts_and_same_indices_for_columns(self) -> None:
        counts = np.array([1, 3, 2, 1, 4])
        sums = np.array([[0.2, 0.4], [0.9, 1.8], [-0.4, -0.8], [0.1, 0.2], [1.6, 3.2]])
        draws, actual_counts = resample_daily_means(sums, counts, 2, 12, seed=17)
        indices = moving_block_indices(5, 2, 12, np.random.default_rng(17))
        expected_counts = counts[indices].sum(axis=1)
        expected = sums[indices].sum(axis=1) / expected_counts[:, None]
        np.testing.assert_allclose(draws, expected)
        np.testing.assert_array_equal(actual_counts, expected_counts)
        np.testing.assert_allclose(draws[:, 1], 2 * draws[:, 0])
        self.assertGreater(len(set(actual_counts)), 1)

    def test_batch_size_does_not_change_result(self) -> None:
        sums, counts, _ = aggregate_matchdays(self.paired)
        first, first_counts = resample_daily_means(sums, counts, 3, 31, seed=14, batch_size=31)
        second, second_counts = resample_daily_means(sums, counts, 3, 31, seed=14, batch_size=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_counts, second_counts)

    def test_percentile_bounds_and_standard_error_match_direct_calculation(self) -> None:
        sums, counts, keys = aggregate_matchdays(self.paired)
        draws, _ = resample_daily_means(sums, counts, 3, 250, seed=51)
        table = bootstrap_loss_intervals(self.paired, 3, 250, seed=51)
        self.assertEqual(len(table), 24)
        self.assertEqual(list(zip(table.reference_model, table.market)), list(keys))
        expected = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
        np.testing.assert_allclose(table.ci_lower, expected[0], atol=1e-14)
        np.testing.assert_allclose(table.ci_upper, expected[1], atol=1e-14)
        np.testing.assert_allclose(table.bootstrap_standard_error, draws.std(axis=0, ddof=1))

    def test_identical_models_have_zero_difference_and_zero_interval(self) -> None:
        paired = build_paired_losses(sample_predictions(constant_difference=0.0))
        table = bootstrap_loss_intervals(paired, 3, 30)
        self.assertTrue((table.mean_loss_difference == 0).all())
        self.assertTrue((table.ci_lower == 0).all() and (table.ci_upper == 0).all())
        self.assertTrue((table.interval_direction == "includes_zero").all())
        self.assertTrue(table.degenerate_distribution.all())

    def test_constant_signed_difference_keeps_direction(self) -> None:
        for difference, direction in ((-0.25, "new_lower"), (0.25, "new_higher")):
            table = bootstrap_loss_intervals(build_paired_losses(sample_predictions(difference)), 3, 30)
            with self.subTest(difference=difference):
                np.testing.assert_allclose(table.ci_lower, difference)
                np.testing.assert_allclose(table.ci_upper, difference)
                self.assertTrue((table.interval_direction == direction).all())

    def test_invalid_confidence_and_resample_count_are_rejected(self) -> None:
        for confidence in (0, 1, np.nan, True):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, 3, 30, confidence)
        for count in (0, 1, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, 3, count)

    def test_point_estimate_does_not_change_with_block_length(self) -> None:
        a = bootstrap_loss_intervals(self.paired, 3, 100)
        b = bootstrap_loss_intervals(self.paired, 5, 100)
        np.testing.assert_array_equal(a.mean_loss_difference, b.mean_loss_difference)

    def test_sensitivity_reports_all_lengths_and_rejects_missing_length(self) -> None:
        intervals = pd.concat([bootstrap_loss_intervals(self.paired, length, 100) for length in (3, 5, 10)])
        summary = summarize_sensitivity(intervals, (3, 5, 10))
        self.assertEqual(len(summary), 24)
        self.assertTrue({"ci_lower_block_3", "ci_lower_block_5", "ci_lower_block_10"}.issubset(summary.columns))
        with self.assertRaisesRegex(ValueError, "A block length is missing"):
            summarize_sensitivity(intervals.iloc[1:], (3, 5, 10))


if __name__ == "__main__":
    unittest.main()
