"""Six-model pairing, match weights and mathematical validation of the DC bootstrap."""

import unittest

import numpy as np
import pandas as pd

from backtest_models import LOSS_MARKETS
from dixon_coles_backtest import COMPARISON_MODELS, NEW_MODEL, PREVIOUS_MODELS
from dixon_coles_uncertainty import aggregate_matchdays, bootstrap_loss_intervals, build_paired_losses, summarize_sensitivity
from uncertainty_analysis import moving_block_indices


def sample_predictions(constant_difference: float | None = None) -> pd.DataFrame:
    records, fixture_index = [], 0
    for day in range(12):
        for game in range(1 + day % 3):
            common = {"match_id": f"match-{fixture_index:03d}", "Season": "2023/24",
                      "Date": (pd.Timestamp("2023-08-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                      "HomeTeam": f"Home-{game}", "AwayTeam": f"Away-{game}", "FTHG": 2, "FTAG": 1}
            for model_index, model in enumerate(COMPARISON_MODELS):
                row = {**common, "model": model}
                for market_index, market in enumerate(LOSS_MARKETS):
                    new_loss = 1.2 + fixture_index / 100 + market_index / 20
                    difference = (constant_difference if constant_difference is not None else
                                  np.sin(fixture_index) / 10 - 0.04 + model_index / 100 + market_index / 200)
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
        self.assertEqual(len(self.paired), 24 * 5)
        self.assertEqual(set(self.paired.reference_model), set(PREVIOUS_MODELS))

    def test_primary_poisson_pair_has_manual_difference_and_equal_match_weight(self) -> None:
        new = self.predictions.loc[self.predictions.model == NEW_MODEL].set_index("match_id")
        old = self.predictions.loc[self.predictions.model == "fitted_poisson"].set_index("match_id")
        manual = new.one_x_two_log_loss - old.loc[new.index, "one_x_two_log_loss"]
        sums, counts, keys = aggregate_matchdays(self.paired)
        column = keys.index(("fitted_poisson", "one_x_two"))
        self.assertEqual(sums.shape, (12, 30))
        self.assertEqual(counts.tolist(), [1, 2, 3] * 4)
        self.assertAlmostEqual(sums[:, column].sum() / counts.sum(), manual.mean(), places=12)
        self.assertNotAlmostEqual((sums[:, column] / counts).mean(), manual.mean(), places=6)

    def test_input_table_is_not_modified(self) -> None:
        original = self.predictions.copy(deep=True)
        build_paired_losses(self.predictions)
        pd.testing.assert_frame_equal(original, self.predictions)

    def test_missing_fixture_duplicate_fixture_and_missing_model_are_rejected(self) -> None:
        for bad in (self.predictions.drop(index=0), pd.concat([self.predictions, self.predictions.iloc[:1]]),
                    self.predictions.loc[self.predictions.model != "fitted_poisson"]):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                build_paired_losses(bad)

    def test_changed_score_date_or_team_is_rejected(self) -> None:
        for column, value in (("FTHG", 3), ("Date", "2023-08-02"), ("HomeTeam", "Changed")):
            bad = self.predictions.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Match data differ"):
                build_paired_losses(bad)

    def test_seasons_cannot_be_pooled(self) -> None:
        bad = self.predictions.copy()
        bad.loc[0, "Season"] = "2024/25"
        with self.assertRaisesRegex(ValueError, "one season"):
            build_paired_losses(bad)

    def test_nonfinite_negative_and_missing_loss_are_rejected(self) -> None:
        for value in (np.nan, np.inf, -0.2):
            bad = self.predictions.copy()
            bad.loc[0, "one_x_two_log_loss"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_paired_losses(bad)
        with self.assertRaises(ValueError):
            build_paired_losses(self.predictions.drop(columns=["btts_log_loss"]))

    def test_tampered_difference_or_missing_reference_is_rejected(self) -> None:
        bad = self.paired.copy()
        bad.loc[0, "one_x_two_loss_difference"] += 0.1
        with self.assertRaisesRegex(ValueError, "difference"):
            aggregate_matchdays(bad)
        with self.assertRaisesRegex(ValueError, "model set"):
            aggregate_matchdays(self.paired.loc[self.paired.reference_model != PREVIOUS_MODELS[0]])

    def test_dc_loss_cannot_change_between_references_even_if_difference_is_updated(self) -> None:
        bad = self.paired.copy()
        index = bad.loc[bad.reference_model == "fitted_poisson"].index[0]
        bad.loc[index, "new_one_x_two_log_loss"] += 0.1
        bad.loc[index, "one_x_two_loss_difference"] += 0.1
        with self.assertRaisesRegex(ValueError, "DC loss"):
            aggregate_matchdays(bad)


class BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = sample_predictions()
        self.paired = build_paired_losses(self.predictions)

    def test_all_thirty_bounds_match_direct_same_block_calculation(self) -> None:
        sums, counts, keys = aggregate_matchdays(self.paired)
        # The daily match counts 1–2–3 are periodic: with complete three-day blocks,
        # every block total would be the same. Five-day blocks also vary the denominator.
        indices = moving_block_indices(len(counts), 5, 250, np.random.default_rng(51))
        sampled_counts = counts[indices].sum(axis=1)
        draws = sums[indices].sum(axis=1) / sampled_counts[:, None]
        table = bootstrap_loss_intervals(self.paired, 5, 250, seed=51)
        bounds = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
        self.assertEqual(len(table), 30)
        self.assertEqual(list(zip(table.reference_model, table.market)), list(keys))
        np.testing.assert_allclose(table.ci_lower, bounds[0], atol=1e-14)
        np.testing.assert_allclose(table.ci_upper, bounds[1], atol=1e-14)
        np.testing.assert_allclose(table.bootstrap_standard_error, draws.std(axis=0, ddof=1))
        self.assertTrue(table.resampled_matches_min.eq(sampled_counts.min()).all())
        self.assertTrue(table.resampled_matches_max.eq(sampled_counts.max()).all())
        self.assertGreater(len(set(sampled_counts)), 1)

    def test_same_seed_is_reproducible_and_changed_seed_changes_intervals(self) -> None:
        first = bootstrap_loss_intervals(self.paired, 3, 300, seed=7)
        same = bootstrap_loss_intervals(self.paired, 3, 300, seed=7)
        other = bootstrap_loss_intervals(self.paired, 3, 300, seed=8)
        pd.testing.assert_frame_equal(first, same)
        self.assertFalse(np.allclose(first.ci_lower, other.ci_lower, atol=1e-10))

    def test_identical_models_have_zero_difference_and_zero_interval(self) -> None:
        table = bootstrap_loss_intervals(build_paired_losses(sample_predictions(0.0)), 3, 30)
        self.assertTrue(table.mean_loss_difference.eq(0).all())
        self.assertTrue(table.ci_lower.eq(0).all() and table.ci_upper.eq(0).all())
        self.assertTrue(table.interval_direction.eq("includes_zero").all())
        self.assertTrue(table.degenerate_distribution.all())

    def test_constant_positive_or_negative_difference_keeps_direction(self) -> None:
        for difference, direction in ((-0.25, "new_lower"), (0.25, "new_higher")):
            table = bootstrap_loss_intervals(build_paired_losses(sample_predictions(difference)), 3, 30)
            with self.subTest(difference=difference):
                np.testing.assert_allclose(table.ci_lower, difference)
                np.testing.assert_allclose(table.ci_upper, difference)
                self.assertTrue(table.interval_direction.eq(direction).all())

    def test_swapping_primary_model_losses_reverses_mean_and_interval(self) -> None:
        changed = self.predictions.copy()
        dc = changed.loc[changed.model == NEW_MODEL].set_index("match_id")
        poisson = changed.loc[changed.model == "fitted_poisson"].set_index("match_id")
        columns = [f"{market}_log_loss" for market in LOSS_MARKETS]
        for index, row in changed.loc[changed.model.isin([NEW_MODEL, "fitted_poisson"])].iterrows():
            source = poisson if row.model == NEW_MODEL else dc
            changed.loc[index, columns] = source.loc[row.match_id, columns].to_numpy()
        original = bootstrap_loss_intervals(self.paired, 3, 250, seed=51)
        reversed_table = bootstrap_loss_intervals(build_paired_losses(changed), 3, 250, seed=51)
        original = original.loc[original.reference_model == "fitted_poisson"].set_index("market")
        reversed_table = reversed_table.loc[reversed_table.reference_model == "fitted_poisson"].set_index("market")
        np.testing.assert_allclose(reversed_table.mean_loss_difference, -original.mean_loss_difference)
        np.testing.assert_allclose(reversed_table.ci_lower, -original.ci_upper)
        np.testing.assert_allclose(reversed_table.ci_upper, -original.ci_lower)

    def test_invalid_confidence_lengths_seeds_and_resample_counts_are_rejected(self) -> None:
        for confidence in (0, 1, np.nan, True):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, 3, 30, confidence)
        for length in (0, 12, 13, 1.5, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, length, 30)
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, 3, 30, seed=seed)
        for count in (0, 1, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                bootstrap_loss_intervals(self.paired, 3, count)

    def test_point_estimate_is_independent_of_block_length(self) -> None:
        short = bootstrap_loss_intervals(self.paired, 3, 100)
        long = bootstrap_loss_intervals(self.paired, 10, 100)
        np.testing.assert_array_equal(short.mean_loss_difference, long.mean_loss_difference)

    def test_sensitivity_covers_all_lengths_and_rejects_missing_length(self) -> None:
        intervals = pd.concat([bootstrap_loss_intervals(self.paired, length, 100) for length in (3, 5, 10)])
        sensitivity = summarize_sensitivity(intervals, (3, 5, 10))
        self.assertEqual(len(sensitivity), 30)
        self.assertTrue(sensitivity.new_model.eq(NEW_MODEL).all())
        with self.assertRaisesRegex(ValueError, "A block length is missing"):
            summarize_sensitivity(intervals.iloc[1:], (3, 5, 10))


if __name__ == "__main__":
    unittest.main()
