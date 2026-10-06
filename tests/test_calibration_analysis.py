"""Hand-calculated examples and report validation without external season data."""

import json
from math import log
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from calibration_analysis import (
    EVENTS, HISTORY_GROUPS, MODEL_NAMES, analyze_calibration, calibration_bins,
    event_observations, history_group, history_group_metrics, largest_one_x_two_errors,
    load_backtest_reports, probability_bin, validate_predictions,
)
from evaluation_metrics import binary_metrics, one_x_two_metrics, realized_outcome


def sample_predictions() -> pd.DataFrame:
    records = []
    probabilities = {
        "league_average": {"home": 0.55, "draw": 0.25, "away": 0.2},
        "raw": {"home": 0.7, "draw": 0.1, "away": 0.2},
        "smoothed": {"home": 0.6, "draw": 0.2, "away": 0.2},
        "time_weighted": {"home": 0.65, "draw": 0.2, "away": 0.15},
    }
    fixtures = [(2, 1, 0, 2), (1, 1, 2, 3), (0, 2, 7, 8)]
    for index, (home_goals, away_goals, home_history, away_history) in enumerate(fixtures):
        date = pd.Timestamp("2023-08-12") + pd.Timedelta(days=index)
        outcome = realized_outcome(home_goals, away_goals)
        for name in MODEL_NAMES:
            one_x_two = one_x_two_metrics(probabilities[name], outcome)
            record = {
                "match_id": f"EPL|2023/24|{date.date()}|Home{index}|Away{index}",
                "Season": "2023/24", "Date": date,
                "HomeTeam": f"Home{index}", "AwayTeam": f"Away{index}",
                "model": name, "FTHG": home_goals, "FTAG": away_goals,
                "history_start": date - pd.Timedelta(days=730),
                "last_history_date": date - pd.Timedelta(days=1), "history_matches": 100,
                "home_history_matches": home_history, "away_history_matches": away_history,
                "actual_one_x_two": outcome, "p_home": probabilities[name]["home"],
                "p_draw": probabilities[name]["draw"], "p_away": probabilities[name]["away"],
                "one_x_two_log_loss": one_x_two["log_loss"], "one_x_two_brier": one_x_two["brier"],
                "one_x_two_hit": int(one_x_two["hit"]), "one_x_two_log_loss_clipped": 0,
            }
            for line, probability in [(1.5, 0.8), (2.5, 0.55), (3.5, 0.35)]:
                key = "over_" + str(line).replace(".", "_")
                actual = int(home_goals + away_goals > line)
                metrics = binary_metrics(probability, actual)
                record[f"p_{key}"] = probability
                record[f"actual_{key}"] = actual
                if line == 2.5:
                    record[f"{key}_log_loss"] = metrics["log_loss"]
                    record[f"{key}_brier"] = metrics["brier"]
            actual_btts = int(home_goals > 0 and away_goals > 0)
            btts = binary_metrics(0.6, actual_btts)
            record.update({"p_btts": 0.6, "actual_btts": actual_btts,
                           "btts_log_loss": btts["log_loss"], "btts_brier": btts["brier"],
                           "btts_log_loss_clipped": int(btts["clipped"])})
            records.append(record)
    return pd.DataFrame.from_records(records)


class CalibrationTests(unittest.TestCase):
    def test_bin_boundaries_include_zero_and_one(self):
        for probability, expected in [(0.0, 0), (0.099999, 0), (0.1, 1),
                                      (0.5, 5), (0.9, 9), (1.0, 9)]:
            with self.subTest(probability=probability):
                self.assertEqual(probability_bin(probability, 10), expected)

    def test_hand_calculated_bins_and_bias_cancellation(self):
        bins = calibration_bins(pd.Series([0, 0.25, 0.75, 1]), pd.Series([0, 1, 0, 1]), 2, 2)
        self.assertEqual(list(bins["matches"]), [2, 2])
        self.assertEqual(list(bins["mean_probability"]), [0.125, 0.875])
        self.assertEqual(list(bins["observed_rate"]), [0.5, 0.5])
        self.assertEqual(list(bins["bias"]), [-0.375, 0.375])
        # Full-sample averages agree, but both bins deviate.
        ece = (bins["matches"] * bins["bias"].abs()).sum() / 4
        self.assertEqual(ece, 0.375)

    def test_empty_bins_are_missing_not_zero(self):
        bins = calibration_bins(pd.Series([0.61]), pd.Series([1]), 10, 20)
        empty = bins.loc[bins["empty"]]
        self.assertEqual(len(empty), 9)
        self.assertTrue(empty["mean_probability"].isna().all())
        self.assertTrue(empty["observed_rate"].isna().all())
        self.assertEqual(int(bins["matches"].sum()), 1)
        self.assertTrue(bins.loc[bins["matches"] == 1, "sparse"].iloc[0])

    def test_probability_and_observation_positions_are_paired(self):
        probabilities = pd.Series([0.2, 0.8], index=[10, 20])
        observations = pd.Series([0, 1], index=[90, 80])
        bins = calibration_bins(probabilities, observations, 2, 1)
        self.assertEqual(list(bins["observed_rate"]), [0.0, 1.0])

    def test_invalid_bins_probabilities_and_observations(self):
        for count in [0, 1, 2.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                probability_bin(0.5, count)
        for probability in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                probability_bin(probability, 10)
        cases = [(pd.Series([], dtype=float), pd.Series([], dtype=int)),
                 (pd.Series([0.5]), pd.Series([0, 1])),
                 (pd.Series([0.5]), pd.Series([2])),
                 (pd.Series([float("nan")]), pd.Series([1]))]
        for probabilities, observations in cases:
            with self.assertRaises(ValueError):
                calibration_bins(probabilities, observations)

    def test_seven_events_and_all_models_keep_full_coverage(self):
        predictions = validate_predictions(sample_predictions())
        bins, summary = analyze_calibration(predictions)
        self.assertEqual(len(bins), 4 * 7 * 10)
        self.assertEqual(len(summary), 4 * 7)
        self.assertEqual(set(summary["matches"]), {3})
        self.assertTrue((bins.groupby(["model", "event"])["matches"].sum() == 3).all())
        for event in ("home_win", "draw", "away_win"):
            rows = predictions.loc[predictions["model"] == "raw"]
            self.assertEqual(int(event_observations(rows, event).sum()), 1)

    def test_small_bin_flag_does_not_remove_matches_from_ece(self):
        predictions = validate_predictions(sample_predictions())
        _, low = analyze_calibration(predictions, min_bin_matches=1)
        _, high = analyze_calibration(predictions, min_bin_matches=100)
        pd.testing.assert_series_equal(low["ece"], high["ece"])
        self.assertEqual(int(low["matches_in_sparse_bins"].sum()), 0)
        self.assertEqual(set(high["matches_in_sparse_bins"]), {3})

    def test_history_groups_have_exclusive_boundaries(self):
        self.assertEqual(history_group(0, 10, 5), "missing_history")
        self.assertEqual(history_group(4, 10, 5), "limited_history")
        self.assertEqual(history_group(5, 5, 5), "enough_history")
        for args in [(-1, 5, 5), (5, 5, 0), (True, 5, 5)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                history_group(*args)

    def test_history_groups_cover_every_fixture_once(self):
        predictions = validate_predictions(sample_predictions())
        summary = history_group_metrics(predictions)
        self.assertEqual(len(summary), 12)
        self.assertEqual(set(summary["history_group"]), set(HISTORY_GROUPS))
        self.assertEqual(set(summary["matches"]), {1})
        self.assertTrue((summary.groupby("model")["matches"].sum() == 3).all())
        # The raw model's draw belongs to the middle group.
        row = summary.loc[(summary["model"] == "raw") & (summary["history_group"] == "limited_history")].iloc[0]
        self.assertAlmostEqual(row["one_x_two_log_loss"], -log(0.1))

    def test_largest_losses_and_deterministic_ties(self):
        predictions = validate_predictions(sample_predictions())
        errors = largest_one_x_two_errors(predictions, 2)
        raw = errors.loc[errors["model"] == "raw"]
        self.assertEqual(list(raw["actual_one_x_two"]), ["draw", "away"])
        self.assertEqual(list(raw["rank"]), [1, 2])
        tied = predictions.copy()
        tied["one_x_two_log_loss"] = 1.0
        ordered = largest_one_x_two_errors(tied.sample(frac=1, random_state=4), 3)
        for name in MODEL_NAMES:
            dates = list(ordered.loc[ordered["model"] == name, "Date"])
            self.assertEqual(dates, sorted(dates))

    def test_analysis_does_not_change_input(self):
        predictions = sample_predictions()
        before = predictions.copy(deep=True)
        validated = validate_predictions(predictions)
        analyze_calibration(validated)
        history_group_metrics(validated)
        largest_one_x_two_errors(validated)
        pd.testing.assert_frame_equal(predictions, before)


class ReportValidationTests(unittest.TestCase):
    def test_duplicate_or_incomplete_model_coverage_is_rejected(self):
        sample = sample_predictions()
        cases = [pd.concat([sample, sample.iloc[[0]]], ignore_index=True),
                 sample.drop(sample.index[-1]), sample.iloc[0:0], sample.drop(columns="p_btts")]
        for table in cases:
            with self.subTest(rows=len(table)), self.assertRaises(ValueError):
                validate_predictions(table)

    def test_invalid_probability_normalization_and_score_are_rejected(self):
        for column, value in [("p_home", -0.1), ("p_btts", float("nan")),
                              ("p_home", 0.99), ("one_x_two_brier", 2.5),
                              ("btts_log_loss", float("inf"))]:
            changed = sample_predictions()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_predictions(changed)

    def test_actual_events_must_match_goals(self):
        for column, value in [("actual_one_x_two", "away"), ("actual_btts", 0), ("actual_over_2_5", 0)]:
            changed = sample_predictions()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_predictions(changed)

    def test_future_history_and_inconsistent_fixture_are_rejected(self):
        changed = sample_predictions()
        changed.loc[0, "last_history_date"] = changed.loc[0, "Date"]
        with self.assertRaises(ValueError):
            validate_predictions(changed)
        changed = sample_predictions()
        changed.loc[0, "home_history_matches"] = 1
        with self.assertRaisesRegex(ValueError, "differs between models"):
            validate_predictions(changed)

    def test_report_and_settings_must_have_same_season_and_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            csv_path = path / "predictions.csv"
            settings_path = path / "settings.json"
            sample_predictions().to_csv(csv_path, index=False)
            settings = {"evaluation_season": "2023/24", "models": list(MODEL_NAMES), "evaluated_matches": 3}
            settings_path.write_text(json.dumps(settings), encoding="utf-8")
            predictions, loaded = load_backtest_reports(csv_path, settings_path, "2023/24")
            self.assertEqual(predictions["match_id"].nunique(), 3)
            self.assertEqual(loaded["evaluated_matches"], 3)
            settings["evaluated_matches"] = 4
            settings_path.write_text(json.dumps(settings), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_backtest_reports(csv_path, settings_path, "2023/24")
            settings["evaluation_season"] = "2022/23"
            settings_path.write_text(json.dumps(settings), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_backtest_reports(csv_path, settings_path, "2023/24")

    def test_missing_report_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaisesRegex(FileNotFoundError, "run_backtest.py"):
                load_backtest_reports(path / "missing.csv", path / "missing.json", "2023/24")


if __name__ == "__main__":
    unittest.main()
