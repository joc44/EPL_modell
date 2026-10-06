"""Small synthetic data set for checking chronology and complete coverage."""

import unittest

import pandas as pd

from backtest_models import (
    MODEL_NAMES,
    evaluate_fixture,
    predict_fixture_models,
    run_backtest,
    summarize_predictions,
)


def sample_matches() -> pd.DataFrame:
    examples = [
        ("2021-08-11", "X", "Y", 8, 9),  # Already outside the history window.
        ("2021-08-12", "A", "B", 2, 1),  # Exactly on the lower boundary.
        ("2022-03-10", "A", "C", 3, 2),
        ("2022-06-10", "D", "B", 1, 1),
        ("2023-08-11", "C", "D", 2, 2),
        ("2023-08-12", "A", "B", 2, 1),
        ("2023-08-12", "C", "D", 3, 3),
        ("2023-08-13", "A", "B", 0, 4),
    ]
    records = []
    for date, home, away, hg, ag in examples:
        timestamp = pd.Timestamp(date)
        year = timestamp.year if timestamp.month >= 7 else timestamp.year - 1
        season = f"{year}/{str(year + 1)[-2:]}"
        records.append({
            "match_id": f"EPL|{season}|{date}|{home}|{away}",
            "Season": season, "Date": timestamp, "HomeTeam": home,
            "AwayTeam": away, "FTHG": hg, "FTAG": ag,
        })
    return pd.DataFrame(records)


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.matches = sample_matches()

    def test_date_boundaries_and_league_baseline(self):
        forecast = predict_fixture_models(self.matches, "A", "B", "2023-08-12")
        self.assertEqual(forecast["history_start"], "2021-08-12")
        self.assertEqual(forecast["last_history_date"], "2023-08-11")
        self.assertEqual(forecast["history_matches"], 4)
        self.assertEqual(forecast["home_history_matches"], 2)
        self.assertEqual(forecast["away_history_matches"], 2)
        baseline = forecast["models"]["league_average"]
        self.assertEqual(baseline["lambda_home"], 2.0)
        self.assertEqual(baseline["lambda_away"], 1.5)

    def test_target_same_day_and_future_results_cannot_change_forecast(self):
        before = predict_fixture_models(self.matches, "A", "B", "2023-08-12")
        changed = self.matches.copy(deep=True)
        mask = changed["Date"] >= pd.Timestamp("2023-08-12")
        changed.loc[mask, ["FTHG", "FTAG"]] = [40, 35]
        # Additional future matches must not leak back into the forecast either.
        extra = changed.iloc[-1].copy()
        extra["Date"] = pd.Timestamp("2023-09-01")
        extra["match_id"] = "EPL|2023/24|2023-09-01|A|B"
        changed = pd.concat([changed, extra.to_frame().T], ignore_index=True)
        changed["Date"] = pd.to_datetime(changed["Date"])
        after = predict_fixture_models(changed, "A", "B", "2023-08-12")
        self.assertEqual(before, after)

    def test_evaluation_changes_scores_but_keeps_predictions(self):
        forecast = predict_fixture_models(self.matches, "A", "B", "2023-08-12")
        home_win = evaluate_fixture(forecast, 2, 1)
        away_win = evaluate_fixture(forecast, 0, 3)
        for before, after in zip(home_win, away_win):
            for column in ("lambda_home", "lambda_away", "p_home", "p_draw", "p_away", "p_btts"):
                self.assertEqual(before[column], after[column])
            self.assertNotEqual(before["one_x_two_log_loss"], after["one_x_two_log_loss"])
            self.assertNotEqual(before["actual_btts"], after["actual_btts"])

    def test_same_day_exclusion_and_next_day_update(self):
        predictions = run_backtest(self.matches)
        same_day = predictions.loc[predictions["Date"] == "2023-08-12"]
        self.assertEqual(set(same_day["history_matches"]), {4})
        self.assertEqual(set(same_day["last_history_date"]), {"2023-08-11"})
        next_day = predictions.loc[predictions["Date"] == "2023-08-13"]
        # On the next day, the window start also advances: the 2021-08-12
        # match leaves and both 2023-08-12 results enter (4 - 1 + 2).
        self.assertEqual(set(next_day["history_matches"]), {5})
        self.assertEqual(set(next_day["last_history_date"]), {"2023-08-12"})

    def test_all_fixtures_all_models_and_progress(self):
        progress = []
        predictions = run_backtest(self.matches, progress=lambda done, total: progress.append((done, total)))
        # The historical date 2023-08-11 also belongs to the evaluation season.
        self.assertEqual(len(predictions), 4 * 4)
        self.assertEqual(predictions["match_id"].nunique(), 4)
        self.assertEqual(set(predictions["model"]), set(MODEL_NAMES))
        self.assertFalse(predictions.duplicated(["match_id", "model"]).any())
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
        summary = summarize_predictions(predictions)
        self.assertEqual(list(summary["model"]), list(MODEL_NAMES))
        self.assertEqual(list(summary["matches"]), [4, 4, 4, 4])

    def test_input_row_order_does_not_affect_results(self):
        reference = run_backtest(self.matches)
        shuffled = run_backtest(self.matches.sample(frac=1, random_state=17))
        pd.testing.assert_frame_equal(reference, shuffled)

    def test_input_matches_are_unchanged(self):
        before = self.matches.copy(deep=True)
        run_backtest(self.matches)
        pd.testing.assert_frame_equal(before, self.matches)

    def test_disabling_decay_restores_smoothed_model(self):
        predictions = run_backtest(self.matches, half_life_days=None)
        smoothed = predictions.loc[predictions["model"] == "smoothed"].reset_index(drop=True)
        weighted = predictions.loc[predictions["model"] == "time_weighted"].reset_index(drop=True)
        for column in ("lambda_home", "lambda_away", "p_home", "p_draw", "p_away", "p_btts"):
            for first, second in zip(smoothed[column], weighted[column]):
                self.assertAlmostEqual(first, second, places=12)

    def test_zero_raw_lambda_is_reported_and_smoothing_remains_positive(self):
        matches = self.matches.copy(deep=True)
        mask = (matches["HomeTeam"] == "A") & (matches["Date"] < pd.Timestamp("2023-08-12"))
        matches.loc[mask, "FTHG"] = 0
        forecast = predict_fixture_models(matches, "A", "B", "2023-08-12")
        evaluated = {row["model"]: row for row in evaluate_fixture(forecast, 2, 1)}
        raw = evaluated["raw"]
        self.assertEqual(raw["lambda_home"], 0.0)
        for market in ("one_x_two", "btts", "exact_score"):
            self.assertEqual(raw[f"{market}_log_loss_clipped"], 1)
        self.assertEqual(raw["p_home"], 0.0)
        self.assertGreater(evaluated["smoothed"]["lambda_home"], 0.0)

    def test_incomplete_or_duplicate_evaluation_is_rejected(self):
        predictions = run_backtest(self.matches)
        incomplete = predictions.drop(predictions.index[-1])
        duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
        for table in (incomplete, duplicate, predictions.iloc[0:0]):
            with self.subTest(rows=len(table)), self.assertRaises(ValueError):
                summarize_predictions(table)

    def test_missing_season_or_history_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "No matches"):
            run_backtest(self.matches, evaluation_season="2099/00")
        with self.assertRaisesRegex(ValueError, "Backtest failed"):
            run_backtest(self.matches, evaluation_season="2021/22")


if __name__ == "__main__":
    unittest.main()
