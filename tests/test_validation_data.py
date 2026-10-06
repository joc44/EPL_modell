"""Validate data preservation, season boundaries and fixed evaluation."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from backtest_models import MODEL_NAMES, run_backtest, summarize_predictions
from validation_data import (
    BASE_SEASONS, VALIDATION_SEASON, assert_base_preserved, check_frozen_settings,
    expected_match_ids, extend_matches, file_sha256, load_development_checkpoint,
    validate_extended_matches, validate_full_season,
)


def season_table(season: str) -> pd.DataFrame:
    # A synthetic but complete home-and-away pairing schedule for validation.
    teams = ["Manchester City", "Nottingham Forest", *[f"Team {i:02d}" for i in range(18)]]
    rows = []
    start = pd.Timestamp(f"{season[:4]}-08-01")
    for home in teams:
        for away in teams:
            if home != away:
                rows.append({
                    "Season": season, "Date": start + pd.Timedelta(days=len(rows) // 10),
                    "HomeTeam": home, "AwayTeam": away, "FTHG": 2, "FTAG": 1,
                })
    result = pd.DataFrame(rows)
    result["match_id"] = expected_match_ids(result)
    return result[["match_id", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]


def settings_for(base_path: Path) -> dict:
    return {
        "evaluation_season": "2023/24", "evaluation_role": "development_backtest",
        "models": list(MODEL_NAMES), "evaluated_matches": 380,
        "lookback_days": 730, "prior_matches": 5.0, "half_life_days": 180.0,
        "history_rule": "match_date - lookback_days <= Date < match_date",
        "same_day_results_used": False, "evaluation_weighting": "equal_weight_per_match",
        "log_loss_probability_floor": 1e-15, "one_x_two_brier_scale": "0_to_2",
        "binary_brier_scale": "0_to_1", "data_file": "matches_clean.csv",
        "data_sha256": file_sha256(base_path),
    }


class SeasonDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.raw_path = Path(self.temporary.name) / "new.csv"
        self.base = pd.concat([season_table(season) for season in BASE_SEASONS], ignore_index=True)
        self.new = season_table(VALIDATION_SEASON)

    def write_raw(self, rows: pd.DataFrame) -> None:
        rows[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].to_csv(
            self.raw_path, index=False, date_format="%d/%m/%Y", encoding="utf-8"
        )

    def test_extension_preserves_every_base_field_and_input(self) -> None:
        original = self.base.copy(deep=True)
        self.write_raw(self.new)
        extended, removed = extend_matches(self.base, self.raw_path)
        self.assertEqual(len(extended), 1520)
        self.assertEqual(removed, 0)
        assert_base_preserved(original, extended)
        pd.testing.assert_frame_equal(original, self.base)

    def test_aliases_and_exact_duplicate_are_cleaned(self) -> None:
        rows = pd.concat([self.new, self.new.iloc[:1]], ignore_index=True)
        for column in ("HomeTeam", "AwayTeam"):
            rows[column] = rows[column].replace({"Manchester City": "Man City", "Nottingham Forest": "Nott'm Forest"})
        self.write_raw(rows)
        extended, removed = extend_matches(self.base, self.raw_path)
        self.assertEqual(removed, 1)
        self.assertEqual(len(extended), 1520)
        self.assertIn("Manchester City", set(extended.HomeTeam))
        self.assertNotIn("Man City", set(extended.HomeTeam))

    def test_conflicting_duplicate_is_rejected(self) -> None:
        conflicting = self.new.iloc[:1].copy()
        conflicting["FTHG"] = 8
        self.write_raw(pd.concat([self.new, conflicting], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "Conflicting final scores"):
            extend_matches(self.base, self.raw_path)

    def test_incomplete_season_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "380 matches"):
            validate_full_season(self.new.iloc[1:], VALIDATION_SEASON)

    def test_duplicate_fixture_on_another_date_is_rejected(self) -> None:
        broken = self.new.copy()
        broken.loc[1, ["HomeTeam", "AwayTeam"]] = broken.loc[0, ["HomeTeam", "AwayTeam"]].values
        broken.loc[1, "Date"] += pd.Timedelta(days=1)
        broken["match_id"] = expected_match_ids(broken)
        with self.assertRaisesRegex(ValueError, "pairing"):
            validate_full_season(broken, VALIDATION_SEASON)

    def test_date_outside_season_is_rejected(self) -> None:
        broken = self.new.copy()
        broken.loc[0, "Date"] = pd.Timestamp("2025-07-01")
        broken["match_id"] = expected_match_ids(broken)
        with self.assertRaisesRegex(ValueError, "date"):
            validate_full_season(broken, VALIDATION_SEASON)

    def test_changed_base_score_is_rejected(self) -> None:
        extended = pd.concat([self.base, self.new], ignore_index=True)
        extended.loc[0, "FTHG"] = 7
        with self.assertRaisesRegex(ValueError, "modifies"):
            assert_base_preserved(self.base, extended)

    def test_future_season_is_rejected(self) -> None:
        extended = pd.concat([self.base, self.new, season_table("2025/26")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_extended_matches(extended)


class DevelopmentCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = pd.concat([season_table(season) for season in BASE_SEASONS], ignore_index=True)
        fixtures = cls.base.loc[cls.base.Season == "2023/24"]
        first_only = pd.concat([cls.base.loc[cls.base.Season != "2023/24"], fixtures.iloc[:1]])
        sample = run_backtest(first_only, "2023/24", 730, 5.0, 180.0)
        # Extend the sample report to the full synthetic season for file validation.
        cls.predictions = pd.concat([
            sample.assign(**{column: row[column] for column in (
                "match_id", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"
            )}) for row in fixtures.to_dict("records")
        ], ignore_index=True)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base_path = self.root / "matches_clean.csv"
        self.base.to_csv(self.base_path, index=False, date_format="%Y-%m-%d", encoding="utf-8")
        self.settings = settings_for(self.base_path)
        self.settings_path = self.root / "backtest_settings_2023_24.json"
        self.write_settings()
        self.predictions.to_csv(self.root / "backtest_predictions_2023_24.csv", index=False, date_format="%Y-%m-%d")
        summarize_predictions(self.predictions).to_csv(self.root / "backtest_summary_2023_24.csv", index=False)

    def write_settings(self) -> None:
        self.settings_path.write_text(json.dumps(self.settings), encoding="utf-8")

    def test_consistent_checkpoint_is_accepted(self) -> None:
        predictions, summary, settings = load_development_checkpoint(self.base_path, self.root)
        self.assertEqual(predictions.match_id.nunique(), 380)
        self.assertEqual(len(summary), 4)
        self.assertEqual(settings["half_life_days"], 180.0)

    def test_changed_settings_are_rejected(self) -> None:
        for key, changed in (("lookback_days", 365), ("prior_matches", 8.0),
                             ("half_life_days", 90.0), ("same_day_results_used", True)):
            with self.subTest(key=key):
                altered = deepcopy(self.settings)
                altered[key] = changed
                with self.assertRaisesRegex(ValueError, "A development setting differs"):
                    check_frozen_settings(altered)

    def test_changed_base_file_is_rejected_by_hash(self) -> None:
        changed = self.base.copy()
        changed.loc[0, "FTHG"] = 7
        changed.to_csv(self.base_path, index=False, date_format="%Y-%m-%d")
        with self.assertRaisesRegex(ValueError, "differs from the file used in Step 8"):
            load_development_checkpoint(self.base_path, self.root)

    def test_inconsistent_summary_is_rejected(self) -> None:
        path = self.root / "backtest_summary_2023_24.csv"
        summary = pd.read_csv(path)
        summary.loc[0, "one_x_two_log_loss"] += 0.1
        summary.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "summary does not match"):
            load_development_checkpoint(self.base_path, self.root)

    def test_report_fixture_must_match_base_file(self) -> None:
        predictions = self.predictions.copy()
        predictions.loc[predictions.match_id == predictions.iloc[0].match_id, "HomeTeam"] = "Wrong Team"
        predictions.to_csv(self.root / "backtest_predictions_2023_24.csv", index=False, date_format="%Y-%m-%d")
        with self.assertRaisesRegex(ValueError, "match fixtures differ"):
            load_development_checkpoint(self.base_path, self.root)


if __name__ == "__main__":
    unittest.main()
