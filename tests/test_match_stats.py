import tempfile
import unittest
from pathlib import Path

import pandas as pd

import analyze_match


def example_matches() -> pd.DataFrame:
    """Synthetic matches for testing temporal filtering."""
    matches = pd.DataFrame([
        ["old", "2023/24", "2023-08-08", "Arsenal", "Nottingham Forest", 7, 8],
        ["lower", "2023/24", "2023-08-09", "Arsenal", "Nottingham Forest", 2, 0],
        ["previous", "2023/24", "2023-08-11", "Arsenal", "Nottingham Forest", 1, 4],
        ["target", "2023/24", "2023-08-12", "Arsenal", "Nottingham Forest", 99, 88],
        ["future", "2023/24", "2023-08-13", "Arsenal", "Nottingham Forest", 55, 44],
    ], columns=analyze_match.COLUMNS)
    matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d")
    return matches


class MatchStatisticsTests(unittest.TestCase):
    def test_lower_boundary_included_and_entire_target_day_excluded(self):
        history = analyze_match.history_before(
            example_matches(), pd.Timestamp("2023-08-12 15:00"), 3
        )
        self.assertEqual(history["match_id"].tolist(), ["lower", "previous"])

    def test_target_result_does_not_change_pre_match_statistics(self):
        matches = example_matches()
        before = analyze_match.analyze_fixture(matches, "Arsenal", "Nottingham Forest", "2023-08-12", 3)
        changed = matches.copy()
        changed.loc[changed["match_id"] == "target", ["FTHG", "FTAG"]] = [0, 0]
        after = analyze_match.analyze_fixture(changed, "Arsenal", "Nottingham Forest", "2023-08-12", 3)
        self.assertEqual(before, after)

    def test_future_results_and_new_future_rows_do_not_change_statistics(self):
        matches = example_matches()
        before = analyze_match.analyze_fixture(matches, "Arsenal", "Nottingham Forest", "2023-08-12", 3)
        changed = matches.copy()
        changed.loc[changed["match_id"] == "future", ["FTHG", "FTAG"]] = [0, 0]
        extra = matches.loc[matches["match_id"] == "future"].copy()
        extra["match_id"] = "another_future"
        extra["Date"] = pd.Timestamp("2023-08-20")
        changed = pd.concat([changed, extra], ignore_index=True)
        after = analyze_match.analyze_fixture(changed, "Arsenal", "Nottingham Forest", "2023-08-12", 3)
        self.assertEqual(before, after)

    def test_home_and_away_roles_use_correct_goals_and_matches(self):
        history = analyze_match.history_before(example_matches(), pd.Timestamp("2023-08-12"), 3)
        reversed_venue = history.iloc[[0]].copy()
        reversed_venue["HomeTeam"] = "Nottingham Forest"
        reversed_venue["AwayTeam"] = "Arsenal"
        reversed_venue["FTHG"] = 8
        reversed_venue["FTAG"] = 9
        history = pd.concat([history, reversed_venue], ignore_index=True)
        home = analyze_match.team_stats(history, "Arsenal", "home")
        away = analyze_match.team_stats(history, "Nottingham Forest", "away")
        self.assertEqual(home["matches"], 2)
        self.assertEqual(home["goals_for_total"], 3)
        self.assertEqual(home["goals_against_total"], 4)
        self.assertEqual(away["matches"], 2)
        self.assertEqual(away["goals_for_mean"], 2.0)
        self.assertEqual(away["goals_against_mean"], 1.5)

    def test_league_means_use_filtered_matches(self):
        history = analyze_match.history_before(example_matches(), pd.Timestamp("2023-08-12"), 3)
        league = analyze_match.league_stats(history)
        self.assertEqual(league, {"matches": 2, "home_goal_mean": 1.5, "away_goal_mean": 2.0})

    def test_missing_team_history_has_no_estimated_average(self):
        stats = analyze_match.team_stats(example_matches(), "Unknown team", "home")
        self.assertEqual(stats["matches"], 0)
        self.assertIsNone(stats["goals_for_mean"])
        self.assertIsNone(stats["goals_against_mean"])

    def test_zero_goals_with_real_history_is_a_zero_average(self):
        history = example_matches().iloc[[1]].copy()
        history["FTHG"] = 0
        stats = analyze_match.team_stats(history, "Arsenal", "home")
        self.assertEqual(stats["matches"], 1)
        self.assertEqual(stats["goals_for_mean"], 0.0)

    def test_no_history_cannot_produce_league_means(self):
        with self.assertRaisesRegex(ValueError, "No previous matches"):
            analyze_match.league_stats(example_matches().iloc[0:0])

    def test_invalid_windows_are_rejected(self):
        for invalid_window in [0, -1, 3.5, True]:
            with self.subTest(window=invalid_window):
                with self.assertRaises(ValueError):
                    analyze_match.history_before(example_matches(), pd.Timestamp("2023-08-12"), invalid_window)

    def test_repeated_match_in_clean_file_is_rejected(self):
        matches = example_matches()
        repeated = pd.concat([matches, matches.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matches_clean.csv"
            repeated.to_csv(path, index=False, date_format="%Y-%m-%d")
            with self.assertRaisesRegex(ValueError, "duplicate match"):
                analyze_match.load_clean_matches(path)


if __name__ == "__main__":
    unittest.main()
