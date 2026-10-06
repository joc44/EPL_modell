import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main


def example_matches(rows: list[list]) -> pd.DataFrame:
    """Synthetic matches for testing data cleaning."""
    return pd.DataFrame(rows, columns=main.COLUMNS)


class CleaningTests(unittest.TestCase):
    def test_date_is_day_month_year(self):
        source = example_matches([["02/03/2022", "Arsenal", "Chelsea", 2, 1]])
        cleaned, removed = main.clean_matches(source, "2021/22")
        self.assertEqual(cleaned.iloc[0]["Date"], pd.Timestamp("2022-03-02"))
        self.assertEqual(removed, 0)

    def test_invalid_date_is_rejected(self):
        source = example_matches([["31/02/2022", "Arsenal", "Chelsea", 2, 1]])
        with self.assertRaises(ValueError):
            main.clean_matches(source, "2021/22")

    def test_mislabelled_season_is_rejected(self):
        source = example_matches([["11/08/2023", "Arsenal", "Chelsea", 2, 1]])
        with self.assertRaisesRegex(ValueError, "do not belong to season 2021/22"):
            main.clean_matches(source, "2021/22")

    def test_aliases_and_spaces_do_not_duplicate_a_match(self):
        source = example_matches([
            ["02/03/2022", " Man Utd ", " Arsenal ", 2, 1],
            ["02/03/2022", "Man United", "Arsenal", 2, 1],
        ])
        original = source.copy(deep=True)
        cleaned, removed = main.clean_matches(source, "2021/22")
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(removed, 1)
        self.assertEqual(cleaned.iloc[0]["HomeTeam"], "Manchester United")
        self.assertEqual(int(cleaned["FTHG"].sum()), 2)
        pd.testing.assert_frame_equal(source, original)

    def test_conflicting_results_are_rejected(self):
        source = example_matches([
            ["02/03/2022", "Arsenal", "Chelsea", 2, 1],
            ["02/03/2022", "Arsenal", "Chelsea", 3, 1],
        ])
        with self.assertRaisesRegex(ValueError, "Conflicting final scores"):
            main.clean_matches(source, "2021/22")

    def test_different_matches_on_same_date_are_kept(self):
        source = example_matches([
            ["02/03/2022", "Arsenal", "Chelsea", 2, 1],
            ["02/03/2022", "Everton", "Liverpool", 0, 1],
        ])
        cleaned, removed = main.clean_matches(source, "2021/22")
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(removed, 0)

    def test_whitespace_only_team_is_rejected(self):
        source = example_matches([["02/03/2022", "   ", "Arsenal", 2, 1]])
        with self.assertRaisesRegex(ValueError, "Empty team name"):
            main.clean_matches(source, "2021/22")

    def test_same_team_with_two_aliases_is_rejected(self):
        source = example_matches([["02/03/2022", "Man Utd", "Man United", 2, 1]])
        with self.assertRaisesRegex(ValueError, "against itself"):
            main.clean_matches(source, "2021/22")


class PipelineTests(unittest.TestCase):
    def test_three_seasons_are_sorted_without_duplicate_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            for season, filename in main.SEASON_FILES.items():
                year = season[:4]
                source = example_matches([
                    [f"14/08/{year}", "Chelsea", "Liverpool", 2, 1],
                    [f"13/08/{year}", "Arsenal", "Everton", 2, 1],
                ])
                if season == "2021/22":
                    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
                source.to_csv(raw_dir / filename, index=False)

            with patch.object(main, "RAW_DIR", raw_dir):
                matches, removed = main.load_seasons()

        self.assertEqual(len(matches), 6)
        self.assertEqual(removed, 1)
        self.assertTrue(matches["Date"].is_monotonic_increasing)
        self.assertTrue(matches["match_id"].is_unique)
        self.assertEqual(matches.index.tolist(), list(range(6)))
        self.assertEqual(int((matches["FTHG"] + matches["FTAG"]).sum()), 18)


if __name__ == "__main__":
    unittest.main()
