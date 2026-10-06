"""Complete seasons, original history and data validation before protocol creation."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from analyze_match import COLUMNS
from final_test_data import (
    DATA_MANIFEST, STEP18_SOURCES, download_final_raw, extend_final_matches, prepare_final_data,
    validate_final_matches, write_outputs_once,
)
from final_test_protocol import PROTOCOL_FILENAME, RUNTIME_SOURCES, SELECTION_SOURCES, build_protocol_payload, freeze_or_verify_protocol
from validation_data import expected_match_ids, file_sha256


def full_season(season: str) -> pd.DataFrame:
    rows = []
    teams = [f"Team-{index:02d}" for index in range(20)]
    start = pd.Timestamp(f"{season[:4]}-08-01")
    for home in teams:
        for away in teams:
            if home != away:
                rows.append({"Season": season, "Date": start + pd.Timedelta(days=len(rows) // 4),
                             "HomeTeam": home, "AwayTeam": away, "FTHG": 2, "FTAG": 1})
    table = pd.DataFrame(rows)
    table["match_id"] = expected_match_ids(table)
    return table[COLUMNS]


class FinalTestDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        for name in {*RUNTIME_SOURCES, *SELECTION_SOURCES, *STEP18_SOURCES}:
            (self.project / name).write_text(f"# test source {name}\n", encoding="utf-8")
        self.base = pd.concat([full_season(season) for season in ("2021/22", "2022/23", "2023/24", "2024/25")], ignore_index=True)
        self.base_path = self.project / "data/processed/matches_validation.csv"
        self.base_path.parent.mkdir(parents=True)
        self.base.to_csv(self.base_path, index=False, date_format="%Y-%m-%d")
        context = {"input_sha256": {"data/processed/matches_validation.csv": file_sha256(self.base_path)},
                   "source_sha256": {"main.py": file_sha256(self.project / "main.py")}}
        self.payload = build_protocol_payload(self.project, context)
        self.protocol_path = self.project / "data/reports/model_selection" / PROTOCOL_FILENAME
        freeze_or_verify_protocol(self.protocol_path, self.payload)
        self.new = full_season("2025/26")
        self.raw_path = self.project / "data/raw/E0_2025_2026.csv"
        self.raw_path.parent.mkdir(parents=True)
        self.write_raw(self.new)

    def raw_bytes(self, table: pd.DataFrame) -> bytes:
        return table[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].to_csv(index=False, date_format="%d/%m/%Y").encode("utf-8")

    def write_raw(self, table: pd.DataFrame) -> None:
        self.raw_path.write_bytes(self.raw_bytes(table))

    def test_final_season_is_added_and_all_1520_old_matches_are_preserved(self) -> None:
        original = self.base.copy(deep=True)
        combined, removed = extend_final_matches(self.base, self.raw_path, self.payload)
        self.assertEqual((len(combined), removed), (1900, 0))
        pd.testing.assert_frame_equal(self.base, original)
        validate_final_matches(combined, self.base, self.payload)

    def test_identical_duplicates_are_removed_and_counted(self) -> None:
        self.write_raw(pd.concat([self.new, self.new.iloc[:1]], ignore_index=True))
        combined, removed = extend_final_matches(self.base, self.raw_path, self.payload)
        self.assertEqual((len(combined), removed), (1900, 1))

    def test_conflicting_duplicate_result_is_rejected(self) -> None:
        duplicate = self.new.iloc[:1].copy()
        duplicate["FTHG"] = 9
        self.write_raw(pd.concat([self.new, duplicate], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "Conflicting final scores"):
            extend_final_matches(self.base, self.raw_path, self.payload)

    def test_incomplete_season_and_wrong_season_date_are_rejected(self) -> None:
        for bad in (self.new.iloc[:-1], self.new.assign(Date=pd.Timestamp("2024-08-01"))):
            self.write_raw(bad)
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                extend_final_matches(self.base, self.raw_path, self.payload)

    def test_noninteger_or_missing_raw_goals_are_rejected(self) -> None:
        for value in (2.5, None):
            bad = self.new.copy()
            bad["FTHG"] = bad.FTHG.astype(float)
            bad.loc[0, "FTHG"] = value
            self.write_raw(bad)
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                extend_final_matches(self.base, self.raw_path, self.payload)

    def test_changes_to_old_matches_are_rejected(self) -> None:
        combined, _ = extend_final_matches(self.base, self.raw_path, self.payload)
        combined.loc[0, "FTHG"] = 8
        with self.assertRaisesRegex(ValueError, "1520"):
            validate_final_matches(combined, self.base, self.payload)

    def test_prepare_and_repeat_preserve_old_data_protocol_and_new_files(self) -> None:
        old_data, old_protocol = self.base_path.read_bytes(), self.protocol_path.read_bytes()
        first, manifest, protocol = prepare_final_data(self.project)
        data_path = self.base_path.parent / "matches_final_test.csv"
        manifest_path = self.base_path.parent / DATA_MANIFEST
        new_data, new_manifest = data_path.read_bytes(), manifest_path.read_bytes()
        second, repeated, same_protocol = prepare_final_data(self.project)
        self.assertEqual(manifest, repeated)
        self.assertEqual(protocol, same_protocol)
        self.assertEqual((old_data, old_protocol), (self.base_path.read_bytes(), self.protocol_path.read_bytes()))
        self.assertEqual((new_data, new_manifest), (data_path.read_bytes(), manifest_path.read_bytes()))
        self.assertEqual(len(first), len(second))

    def test_protocol_failure_prevents_any_download(self) -> None:
        self.raw_path.unlink()
        with patch("final_test_data.checked_protocol", side_effect=ValueError("changed protocol")), \
                patch("final_test_data.download_final_raw") as download:
            with self.assertRaisesRegex(ValueError, "changed protocol"):
                prepare_final_data(self.project)
        download.assert_not_called()

    def test_changed_raw_or_processed_data_is_not_overwritten(self) -> None:
        prepare_final_data(self.project)
        output = self.base_path.parent / "matches_final_test.csv"
        original = output.read_bytes()
        changed = self.new.copy()
        changed.loc[0, "FTHG"] = 8
        self.write_raw(changed)
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            prepare_final_data(self.project)
        self.assertEqual(original, output.read_bytes())

    def test_changed_step18_source_is_detected_before_reading_target_data(self) -> None:
        prepare_final_data(self.project)
        (self.project / STEP18_SOURCES[0]).write_text("changed source")
        with patch("final_test_data.extend_final_matches") as extend, self.assertRaisesRegex(ValueError, "Step 18 source"):
            prepare_final_data(self.project)
        extend.assert_not_called()

    def test_changed_historical_source_or_runtime_is_rejected(self) -> None:
        (self.project / "main.py").write_text("changed old source")
        with self.assertRaisesRegex(ValueError, "main.py"):
            prepare_final_data(self.project)
        (self.project / "main.py").write_text("# test source main.py\n")
        changed = {**self.payload["runtime"], "numpy": "different"}
        with patch("final_test_protocol.current_runtime", return_value=changed), self.assertRaisesRegex(ValueError, "venv"):
            prepare_final_data(self.project)

    def test_missing_raw_with_download_disabled_is_actionable(self) -> None:
        self.raw_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "prepare_final_test_data.py"):
            prepare_final_data(self.project, allow_download=False)

    def test_download_checks_complete_season_before_saving_and_does_not_replace_existing(self) -> None:
        self.raw_path.unlink()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = self.raw_bytes(self.new)
        with patch("final_test_data.urlopen", return_value=response) as opened:
            download_final_raw(self.raw_path, "2025/26")
            download_final_raw(self.raw_path, "2025/26")
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(self.raw_path.read_bytes(), self.raw_bytes(self.new))

    def test_invalid_download_is_not_saved(self) -> None:
        self.raw_path.unlink()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = self.raw_bytes(self.new.iloc[:2])
        with patch("final_test_data.urlopen", return_value=response), self.assertRaisesRegex(ValueError, "380"):
            download_final_raw(self.raw_path, "2025/26")
        self.assertFalse(self.raw_path.exists())

    def test_output_precheck_prevents_partial_write_and_preserves_existing_content(self) -> None:
        existing, new = self.project / "existing.txt", self.project / "new.txt"
        existing.write_bytes(b"keep")
        with self.assertRaises(ValueError):
            write_outputs_once({new: b"new", existing: b"different"})
        self.assertFalse(new.exists())
        self.assertEqual(existing.read_bytes(), b"keep")
        write_outputs_once({existing: b"keep", new: b"new"})
        self.assertEqual(new.read_bytes(), b"new")


if __name__ == "__main__":
    unittest.main()
