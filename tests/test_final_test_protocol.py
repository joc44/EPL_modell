"""Check freezing, unchanged reruns and modified contexts."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from final_test_protocol import (
    PROTOCOL_FILENAME, RUNTIME_SOURCES, SELECTION_SOURCES, assert_frozen_context_unchanged,
    build_protocol_payload, canonical_json, freeze_or_verify_protocol, load_frozen_protocol,
    validate_protocol_payload,
)
from validation_data import file_sha256


class FinalTestProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        for name in (*RUNTIME_SOURCES, *SELECTION_SOURCES, "checked_previous.py"):
            (self.project / name).write_text(f"# unchanged test source: {name}\n", encoding="utf-8")
        self.data = self.project / "data" / "processed" / "matches_validation.csv"
        self.data.parent.mkdir(parents=True)
        self.data.write_text("Season\n2023/24\n2024/25\n", encoding="utf-8")
        self.context = {
            "input_sha256": {"data/processed/matches_validation.csv": file_sha256(self.data)},
            "source_sha256": {"checked_previous.py": file_sha256(self.project / "checked_previous.py")},
        }
        self.payload = build_protocol_payload(self.project, self.context)
        self.path = self.project / "reports" / PROTOCOL_FILENAME

    def rewrite_envelope(self, envelope: dict) -> None:
        raw = (json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8")
        self.path.write_bytes(raw)
        self.path.with_suffix(".sha256").write_text(sha256(raw).hexdigest() + "\n", encoding="ascii")

    def test_canonical_identifier_is_independent_of_dictionary_order(self) -> None:
        self.assertEqual(canonical_json({"b": 2, "a": [1, False]}), canonical_json({"a": [1, False], "b": 2}))
        self.assertNotEqual(canonical_json(False), canonical_json(0))
        with self.assertRaises(ValueError):
            canonical_json(float("nan"))

    def test_build_keeps_previous_hashes_and_freezes_runtime_and_all_required_sources(self) -> None:
        validate_protocol_payload(self.payload)
        frozen = self.payload["frozen_context"]
        self.assertEqual(frozen["input_sha256"], self.context["input_sha256"])
        self.assertTrue(set((*RUNTIME_SOURCES, *SELECTION_SOURCES, "checked_previous.py")) <= set(frozen["source_sha256"]))
        self.context["input_sha256"].clear()
        self.assertEqual(len(frozen["input_sha256"]), 1)
        self.assertEqual(set(self.payload["runtime"]), {"python", "numpy", "pandas", "scipy"})

    def test_changed_test_rules_are_rejected(self) -> None:
        for key, value in (("half_life_days", 90.0), ("ridge_strength", 2.0), ("lookback_days", 365)):
            bad = deepcopy(self.payload)
            bad["selected_parameters"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "test rules"):
                validate_protocol_payload(bad)
        bad = deepcopy(self.payload)
        bad["evaluation_policy"]["primary_metric"] = "accuracy"
        with self.assertRaises(ValueError):
            validate_protocol_payload(bad)

    def test_boolean_and_integer_policy_fields_are_not_interchangeable(self) -> None:
        bad = deepcopy(self.payload)
        bad["final_test_results_used_for_selection"] = 0
        with self.assertRaises(ValueError):
            validate_protocol_payload(bad)

    def test_missing_or_extra_policy_fields_are_rejected(self) -> None:
        for change in ("missing", "extra"):
            bad = deepcopy(self.payload)
            if change == "missing":
                del bad["fit_policy"]
            else:
                bad["unplanned_setting"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_protocol_payload(bad)

    def test_missing_required_source_and_invalid_runtime_are_rejected(self) -> None:
        bad = deepcopy(self.payload)
        del bad["frozen_context"]["source_sha256"]["fitted_poisson_model.py"]
        with self.assertRaisesRegex(ValueError, "source-code"):
            validate_protocol_payload(bad)
        for runtime in ({"python": "3.12"}, {**self.payload["runtime"], "numpy": 1},
                        {**self.payload["runtime"], "scipy": ""}):
            bad = deepcopy(self.payload)
            bad["runtime"] = runtime
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, "environment description"):
                validate_protocol_payload(bad)

    def test_absolute_escaping_paths_and_invalid_checksums_are_rejected(self) -> None:
        for name in ("../outside.csv", "data/../../outside.csv", "/outside.csv", "C:\\outside.csv", ".", ""):
            bad = deepcopy(self.payload)
            bad["frozen_context"]["input_sha256"] = {name: "a" * 64}
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_protocol_payload(bad)
        for digest in ("short", "A" * 64, "z" * 64, None):
            bad = deepcopy(self.payload)
            bad["frozen_context"]["input_sha256"] = {"data.csv": digest}
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                validate_protocol_payload(bad)

    def test_windows_relative_paths_are_portable(self) -> None:
        portable = deepcopy(self.payload)
        hashes = portable["frozen_context"]["input_sha256"]
        portable["frozen_context"]["input_sha256"] = {name.replace("/", "\\"): value for name, value in hashes.items()}
        assert_frozen_context_unchanged(self.project, portable)

    def test_first_freeze_creates_valid_protocol_and_checksum(self) -> None:
        result, created = freeze_or_verify_protocol(self.path, self.payload)
        self.assertTrue(created)
        self.assertEqual(result, load_frozen_protocol(self.path))
        self.assertEqual(result["protocol_id"], sha256(canonical_json(self.payload)).hexdigest())
        self.assertEqual(self.path.with_suffix(".sha256").read_text().strip(), sha256(self.path.read_bytes()).hexdigest())

    def test_second_freeze_preserves_both_files_timestamp_and_identifier(self) -> None:
        first, _ = freeze_or_verify_protocol(self.path, self.payload)
        before = (self.path.read_bytes(), self.path.with_suffix(".sha256").read_bytes())
        second, created = freeze_or_verify_protocol(self.path, deepcopy(self.payload))
        self.assertFalse(created)
        self.assertEqual(first, second)
        self.assertEqual(before, (self.path.read_bytes(), self.path.with_suffix(".sha256").read_bytes()))

    def test_different_context_cannot_replace_an_existing_protocol(self) -> None:
        freeze_or_verify_protocol(self.path, self.payload)
        before = self.path.read_bytes()
        bad = deepcopy(self.payload)
        bad["frozen_context"]["input_sha256"]["data/processed/matches_validation.csv"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            freeze_or_verify_protocol(self.path, bad)
        self.assertEqual(before, self.path.read_bytes())

    def test_changed_runtime_cannot_replace_an_existing_protocol(self) -> None:
        freeze_or_verify_protocol(self.path, self.payload)
        bad = deepcopy(self.payload)
        bad["runtime"]["scipy"] = "different-version"
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            freeze_or_verify_protocol(self.path, bad)

    def test_file_tampering_is_detected_by_checksum(self) -> None:
        freeze_or_verify_protocol(self.path, self.payload)
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_frozen_protocol(self.path)

    def test_identifier_tampering_is_detected_even_with_updated_file_checksum(self) -> None:
        envelope, _ = freeze_or_verify_protocol(self.path, self.payload)
        envelope["protocol_id"] = "0" * 64
        self.rewrite_envelope(envelope)
        with self.assertRaisesRegex(ValueError, "identifier"):
            load_frozen_protocol(self.path)

    def test_non_utc_timestamp_is_rejected(self) -> None:
        envelope, _ = freeze_or_verify_protocol(self.path, self.payload)
        envelope["created_at_utc"] = "2026-10-06T12:00:00+02:00"
        self.rewrite_envelope(envelope)
        with self.assertRaisesRegex(ValueError, "UTC"):
            load_frozen_protocol(self.path)

    def test_incomplete_existing_pair_is_not_overwritten(self) -> None:
        self.path.parent.mkdir(parents=True)
        for survivor in (self.path, self.path.with_suffix(".sha256")):
            survivor.write_bytes(b"existing")
            with self.subTest(survivor=survivor.name), self.assertRaises(FileNotFoundError):
                freeze_or_verify_protocol(self.path, self.payload)
            self.assertEqual(survivor.read_bytes(), b"existing")
            survivor.unlink()

    def test_failed_checksum_write_removes_only_files_created_by_this_call(self) -> None:
        unrelated = self.project / "keep.txt"
        unrelated.write_text("keep")
        original_open = Path.open
        checksum_path = self.path.with_suffix(".sha256")

        def fail_checksum(path, *args, **kwargs):
            if path == checksum_path and args and args[0] == "xb":
                raise PermissionError("test disk-write failure")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_checksum), self.assertRaises(PermissionError):
            freeze_or_verify_protocol(self.path, self.payload)
        self.assertFalse(self.path.exists())
        self.assertFalse(checksum_path.exists())
        self.assertEqual(unrelated.read_text(), "keep")

    def test_exclusive_creation_preserves_a_competing_checksum_file(self) -> None:
        original_open = Path.open
        checksum_path = self.path.with_suffix(".sha256")

        def competing_write(path, *args, **kwargs):
            if path == checksum_path and args and args[0] == "xb":
                with original_open(path, "wb") as output:
                    output.write(b"competing-existing-file")
                raise FileExistsError("test competing write")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", competing_write), self.assertRaises(FileExistsError):
            freeze_or_verify_protocol(self.path, self.payload)
        self.assertFalse(self.path.exists())
        self.assertEqual(checksum_path.read_bytes(), b"competing-existing-file")

    def test_changed_historical_data_is_rejected(self) -> None:
        self.data.write_text("changed historical data")
        with self.assertRaisesRegex(ValueError, "matches_validation.csv"):
            assert_frozen_context_unchanged(self.project, self.payload)

    def test_changed_source_code_is_rejected(self) -> None:
        (self.project / "fitted_poisson_model.py").write_text("# changed model")
        with self.assertRaisesRegex(ValueError, "fitted_poisson_model.py"):
            assert_frozen_context_unchanged(self.project, self.payload)

    def test_missing_frozen_file_is_rejected(self) -> None:
        self.data.unlink()
        with self.assertRaisesRegex(ValueError, "matches_validation.csv"):
            assert_frozen_context_unchanged(self.project, self.payload)

    def test_changed_installed_runtime_is_rejected(self) -> None:
        changed = {**self.payload["runtime"], "numpy": "different"}
        with patch("final_test_protocol.current_runtime", return_value=changed), self.assertRaisesRegex(ValueError, "venv"):
            assert_frozen_context_unchanged(self.project, self.payload)


if __name__ == "__main__":
    unittest.main()
