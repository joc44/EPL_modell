"""Prepare the separate test season after validating the Step 17 protocol."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from analyze_match import COLUMNS, load_clean_matches
from final_test_protocol import PROTOCOL_FILENAME, assert_frozen_context_unchanged, load_frozen_protocol
from main import clean_matches, load_matches
from validation_data import expected_match_ids, file_sha256, validate_extended_matches, validate_full_season


SOURCE_URL = "https://football-data.co.uk/mmz4281/2526/E0.csv"
DATA_MANIFEST = "final_test_data_settings.json"
STEP18_SOURCES = (
    "final_test_data.py", "prepare_final_test_data.py", "final_test_backtest.py", "final_test_reports.py", "run_final_test.py",
)


def checked_protocol(project: Path) -> dict:
    protocol = load_frozen_protocol(project / "data/reports/model_selection" / PROTOCOL_FILENAME)
    assert_frozen_context_unchanged(project, protocol["payload"])
    return protocol


def step18_sources(project: Path) -> dict:
    return {name: file_sha256(project / name) for name in STEP18_SOURCES}


def write_outputs_once(outputs: dict[Path, bytes]) -> None:
    """Accept identical content; never replace an existing file with different content."""
    for path, content in outputs.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise ValueError(f"A final-test file with different content already exists; it will not be overwritten: {path.name}.")
    created = []
    try:
        for path, content in outputs.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                created.append(path)
                output.write(content)
    except OSError:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def download_final_raw(raw_path: Path, season: str) -> None:
    """Save a complete, validated season without replacing existing raw data."""
    if raw_path.exists():
        return
    try:
        request = Request(SOURCE_URL, headers={"User-Agent": "EPL-portfolio-project/1.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read(10 * 1024 * 1024 + 1)
    except (OSError, URLError) as error:
        raise OSError(f"Download failed. Download from: {SOURCE_URL}, and save to: {raw_path}. {error}") from error
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("The downloaded CSV is unexpectedly large.")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="epl-final-", dir=raw_path.parent) as temporary:
        candidate = Path(temporary) / "candidate.csv"
        candidate.write_bytes(content)
        clean, _ = clean_matches(load_matches(candidate), season)
        clean["match_id"] = expected_match_ids(clean)
        validate_full_season(clean, season)
    write_outputs_once({raw_path: content})


def extend_final_matches(base: pd.DataFrame, raw_path: Path, payload: dict) -> tuple[pd.DataFrame, int]:
    validate_extended_matches(base)
    season = payload["final_test_season"]
    new, removed = clean_matches(load_matches(raw_path), season)
    new["match_id"] = expected_match_ids(new)
    validate_full_season(new, season)
    result = pd.concat([base[COLUMNS], new[COLUMNS]], ignore_index=True)
    result = result.sort_values(["Date", "match_id"], kind="stable").reset_index(drop=True)
    validate_final_matches(result, base, payload)
    return result, removed


def validate_final_matches(matches: pd.DataFrame, base: pd.DataFrame, payload: dict) -> None:
    old_seasons = payload["data_policy"]["initial_history_seasons"]
    if set(matches.Season) != {*old_seasons, payload["final_test_season"]}:
        raise ValueError("The final data file must contain exactly the 2021/22–2025/26 seasons.")
    validate_extended_matches(base)
    for season in (*old_seasons, payload["final_test_season"]):
        validate_full_season(matches, season)
    old = matches.loc[matches.Season.isin(old_seasons), COLUMNS].sort_values("match_id").reset_index(drop=True)
    original = base[COLUMNS].sort_values("match_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(old, original, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("The final data file modifies data for one of the previous 1520 matches.") from error


def prepare_final_data(project: Path, allow_download: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    # Validate before reading or downloading any test data.
    protocol = checked_protocol(project)
    payload = protocol["payload"]
    sources = step18_sources(project)
    base_path = project / "data/processed/matches_validation.csv"
    raw_path = project / "data/raw" / payload["data_policy"]["raw_filename"]
    output_path = project / "data/processed" / payload["data_policy"]["processed_filename"]
    manifest_path = output_path.parent / DATA_MANIFEST
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("protocol_id") != protocol["protocol_id"] or saved.get("step18_source_sha256") != sources:
            raise ValueError("The final-data preparation belongs to another protocol or Step 18 source version.")
    base_hash = file_sha256(base_path)
    base = load_clean_matches(base_path)
    validate_extended_matches(base)
    if not raw_path.is_file():
        if not allow_download:
            raise FileNotFoundError("Raw test data are missing. Run prepare_final_test_data.py first.")
        print("Downloading the complete 2025/26 CSV from football-data.co.uk…", flush=True)
        download_final_raw(raw_path, payload["final_test_season"])
    raw_hash = file_sha256(raw_path)
    matches, removed = extend_final_matches(base, raw_path, payload)
    content = matches.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    protocol_path = project / "data/reports/model_selection" / PROTOCOL_FILENAME
    manifest = {
        "step": "18a", "evaluation_role": "separate_final_test_data", "protocol_id": protocol["protocol_id"],
        "protocol_file_sha256": file_sha256(protocol_path), "final_test_season": payload["final_test_season"],
        "source_url": SOURCE_URL, "raw_file": raw_path.name, "raw_sha256": raw_hash,
        "base_file": base_path.name, "base_sha256": base_hash, "output_file": output_path.name,
        "output_sha256": sha256(content).hexdigest(), "base_matches": len(base), "total_matches": len(matches),
        "final_test_matches": int(matches.Season.eq(payload["final_test_season"]).sum()),
        "removed_exact_duplicates": removed, "step18_source_sha256": sources,
    }
    assert_frozen_context_unchanged(project, payload)
    if file_sha256(base_path) != base_hash or file_sha256(raw_path) != raw_hash or step18_sources(project) != sources:
        raise ValueError("The data or Step 18 source changed during processing.")
    write_outputs_once({output_path: content, manifest_path: (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")})
    return matches, manifest, protocol


def load_final_inputs(project: Path) -> tuple[pd.DataFrame, dict, dict]:
    if not (project / "data/processed" / DATA_MANIFEST).is_file():
        raise FileNotFoundError("Run prepare_final_test_data.py first.")
    return prepare_final_data(project, allow_download=False)
