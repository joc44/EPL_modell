"""Step 10/a: add the later season to a separate data file."""

import json
from pathlib import Path

import pandas as pd

from analyze_match import load_clean_matches
from validation_data import VALIDATION_SEASON, extend_matches, file_sha256


PROJECT_DIR = Path(__file__).resolve().parent
BASE_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
RAW_PATH = PROJECT_DIR / "data" / "raw" / "E0_2024_2025.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "matches_validation.csv"
MANIFEST_PATH = PROJECT_DIR / "data" / "processed" / "validation_data_settings.json"
SOURCE_URL = "https://football-data.co.uk/mmz4281/2425/E0.csv"


def main() -> None:
    print("Step 10/a: prepare the 2024/25 validation data file", flush=True)
    base_hash = file_sha256(BASE_PATH)
    raw_hash = file_sha256(RAW_PATH)
    base = load_clean_matches(BASE_PATH)
    matches, removed = extend_matches(base, RAW_PATH)
    if file_sha256(BASE_PATH) != base_hash or file_sha256(RAW_PATH) != raw_hash:
        raise ValueError("The input file changed during processing. Rerun the preparation.")
    print("\nMatches per season:")
    print(matches.groupby("Season").size().to_string())
    print(f"\nTotal matches: {len(matches)}")
    print(f"Identical duplicates removed from the new season: {removed}")
    new_matches = matches.loc[matches["Season"] == VALIDATION_SEASON]
    print(f"Teams in the new season: {len(set(new_matches['HomeTeam']) | set(new_matches['AwayTeam']))}")
    print(f"Goals in the new season: {int(new_matches['FTHG'].sum() + new_matches['FTAG'].sum())}")
    print(f"Mean home goals: {new_matches['FTHG'].mean():.3f}")
    print(f"Mean away goals: {new_matches['FTAG'].mean():.3f}")
    print(f"Mean total goals: {(new_matches['FTHG'] + new_matches['FTAG']).mean():.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    manifest = {
        "step": "10a",
        "validation_season": VALIDATION_SEASON,
        "source_url": SOURCE_URL,
        "source_columns": ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
        "base_file": BASE_PATH.name,
        "base_sha256": base_hash,
        "raw_file": RAW_PATH.name,
        "raw_sha256": raw_hash,
        "output_file": OUTPUT_PATH.name,
        "output_sha256": file_sha256(OUTPUT_PATH),
        "base_matches": len(base),
        "total_matches": len(matches),
        "removed_exact_duplicates": removed,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSeparate validation data file saved: {OUTPUT_PATH}")
    print(f"Data source and fingerprints saved: {MANIFEST_PATH}")
    print("Every field of the previous 1140 matches is preserved in the new file.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Data-preparation error: {error}")
