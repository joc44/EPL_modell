"""Step 18/a: validated preparation of the separate 2025/26 test season."""

from pathlib import Path

import pandas as pd

from final_test_data import prepare_final_data


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    print("Step 18/a: prepare the separate final-test data", flush=True)
    matches, manifest, protocol = prepare_final_data(PROJECT_DIR)
    print(f"Validated protocol: {protocol['protocol_id']}")
    print("\nMatches per season:")
    print(matches.groupby("Season").size().to_string())
    print(f"\nTotal matches: {len(matches)}")
    print(f"Final-test matches: {manifest['final_test_matches']}")
    print(f"Identical duplicates removed from the new season: {manifest['removed_exact_duplicates']}")
    print("The previous 1520 matches and the Step 17 protocol remain unchanged.")
    print("Data file: " + str(PROJECT_DIR / "data/processed" / manifest["output_file"]))
    print("Next runner: run_final_test.py")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Final-data preparation error: {error}")
