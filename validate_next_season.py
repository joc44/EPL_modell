"""Step 10/b: validate unchanged models on a later season."""

import json
from pathlib import Path
import platform

import pandas as pd

from analyze_match import load_clean_matches
from backtest_models import MODEL_LABELS, MODEL_NAMES, run_backtest, summarize_predictions
from calibration_analysis import EVENTS, analyze_calibration, validate_predictions
from run_backtest import print_summary, show_progress
from validation_data import (
    DEVELOPMENT_SEASON, HALF_LIFE_DAYS, LOG_LOSS_FLOOR, LOOKBACK_DAYS,
    MIN_BIN_MATCHES, N_BINS, PRIOR_MATCHES, VALIDATION_SEASON,
    assert_base_preserved, file_sha256, load_development_checkpoint,
    validate_extended_matches,
)


PROJECT_DIR = Path(__file__).resolve().parent
BASE_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_validation.csv"
RAW_PATH = PROJECT_DIR / "data" / "raw" / "E0_2024_2025.csv"
MANIFEST_PATH = PROJECT_DIR / "data" / "processed" / "validation_data_settings.json"
REPORTS_DIR = PROJECT_DIR / "data" / "reports"


def print_season_comparison(development: pd.DataFrame, validation: pd.DataFrame) -> None:
    """The same metric on two seasons, paired by model."""
    previous = development.set_index("model")
    current = validation.set_index("model")
    print("\nTwo-season comparison – log loss (lower is better):")
    for metric, label in (("one_x_two_log_loss", "1X2"),
                          ("over_2_5_log_loss", "Over / under 2.5"),
                          ("btts_log_loss", "BTTS"),
                          ("exact_score_log_loss", "Exact score")):
        print(f"\n{label}:")
        print(f"{'Model':<22} {'2023/24':>10} {'2024/25':>10} {'Difference':>10}")
        for name in MODEL_NAMES:
            old, new = previous.loc[name, metric], current.loc[name, metric]
            print(f"{MODEL_LABELS[name]:<22} {old:>10.4f} {new:>10.4f} {new - old:>+10.4f}")
    print("Difference = 2024/25 − 2023/24; season difficulty and event rates may differ.")


def print_calibration_comparison(comparison: pd.DataFrame) -> None:
    print("\nTime-weighted model – calibration on two seasons:")
    print(f"{'Event / season':<30} {'Predicted':>9} {'Observed':>10} {'Difference':>12} {'ECE':>10}")
    for event in ("home_win", "over_2_5", "btts"):
        print(EVENTS[event][0] + ":")
        rows = comparison.loc[(comparison["model"] == "time_weighted") & (comparison["event"] == event)]
        for row in rows.to_dict("records"):
            print(f"{row['Season']:<30} {row['mean_probability']:>8.2%} "
                  f"{row['observed_rate']:>9.2%} {100 * row['bias']:>+8.2f} pp "
                  f"{100 * row['ece']:>6.2f} pp")
    print("Difference = mean forecast − observed rate; pp = percentage points. ECE is descriptive.")


def main() -> None:
    print("Step 10/b: validation on the later 2024/25 season", flush=True)
    old_predictions, old_summary, old_settings = load_development_checkpoint(BASE_PATH, REPORTS_DIR)
    if not DATA_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Run prepare_validation_data.py first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data_hash = file_sha256(DATA_PATH)
    if (manifest.get("output_sha256") != data_hash
            or manifest.get("base_sha256") != file_sha256(BASE_PATH)
            or manifest.get("raw_sha256") != file_sha256(RAW_PATH)):
        raise ValueError("The prepared-data fingerprint differs. Rerun prepare_validation_data.py.")
    matches = load_clean_matches(DATA_PATH)
    validate_extended_matches(matches)
    assert_base_preserved(load_clean_matches(BASE_PATH), matches)
    print(f"Development season: {DEVELOPMENT_SEASON} | Validation season: {VALIDATION_SEASON}")
    print(f"History window: {LOOKBACK_DAYS} days | Prior weight: {PRIOR_MATCHES:g} | Half-life: {HALF_LIFE_DAYS:g} days")
    print("Model settings are fixed. Pre-match statistics update chronologically.")
    print("Results from all matches on the target matchday are excluded.", flush=True)

    predictions = run_backtest(
        matches, VALIDATION_SEASON, LOOKBACK_DAYS, PRIOR_MATCHES, HALF_LIFE_DAYS, show_progress
    )
    checked_predictions = validate_predictions(predictions)
    summary = summarize_predictions(predictions)
    if file_sha256(DATA_PATH) != data_hash:
        raise ValueError("The validation data file changed during the run.")
    print(f"\nEvaluated matches: {predictions['match_id'].nunique()}")
    print(f"Prediction rows: {len(predictions)} (4 models × 380 matches)")
    print_summary(summary)
    print_season_comparison(old_summary, summary)

    new_bins, new_calibration = analyze_calibration(checked_predictions, N_BINS, MIN_BIN_MATCHES)
    _, old_calibration = analyze_calibration(old_predictions, N_BINS, MIN_BIN_MATCHES)
    comparison = pd.concat([
        old_summary.assign(Season=DEVELOPMENT_SEASON),
        summary.assign(Season=VALIDATION_SEASON),
    ], ignore_index=True)
    calibration_comparison = pd.concat([
        old_calibration.assign(Season=DEVELOPMENT_SEASON),
        new_calibration.assign(Season=VALIDATION_SEASON),
    ], ignore_index=True)
    print_calibration_comparison(calibration_comparison)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = VALIDATION_SEASON.replace("/", "_")
    predictions_path = REPORTS_DIR / f"backtest_predictions_{slug}.csv"
    summary_path = REPORTS_DIR / f"backtest_summary_{slug}.csv"
    settings_path = REPORTS_DIR / f"backtest_settings_{slug}.json"
    comparison_path = REPORTS_DIR / "season_comparison_2023_24_2024_25.csv"
    calibration_path = REPORTS_DIR / "calibration_comparison_2023_24_2024_25.csv"
    diagnostics = REPORTS_DIR / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    calibration_comparison.to_csv(calibration_path, index=False, encoding="utf-8-sig")
    new_bins.to_csv(diagnostics / f"calibration_bins_{slug}.csv", index=False, encoding="utf-8-sig")
    new_calibration.to_csv(diagnostics / f"calibration_summary_{slug}.csv", index=False, encoding="utf-8-sig")
    settings = {
        **old_settings,
        "step": 10,
        "evaluation_season": VALIDATION_SEASON,
        "evaluation_role": "later_season_validation",
        "data_file": DATA_PATH.name,
        "data_sha256": data_hash,
        "development_data_sha256": file_sha256(BASE_PATH),
        "development_settings_sha256": file_sha256(REPORTS_DIR / "backtest_settings_2023_24.json"),
        "development_predictions_sha256": file_sha256(REPORTS_DIR / "backtest_predictions_2023_24.csv"),
        "development_summary_sha256": file_sha256(REPORTS_DIR / "backtest_summary_2023_24.csv"),
        "validation_raw_sha256": file_sha256(RAW_PATH),
        "data_preparation": manifest,
        "calibration_bins": N_BINS,
        "min_bin_matches": MIN_BIN_MATCHES,
        "model_source_sha256": {
            name: file_sha256(PROJECT_DIR / name) for name in (
                "analyze_match.py", "goal_model.py", "score_probabilities.py", "shrinkage_model.py",
                "time_weighted_model.py", "evaluation_metrics.py", "backtest_models.py",
                "validation_data.py", "validate_next_season.py",
            )
        },
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
    }
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDetailed predictions saved: {predictions_path}")
    print(f"Summary metrics saved: {summary_path}")
    print(f"Two-season metrics saved: {comparison_path}")
    print(f"Two-season calibration saved: {calibration_path}")
    print(f"Run settings saved: {settings_path}")
    print("\nThis is validation on a later season. No parameters were tuned from the 2024/25 results.")
    print("If these results inform model changes, keep 2025/26 for a separate final test.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as error:
        raise SystemExit(f"Season-validation error: {error}")
