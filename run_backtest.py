"""Step 8: chronological evaluation of a complete season."""

from hashlib import sha256
import json
from pathlib import Path
import platform

import pandas as pd

from analyze_match import load_clean_matches
from backtest_models import LOSS_MARKETS, MODEL_LABELS, MODEL_NAMES, run_backtest, summarize_predictions
from evaluation_metrics import LOG_LOSS_FLOOR
from predict_match import DATA_PATH, LOOKBACK_DAYS
from predict_smoothed import PRIOR_MATCHES
from predict_time_weighted import HALF_LIFE_DAYS


EVALUATION_SEASON = "2023/24"
EXPECTED_MATCHES = 380
REPORTS_DIR = Path(__file__).resolve().parent / "data" / "reports"


def show_progress(completed: int, total: int) -> None:
    if completed % 100 == 0 or completed == total:
        print(f"Processed: {completed}/{total} matches", flush=True)


def print_summary(summary: pd.DataFrame) -> None:
    print("\n1X2 – lower log loss and Brier are better:")
    print(f"{'Model':<22} {'Matches':>6} {'Log loss':>10} {'Brier':>10} {'Accuracy':>10}")
    for row in summary.to_dict("records"):
        print(f"{MODEL_LABELS[row['model']]:<22} {row['matches']:>6} "
              f"{row['one_x_two_log_loss']:>10.4f} {row['one_x_two_brier']:>10.4f} "
              f"{row['one_x_two_accuracy']:>9.2%}")

    for key, label in [("over_1_5", "Over / under 1.5"), ("over_2_5", "Over / under 2.5"),
                       ("over_3_5", "Over / under 3.5"), ("btts", "Both teams to score")]:
        print(f"\n{label} – lower values are better:")
        print(f"{'Model':<22} {'Log loss':>10} {'Brier':>10}")
        for row in summary.to_dict("records"):
            print(f"{MODEL_LABELS[row['model']]:<22} {row[f'{key}_log_loss']:>10.4f} "
                  f"{row[f'{key}_brier']:>10.4f}")

    print("\nExact score – lower log loss is better:")
    print(f"{'Model':<22} {'Log loss':>10} {'Top-1 accuracy':>15}")
    for row in summary.to_dict("records"):
        print(f"{MODEL_LABELS[row['model']]:<22} {row['exact_score_log_loss']:>10.4f} "
              f"{row['exact_score_accuracy']:>14.2%}")

    print("\nProbability-floor usage (match counts by market):")
    market_labels = {
        "one_x_two": "1X2", "over_1_5": "over 1.5", "over_2_5": "over 2.5",
        "over_3_5": "over 3.5", "btts": "BTTS", "exact_score": "exact score",
    }
    for row in summary.to_dict("records"):
        counts = ", ".join(f"{market_labels[market]}: {row[f'{market}_log_loss_clipped']}" for market in LOSS_MARKETS)
        print(f"{MODEL_LABELS[row['model']]}: {counts}")
    first = summary.iloc[0]
    print(f"\nMatches without home history: {int(first['missing_home_history'])}")
    print(f"Matches without away history: {int(first['missing_away_history'])}")


def main() -> None:
    # The fingerprint and read refer to the same unchanged local data file.
    data_sha256 = sha256(DATA_PATH.read_bytes()).hexdigest()
    matches = load_clean_matches(DATA_PATH)
    fixtures = matches.loc[matches["Season"] == EVALUATION_SEASON]
    if len(fixtures) != EXPECTED_MATCHES:
        raise ValueError(f"For a complete EPL season, {EXPECTED_MATCHES} matches expected; found: {len(fixtures)}.")
    half_life_text = f"{HALF_LIFE_DAYS:g} days" if HALF_LIFE_DAYS is not None else "disabled"
    print("Step 8: chronological backtest", flush=True)
    print(f"Evaluation season: {EVALUATION_SEASON}")
    print(f"Number of matches: {len(fixtures)}")
    print(f"History window: {LOOKBACK_DAYS} days | Prior weight: {PRIOR_MATCHES:g} | Half-life: {half_life_text}")
    print("Every forecast uses only dates before the target matchday.")
    print(f"Log loss probability floor: {LOG_LOSS_FLOOR:g}", flush=True)

    predictions = run_backtest(
        matches, EVALUATION_SEASON, LOOKBACK_DAYS, PRIOR_MATCHES, HALF_LIFE_DAYS, show_progress
    )
    summary = summarize_predictions(predictions)
    print(f"\nEvaluated matches: {predictions['match_id'].nunique()}")
    print(f"Prediction rows: {len(predictions)} (4 models × {len(fixtures)} matches)")
    print_summary(summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    season_slug = EVALUATION_SEASON.replace("/", "_")
    predictions_path = REPORTS_DIR / f"backtest_predictions_{season_slug}.csv"
    summary_path = REPORTS_DIR / f"backtest_summary_{season_slug}.csv"
    settings_path = REPORTS_DIR / f"backtest_settings_{season_slug}.json"
    # UTF-8 BOM supports spreadsheet readers. CSV uses commas and a decimal point.
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    settings = {
        "step": 8,
        "evaluation_season": EVALUATION_SEASON,
        "evaluation_role": "development_backtest",
        "models": list(MODEL_NAMES),
        "evaluated_matches": len(fixtures),
        "lookback_days": LOOKBACK_DAYS,
        "prior_matches": PRIOR_MATCHES,
        "half_life_days": HALF_LIFE_DAYS,
        "history_rule": "match_date - lookback_days <= Date < match_date",
        "same_day_results_used": False,
        "evaluation_weighting": "equal_weight_per_match",
        "log_loss_probability_floor": LOG_LOSS_FLOOR,
        "one_x_two_brier_scale": "0_to_2",
        "binary_brier_scale": "0_to_1",
        "data_file": DATA_PATH.name,
        "data_sha256": data_sha256,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
    }
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDetailed predictions saved: {predictions_path}")
    print(f"Summary metrics saved: {summary_path}")
    print(f"Run settings saved: {settings_path}")
    print("\nThis is a development backtest. Evaluation on a later, separate season remains to be done.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Backtest error: {error}")
