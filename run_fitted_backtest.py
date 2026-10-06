"""Step 12: development comparison of five models on two complete seasons."""

from hashlib import sha256
import json
from pathlib import Path
import platform

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "reports" / "fitted_backtest"
LOOKBACK_DAYS = 730
HALF_LIFE_DAYS = 180.0
RIDGE_STRENGTH = 1.0
N_BINS = 10
MIN_BIN_MATCHES = 20


def show_progress(completed: int, total: int) -> None:
    if completed % 100 == 0 or completed == total:
        print(f"Processed: {completed}/{total} matches", flush=True)


def print_summary(summary: pd.DataFrame, labels: dict) -> None:
    print("\n1X2 – lower log loss and Brier are better:")
    print(f"{'Model':<22} {'Matches':>6} {'Log loss':>10} {'Brier':>10} {'Accuracy':>10}")
    for row in summary.to_dict("records"):
        print(f"{labels[row['model']]:<22} {row['matches']:>6} {row['one_x_two_log_loss']:>10.4f} "
              f"{row['one_x_two_brier']:>10.4f} {row['one_x_two_accuracy']:>9.2%}")
    for market, label in (("over_1_5", "Over / under 1.5"), ("over_2_5", "Over / under 2.5"),
                          ("over_3_5", "Over / under 3.5"), ("btts", "Both teams to score")):
        print(f"\n{label} – lower values are better:")
        print(f"{'Model':<22} {'Log loss':>10} {'Brier':>10}")
        for row in summary.to_dict("records"):
            print(f"{labels[row['model']]:<22} {row[f'{market}_log_loss']:>10.4f} {row[f'{market}_brier']:>10.4f}")
    print("\nExact score – lower log loss is better:")
    print(f"{'Model':<22} {'Log loss':>10} {'Top-1 accuracy':>15}")
    for row in summary.to_dict("records"):
        print(f"{labels[row['model']]:<22} {row['exact_score_log_loss']:>10.4f} {row['exact_score_accuracy']:>14.2%}")


def main() -> None:
    try:
        import numpy as np
        import scipy
        from backtest_models import LOSS_MARKETS
        from calibration_analysis import EVENTS
        from comparison_reports import EVALUATION_SEASONS, assert_inputs_unchanged, load_comparison_inputs
        from evaluation_metrics import LOG_LOSS_FLOOR
        from fitted_backtest import (
            COMPARISON_LABELS, COMPARISON_MODELS, NEW_MODEL, calibration_models,
            paired_loss_differences, run_fitted_backtest, summarize_models, validate_comparison,
        )
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in ("numpy", "scipy"):
            raise SystemExit("SciPy is missing. Install the Step 11 requirements_step_11.txt dependency first.") from None
        raise
    print("Step 12: chronological backtest of the jointly fitted model", flush=True)
    print(f"Seasons: {', '.join(EVALUATION_SEASONS)} | Models: {len(COMPARISON_MODELS)}")
    print(f"History window: {LOOKBACK_DAYS} days | Half-life: {HALF_LIFE_DAYS:g} days | L2 strength: {RIDGE_STRENGTH:g}")
    print("Reuse the previous four models' saved predictions after validation.")
    print("Refit the new model before every matchday; same-day results are excluded.")
    print("Both seasons are development comparisons. The separate final test remains 2025/26.", flush=True)
    matches, references, context = load_comparison_inputs(PROJECT_DIR)
    if LOG_LOSS_FLOOR != 1e-15:
        raise ValueError("The log loss floor differs from the fixed comparison setting.")
    bundles = {}
    for season in EVALUATION_SEASONS:
        print(f"\nEvaluation season: {season}", flush=True)
        new_predictions, diagnostics = run_fitted_backtest(
            matches, season, LOOKBACK_DAYS, HALF_LIFE_DAYS, RIDGE_STRENGTH, show_progress
        )
        fixtures = matches.loc[matches.Season == season]
        combined = validate_comparison(pd.concat([references[season], new_predictions], ignore_index=True), fixtures)
        summary = summarize_models(combined)
        bins, calibration = calibration_models(combined, N_BINS, MIN_BIN_MATCHES)
        paired = paired_loss_differences(combined)
        print(f"\nEvaluated matches: {combined.match_id.nunique()}")
        print(f"Prediction rows: {len(combined)} (5 models × 380 matches)")
        print(f"Successful matchday fits: {len(diagnostics)}/{len(diagnostics)}")
        print(f"Matchdays using numerical refinement: {int((diagnostics.refinement_steps > 0).sum())}")
        print_summary(summary, COMPARISON_LABELS)
        print("\nNew model minus the previous time-weighted model – paired log loss:")
        for row in paired.loc[paired.reference_model == "time_weighted"].to_dict("records"):
            print(f"{row['market']}: {row['mean_loss_difference']:+.4f}")
        print("Difference = new model − previous model; negative favours the new model.")
        print("\nJointly fitted model – calibration:")
        for row in calibration.loc[(calibration.model == NEW_MODEL) & calibration.event.isin(("home_win", "over_2_5", "btts"))].to_dict("records"):
            print(f"{EVENTS[row['event']][0]}: predicted {row['mean_probability']:.2%} | "
                  f"observed {row['observed_rate']:.2%} | difference {100 * row['bias']:+.2f} pp | ECE {100 * row['ece']:.2f} pp")
        print("\nNew-model log loss probability-floor usage:")
        print(", ".join(f"{market}: {int(new_predictions[f'{market}_log_loss_clipped'].sum())}" for market in LOSS_MARKETS))
        print(f"New predictions without full-team history: home {int(new_predictions.home_unknown.sum())}, "
              f"away {int(new_predictions.away_unknown.sum())}")
        bundles[season] = {"predictions": combined, "summary": summary, "bins": bins,
                           "calibration": calibration, "paired": paired, "diagnostics": diagnostics}
    assert_inputs_unchanged(PROJECT_DIR, context)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for season, bundle in bundles.items():
        slug = season.replace("/", "_")
        for name, key in (("comparison_predictions", "predictions"), ("comparison_summary", "summary"),
                          ("calibration_bins", "bins"), ("calibration_summary", "calibration"),
                          ("paired_loss_differences", "paired"), ("fitting_diagnostics", "diagnostics")):
            bundle[key].to_csv(OUTPUT_DIR / f"{name}_{slug}.csv", index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")
    all_summary = pd.concat([bundle["summary"].assign(Season=season) for season, bundle in bundles.items()], ignore_index=True)
    all_summary.to_csv(OUTPUT_DIR / "two_seasons_summary_2023_24_2024_25.csv", index=False, encoding="utf-8-sig")
    settings = {
        "step": 12, "evaluation_role": "development_comparison",
        "evaluation_seasons": list(EVALUATION_SEASONS), "models": list(COMPARISON_MODELS),
        "evaluated_matches": 760, "prediction_rows": 3800,
        "lookback_days": LOOKBACK_DAYS, "half_life_days": HALF_LIFE_DAYS, "ridge_strength": RIDGE_STRENGTH,
        "reference_prior_matches": 5.0, "log_loss_probability_floor": LOG_LOSS_FLOOR,
        "evaluation_weighting": "equal_weight_per_match", "one_x_two_brier_scale": "0_to_2", "binary_brier_scale": "0_to_1",
        "history_rule": "match_date - lookback_days <= Date < match_date", "same_day_results_used": False,
        "fitting_frequency": "once_per_target_matchday", "failed_matches_skipped": 0,
        "same_fixtures_for_all_models": True, "final_test_season_reserved": "2025/26",
        "calibration_bins": N_BINS, "min_bin_matches": MIN_BIN_MATCHES,
        "fits_by_season": {season: len(bundle["diagnostics"]) for season, bundle in bundles.items()},
        "inputs": context,
        "source_sha256": {
            name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest() for name in (
                "analyze_match.py", "time_weighted_model.py", "score_probabilities.py", "evaluation_metrics.py",
                "fitted_poisson_model.py", "fitted_backtest.py", "comparison_reports.py", "run_fitted_backtest.py",
            )
        },
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    }
    (OUTPUT_DIR / "fitted_backtest_settings_2023_24_2024_25.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nTotal evaluated matches: 760 | Prediction rows: 3800")
    print(f"New comparison reports saved in: {OUTPUT_DIR}")
    print("Previous backtests and one-match demonstration reports are retained.")
    print("\nUncertainty in small metric differences has not yet been estimated. No parameters were tuned.")
    print("The separate final-season test follows selection of the final model version.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, pd.errors.ParserError) as error:
        raise SystemExit(f"Comparison error: {error}")
