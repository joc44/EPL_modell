"""Step 15: Dixon–Coles development backtest on two complete seasons."""

import json
from pathlib import Path
import platform

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "reports" / "dixon_coles_backtest"
LOOKBACK_DAYS = 730
HALF_LIFE_DAYS = 180.0
RIDGE_STRENGTH = 1.0
N_BINS = 10
MIN_BIN_MATCHES = 20


def main() -> None:
    try:
        import numpy as np
        import scipy
        from backtest_models import LOSS_MARKETS
        from calibration_analysis import EVENTS
        from comparison_reports import EVALUATION_SEASONS
        from dixon_coles_backtest import (
            COMPARISON_LABELS, COMPARISON_MODELS, NEW_MODEL, calibration_models, correction_effects,
            paired_loss_differences, run_dixon_coles_backtest, validate_dixon_coles_comparison,
        )
        from dixon_coles_model import MINIMUM_TAU
        from dixon_coles_reports import load_dixon_coles_inputs, validate_dixon_coles_diagnostics
        from evaluation_metrics import LOG_LOSS_FLOOR
        from fitted_backtest import summarize_models
        from fitted_poisson_model import GRADIENT_TOLERANCE
        from run_fitted_backtest import print_summary, show_progress
        from uncertainty_reports import assert_uncertainty_inputs_unchanged
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in ("numpy", "scipy"):
            raise SystemExit("Missing dependency. Install requirements_step_11.txt in the project virtual environment.") from None
        raise
    print("Step 15: chronological Dixon–Coles backtest", flush=True)
    print(f"Seasons: {', '.join(EVALUATION_SEASONS)} | Models: {len(COMPARISON_MODELS)}")
    print(f"History window: {LOOKBACK_DAYS} days | Half-life: {HALF_LIFE_DAYS:g} days | L2 strength: {RIDGE_STRENGTH:g}")
    print("Reuse the previous five models' saved predictions after validation.")
    print("Fit a new joint DC model before every matchday; same-day results are excluded.")
    print("Both seasons are development comparisons. The separate final test remains 2025/26.", flush=True)
    matches, references, context = load_dixon_coles_inputs(PROJECT_DIR)
    parameters = context["model_parameters"]
    if (LOG_LOSS_FLOOR != 1e-15 or parameters["lookback_days"] != LOOKBACK_DAYS
            or parameters["half_life_days"] != HALF_LIFE_DAYS or parameters["ridge_strength"] != RIDGE_STRENGTH):
        raise ValueError("The new run's parameters differ from the validated references.")
    bundles = {}
    for season in EVALUATION_SEASONS:
        print(f"\nEvaluation season: {season}", flush=True)
        predictions, diagnostics = run_dixon_coles_backtest(
            matches, season, LOOKBACK_DAYS, HALF_LIFE_DAYS, RIDGE_STRENGTH, show_progress
        )
        fixtures = matches.loc[matches.Season == season]
        combined = validate_dixon_coles_comparison(pd.concat([references[season], predictions], ignore_index=True), fixtures)
        validate_dixon_coles_diagnostics(diagnostics, combined, fixtures, season)
        summary = summarize_models(combined, COMPARISON_MODELS)
        bins, calibration = calibration_models(combined, N_BINS, MIN_BIN_MATCHES)
        paired, effects = paired_loss_differences(combined), correction_effects(combined)
        print(f"\nEvaluated matches: {len(fixtures)}")
        print(f"Prediction rows: {len(combined)} ({len(COMPARISON_MODELS)} models × {len(fixtures)} matches)")
        print(f"Successful matchday DC fits: {len(diagnostics)}/{len(diagnostics)}")
        print(f"Matchdays with active tau constraints: {int((diagnostics.active_tau_constraints > 0).sum())}")
        print(f"Rho – minimum: {diagnostics.rho.min():+.6f} | median: {diagnostics.rho.median():+.6f} | maximum: {diagnostics.rho.max():+.6f}")
        print(f"Largest stationarity residual: {diagnostics.stationarity_max_abs.max():.3e}")
        print(f"Minimum tau across all daily pairings: {diagnostics.minimum_tau_all_pairs.min():.6f}")
        print_summary(summary, COMPARISON_LABELS)
        print("\nDixon–Coles minus jointly fitted Poisson – paired log loss:")
        for row in paired.loc[paired.reference_model == "fitted_poisson"].to_dict("records"):
            print(f"{row['market']}: {row['mean_loss_difference']:+.4f}")
        print("A negative difference favours the Dixon–Coles model.")
        print("\nChange decomposition – new lambdas | tau correction | total difference:")
        for row in effects.to_dict("records"):
            print(f"{row['market']}: {row['lambda_refit_loss_difference']:+.6f} | "
                  f"{row['tau_correction_loss_difference']:+.6f} | {row['total_loss_difference']:+.6f}")
        print("Uncorrected DC lambdas are a diagnostic decomposition, not a separately trained model.")
        print("\nDixon–Coles model – calibration:")
        for row in calibration.loc[(calibration.model == NEW_MODEL) & calibration.event.isin(("home_win", "over_2_5", "btts"))].to_dict("records"):
            print(f"{EVENTS[row['event']][0]}: predicted {row['mean_probability']:.2%} | observed {row['observed_rate']:.2%} | "
                  f"difference {100 * row['bias']:+.2f} pp | ECE {100 * row['ece']:.2f} pp")
        print("\nDC log loss probability-floor usage:")
        print(", ".join(f"{market}: {int(predictions[f'{market}_log_loss_clipped'].sum())}" for market in LOSS_MARKETS))
        print(f"DC predictions without full-team history: home {int(predictions.home_unknown.sum())}, "
              f"away {int(predictions.away_unknown.sum())}")
        bundles[season] = {"predictions": combined, "summary": summary, "bins": bins,
                           "calibration": calibration, "paired": paired, "effects": effects, "diagnostics": diagnostics}
    assert_uncertainty_inputs_unchanged(PROJECT_DIR, context)
    # Write reports only after complete, validated runs for both seasons.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for season, bundle in bundles.items():
        slug = season.replace("/", "_")
        for name, key in (("comparison_predictions", "predictions"), ("comparison_summary", "summary"),
                          ("calibration_bins", "bins"), ("calibration_summary", "calibration"),
                          ("paired_loss_differences", "paired"), ("correction_effects", "effects"),
                          ("fitting_diagnostics", "diagnostics")):
            bundle[key].to_csv(OUTPUT_DIR / f"{name}_{slug}.csv", index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")
    pd.concat([bundle["summary"].assign(Season=season) for season, bundle in bundles.items()], ignore_index=True).to_csv(
        OUTPUT_DIR / "two_seasons_summary_2023_24_2024_25.csv", index=False, encoding="utf-8-sig"
    )
    evaluated = sum(int(bundle["predictions"].match_id.nunique()) for bundle in bundles.values())
    prediction_rows = sum(len(bundle["predictions"]) for bundle in bundles.values())
    settings = {
        "step": 15, "evaluation_role": "development_comparison", "evaluation_seasons": list(EVALUATION_SEASONS),
        "models": list(COMPARISON_MODELS), "evaluated_matches": evaluated, "prediction_rows": prediction_rows,
        "lookback_days": LOOKBACK_DAYS, "half_life_days": HALF_LIFE_DAYS, "ridge_strength": RIDGE_STRENGTH,
        "reference_prior_matches": parameters["reference_prior_matches"], "log_loss_probability_floor": LOG_LOSS_FLOOR,
        "evaluation_weighting": "equal_weight_per_match", "one_x_two_brier_scale": "0_to_2", "binary_brier_scale": "0_to_1",
        "history_rule": "match_date - lookback_days <= Date < match_date", "same_day_results_used": False,
        "fitting_frequency": "once_per_target_matchday", "failed_matches_skipped": 0,
        "same_fixtures_for_all_models": True, "final_test_season_reserved": "2025/26",
        "dc_exact_score_rule": "analytic_untruncated_poisson_product_times_tau",
        "dc_joint_fit": True, "rho_penalized": False, "tau_constraint_floor": MINIMUM_TAU,
        "stationarity_acceptance_tolerance": GRADIENT_TOLERANCE,
        "correction_effects_role": "diagnostic_decomposition_not_separate_fitted_model",
        "calibration_bins": N_BINS, "min_bin_matches": MIN_BIN_MATCHES,
        "fits_by_season": {season: len(bundle["diagnostics"]) for season, bundle in bundles.items()},
        "inputs": context, "source_sha256": context["source_sha256"],
        "uncertainty_intervals_computed": False, "hyperparameters_tuned": False,
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    }
    (OUTPUT_DIR / "dixon_coles_backtest_settings_2023_24_2024_25.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nTotal evaluated matches: {evaluated} | Prediction rows: {prediction_rows}")
    print(f"New reports saved in: {OUTPUT_DIR}")
    print("Previous reports are retained. Predictions from the previous five models are unchanged.")
    print("\nThis is a development comparison. No parameters were tuned in this step.")
    print("Uncertainty in the DC differences has not yet been estimated; Step 13 intervals measured a different comparison.")
    print("The separate 2025/26 final test follows selection of the model version.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, pd.errors.ParserError) as error:
        raise SystemExit(f"Dixon–Coles backtest error: {error}")
