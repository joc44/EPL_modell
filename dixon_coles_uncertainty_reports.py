"""Validate the complete Step 15 report chain without fitting models."""

import json
from pathlib import Path

import pandas as pd

from comparison_reports import EVALUATION_SEASONS
from dixon_coles_backtest import (
    COMPARISON_MODELS, calibration_models, correction_effects, paired_loss_differences,
    validate_dixon_coles_comparison,
)
from dixon_coles_reports import load_dixon_coles_inputs, validate_dixon_coles_diagnostics
from fitted_backtest import summarize_models
from uncertainty_reports import _same_table, assert_uncertainty_inputs_unchanged
from validation_data import file_sha256


def validate_step15_settings(settings: dict, previous: dict) -> None:
    """Fixed evaluation rules, inputs and source-code identity."""
    expected = {
        "step": 15, "evaluation_role": "development_comparison", "evaluation_seasons": list(EVALUATION_SEASONS),
        "models": list(COMPARISON_MODELS), "evaluated_matches": 760, "prediction_rows": 4560,
        "lookback_days": 730, "half_life_days": 180.0, "ridge_strength": 1.0, "reference_prior_matches": 5.0,
        "log_loss_probability_floor": 1e-15, "evaluation_weighting": "equal_weight_per_match",
        "one_x_two_brier_scale": "0_to_2", "binary_brier_scale": "0_to_1",
        "history_rule": "match_date - lookback_days <= Date < match_date", "same_day_results_used": False,
        "fitting_frequency": "once_per_target_matchday", "failed_matches_skipped": 0,
        "same_fixtures_for_all_models": True, "final_test_season_reserved": "2025/26",
        "dc_exact_score_rule": "analytic_untruncated_poisson_product_times_tau", "dc_joint_fit": True,
        "rho_penalized": False, "tau_constraint_floor": 1e-8, "stationarity_acceptance_tolerance": 1e-4,
        "correction_effects_role": "diagnostic_decomposition_not_separate_fitted_model",
        "calibration_bins": 10, "min_bin_matches": 20, "hyperparameters_tuned": False,
        "uncertainty_intervals_computed": False,
    }
    if not isinstance(settings, dict):
        raise ValueError("The Step 15 settings file must contain an object.")
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"A Step 15 setting differs: {key}.")
    inputs = settings.get("inputs", {})
    if not isinstance(inputs, dict) or inputs.get("input_sha256") != previous["input_sha256"]:
        raise ValueError("The Step 15 inputs differ from the validated history.")
    if (settings.get("source_sha256") != previous["source_sha256"]
            or inputs.get("source_sha256") != previous["source_sha256"]):
        raise ValueError("The Step 15 source-code fingerprints differ.")
    if settings.get("fits_by_season") != previous["matchdays_by_season"]:
        raise ValueError("The Step 15 matchday fit counts differ.")


def load_dixon_coles_uncertainty_inputs(project_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Validate six-model probabilities, the previous five models, metrics and daily diagnostics."""
    matches, references, previous = load_dixon_coles_inputs(project_dir)
    root = project_dir / "data" / "reports" / "dixon_coles_backtest"
    settings_path = root / "dixon_coles_backtest_settings_2023_24_2024_25.json"
    if not settings_path.is_file():
        raise FileNotFoundError("The Step 15 report is missing. Run run_dixon_coles_backtest.py first.")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    validate_step15_settings(settings, previous)
    bundles, paths = {}, [settings_path]
    for season in EVALUATION_SEASONS:
        slug = season.replace("/", "_")
        season_paths = {name: root / f"{name}_{slug}.csv" for name in (
            "comparison_predictions", "comparison_summary", "calibration_bins", "calibration_summary",
            "paired_loss_differences", "correction_effects", "fitting_diagnostics",
        )}
        for path in season_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"A Step 15 report is missing: {path.name}.")
        fixtures = matches.loc[matches.Season == season]
        read = lambda name: pd.read_csv(season_paths[name], encoding="utf-8-sig")
        predictions = validate_dixon_coles_comparison(read("comparison_predictions"), fixtures)
        reference = references[season]
        old = predictions.loc[predictions.model != "dixon_coles", reference.columns]
        _same_table(old, reference, ["match_id", "model"], "previous five-model predictions")
        _same_table(summarize_models(predictions, COMPARISON_MODELS), read("comparison_summary"), ["model"], "DC summary")
        bins, calibration = calibration_models(predictions, 10, 20)
        _same_table(bins, read("calibration_bins"), ["model", "event", "bin"], "DC calibration bins")
        _same_table(calibration, read("calibration_summary"), ["model", "event"], "DC calibration summary")
        _same_table(paired_loss_differences(predictions), read("paired_loss_differences"),
                    ["reference_model", "market"], "DC paired means")
        _same_table(correction_effects(predictions), read("correction_effects"), ["market"], "DC correction decomposition")
        validate_dixon_coles_diagnostics(read("fitting_diagnostics"), predictions, fixtures, season)
        bundles[season] = predictions
        paths.extend(season_paths.values())
    combined_path = root / "two_seasons_summary_2023_24_2024_25.csv"
    if not combined_path.is_file():
        raise FileNotFoundError("The Step 15 two-season summary is missing.")
    calculated = pd.concat([summarize_models(table, COMPARISON_MODELS).assign(Season=season)
                            for season, table in bundles.items()], ignore_index=True)
    _same_table(calculated, pd.read_csv(combined_path, encoding="utf-8-sig"), ["Season", "model"], "two-season DC summary")
    paths.append(combined_path)
    context = {
        **previous,
        "input_sha256": {**previous["input_sha256"], **{str(path.relative_to(project_dir)): file_sha256(path) for path in paths}},
    }
    assert_uncertainty_inputs_unchanged(project_dir, context)
    return bundles, context
