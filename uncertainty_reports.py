"""Validate the saved Step 12 reports without refitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparison_reports import EVALUATION_SEASONS, load_comparison_inputs
from fitted_backtest import (
    COMPARISON_MODELS, paired_loss_differences, summarize_models, validate_comparison,
)
from validation_data import file_sha256


STEP12_SOURCES = (
    "analyze_match.py", "time_weighted_model.py", "score_probabilities.py", "evaluation_metrics.py",
    "fitted_poisson_model.py", "fitted_backtest.py", "comparison_reports.py", "run_fitted_backtest.py",
)


def validate_step12_settings(settings: dict, previous_context: dict, project_dir: Path) -> None:
    expected = {
        "step": 12, "evaluation_role": "development_comparison",
        "evaluation_seasons": list(EVALUATION_SEASONS), "models": list(COMPARISON_MODELS),
        "evaluated_matches": 760, "prediction_rows": 3800,
        "lookback_days": 730, "half_life_days": 180.0, "ridge_strength": 1.0,
        "reference_prior_matches": 5.0, "log_loss_probability_floor": 1e-15,
        "evaluation_weighting": "equal_weight_per_match",
        "one_x_two_brier_scale": "0_to_2", "binary_brier_scale": "0_to_1",
        "history_rule": "match_date - lookback_days <= Date < match_date",
        "same_day_results_used": False, "failed_matches_skipped": 0,
        "same_fixtures_for_all_models": True, "fitting_frequency": "once_per_target_matchday",
        "final_test_season_reserved": "2025/26",
    }
    if not isinstance(settings, dict):
        raise ValueError("The Step 12 settings file must contain an object.")
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"A Step 12 setting differs: {key}.")
    if settings.get("inputs", {}).get("input_sha256") != previous_context["input_sha256"]:
        raise ValueError("The Step 12 inputs differ from the earlier validated reports.")
    hashes = settings.get("source_sha256", {})
    if set(hashes) != set(STEP12_SOURCES):
        raise ValueError("Incomplete source-code fingerprint in the Step 12 settings.")
    for name in STEP12_SOURCES:
        if hashes[name] != file_sha256(project_dir / name):
            raise ValueError(f"The Step 12 source code changed: {name}.")


def _same_table(calculated: pd.DataFrame, saved: pd.DataFrame, keys: list[str], name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            calculated.sort_values(keys).reset_index(drop=True),
            saved.sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except (AssertionError, KeyError) as error:
        raise ValueError(f"The detailed forecasts and saved report differ: {name}.") from error


def validate_fit_diagnostics(diagnostics: pd.DataFrame, fixtures: pd.DataFrame, season: str) -> None:
    required = {"Season", "fit_date", "fixtures", "history_start", "last_history_date",
                "success", "gradient_max_abs"}
    if (diagnostics.empty or not required.issubset(diagnostics.columns)
            or diagnostics[list(required)].isna().any().any()):
        raise ValueError("Incomplete fitting diagnostics.")
    if (set(diagnostics.Season) != {season} or diagnostics.fit_date.duplicated().any()
            or not diagnostics.success.eq(True).all()):
        raise ValueError("Duplicate or unsuccessful matchday fit.")
    expected = fixtures.groupby("Date").size().sort_index()
    fit_dates = pd.to_datetime(diagnostics.fit_date, format="%Y-%m-%d", errors="raise")
    actual = pd.Series(diagnostics.fixtures.to_numpy(), index=fit_dates).sort_index()
    if not (actual.index.equals(expected.index) and np.array_equal(actual.to_numpy(), expected.to_numpy())):
        raise ValueError("The diagnostic matchdays or match counts differ.")
    last = pd.to_datetime(diagnostics.last_history_date, format="%Y-%m-%d", errors="raise")
    start = pd.to_datetime(diagnostics.history_start, format="%Y-%m-%d", errors="raise")
    if not ((start <= last) & (last < fit_dates)).all():
        raise ValueError("Invalid fitting date boundary.")
    gradient = diagnostics.gradient_max_abs.to_numpy(dtype=float)
    if not (np.isfinite(gradient) & (gradient >= 0) & (gradient <= 1e-4)).all():
        raise ValueError("The saved fit does not meet the numerical accuracy threshold.")


def load_uncertainty_inputs(project_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Compare the five-model tables with the complete fixture list and previous reports."""
    matches, references, previous = load_comparison_inputs(project_dir)
    root = project_dir / "data" / "reports" / "fitted_backtest"
    settings_path = root / "fitted_backtest_settings_2023_24_2024_25.json"
    if not settings_path.is_file():
        raise FileNotFoundError("The Step 12 settings file is missing. Run run_fitted_backtest.py.")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    validate_step12_settings(settings, previous, project_dir)
    paths = [settings_path]
    bundles = {}
    matchdays = {}
    for season in EVALUATION_SEASONS:
        slug = season.replace("/", "_")
        season_paths = {
            name: root / f"{name}_{slug}.csv" for name in (
                "comparison_predictions", "comparison_summary", "paired_loss_differences", "fitting_diagnostics",
            )
        }
        for path in season_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"A Step 12 report is missing: {path.name}.")
        fixtures = matches.loc[matches.Season == season]
        predictions = validate_comparison(
            pd.read_csv(season_paths["comparison_predictions"], encoding="utf-8-sig"), fixtures
        )
        reference = references[season]
        old = predictions.loc[predictions.model != "fitted_poisson", reference.columns]
        _same_table(old, reference, ["match_id", "model"], "previous-model predictions")
        _same_table(
            summarize_models(predictions),
            pd.read_csv(season_paths["comparison_summary"], encoding="utf-8-sig"),
            ["model"], "summary metrics",
        )
        _same_table(
            paired_loss_differences(predictions),
            pd.read_csv(season_paths["paired_loss_differences"], encoding="utf-8-sig"),
            ["reference_model", "market"], "paired means",
        )
        diagnostics = pd.read_csv(season_paths["fitting_diagnostics"], encoding="utf-8-sig")
        validate_fit_diagnostics(diagnostics, fixtures, season)
        if settings.get("fits_by_season", {}).get(season) != len(diagnostics):
            raise ValueError("The number of fits in the settings differs.")
        matchdays[season] = len(diagnostics)
        bundles[season] = predictions
        paths.extend(season_paths.values())
    context = {
        "input_sha256": {
            **previous["input_sha256"],
            **{str(path.relative_to(project_dir)): file_sha256(path) for path in paths},
        },
        "source_sha256": settings["source_sha256"], "matchdays_by_season": matchdays,
        "model_parameters": {key: settings[key] for key in (
            "lookback_days", "half_life_days", "ridge_strength", "reference_prior_matches",
        )},
    }
    return bundles, context


def assert_uncertainty_inputs_unchanged(project_dir: Path, context: dict) -> None:
    for field in ("input_sha256", "source_sha256"):
        for name, original in context[field].items():
            if file_sha256(project_dir / name) != original:
                raise ValueError(f"An input changed during the run: {name}.")
