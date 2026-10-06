"""Reproduce the Step 16 reports and model-selection evidence."""

import json
from pathlib import Path

import pandas as pd

from backtest_models import LOSS_MARKETS, MODEL_NAMES
from dixon_coles_backtest import COMPARISON_MODELS, PREVIOUS_MODELS
from dixon_coles_uncertainty import bootstrap_loss_intervals, build_paired_losses, summarize_sensitivity
from dixon_coles_uncertainty_reports import load_dixon_coles_uncertainty_inputs
from fitted_backtest import summarize_models
from uncertainty_reports import _same_table, assert_uncertainty_inputs_unchanged
from validation_data import file_sha256


STEP16_SOURCES = (
    "uncertainty_analysis.py", "dixon_coles_uncertainty.py", "dixon_coles_uncertainty_reports.py",
    "analyze_dixon_coles_uncertainty.py",
)
BLOCK_LENGTHS = (3, 5, 10)


def validate_step16_settings(settings: dict, previous: dict, project_dir: Path) -> None:
    expected = {
        "step": 16, "evaluation_role": "exploratory_development_diagnostics",
        "evaluation_seasons": ["2023/24", "2024/25"], "new_model": "dixon_coles",
        "reference_models": list(PREVIOUS_MODELS), "focus_reference": "fitted_poisson",
        "markets": list(LOSS_MARKETS), "evaluated_matches": 760, "paired_match_rows": 3800, "interval_rows": 180,
        "block_lengths_matchdays": list(BLOCK_LENGTHS), "primary_block_length_matchdays": 5,
        "n_resamples_per_season_and_length": 5000, "confidence_level": 0.95, "base_seed": 42,
        "seed_rule": "base_seed + 1000 * sorted_season_index + block_length",
        "bootstrap_method": "non_circular_overlapping_moving_blocks_of_matchdays",
        "interval_method": "percentile", "quantile_method": "linear", "pairing": "match_id",
        "difference_sign": "dixon_coles_minus_reference", "resampling_unit": "complete_matchday",
        "statistic_weighting": "equal_weight_per_resampled_match", "resampled_matchday_count": "same_as_original_season",
        "resampled_match_count": "varies_with_number_of_matches_on_selected_days",
        "last_sampled_block_truncated": True, "blocks_cross_season_boundary": False,
        "same_resample_indices_for_all_comparisons": True, "intervals_are_simultaneous": False,
        "multiple_comparison_adjustment": False, "bootstrap_negative_fraction_is_reported_as_probability": False,
        "models_refitted_in_bootstrap": False, "model_parameters_changed": False,
        "conditioning": "fixed_saved_forecasts; excludes model_selection_and_refitting_uncertainty",
        "assumption": "approximately stable loss-difference process with short-range dependence",
        "final_test_season_reserved": "2025/26",
    }
    if not isinstance(settings, dict):
        raise ValueError("The Step 16 settings file must contain an object.")
    for key, value in expected.items():
        if json.dumps(settings.get(key), sort_keys=True, allow_nan=False) != json.dumps(value, sort_keys=True, allow_nan=False):
            raise ValueError(f"A Step 16 setting differs: {key}.")
    source_hashes = {**previous["source_sha256"],
                     **{name: file_sha256(project_dir / name) for name in STEP16_SOURCES}}
    expected_context = {**previous, "source_sha256": source_hashes}
    if settings.get("source_context") != expected_context or settings.get("source_sha256") != source_hashes:
        raise ValueError("The Step 16 input or source-code fingerprints differ.")


def load_selection_inputs(project_dir: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Validate the complete report chains for both seasons without refitting."""
    predictions, previous = load_dixon_coles_uncertainty_inputs(project_dir)
    root = project_dir / "data" / "reports" / "dixon_coles_uncertainty"
    settings_path = root / "dixon_coles_uncertainty_settings_2023_24_2024_25.json"
    if not settings_path.is_file():
        raise FileNotFoundError("The Step 16 report is missing. Run analyze_dixon_coles_uncertainty.py.")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    validate_step16_settings(settings, previous, project_dir)
    paths, tables = [settings_path], []
    for season_index, (season, table) in enumerate(sorted(predictions.items())):
        slug = season.replace("/", "_")
        season_paths = {name: root / f"{name}_{slug}.csv" for name in (
            "paired_match_losses", "bootstrap_intervals", "block_sensitivity",
        )}
        for path in season_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"A Step 16 report is missing: {path.name}.")
        paired = build_paired_losses(table)
        saved_paired = pd.read_csv(season_paths["paired_match_losses"], encoding="utf-8-sig")
        saved_paired["Date"] = pd.to_datetime(saved_paired.Date, format="%Y-%m-%d", errors="raise")
        _same_table(paired, saved_paired, ["match_id", "reference_model"], "Step 16 paired losses")
        by_length = [bootstrap_loss_intervals(paired, length, 5000, 0.95, 42 + 1000 * season_index + length)
                     for length in BLOCK_LENGTHS]
        calculated = pd.concat(by_length, ignore_index=True)
        _same_table(calculated, pd.read_csv(season_paths["bootstrap_intervals"], encoding="utf-8-sig"),
                    ["reference_model", "market", "block_length_matchdays"], "Step 16 intervals")
        _same_table(summarize_sensitivity(calculated, BLOCK_LENGTHS),
                    pd.read_csv(season_paths["block_sensitivity"], encoding="utf-8-sig"),
                    ["reference_model", "market"], "Step 16 block-length sensitivity")
        tables.append(calculated)
        paths.extend(season_paths.values())
    intervals = pd.concat(tables, ignore_index=True)
    combined_path = root / "bootstrap_intervals_2023_24_2024_25.csv"
    figure_path = root / "uncertainty_dixon_coles_vs_poisson_2023_24_2024_25.png"
    if not combined_path.is_file() or not figure_path.is_file():
        raise FileNotFoundError("The combined Step 16 interval table or figure is missing.")
    _same_table(intervals, pd.read_csv(combined_path, encoding="utf-8-sig"),
                ["Season", "reference_model", "market", "block_length_matchdays"], "two-season intervals")
    if not figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The Step 16 figure is not a PNG file.")
    paths.extend((combined_path, figure_path))
    context = {
        **settings["source_context"],
        "input_sha256": {**previous["input_sha256"], **{str(path.relative_to(project_dir)): file_sha256(path) for path in paths}},
    }
    assert_uncertainty_inputs_unchanged(project_dir, context)
    return predictions, intervals, context


def selection_evidence(predictions: dict, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validated reasons for selecting Poisson; no claim that DC is the winner."""
    if set(predictions) != {"2023/24", "2024/25"}:
        raise ValueError("Both development seasons are required for model selection.")
    focus = intervals.loc[(intervals.new_model == "dixon_coles") & (intervals.reference_model == "fitted_poisson")].copy()
    keys = ["Season", "market", "block_length_matchdays"]
    expected = {(season, market, length) for season in predictions for market in LOSS_MARKETS for length in BLOCK_LENGTHS}
    if focus.duplicated(keys).any() or set(map(tuple, focus[keys].to_numpy())) != expected:
        raise ValueError("Incomplete DC–Poisson uncertainty evidence.")
    primary = focus.loc[focus.market == "one_x_two"]
    if not ((primary.ci_lower <= 0) & (primary.ci_upper >= 0)).all():
        raise ValueError("The 1X2 uncertainty result has changed; review the selection rationale.")
    summaries = []
    for season, table in sorted(predictions.items()):
        summary = summarize_models(table, COMPARISON_MODELS)
        indexed = summary.set_index("model")
        selected_loss = indexed.loc["fitted_poisson", "one_x_two_log_loss"]
        if not all(selected_loss < indexed.loc[name, "one_x_two_log_loss"] for name in MODEL_NAMES):
            raise ValueError("The development Poisson 1X2 result has changed; review the selection rationale.")
        summary["development_one_x_two_rank"] = summary.one_x_two_log_loss.rank(method="min").astype(int)
        summary["selected_for_final_test"] = summary.model.eq("fitted_poisson")
        summary["final_reference"] = summary.model.isin(("league_average", "time_weighted"))
        summary["Season"] = season
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True), focus.sort_values(keys).reset_index(drop=True)
