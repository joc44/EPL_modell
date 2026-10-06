"""Validate previous reports before the five-model comparison."""

import json
from pathlib import Path

import pandas as pd

from analyze_match import load_clean_matches
from backtest_models import MODEL_NAMES
from fitted_backtest import summarize_models, validate_comparison
from validation_data import (
    assert_base_preserved, file_sha256, load_development_checkpoint, validate_extended_matches,
)


EVALUATION_SEASONS = ("2023/24", "2024/25")


def load_comparison_inputs(project_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Require the same data and evaluation rules for the old and new models."""
    processed = project_dir / "data" / "processed"
    reports = project_dir / "data" / "reports"
    base_path = processed / "matches_clean.csv"
    data_path = processed / "matches_validation.csv"
    _, _, settings_2023 = load_development_checkpoint(base_path, reports)
    if not data_path.is_file():
        raise FileNotFoundError("matches_validation.csv is missing. Run Step 10/a first.")
    matches = load_clean_matches(data_path)
    validate_extended_matches(matches)
    assert_base_preserved(load_clean_matches(base_path), matches)
    settings_path = reports / "backtest_settings_2024_25.json"
    if not settings_path.is_file():
        raise FileNotFoundError("The 2024/25 report is missing. Run validate_next_season.py first.")
    settings_2024 = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings_2024, dict) or settings_2024.get("evaluation_season") != "2024/25":
        raise ValueError("The settings file for the later season is unusable.")
    if settings_2024.get("evaluation_role") != "later_season_validation":
        raise ValueError("The 2024/25 source report must come from the Step 10 validation.")
    shared_keys = (
        "models", "evaluated_matches", "lookback_days", "prior_matches", "half_life_days",
        "history_rule", "same_day_results_used", "evaluation_weighting", "log_loss_probability_floor",
        "one_x_two_brier_scale", "binary_brier_scale",
    )
    for key in shared_keys:
        if settings_2024.get(key) != settings_2023[key]:
            raise ValueError(f"Settings differ between the previous seasons: {key}.")
    if settings_2024.get("data_file") != data_path.name or settings_2024.get("data_sha256") != file_sha256(data_path):
        raise ValueError("The extended data file differs from the file used in Step 10.")
    for key, path in {
        "development_data_sha256": base_path,
        "development_settings_sha256": reports / "backtest_settings_2023_24.json",
        "development_predictions_sha256": reports / "backtest_predictions_2023_24.csv",
        "development_summary_sha256": reports / "backtest_summary_2023_24.csv",
    }.items():
        if settings_2024.get(key) != file_sha256(path):
            raise ValueError(f"The development source has changed since Step 10: {path.name}.")
    # These modules are also used in the new calculation. Changes to the old code
    # require a separate run; do not mix them with earlier saved predictions.
    source_hashes = settings_2024.get("model_source_sha256", {})
    for name in ("analyze_match.py", "goal_model.py", "score_probabilities.py", "shrinkage_model.py",
                 "time_weighted_model.py", "evaluation_metrics.py", "backtest_models.py"):
        if source_hashes.get(name) != file_sha256(project_dir / name):
            raise ValueError(f"The previous model module has changed since Step 10: {name}.")

    input_paths = [base_path, data_path]
    reference_predictions = {}
    for season in EVALUATION_SEASONS:
        slug = season.replace("/", "_")
        predictions_path = reports / f"backtest_predictions_{slug}.csv"
        summary_path = reports / f"backtest_summary_{slug}.csv"
        season_settings_path = reports / f"backtest_settings_{slug}.json"
        for path in (predictions_path, summary_path, season_settings_path):
            if not path.is_file():
                raise FileNotFoundError(f"A previous report is missing: {path}.")
        fixtures = matches.loc[matches.Season == season]
        reference = validate_comparison(pd.read_csv(predictions_path, encoding="utf-8-sig"), fixtures, MODEL_NAMES)
        calculated = summarize_models(reference, MODEL_NAMES).set_index("model").sort_index()
        saved = pd.read_csv(summary_path, encoding="utf-8-sig").set_index("model").sort_index()
        try:
            pd.testing.assert_frame_equal(calculated, saved, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValueError(f"A {season} summary differs from the values calculated from the detailed report.") from error
        reference_predictions[season] = reference
        input_paths.extend((predictions_path, summary_path, season_settings_path))
    context = {
        "input_sha256": {str(path.relative_to(project_dir)): file_sha256(path) for path in input_paths},
        "source_settings": {"2023/24": settings_2023, "2024/25": settings_2024},
    }
    return matches, reference_predictions, context


def assert_inputs_unchanged(project_dir: Path, context: dict) -> None:
    for relative_path, original_hash in context["input_sha256"].items():
        if file_sha256(project_dir / relative_path) != original_hash:
            raise ValueError(f"An input changed during the run: {relative_path}.")
