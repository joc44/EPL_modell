"""Verifiable final-test protocol that remains unchanged on reruns."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import platform

from validation_data import file_sha256


RUNTIME_SOURCES = (
    "main.py", "analyze_match.py", "goal_model.py", "shrinkage_model.py", "time_weighted_model.py",
    "score_probabilities.py", "evaluation_metrics.py", "backtest_models.py", "fitted_poisson_model.py",
    "fitted_backtest.py", "validation_data.py",
)
SELECTION_SOURCES = ("model_selection_reports.py", "final_test_protocol.py", "freeze_final_test.py")
PROTOCOL_FILENAME = "final_test_protocol_2025_26.json"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def current_runtime() -> dict:
    import numpy as np
    import pandas as pd
    import scipy
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


def fixed_policy() -> dict:
    """Selection and evaluation rules, without using the separate test results."""
    return {
        "protocol_schema_version": 1, "protocol_name": "EPL_fitted_poisson_final_test_v1", "step": 17,
        "selection_role": "development_choice_before_separate_test",
        "development_seasons_reviewed": ["2023/24", "2024/25"], "final_test_season": "2025/26",
        "final_test_results_used_for_selection": False, "selected_model": "fitted_poisson",
        "reference_models": ["league_average", "time_weighted"], "final_test_models": ["league_average", "time_weighted", "fitted_poisson"],
        "selected_parameters": {"lookback_days": 730, "half_life_days": 180.0, "ridge_strength": 1.0},
        "reference_parameters": {
            "league_average": {"lookback_days": 730, "history_weighting": "equal_weight_per_history_match"},
            "time_weighted": {"lookback_days": 730, "half_life_days": 180.0, "prior_matches": 5.0},
        },
        "fit_policy": {
            "fitting_frequency": "once_per_target_matchday", "history_rule": "match_date - lookback_days <= Date < match_date",
            "same_day_results_used": False, "past_final_test_matches_may_enter_later_history": True,
            "hyperparameters_updated_during_final_test": False, "team_universe": "history_only",
            "unknown_team_policy": "neutral_log_effects_zero_and_explicit_unknown_flag",
            "objective_weighting": "sum_of_time_weighted_match_log_likelihood_terms",
            "ridge_penalty": "0.5 * ridge_strength * sum(attack_log**2 + concede_log**2)",
            "attack_and_concede_log_sums": 0.0, "optimizer": "L-BFGS-B_with_existing_newton_refinement",
            "required_gradient_max_abs": 1e-4, "failed_fit_policy": "stop_without_skipping_or_substitution",
            "failed_matches_skipped": 0,
        },
        "data_policy": {
            "provider": "football-data.co.uk", "league_code": "E0", "raw_filename": "E0_2025_2026.csv",
            "processed_filename": "matches_final_test.csv", "raw_date_format": "%d/%m/%Y",
            "required_raw_columns": ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
            "expected_final_test_matches": 380, "expected_final_test_teams": 20,
            "fixture_rule": "each_distinct_ordered_home_away_team_pair_exactly_once",
            "initial_history_seasons": ["2021/22", "2022/23", "2023/24", "2024/25"],
            "previous_processed_data_must_remain_unchanged": True, "identical_duplicates": "remove_and_report_count",
            "conflicting_duplicate_or_missing_required_data": "stop", "same_fixtures_for_all_models": True,
        },
        "evaluation_policy": {
            "primary_market": "one_x_two", "primary_metric": "log_loss", "primary_reference": "league_average",
            "secondary_markets": ["over_1_5", "over_2_5", "over_3_5", "btts", "exact_score"],
            "other_metrics": ["brier", "one_x_two_accuracy", "exact_score_top_one_accuracy", "calibration"],
            "all_secondary_results_reported": True, "evaluation_weighting": "equal_weight_per_match",
            "difference_sign": "selected_model_minus_reference", "lower_loss_is_better": True,
            "log_loss_probability_floor": 1e-15, "clip_counts_reported_per_market": True,
            "one_x_two_brier_scale": "sum_of_three_squares_0_to_2", "binary_brier_scale": "single_square_0_to_1",
            "exact_score_probability": "analytic_untruncated_independent_poisson_product",
            "goal_lines": [1.5, 2.5, 3.5], "argmax_one_x_two_tie_order": ["home", "draw", "away"],
            "top_score_tie_order": ["home_goals_ascending", "away_goals_ascending"],
            "grid_tail_tolerance_per_team": 1e-10, "grid_max_goals": 200, "grid_normalized": True,
            "calibration_bins": 10, "min_bin_matches": 20,
        },
        "uncertainty_policy": {
            "method": "non_circular_overlapping_moving_blocks_of_matchdays", "pairing": "match_id",
            "block_lengths_matchdays": [3, 5, 10], "primary_block_length_matchdays": 5,
            "n_resamples_per_length": 5000, "confidence_level": 0.95, "interval_method": "percentile",
            "quantile_method": "linear", "base_seed": 42, "seed_rule": "base_seed + block_length",
            "resampling_unit": "complete_matchday", "statistic_weighting": "equal_weight_per_resampled_match",
            "resampled_matchday_count": "same_as_original_season",
            "resampled_match_count": "varies_with_number_of_matches_on_selected_days",
            "last_sampled_block_truncated": True, "blocks_cross_season_boundary": False,
            "same_resample_indices_for_all_comparisons": True, "models_refitted_in_bootstrap": False,
            "intervals_are_simultaneous": False, "multiple_comparison_adjustment": False,
            "conditioning": "fixed_final_test_forecasts; excludes_model_selection_and_refitting_uncertainty",
            "assumption": "approximately stable loss-difference process with short-range dependence",
        },
        "interpretation_policy": {
            "primary_point_improvement": "one_x_two_mean_loss_difference_vs_league_average < 0",
            "primary_interval_support": "primary_5_matchday_ci_upper_vs_league_average < 0",
            "block_length_robust_support": "ci_upper_vs_league_average < 0_for_all_3_5_10_lengths",
            "no_automatic_success_guarantee": True, "zero_in_interval_does_not_establish_equivalence": True,
            "no_model_reranking_after_test": True, "changes_after_reviewing_test_require_new_development_cycle": True,
            "same_test_rerun_role": "reproduction_only_not_new_independent_test",
        },
        "selection_rationale": [
            "fitted_poisson_has_lower_development_1x2_loss_than_the_four_original_models_on_both_seasons",
            "dc_minus_poisson_1x2_intervals_include_zero_on_both_seasons_and_all_three_block_lengths",
            "prefer_simpler_joint_fit_without_additional_rho_or_tau_constraints_for_this_version",
            "single_coherent_score_distribution_for_all_reported_markets",
        ],
    }


def _validate_hash_map(mapping: dict) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A non-empty file-fingerprint table is required.")
    for name, digest in mapping.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid filename in the fingerprints.")
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in name or str(path) == ".":
            raise ValueError("Protocol file references must be paths relative to the project.")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid SHA-256 fingerprint.")


def validate_protocol_payload(payload: dict) -> None:
    policy = fixed_policy()
    if not isinstance(payload, dict) or set(payload) != {*policy, "frozen_context", "runtime"}:
        raise ValueError("Incomplete or unexpected protocol field.")
    if canonical_json({key: payload[key] for key in policy}) != canonical_json(policy):
        raise ValueError("The protocol differs from the fixed model-selection and test rules.")
    context = payload["frozen_context"]
    if not isinstance(context, dict) or set(context) != {"input_sha256", "source_sha256"}:
        raise ValueError("Incomplete frozen input context.")
    for hashes in context.values():
        _validate_hash_map(hashes)
    if not {*RUNTIME_SOURCES, *SELECTION_SOURCES}.issubset(context["source_sha256"]):
        raise ValueError("Incomplete runtime or model-selection source-code fingerprints.")
    runtime = payload["runtime"]
    if (not isinstance(runtime, dict) or set(runtime) != {"python", "pandas", "numpy", "scipy"}
            or not all(isinstance(value, str) and value for value in runtime.values())):
        raise ValueError("Incomplete numerical environment description.")


def build_protocol_payload(project_dir: Path, context: dict) -> dict:
    sources = {**context["source_sha256"],
               **{name: file_sha256(project_dir / name) for name in (*RUNTIME_SOURCES, *SELECTION_SOURCES)}}
    payload = {**fixed_policy(), "frozen_context": {"input_sha256": dict(context["input_sha256"]), "source_sha256": sources},
               "runtime": current_runtime()}
    validate_protocol_payload(payload)
    return payload


def assert_frozen_context_unchanged(project_dir: Path, payload: dict, check_runtime: bool = True) -> None:
    validate_protocol_payload(payload)
    for mapping in payload["frozen_context"].values():
        for name, digest in mapping.items():
            path = project_dir / Path(*PurePosixPath(name.replace("\\", "/")).parts)
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"A frozen input or source file changed: {name}.")
    if check_runtime and current_runtime() != payload["runtime"]:
        raise ValueError("The numerical environment differs from the protocol; use the frozen virtual environment (venv).")


def load_frozen_protocol(path: Path) -> dict:
    checksum_path = path.with_suffix(".sha256")
    if not path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("The protocol and its .sha256 file are both required.")
    raw = path.read_bytes()
    if checksum_path.read_text(encoding="utf-8").strip() != sha256(raw).hexdigest():
        raise ValueError("The frozen protocol fingerprint differs.")
    result = json.loads(raw.decode("utf-8"))
    if not isinstance(result, dict) or set(result) != {"protocol_id", "created_at_utc", "payload"}:
        raise ValueError("Invalid protocol envelope.")
    validate_protocol_payload(result["payload"])
    if result["protocol_id"] != sha256(canonical_json(result["payload"])).hexdigest():
        raise ValueError("The protocol identifier differs from its content.")
    created = datetime.fromisoformat(result["created_at_utc"])
    if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
        raise ValueError("The freezing timestamp must be in UTC.")
    return result


def freeze_or_verify_protocol(path: Path, payload: dict) -> tuple[dict, bool]:
    """Create on the first run; validate later without rewriting the protocol."""
    validate_protocol_payload(payload)
    checksum_path = path.with_suffix(".sha256")
    if path.exists() or checksum_path.exists():
        existing = load_frozen_protocol(path)
        if canonical_json(existing["payload"]) != canonical_json(payload):
            raise ValueError("A different frozen protocol already exists. It will not be overwritten.")
        return existing, False
    result = {"protocol_id": sha256(canonical_json(payload)).hexdigest(),
              "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "payload": payload}
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    created_paths = []
    try:
        with path.open("xb") as output:
            created_paths.append(path)
            output.write(raw)
        with checksum_path.open("xb") as output:
            created_paths.append(checksum_path)
            output.write((sha256(raw).hexdigest() + "\n").encode("ascii"))
    except OSError:
        for created_path in reversed(created_paths):
            created_path.unlink(missing_ok=True)
        raise
    return result, True
