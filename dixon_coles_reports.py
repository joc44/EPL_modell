"""Load the previous five-model reports and validate the DC fits."""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_match import load_clean_matches
from dixon_coles_backtest import NEW_MODEL
from dixon_coles_model import CONSTRAINT_TOLERANCE, MINIMUM_TAU
from fitted_poisson_model import GRADIENT_TOLERANCE
from uncertainty_reports import assert_uncertainty_inputs_unchanged, load_uncertainty_inputs
from validation_data import file_sha256


NEW_SOURCES = (
    "dixon_coles_model.py", "dixon_coles_probabilities.py", "dixon_coles_backtest.py",
    "dixon_coles_reports.py", "run_dixon_coles_backtest.py", "uncertainty_reports.py",
    "calibration_analysis.py",
)


def load_dixon_coles_inputs(project_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Use the Step 13 validation chain without refitting the previous models."""
    references, previous = load_uncertainty_inputs(project_dir)
    matches = load_clean_matches(project_dir / "data" / "processed" / "matches_validation.csv")
    context = {
        **previous,
        "source_sha256": {
            **previous["source_sha256"],
            **{name: file_sha256(project_dir / name) for name in NEW_SOURCES},
        },
    }
    assert_uncertainty_inputs_unchanged(project_dir, context)
    return matches, references, context


def validate_dixon_coles_diagnostics(
    diagnostics: pd.DataFrame, predictions: pd.DataFrame, fixtures: pd.DataFrame, season: str,
) -> None:
    """Complete matchday coverage, temporal boundaries, tau and constrained stationarity."""
    required = {
        "Season", "fit_date", "fixtures", "history_start", "last_history_date", "history_matches",
        "history_weight_sum", "low_score_history_matches", "low_score_history_weight_sum",
        "teams", "free_parameters", "rho", "rho_lower", "rho_upper", "success", "method", "iterations",
        "gradient_max_abs", "stationarity_max_abs", "active_tau_constraints", "minimum_tau_all_pairs",
        "tau_constraint_floor", "initial_penalized_nll", "final_penalized_nll", "attack_log_sum", "concede_log_sum",
        "poisson_start_success", "poisson_start_gradient_max_abs", "poisson_start_refinement_steps",
    }
    if (diagnostics.empty or not required.issubset(diagnostics.columns)
            or diagnostics[list(required)].isna().any().any()):
        raise ValueError("Incomplete Dixon–Coles fitting diagnostics.")
    if (set(diagnostics.Season) != {season} or diagnostics.fit_date.duplicated().any()
            or not diagnostics.success.eq(True).all() or not diagnostics.poisson_start_success.eq(True).all()
            or not diagnostics.method.eq("SLSQP").all()):
        raise ValueError("Duplicate or unsuccessful matchday fit.")
    expected = fixtures.groupby("Date").size().sort_index()
    fit_dates = pd.to_datetime(diagnostics.fit_date, format="%Y-%m-%d", errors="raise")
    actual = pd.Series(diagnostics.fixtures.to_numpy(), index=fit_dates).sort_index()
    if not (actual.index.equals(expected.index) and np.array_equal(actual.to_numpy(), expected.to_numpy())):
        raise ValueError("The matchdays or match counts in the DC diagnostics differ.")
    last = pd.to_datetime(diagnostics.last_history_date, format="%Y-%m-%d", errors="raise")
    start = pd.to_datetime(diagnostics.history_start, format="%Y-%m-%d", errors="raise")
    if not ((start <= last) & (last < fit_dates)).all():
        raise ValueError("Invalid date boundary in the DC fit.")
    numeric = sorted(required - {"Season", "fit_date", "history_start", "last_history_date", "success",
                                 "poisson_start_success", "method"})
    if not np.isfinite(diagnostics[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite fitting diagnostics.")
    count_fields = ("fixtures", "history_matches", "low_score_history_matches", "teams", "free_parameters",
                    "iterations", "active_tau_constraints", "poisson_start_refinement_steps")
    for column in count_fields:
        values = diagnostics[column].to_numpy(dtype=float)
        if not ((values >= 0) & (values % 1 == 0)).all():
            raise ValueError(f"A non-negative integer diagnostic field is required: {column}.")
    if ((diagnostics.history_matches <= 0).any() or (diagnostics.low_score_history_matches <= 0).any()
            or (diagnostics.low_score_history_matches > diagnostics.history_matches).any()
            or (diagnostics.history_weight_sum <= 0).any() or (diagnostics.low_score_history_weight_sum <= 0).any()
            or (diagnostics.low_score_history_weight_sum > diagnostics.history_weight_sum).any()
            or not diagnostics.free_parameters.eq(2 * diagnostics.teams + 1).all()):
        raise ValueError("Invalid history quantity or parameter count.")
    residual = diagnostics.stationarity_max_abs
    initial_gradient = diagnostics.poisson_start_gradient_max_abs
    if (not residual.between(0, GRADIENT_TOLERANCE).all()
            or not initial_gradient.between(0, GRADIENT_TOLERANCE).all()
            or (diagnostics.gradient_max_abs < 0).any()):
        raise ValueError("The fit does not meet the stationarity threshold.")
    no_active = diagnostics.active_tau_constraints.eq(0)
    if not np.allclose(diagnostics.loc[no_active, "gradient_max_abs"], residual.loc[no_active], rtol=0, atol=1e-12):
        raise ValueError("Without an active constraint, stationarity must be calculated from the raw gradient.")
    if (not diagnostics.tau_constraint_floor.eq(MINIMUM_TAU).all()
            or (diagnostics.minimum_tau_all_pairs <= 0).any()
            or (diagnostics.minimum_tau_all_pairs < MINIMUM_TAU - CONSTRAINT_TOLERANCE).any()
            or (diagnostics.rho_lower > diagnostics.rho_upper).any()
            or (diagnostics.rho < diagnostics.rho_lower - CONSTRAINT_TOLERANCE).any()
            or (diagnostics.rho > diagnostics.rho_upper + CONSTRAINT_TOLERANCE).any()):
        raise ValueError("Invalid tau or rho bounds in the fitting diagnostics.")
    if (diagnostics.final_penalized_nll > diagnostics.initial_penalized_nll + 1e-8).any():
        raise ValueError("The fitted objective is worse than the initial Poisson objective.")
    if not (diagnostics[["attack_log_sum", "concede_log_sum"]].abs() <= 1e-10).all().all():
        raise ValueError("The team effects do not satisfy the centring condition.")
    rows = predictions.loc[predictions.model == NEW_MODEL].copy()
    rows["Date"] = pd.to_datetime(rows.Date)
    if rows.empty or set(rows.Season) != {season} or rows.match_id.duplicated().any():
        raise ValueError("Incomplete DC predictions for diagnostic validation.")
    for position, record in enumerate(diagnostics.to_dict("records")):
        day = rows.loc[rows.Date == fit_dates.iloc[position]]
        if len(day) != record["fixtures"] or not np.allclose(day.rho, record["rho"], rtol=0, atol=1e-12):
            raise ValueError("The daily predictions and diagnostic rho differ.")
        if ((day.minimum_tau < record["minimum_tau_all_pairs"] - CONSTRAINT_TOLERANCE).any()
                or not day.history_matches.eq(record["history_matches"]).all()
                or not pd.to_datetime(day.history_start).eq(start.iloc[position]).all()
                or not pd.to_datetime(day.last_history_date).eq(last.iloc[position]).all()):
            raise ValueError("The forecast history or tau differs from the fitted model.")
