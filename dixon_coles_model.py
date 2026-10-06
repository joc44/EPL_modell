"""Time-weighted, L2-regularised, jointly fitted Dixon–Coles model."""

from math import exp

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize, nnls
from scipy.special import gammaln

from analyze_match import history_before
from dixon_coles_probabilities import rho_bounds, tau_factors
from fitted_poisson_model import (
    GRADIENT_TOLERANCE, _poisson_objective, fit_poisson_model, predict_goals_from_model,
)
from time_weighted_model import time_decay_weight


MINIMUM_TAU = 1e-8
CONSTRAINT_TOLERANCE = 1e-9
ACTIVE_CONSTRAINT_TOLERANCE = 1e-6
MAX_ITERATIONS = 1000


def _tau_and_jacobian(parameters: np.ndarray, home_design: np.ndarray, away_design: np.ndarray) -> tuple:
    """The four tau values for every pairing; derivatives for every free parameter."""
    theta, rho = parameters[:-1], parameters[-1]
    with np.errstate(over="raise", invalid="raise"):
        home, away = np.exp(home_design @ theta), np.exp(away_design @ theta)
        product = home * away
        factors = np.column_stack((1 - rho * product, 1 + rho * home,
                                   1 + rho * away, np.full(len(home), 1 - rho)))
        jacobian = np.zeros((len(home), 4, len(parameters)))
        jacobian[:, 0, :-1] = (-rho * product)[:, None] * (home_design + away_design)
        jacobian[:, 1, :-1] = (rho * home)[:, None] * home_design
        jacobian[:, 2, :-1] = (rho * away)[:, None] * away_design
        jacobian[:, 0, -1] = -product
        jacobian[:, 1, -1] = home
        jacobian[:, 2, -1] = away
        jacobian[:, 3, -1] = -1
    return factors, jacobian


def _dixon_coles_objective(
    parameters: np.ndarray, home_design: np.ndarray, away_design: np.ndarray,
    home_goals: np.ndarray, away_goals: np.ndarray, match_weights: np.ndarray,
    ridge_strength: float,
) -> tuple[float, np.ndarray]:
    """Poisson loss minus weighted log(tau); rho receives no L2 penalty."""
    design = np.vstack((home_design, away_design))
    goals, weights = np.r_[home_goals, away_goals], np.tile(match_weights, 2)
    value, poisson_gradient = _poisson_objective(parameters[:-1], design, goals, weights, ridge_strength)
    factors, jacobian = _tau_and_jacobian(parameters, home_design, away_design)
    low = (home_goals <= 1) & (away_goals <= 1)
    indices = 2 * home_goals[low].astype(int) + away_goals[low].astype(int)
    tau = factors[low, indices]
    if not (np.isfinite(tau) & (tau > 0)).all():
        # The optimiser must not accept this out-of-domain trial.
        return float("inf"), np.zeros_like(parameters)
    value -= np.dot(match_weights[low], np.log(tau))
    gradient = np.r_[poisson_gradient, 0.0]
    gradient -= np.sum((match_weights[low] / tau)[:, None] * jacobian[low, indices], axis=0)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Non-finite Dixon–Coles objective or gradient.")
    return float(value), gradient


def _pair_designs(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Known team pairs in the history and cases involving a neutral unknown team."""
    n_teams = basis.shape[0]
    extended = np.vstack((basis, np.zeros((1, basis.shape[1]))))
    pairs = [(home, away) for home in range(n_teams + 1) for away in range(n_teams + 1)
             if home != away or home == n_teams]
    hi, ai = np.array(pairs).T
    home = np.column_stack((np.ones(len(pairs)), np.ones(len(pairs)), extended[hi], extended[ai]))
    away = np.column_stack((np.ones(len(pairs)), np.zeros(len(pairs)), extended[ai], extended[hi]))
    return home, away


def _stationarity_residual(gradient: np.ndarray, constraint_values: np.ndarray, jacobian: np.ndarray) -> tuple:
    """KKT stationarity: an active constraint can allow a non-zero raw gradient."""
    active = constraint_values <= ACTIVE_CONSTRAINT_TOLERANCE
    if not active.any():
        return float(np.max(np.abs(gradient))), 0
    multipliers, _ = nnls(jacobian[active].T, gradient, maxiter=10000)
    residual = gradient - jacobian[active].T @ multipliers
    return float(np.max(np.abs(residual))), int(active.sum())


def fit_dixon_coles_model(
    matches: pd.DataFrame, match_date: str, lookback_days: int = 730,
    half_life_days: float | None = 180.0, ridge_strength: float = 1.0,
) -> dict:
    """Start from the Poisson fit, then estimate team effects and rho jointly."""
    poisson = fit_poisson_model(matches, match_date, lookback_days, half_life_days, ridge_strength)
    target = pd.Timestamp(poisson["match_date"])
    history = history_before(matches, target, lookback_days)
    home_goals = history.FTHG.to_numpy(dtype=float)
    away_goals = history.FTAG.to_numpy(dtype=float)
    low = (home_goals <= 1) & (away_goals <= 1)
    if not low.any():
        raise ValueError("No 0–0, 0–1, 1–0 or 1–1 history is available to estimate rho.")
    names = sorted(poisson["teams"])
    n_teams = len(names)
    basis = helmert(n_teams, full=False).T
    index = {name: position for position, name in enumerate(names)}
    hi = history.HomeTeam.map(index).to_numpy(dtype=int)
    ai = history.AwayTeam.map(index).to_numpy(dtype=int)
    home_design = np.column_stack((np.ones(len(history)), np.ones(len(history)), basis[hi], basis[ai]))
    away_design = np.column_stack((np.ones(len(history)), np.zeros(len(history)), basis[ai], basis[hi]))
    weights = (target - history.Date).dt.days.map(
        lambda age: time_decay_weight(age, half_life_days)
    ).to_numpy(dtype=float)
    if weights[low].sum() <= 0:
        raise ValueError("The low-score history has zero total time weight; rho cannot be estimated.")
    attack = np.array([poisson["teams"][name]["attack_log"] for name in names])
    concede = np.array([poisson["teams"][name]["concede_log"] for name in names])
    initial = np.r_[poisson["intercept_log"], poisson["home_advantage_log"],
                    basis.T @ attack, basis.T @ concede, 0.0]
    args = (home_design, away_design, home_goals, away_goals, weights, ridge_strength)
    pair_home, pair_away = _pair_designs(basis)

    def constraint_values(parameters):
        return _tau_and_jacobian(parameters, pair_home, pair_away)[0].ravel() - MINIMUM_TAU

    def constraint_jacobian(parameters):
        return _tau_and_jacobian(parameters, pair_home, pair_away)[1].reshape(-1, len(parameters))

    initial_value, _ = _dixon_coles_objective(initial, *args)
    try:
        result = minimize(
            _dixon_coles_objective, initial, args=args, jac=True, method="SLSQP",
            constraints=[{"type": "ineq", "fun": constraint_values, "jac": constraint_jacobian}],
            options={"maxiter": MAX_ITERATIONS, "ftol": 1e-12, "disp": False},
        )
    except FloatingPointError as error:
        raise RuntimeError("Numerical error in the Dixon–Coles fit.") from error
    if not result.success:
        raise RuntimeError(f"Dixon–Coles fit failed: {result.message}")
    parameters = result.x
    value, gradient = _dixon_coles_objective(parameters, *args)
    constraints = constraint_values(parameters)
    min_tau = float(constraints.min() + MINIMUM_TAU)
    if min_tau <= 0 or constraints.min() < -CONSTRAINT_TOLERANCE:
        raise RuntimeError("The fit violated the positive-tau condition.")
    residual, active_count = _stationarity_residual(gradient, constraints, constraint_jacobian(parameters))
    if not np.isfinite(residual) or residual > GRADIENT_TOLERANCE:
        raise RuntimeError(f"The constrained fit's stationarity residual is too large: {residual:.3g}.")
    if value > initial_value + 1e-8:
        raise RuntimeError("The fitted Dixon–Coles objective is worse than the initial model.")
    attack = basis @ parameters[2:2 + n_teams - 1]
    concede = basis @ parameters[2 + n_teams - 1:-1]
    team_parameters = {}
    for position, name in enumerate(names):
        team_parameters[name] = {
            **poisson["teams"][name], "attack_log": float(attack[position]),
            "concede_log": float(concede[position]), "attack_factor": exp(float(attack[position])),
            "concede_factor": exp(float(concede[position])),
        }
    constant = float(np.dot(np.tile(weights, 2), gammaln(np.r_[home_goals, away_goals] + 1)))
    penalty = float(0.5 * ridge_strength * np.dot(parameters[2:-1], parameters[2:-1]))
    bounds = [rho_bounds(exp(h @ parameters[:-1]), exp(a @ parameters[:-1]), MINIMUM_TAU)
              for h, a in zip(pair_home, pair_away)]
    return {
        **{key: poisson[key] for key in (
            "schema_version", "match_date", "history_start", "last_history_date", "history_matches",
            "history_weight_sum", "lookback_days", "half_life_days", "ridge_strength",
        )},
        "model": "dixon_coles", "free_parameters": len(initial), "rho": float(parameters[-1]),
        "low_score_history_matches": int(low.sum()), "low_score_history_weight_sum": float(weights[low].sum()),
        "intercept_log": float(parameters[0]), "home_advantage_log": float(parameters[1]),
        "neutral_away_goal_base": exp(float(parameters[0])), "home_advantage_factor": exp(float(parameters[1])),
        "attack_log_sum": float(attack.sum()), "concede_log_sum": float(concede.sum()), "teams": team_parameters,
        "prediction_domain": "all_known_team_pairs_and_neutral_unknown_teams",
        "rho_domain_at_fitted_rates": {"lower": max(bound[0] for bound in bounds),
                                       "upper": min(bound[1] for bound in bounds)},
        "poisson_start_model": poisson,
        "optimizer": {
            "method": "SLSQP", "success": bool(result.success), "iterations": int(result.nit),
            "message": str(result.message), "gradient_max_abs": float(np.max(np.abs(gradient))),
            "stationarity_max_abs": residual, "stationarity_acceptance_tolerance": GRADIENT_TOLERANCE,
            "active_tau_constraints": active_count, "minimum_tau_all_pairs": min_tau,
            "tau_constraint_floor": MINIMUM_TAU, "constraint_pairs": len(pair_home),
            "constraint_count": len(constraints),
            "initial_penalized_nll": initial_value + constant, "final_penalized_nll": value + constant,
            "weighted_negative_log_likelihood": value - penalty + constant, "l2_penalty": penalty,
            "penalty_rule": "0.5 * ridge_strength * sum(attack_log**2 + concede_log**2); rho unpenalized",
        },
    }


def predict_dixon_coles_goals(model: dict, home_team: str, away_team: str) -> dict:
    if model.get("model") != "dixon_coles":
        raise ValueError("A Dixon–Coles model is required for prediction.")
    prediction = predict_goals_from_model(model, home_team, away_team)
    factors = tau_factors(prediction["lambda_home"], prediction["lambda_away"], model["rho"])
    if not all(np.isfinite(value) and value > 0 for value in factors.values()):
        raise ValueError("The forecast tau multipliers are invalid.")
    return {**prediction, "rho": model["rho"], "minimum_tau": min(factors.values())}
