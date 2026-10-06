"""Time-weighted, jointly fitted attack–concede Poisson model.

log(lambda_home) = intercept + home_advantage + attack_home + concede_away
log(lambda_away) = intercept + attack_away + concede_home

Team attack and concede parameters each sum to zero.
The L2 penalty shrinks extreme team parameters towards the neutral value zero.
"""

from math import exp, isfinite, log

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import gammaln

from analyze_match import history_before
from time_weighted_model import time_decay_weight


DEFAULT_RIDGE_STRENGTH = 1.0
MAX_ITERATIONS = 1000
GRADIENT_TOLERANCE = 1e-4


def _poisson_objective(
    parameters: np.ndarray, design: np.ndarray, goals: np.ndarray,
    weights: np.ndarray, ridge_strength: float,
) -> tuple[float, np.ndarray]:
    """Weighted negative log likelihood plus L2 penalty, with an analytic gradient.

    The log(goal!) term is independent of the parameters and can be omitted.
    The first two parameters (intercept and home advantage) are not penalised.
    """
    linear = design @ parameters
    with np.errstate(over="raise", invalid="raise"):
        expected = np.exp(linear)
        value = np.sum(weights * (expected - goals * linear))
        value += 0.5 * ridge_strength * np.dot(parameters[2:], parameters[2:])
        gradient = design.T @ (weights * (expected - goals))
        gradient[2:] += ridge_strength * parameters[2:]
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("The fitting objective or gradient is not finite.")
    return float(value), gradient


def _poisson_hessian(
    parameters: np.ndarray, design: np.ndarray, weights: np.ndarray, ridge_strength: float,
) -> np.ndarray:
    """Second derivative of the objective for a numerical refinement step."""
    with np.errstate(over="raise", invalid="raise"):
        expected = np.exp(design @ parameters)
        hessian = design.T @ ((weights * expected)[:, None] * design)
        penalty_diagonal = np.full(len(parameters), ridge_strength)
        penalty_diagonal[:2] = 0.0
        hessian += np.diag(penalty_diagonal)
    if not np.isfinite(hessian).all():
        raise ValueError("The refinement step's second derivative is not finite.")
    return hessian


def _refine_poisson_fit(
    parameters: np.ndarray, design: np.ndarray, goals: np.ndarray,
    weights: np.ndarray, ridge_strength: float,
) -> tuple[np.ndarray, int]:
    """After successful L-BFGS-B, apply up to five Newton refinements if needed.

    The relative objective change can be small while the gradient still
    exceeds its separate acceptance threshold. Refine the same objective
    rather than relaxing the gradient check.
    """
    current = parameters.copy()
    steps = 0
    for _ in range(5):
        value, gradient = _poisson_objective(current, design, goals, weights, ridge_strength)
        if np.max(np.abs(gradient)) <= GRADIENT_TOLERANCE:
            break
        direction = np.linalg.solve(_poisson_hessian(current, design, weights, ridge_strength), gradient)
        accepted = False
        for reduction in range(20):
            candidate = current - (0.5 ** reduction) * direction
            try:
                next_value, next_gradient = _poisson_objective(candidate, design, goals, weights, ridge_strength)
            except FloatingPointError:
                continue
            roundoff = 32 * np.finfo(float).eps * max(1.0, abs(value))
            if next_value <= value + roundoff and np.max(np.abs(next_gradient)) < np.max(np.abs(gradient)):
                current = candidate
                accepted = True
                steps += 1
                break
        if not accepted:
            raise RuntimeError("The numerical refinement did not find an improving step.")
    return current, steps


def fit_poisson_model(
    matches: pd.DataFrame,
    match_date: str,
    lookback_days: int = 730,
    half_life_days: float | None = 180.0,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict:
    """Fit only on dates before the target matchday and return a JSON-serialisable model.

    Home and away results estimate the same two team parameters.
    The known-team list is also derived exclusively from the history.
    """
    if isinstance(ridge_strength, bool) or not isfinite(ridge_strength) or ridge_strength <= 0:
        raise ValueError("The L2 penalty strength must be positive and finite.")
    time_decay_weight(0.0, half_life_days)  # Validate the setting even when the history is empty.
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(matches.columns):
        raise ValueError("Missing fitting field: Date, HomeTeam, AwayTeam, FTHG and FTAG are required.")
    target = pd.to_datetime(match_date, format="%Y-%m-%d", errors="raise").normalize()
    if pd.isna(target):
        raise ValueError("A valid target matchday is required.")
    history = history_before(matches, target, lookback_days)
    if history.empty:
        raise ValueError("No previous league matches are available for fitting.")
    if history[list(required)].isna().any().any():
        raise ValueError("Incomplete previous matches cannot be used for fitting.")
    if not history["Date"].eq(history["Date"].dt.normalize()).all():
        raise ValueError("Match dates must contain a date only.")
    for column in ("HomeTeam", "AwayTeam"):
        if not history[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("Non-empty text team names are required.")
    if history["HomeTeam"].eq(history["AwayTeam"]).any():
        raise ValueError("A team cannot play against itself.")
    if history.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate previous match found.")
    for column in ("FTHG", "FTAG"):
        values = pd.to_numeric(history[column], errors="raise")
        if not values.map(lambda value: not isinstance(value, bool) and isfinite(value)
                          and value >= 0 and value % 1 == 0).all():
            raise ValueError("Previous goal counts must be non-negative integers.")

    names = sorted(set(history["HomeTeam"]) | set(history["AwayTeam"]))
    team_index = {name: index for index, name in enumerate(names)}
    n_teams = len(names)
    home_index = history["HomeTeam"].map(team_index).to_numpy(dtype=int)
    away_index = history["AwayTeam"].map(team_index).to_numpy(dtype=int)
    ages = (target - history["Date"]).dt.days
    match_weights = ages.map(lambda age: time_decay_weight(age, half_life_days)).to_numpy(dtype=float)
    weight_sum = float(match_weights.sum())
    if not isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("The total league time weight is invalid.")
    home_goals = history["FTHG"].to_numpy(dtype=float)
    away_goals = history["FTAG"].to_numpy(dtype=float)
    home_mean = float(np.dot(match_weights, home_goals) / weight_sum)
    away_mean = float(np.dot(match_weights, away_goals) / weight_sum)
    if home_mean <= 0 or away_mean <= 0:
        raise ValueError("Fitting requires positive league mean home and away goals.")

    # K teams have K-1 independent effects when their sum is zero.
    # The Helmert basis enforces this exactly for both parameter groups.
    # The columns are orthonormal: the penalty treats all teams equally.
    basis = helmert(n_teams, full=False).T
    home_design = np.column_stack([
        np.ones(len(history)), np.ones(len(history)), basis[home_index], basis[away_index]
    ])
    away_design = np.column_stack([
        np.ones(len(history)), np.zeros(len(history)), basis[away_index], basis[home_index]
    ])
    design = np.vstack([home_design, away_design])
    goals = np.concatenate([home_goals, away_goals])
    weights = np.tile(match_weights, 2)
    initial = np.zeros(design.shape[1])
    initial[0] = log(away_mean)
    initial[1] = log(home_mean / away_mean)
    initial_objective, _ = _poisson_objective(initial, design, goals, weights, ridge_strength)

    try:
        result = minimize(
            _poisson_objective, initial, args=(design, goals, weights, ridge_strength),
            method="L-BFGS-B", jac=True,
            options={"maxiter": MAX_ITERATIONS, "gtol": 1e-8, "ftol": 1e-14, "maxls": 40},
        )
    except FloatingPointError as error:
        raise RuntimeError("Numerical fitting error; no prediction is produced.") from error
    if not result.success:
        raise RuntimeError(f"The optimiser did not complete the fit successfully: {result.message}")
    try:
        parameters, refinement_steps = _refine_poisson_fit(result.x, design, goals, weights, ridge_strength)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise RuntimeError("Numerical error in the fitting refinement.") from error
    final_objective, gradient = _poisson_objective(parameters, design, goals, weights, ridge_strength)
    gradient_max = float(np.max(np.abs(gradient)))
    if gradient_max > GRADIENT_TOLERANCE:
        raise RuntimeError(f"The fitted gradient is too large: {gradient_max:.3g}.")
    if final_objective > initial_objective + 1e-8:
        raise RuntimeError("The fitted objective is worse than the initial model.")
    attack = basis @ parameters[2:2 + n_teams - 1]
    concede = basis @ parameters[2 + n_teams - 1:]
    constant = float(np.dot(weights, gammaln(goals + 1)))
    penalty = float(0.5 * ridge_strength * np.dot(parameters[2:], parameters[2:]))
    team_parameters = {}
    for index, name in enumerate(names):
        involved = (home_index == index) | (away_index == index)
        team_parameters[name] = {
            "attack_log": float(attack[index]), "concede_log": float(concede[index]),
            "attack_factor": exp(float(attack[index])), "concede_factor": exp(float(concede[index])),
            "history_matches": int(involved.sum()),
            "history_weight_sum": float(match_weights[involved].sum()),
            "home_history_matches": int((home_index == index).sum()),
            "away_history_matches": int((away_index == index).sum()),
        }
    return {
        "model": "fitted_poisson", "schema_version": 1,
        "match_date": target.strftime("%Y-%m-%d"),
        "history_start": (target - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
        "last_history_date": history["Date"].max().strftime("%Y-%m-%d"),
        "history_matches": len(history), "history_weight_sum": weight_sum,
        "lookback_days": lookback_days, "half_life_days": half_life_days,
        "ridge_strength": ridge_strength, "free_parameters": len(initial),
        "intercept_log": float(parameters[0]), "home_advantage_log": float(parameters[1]),
        "neutral_away_goal_base": exp(float(parameters[0])),
        "home_advantage_factor": exp(float(parameters[1])),
        "attack_log_sum": float(attack.sum()), "concede_log_sum": float(concede.sum()),
        "teams": team_parameters,
        "optimizer": {
            "method": "L-BFGS-B", "success": bool(result.success), "iterations": int(result.nit),
            "refinement_method": "Newton if needed", "refinement_steps": refinement_steps,
            "message": str(result.message), "gradient_max_abs": gradient_max,
            "gradient_acceptance_tolerance": GRADIENT_TOLERANCE,
            "initial_penalized_nll": initial_objective + constant,
            "final_penalized_nll": final_objective + constant,
            "weighted_negative_log_likelihood": final_objective - penalty + constant,
            "l2_penalty": penalty,
        },
    }


def predict_goals_from_model(model: dict, home_team: str, away_team: str) -> dict:
    """Calculate from the fitted model without needing a match result.

    Assign neutral multipliers to a team without history and flag it separately.
    """
    if not all(isinstance(team, str) and bool(team.strip()) for team in (home_team, away_team)):
        raise ValueError("Non-empty team names are required for prediction.")
    if home_team == away_team:
        raise ValueError("The home and away teams must be different.")
    neutral = {"attack_log": 0.0, "concede_log": 0.0,
               "attack_factor": 1.0, "concede_factor": 1.0, "history_matches": 0}
    home = model["teams"].get(home_team, neutral)
    away = model["teams"].get(away_team, neutral)
    lambda_home = exp(model["intercept_log"] + model["home_advantage_log"]
                      + home["attack_log"] + away["concede_log"])
    lambda_away = exp(model["intercept_log"] + away["attack_log"] + home["concede_log"])
    if not all(isfinite(value) and value > 0 for value in (lambda_home, lambda_away)):
        raise ValueError("The model did not produce finite, positive expected goal counts.")
    return {
        "lambda_home": lambda_home, "lambda_away": lambda_away,
        "lambda_total": lambda_home + lambda_away,
        "home_attack_factor": home["attack_factor"], "home_concede_factor": home["concede_factor"],
        "away_attack_factor": away["attack_factor"], "away_concede_factor": away["concede_factor"],
        "home_unknown": home_team not in model["teams"], "away_unknown": away_team not in model["teams"],
        "home_history_matches": home["history_matches"], "away_history_matches": away["history_matches"],
    }
