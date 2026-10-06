"""Dixon–Coles correction for the four low-score cells, with valid probabilities."""

from math import exp, fsum, isfinite, lgamma, log
from numbers import Integral, Real

from score_probabilities import MAX_GOALS, TAIL_TOLERANCE, poisson_goal_probabilities


LOW_SCORES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _rates(lambda_home: float, lambda_away: float) -> tuple[float, float]:
    values = (lambda_home, lambda_away)
    if not all(isinstance(value, Real) and not isinstance(value, bool)
               and isfinite(value) and value >= 0 for value in values):
        raise ValueError("Finite, non-negative expected goal counts are required.")
    home, away = map(float, values)
    if not isfinite(home * away):
        raise ValueError("The product of the two expected goal counts is not finite.")
    return home, away


def tau_factors(lambda_home: float, lambda_away: float, rho: float) -> dict:
    """Raw tau multipliers; the caller checks positivity separately."""
    home, away = _rates(lambda_home, lambda_away)
    if isinstance(rho, bool) or not isinstance(rho, Real) or not isfinite(rho):
        raise ValueError("The rho parameter must be finite.")
    return {(0, 0): 1 - home * away * rho, (0, 1): 1 + home * rho,
            (1, 0): 1 + away * rho, (1, 1): 1 - rho}


def rho_bounds(lambda_home: float, lambda_away: float, minimum_tau: float = 0.0) -> tuple[float, float]:
    """Rho bounds for all strict tau > minimum_tau conditions."""
    home, away = _rates(lambda_home, lambda_away)
    if (isinstance(minimum_tau, bool) or not isinstance(minimum_tau, Real)
            or not isfinite(minimum_tau) or not 0 <= minimum_tau < 1):
        raise ValueError("The lower tau threshold must be between zero and one.")
    lower = max(((minimum_tau - 1) / rate for rate in (home, away) if rate > 0), default=-float("inf"))
    upper = min(1 - minimum_tau, (1 - minimum_tau) / (home * away) if home * away > 0 else float("inf"))
    return lower, upper


def _valid_factors(lambda_home: float, lambda_away: float, rho: float) -> dict:
    factors = tau_factors(lambda_home, lambda_away, rho)
    if not all(isfinite(value) and value > 0 for value in factors.values()):
        raise ValueError("Invalid rho: all four tau multipliers must be positive.")
    return factors


def build_dixon_coles_score_matrix(
    lambda_home: float, lambda_away: float, rho: float,
    tail_tolerance: float = TAIL_TOLERANCE, max_goals: int = MAX_GOALS,
) -> dict:
    """Correct the original Poisson mass, then normalise the finite table."""
    factors = _valid_factors(lambda_home, lambda_away, rho)
    if isinstance(max_goals, bool) or not isinstance(max_goals, Integral) or max_goals < 1:
        raise ValueError("The correction requires a table extending to at least one goal.")
    home = poisson_goal_probabilities(lambda_home, tail_tolerance, int(max_goals))
    away = poisson_goal_probabilities(lambda_away, tail_tolerance, int(max_goals))
    # Keep all four cells present even for very small lambdas.
    if len(home) == 1:
        home.append(home[0] * lambda_home)
    if len(away) == 1:
        away.append(away[0] * lambda_away)
    raw = [[ph * pa for pa in away] for ph in home]
    poisson_mass = fsum(value for row in raw for value in row)
    for (hg, ag), factor in factors.items():
        raw[hg][ag] *= factor
    captured = fsum(value for row in raw for value in row)
    if not isfinite(captured) or captured <= 0 or abs(captured - poisson_mass) > 2e-12:
        raise ValueError("The correction did not preserve probability mass.")
    return {
        "matrix": [[value / captured for value in row] for row in raw],
        "captured_probability": captured, "omitted_probability": max(0.0, 1 - captured),
        "max_home_goals": len(home) - 1, "max_away_goals": len(away) - 1,
        "rho": float(rho), "minimum_tau": min(factors.values()),
        "mass_change_from_correction": captured - poisson_mass,
    }


def dixon_coles_score_probability(
    lambda_home: float, lambda_away: float, rho: float, home_goals: int, away_goals: int,
) -> float:
    """Probability of an exact final score, including scores outside the finite table."""
    factors = _valid_factors(lambda_home, lambda_away, rho)
    if not all(isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
               for value in (home_goals, away_goals)):
        raise ValueError("Final-score goal counts must be non-negative integers.")
    log_probability = 0.0
    for rate, goals in ((lambda_home, home_goals), (lambda_away, away_goals)):
        if rate == 0:
            if goals > 0:
                return 0.0
        else:
            log_probability += goals * log(rate) - rate - lgamma(goals + 1)
    return exp(log_probability) * factors.get((home_goals, away_goals), 1.0)
