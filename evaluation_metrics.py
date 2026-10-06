"""Evaluate probabilistic forecasts independently of the models."""

from math import exp, fsum, isclose, isfinite, lgamma, log


OUTCOMES = ("home", "draw", "away")
LOG_LOSS_FLOOR = 1e-15


def validate_probability(probability: float) -> None:
    if isinstance(probability, bool) or not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The probability must be finite and between zero and one.")


def observed_probability_loss(probability: float) -> dict:
    """Calculate -ln(p) loss from the probability of the observed event.

    Do not modify the model's probability. Apply a lower bound only
    during evaluation, and flag every such case separately.
    """
    validate_probability(probability)
    return {
        "log_loss": -log(max(probability, LOG_LOSS_FLOOR)),
        "clipped": probability < LOG_LOSS_FLOOR,
    }


def validate_goals(home_goals: int, away_goals: int) -> None:
    for goals in (home_goals, away_goals):
        if isinstance(goals, bool) or not isinstance(goals, int) or goals < 0:
            raise ValueError("The observed goal count must be a non-negative integer.")


def realized_outcome(home_goals: int, away_goals: int) -> str:
    validate_goals(home_goals, away_goals)
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def one_x_two_metrics(probabilities: dict, outcome: str) -> dict:
    """1X2 log loss, three-class Brier score (0–2) and accuracy indicator."""
    if set(probabilities) != set(OUTCOMES) or outcome not in OUTCOMES:
        raise ValueError("The 1X2 labels must be home, draw and away.")
    for probability in probabilities.values():
        validate_probability(probability)
    if not isclose(fsum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The 1X2 probabilities must sum to one.")

    predicted = max(OUTCOMES, key=lambda label: probabilities[label])
    return {
        **observed_probability_loss(probabilities[outcome]),
        "brier": fsum((probabilities[label] - int(label == outcome)) ** 2 for label in OUTCOMES),
        "predicted": predicted,
        "hit": predicted == outcome,
    }


def binary_metrics(probability: float, observed: bool | int) -> dict:
    """Binary event: log loss and binary Brier score (0–1)."""
    validate_probability(probability)
    if not isinstance(observed, (bool, int)) or observed not in (0, 1):
        raise ValueError("The observed binary event must be 0, 1, False or True.")
    actual_probability = probability if observed else 1.0 - probability
    return {
        **observed_probability_loss(actual_probability),
        "brier": (probability - int(observed)) ** 2,
    }


def poisson_score_probability(
    lambda_home: float, lambda_away: float, home_goals: int, away_goals: int
) -> float:
    """Probability of the observed score directly from the two Poisson distributions.

    This also evaluates scores outside the truncated score table.
    The log form supports high goal counts; λ=0 is a separate case.
    """
    validate_goals(home_goals, away_goals)
    log_probabilities = []
    for expected, goals in ((lambda_home, home_goals), (lambda_away, away_goals)):
        if isinstance(expected, bool) or not isfinite(expected) or expected < 0:
            raise ValueError("The expected goal count must be finite and non-negative.")
        if expected == 0:
            log_probabilities.append(0.0 if goals == 0 else float("-inf"))
        else:
            log_probabilities.append(-expected + goals * log(expected) - lgamma(goals + 1))
    return exp(fsum(log_probabilities))
