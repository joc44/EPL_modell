"""Joint score table and event probabilities from independent Poisson goal counts."""

from math import exp, fsum, isclose, isfinite


TAIL_TOLERANCE = 1e-10
MAX_GOALS = 200


def poisson_goal_probabilities(
    expected_goals: float,
    tail_tolerance: float = TAIL_TOLERANCE,
    max_goals: int = MAX_GOALS,
) -> list[float]:
    """Build P(0), P(1), ... with at most tail_tolerance omitted mass.

P(0) = exp(-lambda); P(k) = P(k-1) * lambda / k.
Raise an error if the computation limit is reached before sufficient coverage.
"""
    if not isfinite(expected_goals) or expected_goals < 0:
        raise ValueError("The expected goal count must be finite and non-negative.")
    if not isfinite(tail_tolerance) or not 0 < tail_tolerance < 1:
        raise ValueError("The tail tolerance must be between zero and one.")
    if isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 0:
        raise ValueError("The maximum goal count must be a non-negative integer.")

    probabilities = [exp(-expected_goals)]
    if fsum(probabilities) >= 1 - tail_tolerance:
        return probabilities

    for goals in range(1, max_goals + 1):
        probability = probabilities[-1] * expected_goals / goals
        probabilities.append(probability)
        if fsum(probabilities) >= 1 - tail_tolerance:
            return probabilities

    raise ValueError(
        "Poisson coverage is insufficient at the computation limit. "
        "Check the expected goal count and upper limit."
    )


def build_score_matrix(
    lambda_home: float,
    lambda_away: float,
    tail_tolerance: float = TAIL_TOLERANCE,
    max_goals: int = MAX_GOALS,
) -> dict:
    """Rows are home goals, columns are away goals; cells are score probabilities.

Under independence, multiply the two individual probabilities.
After sufficient coverage, normalise the finite table to one.
"""
    home_probabilities = poisson_goal_probabilities(lambda_home, tail_tolerance, max_goals)
    away_probabilities = poisson_goal_probabilities(lambda_away, tail_tolerance, max_goals)

    raw_matrix = []
    for home_probability in home_probabilities:
        row = []
        for away_probability in away_probabilities:
            row.append(home_probability * away_probability)
        raw_matrix.append(row)

    captured_probability = fsum(probability for row in raw_matrix for probability in row)
    if not isfinite(captured_probability) or captured_probability <= 0:
        raise ValueError("The score table's probability mass is invalid.")

    matrix = [
        [probability / captured_probability for probability in row]
        for row in raw_matrix
    ]
    return {
        "matrix": matrix,
        "captured_probability": captured_probability,
        "omitted_probability": max(0.0, 1.0 - captured_probability),
        "max_home_goals": len(home_probabilities) - 1,
        "max_away_goals": len(away_probabilities) - 1,
    }


def market_probabilities(
    matrix: list[list[float]],
    lines: tuple[float, ...] = (1.5, 2.5, 3.5),
    top_n: int = 5,
) -> dict:
    """Every event is a sum from the same normalised table."""
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("The score table must be a non-empty rectangular list.")
    if any(not isfinite(p) or p < 0 for row in matrix for p in row):
        raise ValueError("Score-table probabilities must be finite and non-negative.")
    mass = fsum(p for row in matrix for p in row)
    if not isclose(mass, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The score-table probabilities must sum to one.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("The number of exact scores to display must be a positive integer.")
    for line in lines:
        if not isfinite(line) or line < 0.5 or line % 1 != 0.5:
            raise ValueError("Only positive goal lines ending in .5 are supported in this step.")

    # The list contains each possible score in the table exactly once.
    scores = []
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            scores.append({
                "home_goals": home_goals,
                "away_goals": away_goals,
                "probability": probability,
            })

    home_win = fsum(s["probability"] for s in scores if s["home_goals"] > s["away_goals"])
    draw = fsum(s["probability"] for s in scores if s["home_goals"] == s["away_goals"])
    away_win = fsum(s["probability"] for s in scores if s["home_goals"] < s["away_goals"])

    over_under = {}
    for line in lines:
        over_under[line] = {
            "under": fsum(s["probability"] for s in scores if s["home_goals"] + s["away_goals"] < line),
            "over": fsum(s["probability"] for s in scores if s["home_goals"] + s["away_goals"] > line),
        }

    btts_yes = fsum(s["probability"] for s in scores if s["home_goals"] > 0 and s["away_goals"] > 0)
    btts_no = fsum(s["probability"] for s in scores if s["home_goals"] == 0 or s["away_goals"] == 0)
    ranked_scores = sorted(scores, key=lambda s: (-s["probability"], s["home_goals"], s["away_goals"]))

    return {
        "one_x_two": {"home": home_win, "draw": draw, "away": away_win},
        "over_under": over_under,
        "btts": {"yes": btts_yes, "no": btts_no},
        "top_scores": ranked_scores[:top_n],
    }
