"""Step 11: jointly fitted Poisson model on a known example fixture."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import re

import pandas as pd

from analyze_match import analyze_fixture, load_clean_matches
from score_probabilities import build_score_matrix, market_probabilities
from time_weighted_model import calculate_time_weighted_expected_goals


PROJECT_DIR = Path(__file__).resolve().parent
# The original three-season data file is sufficient for the known example fixture.
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
REPORTS_DIR = PROJECT_DIR / "data" / "reports" / "fitted_poisson"
HOME_TEAM = "Arsenal"
AWAY_TEAM = "Nottingham Forest"
MATCH_DATE = "2023-08-12"
LOOKBACK_DAYS = 730
HALF_LIFE_DAYS = 180.0
PRIOR_MATCHES = 5.0  # Used only for the Step 7 comparison model.
RIDGE_STRENGTH = 1.0  # A new learning setting, rather than one match of prior weight.


def comparison_row(name: str, prediction: dict, markets: dict) -> dict:
    result = {"model": name, **{key: prediction[key] for key in ("lambda_home", "lambda_away", "lambda_total")}}
    result.update({"p_home": markets["one_x_two"]["home"], "p_draw": markets["one_x_two"]["draw"],
                   "p_away": markets["one_x_two"]["away"], "p_btts": markets["btts"]["yes"]})
    for line in (1.5, 2.5, 3.5):
        result["p_over_" + str(line).replace(".", "_")] = markets["over_under"][line]["over"]
    return result


def main() -> None:
    # A separate message when the new dependency is missing from the project environment.
    try:
        import numpy as np
        import scipy
        from fitted_poisson_model import fit_poisson_model, predict_goals_from_model
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in ("numpy", "scipy"):
            raise SystemExit(
                "Step 11 requires SciPy. Run this in the project virtual environment:\n"
                "python -m pip install -r requirements_step_11.txt"
            ) from None
        raise

    print("Step 11: jointly fitted attack–concede Poisson model", flush=True)
    print(f"Match: {HOME_TEAM} – {AWAY_TEAM}, {MATCH_DATE}")
    data_hash = sha256(DATA_PATH.read_bytes()).hexdigest()
    matches = load_clean_matches(DATA_PATH)
    snapshot = analyze_fixture(matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS)
    previous = calculate_time_weighted_expected_goals(
        matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS, HALF_LIFE_DAYS, PRIOR_MATCHES
    )
    model = fit_poisson_model(matches, MATCH_DATE, LOOKBACK_DAYS, HALF_LIFE_DAYS, RIDGE_STRENGTH)
    prediction = predict_goals_from_model(model, HOME_TEAM, AWAY_TEAM)
    if sha256(DATA_PATH.read_bytes()).hexdigest() != data_hash:
        raise ValueError("The input data file changed during the run.")
    previous_grid = build_score_matrix(previous["lambda_home"], previous["lambda_away"])
    grid = build_score_matrix(prediction["lambda_home"], prediction["lambda_away"])
    previous_markets = market_probabilities(previous_grid["matrix"])
    markets = market_probabilities(grid["matrix"])

    print(f"Match identifier: {snapshot['match_id']}")
    print(f"History window: {LOOKBACK_DAYS} days | Half-life: {HALF_LIFE_DAYS:g} days")
    print(f"L2 penalty strength: {RIDGE_STRENGTH:g} (a separate parameter from the earlier smoothing weight)")
    print(f"History-window start: {model['history_start']}")
    print(f"Exclusive upper date boundary: {MATCH_DATE}")
    print(f"Latest matchday used: {model['last_history_date']}")
    print(f"Previous matches: {model['history_matches']} | Total weight: {model['history_weight_sum']:.3f}")
    print(f"Fitted teams: {len(model['teams'])} | Independent parameters: {model['free_parameters']}")

    print("\nFitting diagnostics:")
    optimizer = model["optimizer"]
    print(f"Successful optimisation: {optimizer['success']} | Iterations: {optimizer['iterations']}")
    print(f"Maximum absolute gradient: {optimizer['gradient_max_abs']:.3e}")
    print(f"Penalised objective (initial | fitted): "
          f"{optimizer['initial_penalized_nll']:.3f} | {optimizer['final_penalized_nll']:.3f}")
    print(f"Sum of attack log parameters: {model['attack_log_sum']:.3e}")
    print(f"Sum of concede log parameters: {model['concede_log_sum']:.3e}")
    print("Fitting success concerns numerical computation; predictive quality is evaluated separately.")

    print("\nNeutral-team goal intercept and shared home advantage:")
    print(f"Away goal baseline: {model['neutral_away_goal_base']:.3f}")
    print(f"Home multiplier: {model['home_advantage_factor']:.3f}")
    print("The goal baseline is a fitted parameter and may differ from the league's arithmetic goal average.")
    print("\nTeam parameters – from matches at both venues:")
    for team in (HOME_TEAM, AWAY_TEAM):
        parameters = model["teams"].get(team)
        if parameters is None:
            print(f"{team}: no previous matches; neutral attack and concede multipliers: 1.000.")
        else:
            print(f"{team}: {parameters['history_matches']} matches | "
                  f"total weight {parameters['history_weight_sum']:.3f} | "
                  f"attack {parameters['attack_factor']:.3f} | concede {parameters['concede_factor']:.3f}")
    print("Attack multiplier > 1: stronger attack. Concede multiplier > 1: more goals conceded, weaker defence.")

    print("\nExpected goals (Step 7 | jointly fitted):")
    for label, key in (("Home", "lambda_home"), ("Away", "lambda_away"), ("Total", "lambda_total")):
        print(f"{label}: {previous[key]:.3f} | {prediction[key]:.3f}")
    print("\n1X2 (Step 7 | jointly fitted):")
    for label, key in (("Home win", "home"), ("Draw", "draw"), ("Away win", "away")):
        print(f"{label}: {previous_markets['one_x_two'][key]:.2%} | {markets['one_x_two'][key]:.2%}")
    print("\nOver / under (Step 7 | jointly fitted):")
    for line in (1.5, 2.5, 3.5):
        for outcome in ("under", "over"):
            print(f"{line:.1f} {outcome}: {previous_markets['over_under'][line][outcome]:.2%} | "
                  f"{markets['over_under'][line][outcome]:.2%}")
    print("\nBoth teams to score (Step 7 | jointly fitted):")
    for label, key in (("Yes", "yes"), ("No", "no")):
        print(f"{label}: {previous_markets['btts'][key]:.2%} | {markets['btts'][key]:.2%}")
    print("\nThe five most probable scores under the jointly fitted model:")
    for score in markets["top_scores"]:
        print(f"{score['home_goals']}–{score['away_goals']}: {score['probability']:.2%}")
    print(f"Probability omitted from the score table: {grid['omitted_probability']:.3e}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Give another fixture on the same date its own file if settings are changed later.
    safe_teams = re.sub(r"[^a-z0-9]+", "_", f"{HOME_TEAM}_{AWAY_TEAM}".lower()).strip("_")
    slug = f"{MATCH_DATE}_{safe_teams}"
    model_path = REPORTS_DIR / f"fitted_model_{slug}.json"
    teams_path = REPORTS_DIR / f"team_parameters_{slug}.csv"
    comparison_path = REPORTS_DIR / f"match_comparison_{slug}.csv"
    model.update({
        "step": 11, "evaluation_role": "single_fixture_demonstration",
        "match_id": snapshot["match_id"], "home_team": HOME_TEAM, "away_team": AWAY_TEAM,
        "data_file": DATA_PATH.name, "data_sha256": data_hash,
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "history_rule": "match_date - lookback_days <= Date < match_date",
        "same_day_results_used": False,
        "parameter_constraints": "sum(attack_log) = sum(concede_log) = 0",
        "penalty_rule": "0.5 * ridge_strength * sum(attack_log**2 + concede_log**2)",
        "home_unknown": prediction["home_unknown"], "away_unknown": prediction["away_unknown"],
        "prediction": prediction,
        "probabilities": markets,
        "omitted_probability": grid["omitted_probability"],
        "source_sha256": {
            name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest() for name in (
                "analyze_match.py", "time_weighted_model.py", "score_probabilities.py",
                "fitted_poisson_model.py", "predict_fitted_poisson.py",
            )
        },
    })
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([{"team": team, **parameters} for team, parameters in model["teams"].items()]).to_csv(
        teams_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([
        comparison_row("time_weighted", previous, previous_markets),
        comparison_row("fitted_poisson", prediction, markets),
    ]).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"\nFitted model and settings saved: {model_path}")
    print(f"Team parameters saved: {teams_path}")
    print(f"Match predictions for both models saved: {comparison_path}")
    print("\nThis is a one-match demonstration. L2 strength is a learning setting, not a verified optimum.")
    print("The next step also backtests the jointly fitted model chronologically.")
    print("The 2025/26 season remains the separate final test.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError, pd.errors.ParserError) as error:
        raise SystemExit(f"Fitted-model error: {error}")
