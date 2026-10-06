"""Step 14: jointly fitted Dixon–Coles model and the correction's effect."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import re

import pandas as pd

from analyze_match import analyze_fixture, load_clean_matches
from score_probabilities import build_score_matrix, market_probabilities


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "reports" / "dixon_coles"
HOME_TEAM = "Arsenal"
AWAY_TEAM = "Nottingham Forest"
MATCH_DATE = "2023-08-12"
LOOKBACK_DAYS = 730
HALF_LIFE_DAYS = 180.0
RIDGE_STRENGTH = 1.0


def comparison_row(name: str, prediction: dict, markets: dict) -> dict:
    row = {"model": name, **{key: prediction[key] for key in ("lambda_home", "lambda_away", "lambda_total")}}
    row.update({"p_home": markets["one_x_two"]["home"], "p_draw": markets["one_x_two"]["draw"],
                "p_away": markets["one_x_two"]["away"], "p_btts": markets["btts"]["yes"]})
    for line in (1.5, 2.5, 3.5):
        row["p_over_" + str(line).replace(".", "_")] = markets["over_under"][line]["over"]
    return row


def main() -> None:
    try:
        import numpy as np
        import scipy
        from dixon_coles_model import fit_dixon_coles_model, predict_dixon_coles_goals
        from dixon_coles_probabilities import build_dixon_coles_score_matrix, tau_factors
        from fitted_poisson_model import predict_goals_from_model
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in ("numpy", "scipy"):
            raise SystemExit("The previous SciPy dependency is required. Install requirements_step_11.txt.") from None
        raise
    print("Step 14: jointly fitted Dixon–Coles model", flush=True)
    print(f"Match: {HOME_TEAM} – {AWAY_TEAM}, {MATCH_DATE}")
    print(f"History window: {LOOKBACK_DAYS} days | Half-life: {HALF_LIFE_DAYS:g} days | L2 strength: {RIDGE_STRENGTH:g}")
    print("Estimate the team effects, home advantage, goal intercept and rho jointly.", flush=True)
    data_hash = sha256(DATA_PATH.read_bytes()).hexdigest()
    matches = load_clean_matches(DATA_PATH)
    snapshot = analyze_fixture(matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS)
    model = fit_dixon_coles_model(matches, MATCH_DATE, LOOKBACK_DAYS, HALF_LIFE_DAYS, RIDGE_STRENGTH)
    prediction = predict_dixon_coles_goals(model, HOME_TEAM, AWAY_TEAM)
    original = predict_goals_from_model(model["poisson_start_model"], HOME_TEAM, AWAY_TEAM)
    if sha256(DATA_PATH.read_bytes()).hexdigest() != data_hash:
        raise ValueError("The input data file changed during the run.")
    original_grid = build_score_matrix(original["lambda_home"], original["lambda_away"])
    independent_grid = build_score_matrix(prediction["lambda_home"], prediction["lambda_away"])
    corrected_grid = build_dixon_coles_score_matrix(prediction["lambda_home"], prediction["lambda_away"], model["rho"])
    original_markets = market_probabilities(original_grid["matrix"])
    independent_markets = market_probabilities(independent_grid["matrix"])
    corrected_markets = market_probabilities(corrected_grid["matrix"])
    print(f"Match identifier: {snapshot['match_id']}")
    print(f"History-window start: {model['history_start']} | Exclusive upper date boundary: {MATCH_DATE}")
    print(f"Latest matchday used: {model['last_history_date']}")
    print(f"Previous matches: {model['history_matches']} | Total weight: {model['history_weight_sum']:.3f}")
    print(f"Low-score history: {model['low_score_history_matches']} | Total weight: {model['low_score_history_weight_sum']:.3f}")
    print(f"Fitted teams: {len(model['teams'])} | Independent parameters: {model['free_parameters']}")
    print(f"Fitted rho: {model['rho']:+.6f}")
    if model["rho"] < 0:
        print("Negative rho: at fixed lambdas, 0–0 and 1–1 probabilities increase, while 0–1 and 1–0 probabilities decrease.")
    elif model["rho"] > 0:
        print("Positive rho: at fixed lambdas, 0–0 and 1–1 probabilities decrease, while 0–1 and 1–0 probabilities increase.")
    else:
        print("Zero rho: at fixed lambdas, the correction has no effect.")
    optimizer = model["optimizer"]
    print("\nFitting diagnostics:")
    print(f"Successful optimisation: {optimizer['success']} | Method: {optimizer['method']} | Iterations: {optimizer['iterations']}")
    print(f"Maximum raw gradient: {optimizer['gradient_max_abs']:.3e}")
    print(f"Constraint-aware stationarity residual: {optimizer['stationarity_max_abs']:.3e}")
    print(f"Active tau constraints: {optimizer['active_tau_constraints']} | Minimum tau across all pairings: {optimizer['minimum_tau_all_pairs']:.6f}")
    print(f"Penalised objective (initial Poisson | fitted DC): {optimizer['initial_penalized_nll']:.3f} | {optimizer['final_penalized_nll']:.3f}")
    print(f"Sums of attack and concede log parameters: {model['attack_log_sum']:.3e} | {model['concede_log_sum']:.3e}")
    print("Fitting success is a numerical check. Predictive quality requires a separate backtest.")
    print("\nColumns: previous Poisson | DC lambdas without correction | full Dixon–Coles")
    print("Expected goals:")
    for label, key in (("Home", "lambda_home"), ("Away", "lambda_away"), ("Total", "lambda_total")):
        print(f"{label}: {original[key]:.3f} | {prediction[key]:.3f} | {prediction[key]:.3f}")
    print("\n1X2:")
    for label, key in (("Home win", "home"), ("Draw", "draw"), ("Away win", "away")):
        print(f"{label}: {original_markets['one_x_two'][key]:.2%} | {independent_markets['one_x_two'][key]:.2%} | {corrected_markets['one_x_two'][key]:.2%}")
    print("\nOver / under:")
    for line in (1.5, 2.5, 3.5):
        for event in ("under", "over"):
            print(f"{line:.1f} {event}: {original_markets['over_under'][line][event]:.2%} | {independent_markets['over_under'][line][event]:.2%} | {corrected_markets['over_under'][line][event]:.2%}")
    print("\nBoth teams to score:")
    for label, event in (("Yes", "yes"), ("No", "no")):
        print(f"{label}: {original_markets['btts'][event]:.2%} | {independent_markets['btts'][event]:.2%} | {corrected_markets['btts'][event]:.2%}")
    factors = tau_factors(prediction["lambda_home"], prediction["lambda_away"], model["rho"])
    effects = []
    print("\nThe four low-score cells – with fixed DC lambdas:")
    print(f"{'Score':<9} {'Tau':>9} {'Uncorrected':>19} {'Corrected':>12} {'Difference':>12}")
    for (hg, ag), factor in factors.items():
        before, after = independent_grid["matrix"][hg][ag], corrected_grid["matrix"][hg][ag]
        print(f"{hg}–{ag:<7} {factor:>9.6f} {before:>18.2%} {after:>11.2%} {(after-before)*100:>+9.3f} pp")
        effects.append({"home_goals": hg, "away_goals": ag, "tau": factor,
                        "p_original_poisson": original_grid["matrix"][hg][ag],
                        "p_dc_rates_without_correction": before, "p_dixon_coles": after,
                        "correction_difference": after - before})
    print("\nChecks with fixed lambdas:")
    for line in (2.5, 3.5):
        difference = corrected_markets["over_under"][line]["over"] - independent_markets["over_under"][line]["over"]
        if abs(difference) > 1e-12:
            raise ValueError("The low-score correction incorrectly changed the over 2.5/3.5 probability.")
        print(f"Over {line:.1f} correction difference: {difference:+.3e}")
    print(f"Probability-mass change from correction: {corrected_grid['mass_change_from_correction']:+.3e}")
    print(f"Probability omitted from the score table: {corrected_grid['omitted_probability']:.3e}")
    print("\nThe five most probable scores under the full Dixon–Coles model:")
    for score in corrected_markets["top_scores"]:
        print(f"{score['home_goals']}–{score['away_goals']}: {score['probability']:.2%}")
    safe_teams = re.sub(r"[^a-z0-9]+", "_", f"{HOME_TEAM}_{AWAY_TEAM}".lower()).strip("_")
    slug = f"{MATCH_DATE}_{safe_teams}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / f"dixon_coles_model_{slug}.json"
    comparison_path = OUTPUT_DIR / f"match_comparison_{slug}.csv"
    effect_path = OUTPUT_DIR / f"low_score_effects_{slug}.csv"
    teams_path = OUTPUT_DIR / f"team_parameters_{slug}.csv"
    model.update({
        "step": 14, "evaluation_role": "single_fixture_demonstration", "match_id": snapshot["match_id"],
        "home_team": HOME_TEAM, "away_team": AWAY_TEAM, "data_file": DATA_PATH.name, "data_sha256": data_hash,
        "history_rule": "match_date - lookback_days <= Date < match_date", "same_day_results_used": False,
        "final_test_season_reserved": "2025/26", "prediction": prediction, "probabilities": corrected_markets,
        "omitted_probability": corrected_grid["omitted_probability"],
        "source_sha256": {name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest() for name in (
            "analyze_match.py", "time_weighted_model.py", "score_probabilities.py", "fitted_poisson_model.py",
            "dixon_coles_probabilities.py", "dixon_coles_model.py", "predict_dixon_coles.py",
        )},
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "method_source": "https://doi.org/10.1111/1467-9876.00065",
    })
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([
        comparison_row("fitted_poisson", original, original_markets),
        comparison_row("dc_rates_independent", prediction, independent_markets),
        comparison_row("dixon_coles", prediction, corrected_markets),
    ]).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(effects).to_csv(effect_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"team": name, **values} for name, values in model["teams"].items()]).to_csv(
        teams_path, index=False, encoding="utf-8-sig",
    )
    print(f"\nFitted model and settings saved: {model_path}")
    print(f"Match predictions saved: {comparison_path}")
    print(f"Low-score effects saved: {effect_path}")
    print(f"Team parameters saved: {teams_path}")
    print("\nThis is a one-match demonstration. The sign and size of rho come from this history.")
    print("The next step also backtests the Dixon–Coles model chronologically.")
    print("The 2025/26 season remains the separate final test.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError, pd.errors.ParserError) as error:
        raise SystemExit(f"Dixon–Coles error: {error}")
