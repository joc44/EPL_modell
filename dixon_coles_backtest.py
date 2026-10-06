"""Chronological Dixon–Coles backtest, six-model evaluation and correction decomposition."""

from collections.abc import Callable
from math import isclose, isfinite

import pandas as pd

from backtest_models import GOAL_LINES, LOSS_MARKETS
from calibration_analysis import EVENTS, calibration_bins, event_observations
from dixon_coles_model import fit_dixon_coles_model, predict_dixon_coles_goals
from dixon_coles_probabilities import build_dixon_coles_score_matrix, dixon_coles_score_probability
from evaluation_metrics import observed_probability_loss, poisson_score_probability
from fitted_backtest import (
    COMPARISON_LABELS as PREVIOUS_LABELS, COMPARISON_MODELS as PREVIOUS_MODELS,
    evaluate_forecast, validate_comparison,
)
from score_probabilities import build_score_matrix, market_probabilities


NEW_MODEL = "dixon_coles"
COMPARISON_MODELS = (*PREVIOUS_MODELS, NEW_MODEL)
COMPARISON_LABELS = {**PREVIOUS_LABELS, NEW_MODEL: "Dixon–Coles"}
UNCORRECTED_FIELDS = tuple(f"uncorrected_{market}_log_loss" for market in LOSS_MARKETS)


def forecast_from_rates(lambda_home: float, lambda_away: float, rho: float) -> dict:
    """Forecast from the final lambdas without using the observed result."""
    grid = build_dixon_coles_score_matrix(lambda_home, lambda_away, rho)
    return {
        "lambda_home": lambda_home, "lambda_away": lambda_away, "rho": rho,
        "minimum_tau": grid["minimum_tau"], "omitted_probability": grid["omitted_probability"],
        "markets": market_probabilities(grid["matrix"], GOAL_LINES, top_n=1),
    }


def evaluate_dixon_coles_forecast(forecast: dict, home_goals: int, away_goals: int) -> dict:
    """Use the shared metrics for all markets; exact-score probabilities now come from DC."""
    record = evaluate_forecast(forecast, home_goals, away_goals)
    probability = dixon_coles_score_probability(
        forecast["lambda_home"], forecast["lambda_away"], forecast["rho"], home_goals, away_goals
    )
    loss = observed_probability_loss(probability)
    record.update({
        "rho": forecast["rho"], "minimum_tau": forecast["minimum_tau"],
        "p_actual_score": probability, "exact_score_log_loss": loss["log_loss"],
        "exact_score_log_loss_clipped": int(loss["clipped"]),
    })
    # These are the same fitted DC lambdas without tau: a diagnostic decomposition,
    # rather than an independently trained seventh model.
    grid = build_score_matrix(forecast["lambda_home"], forecast["lambda_away"])
    uncorrected = evaluate_forecast({
        "lambda_home": forecast["lambda_home"], "lambda_away": forecast["lambda_away"],
        "omitted_probability": grid["omitted_probability"],
        "markets": market_probabilities(grid["matrix"], GOAL_LINES, top_n=1),
    }, home_goals, away_goals)
    record.update({f"uncorrected_{market}_log_loss": uncorrected[f"{market}_log_loss"]
                   for market in LOSS_MARKETS})
    return record


def run_dixon_coles_backtest(
    matches: pd.DataFrame, evaluation_season: str,
    lookback_days: int = 730, half_life_days: float | None = 180.0,
    ridge_strength: float = 1.0, progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a new joint model before each target matchday; stop if a day fails."""
    fixtures = matches.loc[matches.Season == evaluation_season].sort_values(["Date", "match_id"])
    if fixtures.empty or fixtures.match_id.duplicated().any():
        raise ValueError("A non-empty season with unique match identifiers is required.")
    records, diagnostics = [], []
    completed = 0
    for date, day_fixtures in fixtures.groupby("Date", sort=True):
        date_text = date.strftime("%Y-%m-%d")
        try:
            model = fit_dixon_coles_model(matches, date_text, lookback_days, half_life_days, ridge_strength)
            if model["last_history_date"] >= date_text:
                raise ValueError("The fit used same-day or later data.")
            if not model["optimizer"]["success"]:
                raise RuntimeError("Optimisation was unsuccessful.")
        except (ValueError, TypeError, RuntimeError, OverflowError) as error:
            raise ValueError(f"Dixon–Coles fit failed on this matchday: {date_text}. {error}") from error
        optimizer, start = model["optimizer"], model["poisson_start_model"]["optimizer"]
        diagnostics.append({
            "Season": evaluation_season, "fit_date": date_text, "fixtures": len(day_fixtures),
            "history_start": model["history_start"], "last_history_date": model["last_history_date"],
            "history_matches": model["history_matches"], "history_weight_sum": model["history_weight_sum"],
            "low_score_history_matches": model["low_score_history_matches"],
            "low_score_history_weight_sum": model["low_score_history_weight_sum"],
            "teams": len(model["teams"]), "free_parameters": model["free_parameters"],
            "rho": model["rho"], "rho_lower": model["rho_domain_at_fitted_rates"]["lower"],
            "rho_upper": model["rho_domain_at_fitted_rates"]["upper"],
            "success": optimizer["success"], "method": optimizer["method"],
            "iterations": optimizer["iterations"], "gradient_max_abs": optimizer["gradient_max_abs"],
            "stationarity_max_abs": optimizer["stationarity_max_abs"],
            "active_tau_constraints": optimizer["active_tau_constraints"],
            "minimum_tau_all_pairs": optimizer["minimum_tau_all_pairs"],
            "tau_constraint_floor": optimizer["tau_constraint_floor"],
            "initial_penalized_nll": optimizer["initial_penalized_nll"],
            "final_penalized_nll": optimizer["final_penalized_nll"],
            "attack_log_sum": model["attack_log_sum"], "concede_log_sum": model["concede_log_sum"],
            "poisson_start_success": start["success"],
            "poisson_start_gradient_max_abs": start["gradient_max_abs"],
            "poisson_start_refinement_steps": start.get("refinement_steps", 0),
        })
        # Complete every forecast for a day before evaluating that day's results.
        forecasts = []
        for fixture in day_fixtures[["match_id", "HomeTeam", "AwayTeam"]].itertuples(index=False):
            rates = predict_dixon_coles_goals(model, fixture.HomeTeam, fixture.AwayTeam)
            forecast = forecast_from_rates(rates["lambda_home"], rates["lambda_away"], rates["rho"])
            home, away = model["teams"].get(fixture.HomeTeam, {}), model["teams"].get(fixture.AwayTeam, {})
            metadata = {
                "match_id": fixture.match_id, "model": NEW_MODEL, "Season": evaluation_season,
                "Date": date_text, "HomeTeam": fixture.HomeTeam, "AwayTeam": fixture.AwayTeam,
                "history_start": model["history_start"], "last_history_date": model["last_history_date"],
                "history_matches": model["history_matches"],
                "home_history_matches": home.get("home_history_matches", 0),
                "away_history_matches": away.get("away_history_matches", 0),
                "home_all_history_matches": rates["home_history_matches"],
                "away_all_history_matches": rates["away_history_matches"],
                "home_unknown": int(rates["home_unknown"]), "away_unknown": int(rates["away_unknown"]),
            }
            forecasts.append((fixture.match_id, metadata, forecast))
        actuals = day_fixtures.set_index("match_id")
        for match_id, metadata, forecast in forecasts:
            actual = actuals.loc[match_id]
            records.append({**metadata, **evaluate_dixon_coles_forecast(forecast, int(actual.FTHG), int(actual.FTAG))})
            completed += 1
            if progress is not None:
                progress(completed, len(fixtures))
    return pd.DataFrame(records), pd.DataFrame(diagnostics)


def validate_dixon_coles_comparison(predictions: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Validate the six-model table and recalculate DC probabilities from lambdas and rho."""
    if "model" not in predictions:
        raise ValueError("The model column is missing.")
    dc_mask = predictions.model == NEW_MODEL
    required = {"rho", "minimum_tau", *UNCORRECTED_FIELDS,
                "home_all_history_matches", "away_all_history_matches", "home_unknown", "away_unknown"}
    if (not required.issubset(predictions.columns) or not dc_mask.any()
            or predictions.loc[dc_mask, list(required)].isna().any().any()):
        raise ValueError("Incomplete Dixon–Coles prediction.")
    result = predictions.copy().reset_index(drop=True)
    dc_mask = result.model == NEW_MODEL
    # Retain the previous validator's shared date, fixture-list and metric rules.
    # In a temporary copy, replace only the model-specific exact-score probability
    # with Poisson; validate the original corrected DC values separately below.
    proxy = result.copy()
    for index, row in result.loc[dc_mask].iterrows():
        probability = poisson_score_probability(row.lambda_home, row.lambda_away, int(row.FTHG), int(row.FTAG))
        loss = observed_probability_loss(probability)
        proxy.loc[index, ["p_actual_score", "exact_score_log_loss", "exact_score_log_loss_clipped"]] = (
            probability, loss["log_loss"], int(loss["clipped"])
        )
    checked = validate_comparison(proxy, fixtures, COMPARISON_MODELS)
    for column in ("Date", "history_start", "last_history_date"):
        result[column] = checked[column]
    for row in result.loc[dc_mask].to_dict("records"):
        forecast = forecast_from_rates(row["lambda_home"], row["lambda_away"], row["rho"])
        recomputed = evaluate_dixon_coles_forecast(forecast, int(row["FTHG"]), int(row["FTAG"]))
        for column, expected in recomputed.items():
            value = row[column]
            equal = (value == expected if isinstance(expected, str)
                     else isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12))
            if not equal:
                raise ValueError(f"A value recalculated from the DC probability differs: {column}, {row['match_id']}.")
        for side in ("home", "away"):
            count = row[f"{side}_all_history_matches"]
            unknown = row[f"{side}_unknown"]
            if (not isfinite(count) or count % 1 or not 0 <= count <= row["history_matches"]
                    or unknown not in (0, 1) or unknown != int(count == 0)
                    or count < row[f"{side}_history_matches"]):
                raise ValueError("Invalid full-team history or unknown-team indicator.")
    if result.loc[dc_mask].groupby("Date").rho.nunique().gt(1).any():
        raise ValueError("Each matchday must have a single fitted rho.")
    return result


def calibration_models(predictions: pd.DataFrame, n_bins: int = 10, min_bin_matches: int = 20) -> tuple:
    """Extend the previous ten-bin calibration analysis to all six models."""
    all_bins, summaries = [], []
    for name in COMPARISON_MODELS:
        rows = predictions.loc[predictions.model == name]
        if rows.empty:
            raise ValueError(f"A model required for calibration is missing: {name}.")
        for event, (_, probability_column, _) in EVENTS.items():
            observations = event_observations(rows, event)
            bins = calibration_bins(rows[probability_column], observations, n_bins, min_bin_matches)
            bins.insert(0, "event", event)
            bins.insert(0, "model", name)
            all_bins.append(bins)
            active = bins.loc[~bins["empty"]]
            mean_probability, observed_rate = float(rows[probability_column].mean()), float(observations.mean())
            summaries.append({
                "model": name, "event": event, "matches": len(rows),
                "mean_probability": mean_probability, "observed_rate": observed_rate,
                "bias": mean_probability - observed_rate,
                "ece": float((active.matches * active.bias.abs()).sum() / len(rows)),
                "empty_bins": int(bins["empty"].sum()), "sparse_bins": int(bins["sparse"].sum()),
                "matches_in_sparse_bins": int(bins.loc[bins["sparse"], "matches"].sum()),
            })
    return pd.concat(all_bins, ignore_index=True), pd.DataFrame(summaries)


def paired_loss_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    """DC minus each of the five references, using identical match identifiers only."""
    new = predictions.loc[predictions.model == NEW_MODEL].set_index("match_id")
    if new.empty or new.index.duplicated().any():
        raise ValueError("A unique, non-empty Dixon–Coles match list is required.")
    records = []
    for reference in PREVIOUS_MODELS:
        old = predictions.loc[predictions.model == reference].set_index("match_id")
        if old.index.duplicated().any() or set(new.index) != set(old.index):
            raise ValueError("Pairing requires identical, unique matches.")
        for market in LOSS_MARKETS:
            difference = new[f"{market}_log_loss"] - old.loc[new.index, f"{market}_log_loss"]
            records.append({
                "new_model": NEW_MODEL, "reference_model": reference, "market": market,
                "matches": len(difference), "mean_loss_difference": float(difference.mean()),
                "new_model_lower_loss_matches": int((difference < 0).sum()),
                "equal_loss_matches": int((difference == 0).sum()),
            })
    return pd.DataFrame(records)


def correction_effects(predictions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive three-way decomposition: previous Poisson, new lambdas, full DC."""
    new = predictions.loc[predictions.model == NEW_MODEL].set_index("match_id")
    old = predictions.loc[predictions.model == "fitted_poisson"].set_index("match_id")
    if new.empty or new.index.duplicated().any() or old.index.duplicated().any() or set(new.index) != set(old.index):
        raise ValueError("Correction decomposition requires identical, unique matches.")
    old = old.loc[new.index]
    records = []
    for market in LOSS_MARKETS:
        baseline = float(old[f"{market}_log_loss"].mean())
        uncorrected = float(new[f"uncorrected_{market}_log_loss"].mean())
        corrected = float(new[f"{market}_log_loss"].mean())
        records.append({
            "market": market, "matches": len(new), "fitted_poisson_log_loss": baseline,
            "dc_lambdas_uncorrected_log_loss": uncorrected, "dixon_coles_log_loss": corrected,
            "lambda_refit_loss_difference": uncorrected - baseline,
            "tau_correction_loss_difference": corrected - uncorrected,
            "total_loss_difference": corrected - baseline,
        })
    return pd.DataFrame(records)
