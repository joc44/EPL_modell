"""Chronological evaluation and comparison of the jointly fitted model."""

from collections.abc import Callable
from math import isclose, isfinite

import pandas as pd

from backtest_models import BINARY_MARKETS, GOAL_LINES, LOSS_MARKETS, MODEL_LABELS, MODEL_NAMES
from calibration_analysis import EVENTS, calibration_bins, event_observations
from evaluation_metrics import (
    binary_metrics, observed_probability_loss, one_x_two_metrics,
    poisson_score_probability, realized_outcome,
)
from fitted_poisson_model import fit_poisson_model, predict_goals_from_model
from score_probabilities import build_score_matrix, market_probabilities


NEW_MODEL = "fitted_poisson"
COMPARISON_MODELS = (*MODEL_NAMES, NEW_MODEL)
COMPARISON_LABELS = {**MODEL_LABELS, NEW_MODEL: "Jointly fitted"}
FIXTURE_FIELDS = (
    "match_id", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
    "history_start", "last_history_date", "history_matches",
    "home_history_matches", "away_history_matches",
)


def evaluate_forecast(prediction: dict, home_goals: int, away_goals: int) -> dict:
    """Attach the result to the completed forecast using the previous metrics."""
    markets = prediction["markets"]
    outcome = realized_outcome(home_goals, away_goals)
    one_x_two = one_x_two_metrics(markets["one_x_two"], outcome)
    top = markets["top_scores"][0]
    score_probability = poisson_score_probability(
        prediction["lambda_home"], prediction["lambda_away"], home_goals, away_goals
    )
    score_loss = observed_probability_loss(score_probability)
    record = {
        "FTHG": home_goals, "FTAG": away_goals, "actual_one_x_two": outcome,
        "lambda_home": prediction["lambda_home"], "lambda_away": prediction["lambda_away"],
        "lambda_total": prediction["lambda_home"] + prediction["lambda_away"],
        "p_home": markets["one_x_two"]["home"], "p_draw": markets["one_x_two"]["draw"],
        "p_away": markets["one_x_two"]["away"],
        "one_x_two_predicted": one_x_two["predicted"],
        "one_x_two_log_loss": one_x_two["log_loss"], "one_x_two_brier": one_x_two["brier"],
        "one_x_two_hit": int(one_x_two["hit"]),
        "one_x_two_log_loss_clipped": int(one_x_two["clipped"]),
        "p_actual_score": score_probability,
        "top_score_home": top["home_goals"], "top_score_away": top["away_goals"],
        "exact_score_log_loss": score_loss["log_loss"],
        "exact_score_log_loss_clipped": int(score_loss["clipped"]),
        "exact_score_hit": int((home_goals, away_goals) == (top["home_goals"], top["away_goals"])),
        "omitted_probability": prediction["omitted_probability"],
    }
    for line in GOAL_LINES:
        key = "over_" + str(line).replace(".", "_")
        observed = int(home_goals + away_goals > line)
        probability = markets["over_under"][line]["over"]
        metrics = binary_metrics(probability, observed)
        record.update({
            f"p_{key}": probability, f"actual_{key}": observed,
            f"{key}_log_loss": metrics["log_loss"], f"{key}_brier": metrics["brier"],
            f"{key}_log_loss_clipped": int(metrics["clipped"]),
        })
    btts_observed = int(home_goals > 0 and away_goals > 0)
    btts = binary_metrics(markets["btts"]["yes"], btts_observed)
    record.update({
        "p_btts": markets["btts"]["yes"], "actual_btts": btts_observed,
        "btts_log_loss": btts["log_loss"], "btts_brier": btts["brier"],
        "btts_log_loss_clipped": int(btts["clipped"]),
    })
    return record


def run_fitted_backtest(
    matches: pd.DataFrame, evaluation_season: str,
    lookback_days: int = 730, half_life_days: float | None = 180.0,
    ridge_strength: float = 1.0, progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a new model for each target matchday; all fixtures on that day use it.

    Do not fit once on the entire evaluation season. In this backtest,
    a matchday's model is used only on that day.
    """
    fixtures = matches.loc[matches["Season"] == evaluation_season].sort_values(["Date", "match_id"])
    if fixtures.empty or fixtures["match_id"].duplicated().any():
        raise ValueError("A non-empty season with unique match identifiers is required.")
    records, diagnostics = [], []
    completed = 0
    for date, day_fixtures in fixtures.groupby("Date", sort=True):
        date_text = date.strftime("%Y-%m-%d")
        try:
            model = fit_poisson_model(matches, date_text, lookback_days, half_life_days, ridge_strength)
            if model["last_history_date"] >= date_text:
                raise ValueError("The fit used same-day or later data.")
        except (ValueError, TypeError, RuntimeError, OverflowError) as error:
            raise ValueError(f"Fit failed on this matchday: {date_text}. {error}") from error
        optimizer = model["optimizer"]
        diagnostics.append({
            "Season": evaluation_season, "fit_date": date_text, "fixtures": len(day_fixtures),
            "history_start": model["history_start"], "last_history_date": model["last_history_date"],
            "history_matches": model["history_matches"], "history_weight_sum": model["history_weight_sum"],
            "teams": len(model["teams"]), "free_parameters": model["free_parameters"],
            "success": optimizer["success"], "iterations": optimizer["iterations"],
            "refinement_steps": optimizer.get("refinement_steps", 0),
            "gradient_max_abs": optimizer["gradient_max_abs"],
            "initial_penalized_nll": optimizer["initial_penalized_nll"],
            "final_penalized_nll": optimizer["final_penalized_nll"],
        })
        # Complete all same-day forecasts before evaluating that day's final scores.
        forecasts = []
        for fixture in day_fixtures[["match_id", "HomeTeam", "AwayTeam"]].itertuples(index=False):
            prediction = predict_goals_from_model(model, fixture.HomeTeam, fixture.AwayTeam)
            grid = build_score_matrix(prediction["lambda_home"], prediction["lambda_away"])
            forecast = {**prediction, "omitted_probability": grid["omitted_probability"],
                        "markets": market_probabilities(grid["matrix"], GOAL_LINES, top_n=1)}
            home = model["teams"].get(fixture.HomeTeam, {})
            away = model["teams"].get(fixture.AwayTeam, {})
            common = {
                "match_id": fixture.match_id, "model": NEW_MODEL, "Season": evaluation_season,
                "Date": date_text, "HomeTeam": fixture.HomeTeam, "AwayTeam": fixture.AwayTeam,
                "history_start": model["history_start"], "last_history_date": model["last_history_date"],
                "history_matches": model["history_matches"],
                # Shared venue-specific match counts retain the meaning used in the previous report.
                "home_history_matches": home.get("home_history_matches", 0),
                "away_history_matches": away.get("away_history_matches", 0),
                "home_all_history_matches": prediction["home_history_matches"],
                "away_all_history_matches": prediction["away_history_matches"],
                "home_unknown": int(prediction["home_unknown"]), "away_unknown": int(prediction["away_unknown"]),
            }
            forecasts.append((fixture.match_id, common, forecast))
        actuals = day_fixtures.set_index("match_id")
        for match_id, common, forecast in forecasts:
            actual = actuals.loc[match_id]
            records.append({**common, **evaluate_forecast(forecast, int(actual.FTHG), int(actual.FTAG))})
            completed += 1
            if progress is not None:
                progress(completed, len(fixtures))
    return pd.DataFrame(records), pd.DataFrame(diagnostics)


def validate_comparison(
    predictions: pd.DataFrame, fixtures: pd.DataFrame,
    model_names: tuple[str, ...] = COMPARISON_MODELS,
) -> pd.DataFrame:
    """Identical fixtures, results, temporal boundaries and recalculable metrics."""
    metric_fields = [f"{market}_{suffix}" for market in LOSS_MARKETS for suffix in ("log_loss", "log_loss_clipped")]
    metric_fields += [f"{market}_brier" for market in ("one_x_two", *BINARY_MARKETS)]
    required = {*FIXTURE_FIELDS, "model", "lambda_home", "lambda_away", "lambda_total",
                "p_home", "p_draw", "p_away", "p_btts", "p_actual_score", "actual_one_x_two",
                "one_x_two_predicted", "one_x_two_hit", "exact_score_hit", "top_score_home", "top_score_away",
                "omitted_probability", *metric_fields,
                *[f"p_{key}" for key in BINARY_MARKETS], *[f"actual_{key}" for key in BINARY_MARKETS]}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError("Incomplete comparison prediction table.")
    result = predictions.copy()
    if result[list(required)].isna().any().any():
        raise ValueError("A required value is missing from the predictions.")
    if set(result["model"]) != set(model_names):
        raise ValueError("A model is missing from the comparison, or an unexpected model is present.")
    if result.duplicated(["match_id", "model"]).any():
        raise ValueError("Duplicate match–model pair.")
    if fixtures["match_id"].duplicated().any() or fixtures.empty:
        raise ValueError("A unique, non-empty evaluation match list is required.")
    for column in ("Date", "history_start", "last_history_date"):
        result[column] = pd.to_datetime(result[column], format="%Y-%m-%d", errors="raise")
    if not ((result.history_start <= result.last_history_date) & (result.last_history_date < result.Date)).all():
        raise ValueError("Invalid history boundary: same-day and later data must be excluded.")
    count_fields = ("FTHG", "FTAG", "history_matches", "home_history_matches", "away_history_matches",
                    "top_score_home", "top_score_away")
    for column in count_fields:
        if not result[column].map(lambda value: isfinite(value) and value >= 0 and value % 1 == 0).all():
            raise ValueError(f"A non-negative integer field is required: {column}.")
    if (result.history_matches <= 0).any():
        raise ValueError("Positive league history is required.")
    if ((result.home_history_matches > result.history_matches)
            | (result.away_history_matches > result.history_matches)).any():
        raise ValueError("Team history must not exceed league history.")
    if not result.omitted_probability.map(lambda value: isfinite(value) and 0 <= value <= 1).all():
        raise ValueError("The omitted probability must be between zero and one.")
    fixture_columns = ["match_id", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    expected = fixtures[fixture_columns].sort_values("match_id").reset_index(drop=True).copy()
    expected["Date"] = pd.to_datetime(expected["Date"])
    for name in model_names:
        actual = result.loc[result.model == name, fixture_columns].sort_values("match_id").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError(f"Different match list or result for this model: {name}.") from error
    if (result.groupby("match_id")[list(FIXTURE_FIELDS[1:])].nunique() != 1).any().any():
        raise ValueError("A match result or its history data differ between models.")
    for row in result.to_dict("records"):
        for key in ("lambda_home", "lambda_away", "lambda_total"):
            if not isfinite(row[key]) or row[key] < 0:
                raise ValueError("The expected goal count must be finite and non-negative.")
        forecast = {
            "lambda_home": row["lambda_home"], "lambda_away": row["lambda_away"],
            "omitted_probability": row["omitted_probability"],
            "markets": {
                "one_x_two": {"home": row["p_home"], "draw": row["p_draw"], "away": row["p_away"]},
                "over_under": {line: {"over": row["p_over_" + str(line).replace(".", "_")]}
                               for line in GOAL_LINES},
                "btts": {"yes": row["p_btts"]},
                "top_scores": [{"home_goals": int(row["top_score_home"]), "away_goals": int(row["top_score_away"])}],
            },
        }
        recalculated = evaluate_forecast(forecast, int(row["FTHG"]), int(row["FTAG"]))
        for key, value in recalculated.items():
            given = row[key]
            equal = given == value if isinstance(value, str) else isclose(given, value, rel_tol=1e-10, abs_tol=1e-12)
            if not equal:
                raise ValueError(f"A metric differs from the value calculated from the stored probability: {key}, {row['match_id']}.")
    return result


def summarize_models(predictions: pd.DataFrame, model_names: tuple[str, ...] = COMPARISON_MODELS) -> pd.DataFrame:
    """Equal-match-weight summary for the previous four models and the new model."""
    if set(predictions.model) != set(model_names) or predictions.duplicated(["match_id", "model"]).any():
        raise ValueError("Invalid model set or duplicate prediction.")
    reference_ids = set(predictions.loc[predictions.model == model_names[0], "match_id"])
    records = []
    for name in model_names:
        rows = predictions.loc[predictions.model == name]
        if set(rows.match_id) != reference_ids:
            raise ValueError("Models must be summarised on the same matches.")
        record = {"model": name, "matches": len(rows)}
        for market in LOSS_MARKETS:
            record[f"{market}_log_loss"] = float(rows[f"{market}_log_loss"].mean())
            record[f"{market}_log_loss_clipped"] = int(rows[f"{market}_log_loss_clipped"].sum())
        for market in ("one_x_two", *BINARY_MARKETS):
            record[f"{market}_brier"] = float(rows[f"{market}_brier"].mean())
        for market in ("one_x_two", "exact_score"):
            record[f"{market}_accuracy"] = float(rows[f"{market}_hit"].mean())
        record["missing_home_history"] = int((rows.home_history_matches == 0).sum())
        record["missing_away_history"] = int((rows.away_history_matches == 0).sum())
        record["max_omitted_probability"] = float(rows.omitted_probability.max())
        records.append(record)
    return pd.DataFrame(records)


def calibration_models(predictions: pd.DataFrame, n_bins: int = 10, min_bin_matches: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identical binned calibration calculation for five models and seven events."""
    all_bins, summaries = [], []
    for name in COMPARISON_MODELS:
        rows = predictions.loc[predictions.model == name]
        for event, (_, probability_column, _) in EVENTS.items():
            bins = calibration_bins(rows[probability_column], event_observations(rows, event), n_bins, min_bin_matches)
            bins.insert(0, "event", event)
            bins.insert(0, "model", name)
            all_bins.append(bins)
            active = bins.loc[~bins["empty"]]
            mean_probability = float(rows[probability_column].mean())
            observed_rate = float(event_observations(rows, event).mean())
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
    """Paired match-level differences: new model minus previous model."""
    new = predictions.loc[predictions.model == NEW_MODEL].set_index("match_id")
    records = []
    for reference in MODEL_NAMES:
        old = predictions.loc[predictions.model == reference].set_index("match_id")
        if new.index.duplicated().any() or old.index.duplicated().any() or set(new.index) != set(old.index):
            raise ValueError("Paired comparison requires identical, unique matches.")
        for market in LOSS_MARKETS:
            difference = new[f"{market}_log_loss"] - old.loc[new.index, f"{market}_log_loss"]
            records.append({
                "new_model": NEW_MODEL, "reference_model": reference, "market": market,
                "matches": len(difference), "mean_loss_difference": float(difference.mean()),
                "new_model_lower_loss_matches": int((difference < 0).sum()),
                "equal_loss_matches": int((difference == 0).sum()),
            })
    return pd.DataFrame(records)
