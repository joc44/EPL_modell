"""Chronological backtest: forecast first, then evaluate the result separately."""

from collections.abc import Callable
from math import isfinite

import pandas as pd

from analyze_match import analyze_fixture
from evaluation_metrics import (
    binary_metrics,
    observed_probability_loss,
    one_x_two_metrics,
    poisson_score_probability,
    realized_outcome,
)
from goal_model import calculate_expected_goals
from score_probabilities import build_score_matrix, market_probabilities
from shrinkage_model import calculate_smoothed_expected_goals
from time_weighted_model import calculate_time_weighted_expected_goals


MODEL_NAMES = ("league_average", "raw", "smoothed", "time_weighted")
MODEL_LABELS = {
    "league_average": "League average",
    "raw": "Raw",
    "smoothed": "Smoothed",
    "time_weighted": "Time-weighted",
}
GOAL_LINES = (1.5, 2.5, 3.5)
BINARY_MARKETS = ("over_1_5", "over_2_5", "over_3_5", "btts")
LOSS_MARKETS = ("one_x_two", *BINARY_MARKETS, "exact_score")


def predict_fixture_models(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: str,
    lookback_days: int = 730,
    prior_matches: float = 5.0,
    half_life_days: float | None = 180.0,
) -> dict:
    """Forecast with all four models without reading the target match goals."""
    analysis = analyze_fixture(matches, home_team, away_team, match_date, lookback_days)
    if analysis["last_history_date"] >= analysis["match_date"]:
        raise ValueError("History must not include the target matchday or any later date.")

    league = analysis["league"]
    predictions = {
        "league_average": {
            "lambda_home": league["home_goal_mean"],
            "lambda_away": league["away_goal_mean"],
            "lambda_total": league["home_goal_mean"] + league["away_goal_mean"],
        },
        "raw": calculate_expected_goals(analysis),
        "smoothed": calculate_smoothed_expected_goals(analysis, prior_matches),
        "time_weighted": calculate_time_weighted_expected_goals(
            matches, home_team, away_team, match_date,
            lookback_days, half_life_days, prior_matches,
        ),
    }
    forecast = {}
    for name, prediction in predictions.items():
        distribution = build_score_matrix(prediction["lambda_home"], prediction["lambda_away"])
        forecast[name] = {
            "lambda_home": prediction["lambda_home"],
            "lambda_away": prediction["lambda_away"],
            "lambda_total": prediction["lambda_total"],
            "markets": market_probabilities(distribution["matrix"], GOAL_LINES, top_n=1),
            "omitted_probability": distribution["omitted_probability"],
        }

    return {
        "match_id": analysis["match_id"],
        "history_start": analysis["start_date"].strftime("%Y-%m-%d"),
        "last_history_date": analysis["last_history_date"].strftime("%Y-%m-%d"),
        "history_matches": league["matches"],
        "home_history_matches": analysis["home"]["matches"],
        "away_history_matches": analysis["away"]["matches"],
        "models": forecast,
    }


def evaluate_fixture(forecast: dict, home_goals: int, away_goals: int) -> list[dict]:
    """Attach the observed result only after the forecast has been completed."""
    outcome = realized_outcome(home_goals, away_goals)
    records = []
    for name in MODEL_NAMES:
        prediction = forecast["models"][name]
        markets = prediction["markets"]
        one_x_two = one_x_two_metrics(markets["one_x_two"], outcome)
        top = markets["top_scores"][0]
        score_probability = poisson_score_probability(
            prediction["lambda_home"], prediction["lambda_away"], home_goals, away_goals
        )
        score_loss = observed_probability_loss(score_probability)
        record = {
            "model": name,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "actual_one_x_two": outcome,
            "lambda_home": prediction["lambda_home"],
            "lambda_away": prediction["lambda_away"],
            "lambda_total": prediction["lambda_total"],
            "p_home": markets["one_x_two"]["home"],
            "p_draw": markets["one_x_two"]["draw"],
            "p_away": markets["one_x_two"]["away"],
            "one_x_two_predicted": one_x_two["predicted"],
            "one_x_two_log_loss": one_x_two["log_loss"],
            "one_x_two_brier": one_x_two["brier"],
            "one_x_two_hit": int(one_x_two["hit"]),
            "one_x_two_log_loss_clipped": int(one_x_two["clipped"]),
            "p_actual_score": score_probability,
            "top_score_home": top["home_goals"],
            "top_score_away": top["away_goals"],
            "exact_score_log_loss": score_loss["log_loss"],
            "exact_score_log_loss_clipped": int(score_loss["clipped"]),
            "exact_score_hit": int((home_goals, away_goals) == (top["home_goals"], top["away_goals"])),
            "omitted_probability": prediction["omitted_probability"],
        }
        for line in GOAL_LINES:
            key = "over_" + str(line).replace(".", "_")
            probability = markets["over_under"][line]["over"]
            observed = home_goals + away_goals > line
            metrics = binary_metrics(probability, observed)
            record.update({
                f"p_{key}": probability,
                f"actual_{key}": int(observed),
                f"{key}_log_loss": metrics["log_loss"],
                f"{key}_brier": metrics["brier"],
                f"{key}_log_loss_clipped": int(metrics["clipped"]),
            })
        btts_observed = home_goals > 0 and away_goals > 0
        btts = binary_metrics(markets["btts"]["yes"], btts_observed)
        record.update({
            "p_btts": markets["btts"]["yes"],
            "actual_btts": int(btts_observed),
            "btts_log_loss": btts["log_loss"],
            "btts_brier": btts["brier"],
            "btts_log_loss_clipped": int(btts["clipped"]),
        })
        records.append(record)
    return records


def run_backtest(
    matches: pd.DataFrame,
    evaluation_season: str = "2023/24",
    lookback_days: int = 730,
    prior_matches: float = 5.0,
    half_life_days: float | None = 180.0,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every match in the season, ordered by date and identifier.

    Stop on a failed forecast; never silently omit a match.
    Parameters remain fixed throughout the run.
    """
    fixtures = matches.loc[matches["Season"] == evaluation_season].sort_values(
        ["Date", "match_id"], kind="stable"
    )
    if fixtures.empty:
        raise ValueError(f"No matches from the selected season: {evaluation_season}.")
    if fixtures["match_id"].duplicated().any():
        raise ValueError("The match identifiers to evaluate contain duplicates.")

    records = []
    for completed, fixture in enumerate(fixtures.itertuples(index=False), start=1):
        date = fixture.Date.strftime("%Y-%m-%d")
        try:
            forecast = predict_fixture_models(
                matches, fixture.HomeTeam, fixture.AwayTeam, date,
                lookback_days, prior_matches, half_life_days,
            )
            # Read the result only after the complete forecast is available.
            evaluated = evaluate_fixture(forecast, int(fixture.FTHG), int(fixture.FTAG))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Backtest failed for this match: {fixture.match_id}. {error}") from error

        common = {
            "match_id": forecast["match_id"],
            "Season": evaluation_season,
            "Date": date,
            "HomeTeam": fixture.HomeTeam,
            "AwayTeam": fixture.AwayTeam,
            "history_start": forecast["history_start"],
            "last_history_date": forecast["last_history_date"],
            "history_matches": forecast["history_matches"],
            "home_history_matches": forecast["home_history_matches"],
            "away_history_matches": forecast["away_history_matches"],
        }
        records.extend({**common, **record} for record in evaluated)
        if progress is not None:
            progress(completed, len(fixtures))
    return pd.DataFrame.from_records(records)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Every match has equal weight in the model evaluation."""
    if predictions.empty:
        raise ValueError("Cannot summarise an empty prediction table.")
    if set(predictions["model"]) != set(MODEL_NAMES):
        raise ValueError("All four models are required for the summary.")
    if predictions.duplicated(["match_id", "model"]).any():
        raise ValueError("A match–model pair appears more than once.")
    reference_ids = set(predictions.loc[predictions["model"] == MODEL_NAMES[0], "match_id"])
    summary = []
    for name in MODEL_NAMES:
        rows = predictions.loc[predictions["model"] == name]
        if set(rows["match_id"]) != reference_ids:
            raise ValueError("All models must cover the same matches.")
        record = {"model": name, "matches": len(rows)}
        for market in LOSS_MARKETS:
            loss_column = f"{market}_log_loss"
            if not all(isfinite(value) and value >= 0 for value in rows[loss_column]):
                raise ValueError("Invalid loss value in the prediction table.")
            record[loss_column] = float(rows[loss_column].mean())
            count_column = f"{market}_log_loss_clipped"
            record[count_column] = int(rows[count_column].sum())
        for market in ("one_x_two", *BINARY_MARKETS):
            column = f"{market}_brier"
            record[column] = float(rows[column].mean())
        for market in ("one_x_two", "exact_score"):
            record[f"{market}_accuracy"] = float(rows[f"{market}_hit"].mean())
        record["missing_home_history"] = int((rows["home_history_matches"] == 0).sum())
        record["missing_away_history"] = int((rows["away_history_matches"] == 0).sum())
        record["max_omitted_probability"] = float(rows["omitted_probability"].max())
        summary.append(record)
    return pd.DataFrame.from_records(summary)
