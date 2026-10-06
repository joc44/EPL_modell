"""Chronological final test of the protocol's three models on the same fixtures."""

from collections.abc import Callable
from math import isclose, isfinite

import pandas as pd

from analyze_match import analyze_fixture, history_before
from backtest_models import GOAL_LINES
from final_test_protocol import validate_protocol_payload
from fitted_backtest import evaluate_forecast, run_fitted_backtest, validate_comparison
from score_probabilities import build_score_matrix, market_probabilities
from time_weighted_model import calculate_time_weighted_expected_goals


FINAL_MODELS = ("league_average", "time_weighted", "fitted_poisson")
REFERENCES = FINAL_MODELS[:2]
LABELS = {"league_average": "League average", "time_weighted": "Time-weighted", "fitted_poisson": "Jointly fitted"}


def predict_references(matches: pd.DataFrame, home: str, away: str, date: str, payload: dict) -> tuple[dict, dict]:
    parameters = payload["reference_parameters"]
    analysis = analyze_fixture(matches, home, away, date, parameters["league_average"]["lookback_days"])
    league = analysis["league"]
    weighted = parameters["time_weighted"]
    lambdas = {
        "league_average": {"lambda_home": league["home_goal_mean"], "lambda_away": league["away_goal_mean"]},
        "time_weighted": calculate_time_weighted_expected_goals(
            matches, home, away, date, weighted["lookback_days"], weighted["half_life_days"], weighted["prior_matches"]
        ),
    }
    forecasts = {}
    for name in REFERENCES:
        goals = lambdas[name]
        grid = build_score_matrix(goals["lambda_home"], goals["lambda_away"])
        forecasts[name] = {**goals, "omitted_probability": grid["omitted_probability"],
                           "markets": market_probabilities(grid["matrix"], GOAL_LINES, top_n=1)}
    common = {"match_id": analysis["match_id"], "Date": date, "HomeTeam": home, "AwayTeam": away,
              "history_start": analysis["start_date"].strftime("%Y-%m-%d"),
              "last_history_date": analysis["last_history_date"].strftime("%Y-%m-%d"),
              "history_matches": league["matches"], "home_history_matches": analysis["home"]["matches"],
              "away_history_matches": analysis["away"]["matches"]}
    return common, forecasts


def run_final_backtest(matches: pd.DataFrame, payload: dict, progress: Callable | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_protocol_payload(payload)
    season = payload["final_test_season"]
    fixtures = matches.loc[matches.Season == season].sort_values(["Date", "match_id"])
    if fixtures.empty or fixtures.match_id.duplicated().any():
        raise ValueError("A unique, non-empty final-test match list is required.")
    records = []
    for date, day in fixtures.groupby("Date", sort=True):
        forecasts = []
        for fixture in day[["match_id", "HomeTeam", "AwayTeam"]].itertuples(index=False):
            common, predictions = predict_references(matches, fixture.HomeTeam, fixture.AwayTeam, date.strftime("%Y-%m-%d"), payload)
            forecasts.append((fixture.match_id, common, predictions))
        actuals = day.set_index("match_id")
        for match_id, common, predictions in forecasts:
            actual = actuals.loc[match_id]
            for name in REFERENCES:
                records.append({**common, "model": name, "Season": season,
                                **evaluate_forecast(predictions[name], int(actual.FTHG), int(actual.FTAG))})
    parameters = payload["selected_parameters"]
    fitted, diagnostics = run_fitted_backtest(
        matches, season, parameters["lookback_days"], parameters["half_life_days"], parameters["ridge_strength"], progress
    )
    combined = validate_final_predictions(pd.concat([pd.DataFrame(records), fitted], ignore_index=True), fixtures)
    validate_final_diagnostics(diagnostics, matches, payload)
    return combined, diagnostics


def validate_final_predictions(predictions: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    result = validate_comparison(predictions, fixtures, FINAL_MODELS)
    for row in result.to_dict("records"):
        grid = build_score_matrix(row["lambda_home"], row["lambda_away"])
        probabilities = market_probabilities(grid["matrix"], GOAL_LINES, top_n=1)
        expected = {"p_home": probabilities["one_x_two"]["home"], "p_draw": probabilities["one_x_two"]["draw"],
                    "p_away": probabilities["one_x_two"]["away"], "p_btts": probabilities["btts"]["yes"],
                    "omitted_probability": grid["omitted_probability"],
                    **{f"p_over_{str(line).replace('.', '_')}": probabilities["over_under"][line]["over"] for line in GOAL_LINES}}
        if any(not isclose(row[key], value, rel_tol=1e-10, abs_tol=1e-12) for key, value in expected.items()):
            raise ValueError("A saved event probability differs from the Poisson goal distribution.")
        top = probabilities["top_scores"][0]
        if (row["top_score_home"], row["top_score_away"]) != (top["home_goals"], top["away_goals"]):
            raise ValueError("Invalid most-probable exact score.")
    return result


def validate_final_diagnostics(diagnostics: pd.DataFrame, matches: pd.DataFrame, payload: dict) -> None:
    required = {"Season", "fit_date", "fixtures", "history_start", "last_history_date", "history_matches",
                "success", "gradient_max_abs", "initial_penalized_nll", "final_penalized_nll"}
    if diagnostics.empty or not required.issubset(diagnostics) or diagnostics[list(required)].isna().any().any():
        raise ValueError("Incomplete final fitting diagnostics.")
    dates = pd.to_datetime(diagnostics.fit_date, format="%Y-%m-%d")
    fixtures = matches.loc[matches.Season == payload["final_test_season"]]
    if dates.duplicated().any() or set(dates) != set(fixtures.Date) or set(diagnostics.Season) != {payload["final_test_season"]}:
        raise ValueError("The matchday fit list differs from the final-test matchdays.")
    for row, date in zip(diagnostics.itertuples(index=False), dates):
        history = history_before(matches, date, payload["selected_parameters"]["lookback_days"])
        if (row.success != True or not all(isfinite(value) for value in (row.gradient_max_abs, row.initial_penalized_nll, row.final_penalized_nll))
                or not 0 <= row.gradient_max_abs <= payload["fit_policy"]["required_gradient_max_abs"]
                or row.final_penalized_nll > row.initial_penalized_nll + 1e-8):
            raise ValueError("Unsuccessful or insufficiently precise final matchday fit.")
        if (row.history_matches != len(history) or pd.Timestamp(row.last_history_date) != history.Date.max()
                or pd.Timestamp(row.history_start) != date - pd.Timedelta(days=payload["selected_parameters"]["lookback_days"])
                or row.fixtures != int(fixtures.Date.eq(date).sum())):
            raise ValueError("The final fit's history data disagree with the chronological rules.")
