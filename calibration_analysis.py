"""Calibration and grouped error analysis of saved predictions."""

from bisect import bisect_right
import json
from math import isfinite
from pathlib import Path

import pandas as pd


MODEL_NAMES = ("league_average", "raw", "smoothed", "time_weighted")
MODEL_LABELS = {
    "league_average": "League average", "raw": "Raw", "smoothed": "Smoothed",
    "time_weighted": "Time-weighted",
}
EVENTS = {
    "home_win": ("Home win", "p_home", "home"),
    "draw": ("Draw", "p_draw", "draw"),
    "away_win": ("Away win", "p_away", "away"),
    "over_1_5": ("Over 1.5", "p_over_1_5", "actual_over_1_5"),
    "over_2_5": ("Over 2.5", "p_over_2_5", "actual_over_2_5"),
    "over_3_5": ("Over 3.5", "p_over_3_5", "actual_over_3_5"),
    "btts": ("Both teams to score", "p_btts", "actual_btts"),
}
HISTORY_GROUPS = ("missing_history", "limited_history", "enough_history")
HISTORY_LABELS = {
    "missing_history": "At least one history is missing",
    "limited_history": "Limited history",
    "enough_history": "Both histories reach the threshold",
}
SCORE_COLUMNS = (
    "one_x_two_log_loss", "one_x_two_brier", "over_2_5_log_loss",
    "over_2_5_brier", "btts_log_loss", "btts_brier",
)
COUNT_COLUMNS = (
    "FTHG", "FTAG", "history_matches", "home_history_matches", "away_history_matches",
    "one_x_two_hit", "one_x_two_log_loss_clipped", "btts_log_loss_clipped",
)
FIXTURE_COLUMNS = (
    "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
    "history_start", "last_history_date", "history_matches",
    "home_history_matches", "away_history_matches", "actual_one_x_two",
    "actual_over_1_5", "actual_over_2_5", "actual_over_3_5", "actual_btts",
)
REQUIRED_COLUMNS = tuple(dict.fromkeys((
    "match_id", "model", *FIXTURE_COLUMNS, *COUNT_COLUMNS, *SCORE_COLUMNS,
    *(spec[1] for spec in EVENTS.values()),
)))


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate the report and return a copy without modifying the input."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing columns in the backtest report: {', '.join(missing)}.")
    if predictions.empty:
        raise ValueError("The prediction report is empty.")
    result = predictions.loc[:, REQUIRED_COLUMNS].copy()
    if result.isna().any().any():
        raise ValueError("The prediction report contains missing values.")
    for column in ("match_id", "model", "Season", "HomeTeam", "AwayTeam"):
        if not result[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Invalid text field: {column}.")
    if set(result["model"]) != set(MODEL_NAMES):
        raise ValueError("The report must contain all four known models.")
    if result.duplicated(["match_id", "model"]).any():
        raise ValueError("The report contains a duplicate match–model pair.")
    reference = set(result.loc[result["model"] == MODEL_NAMES[0], "match_id"])
    for name in MODEL_NAMES:
        if set(result.loc[result["model"] == name, "match_id"]) != reference:
            raise ValueError("The models' predictions must cover the same matches.")

    for column in ("Date", "history_start", "last_history_date"):
        result[column] = pd.to_datetime(result[column], format="%Y-%m-%d", errors="raise")
    if not ((result["history_start"] <= result["last_history_date"])
            & (result["last_history_date"] < result["Date"])).all():
        raise ValueError("The temporal boundary of the history data is invalid.")

    for column in COUNT_COLUMNS:
        values = pd.to_numeric(result[column], errors="raise")
        if not values.map(lambda value: isfinite(value) and value >= 0 and value % 1 == 0).all():
            raise ValueError(f"A non-negative integer is required: {column}.")
        result[column] = values.astype("int64")
    if (result["history_matches"] <= 0).any():
        raise ValueError("The league history match count must be positive.")
    for column in ("home_history_matches", "away_history_matches"):
        if (result[column] > result["history_matches"]).any():
            raise ValueError("Team history must not exceed league history.")
    for column in ("one_x_two_hit", "one_x_two_log_loss_clipped", "btts_log_loss_clipped"):
        if not result[column].isin((0, 1)).all():
            raise ValueError(f"A zero-or-one indicator is required: {column}.")

    for _, probability_column, _ in EVENTS.values():
        values = pd.to_numeric(result[probability_column], errors="raise")
        if not values.map(lambda value: isfinite(value) and 0 <= value <= 1).all():
            raise ValueError(f"A finite probability between zero and one is required: {probability_column}.")
        result[probability_column] = values
    probability_sum = result[["p_home", "p_draw", "p_away"]].sum(axis=1)
    if ((probability_sum - 1.0).abs() > 1e-12).any():
        raise ValueError("The 1X2 probabilities do not sum to one.")

    actual = pd.Series("draw", index=result.index)
    actual.loc[result["FTHG"] > result["FTAG"]] = "home"
    actual.loc[result["FTHG"] < result["FTAG"]] = "away"
    if not (actual == result["actual_one_x_two"]).all():
        raise ValueError("The 1X2 outcome indicator disagrees with the observed goal counts.")
    for line in (1.5, 2.5, 3.5):
        column = "actual_over_" + str(line).replace(".", "_")
        values = pd.to_numeric(result[column], errors="raise")
        expected = (result["FTHG"] + result["FTAG"] > line).astype(int)
        if not values.isin((0, 1)).all() or not (values == expected).all():
            raise ValueError(f"Invalid goal-line event indicator: {column}.")
        result[column] = values.astype(int)
    btts = ((result["FTHG"] > 0) & (result["FTAG"] > 0)).astype(int)
    values = pd.to_numeric(result["actual_btts"], errors="raise")
    if not values.isin((0, 1)).all() or not (values == btts).all():
        raise ValueError("The BTTS event indicator disagrees with the observed goal counts.")
    result["actual_btts"] = values.astype(int)

    for column in SCORE_COLUMNS:
        values = pd.to_numeric(result[column], errors="raise")
        if not values.map(lambda value: isfinite(value) and value >= 0).all():
            raise ValueError(f"Invalid metric: {column}.")
        upper = 2.0 if column == "one_x_two_brier" else 1.0
        if column.endswith("brier") and (values > upper).any():
            raise ValueError(f"Brier value outside the permitted scale: {column}.")
        result[column] = values

    if (result.groupby("match_id")[list(FIXTURE_COLUMNS)].nunique() > 1).any().any():
        raise ValueError("A match result or its history differs between models in the report.")
    return result


def load_backtest_reports(
    predictions_path: Path, settings_path: Path, evaluation_season: str
) -> tuple[pd.DataFrame, dict]:
    """Load the Step 8 report and settings."""
    for path in (predictions_path, settings_path):
        if not path.is_file():
            raise FileNotFoundError(f"The Step 8 report is missing: {path}. Run run_backtest.py first.")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("evaluation_season") != evaluation_season:
        raise ValueError("The backtest settings file does not belong to the selected season.")
    if set(settings.get("models", [])) != set(MODEL_NAMES):
        raise ValueError("The models in the settings file differ from the four expected models.")
    predictions = validate_predictions(pd.read_csv(predictions_path, encoding="utf-8-sig"))
    if set(predictions["Season"]) != {evaluation_season}:
        raise ValueError("The prediction report also contains data from another season.")
    count = settings.get("evaluated_matches")
    if isinstance(count, bool) or not isinstance(count, int) or count != predictions["match_id"].nunique():
        raise ValueError("The match counts in the settings file and report differ.")
    return predictions, settings


def positive_integer(value: int, label: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label}: an integer of at least {minimum} is required.")


def probability_bin(probability: float, n_bins: int) -> int:
    """Left-closed bins; a probability of exactly 100% goes into the last bin."""
    positive_integer(n_bins, "Number of bins", minimum=2)
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The probability to assign to a bin must be finite and between zero and one.")
    edges = [index / n_bins for index in range(n_bins + 1)]
    return min(bisect_right(edges, probability) - 1, n_bins - 1)


def event_observations(rows: pd.DataFrame, event: str) -> pd.Series:
    if event not in EVENTS:
        raise ValueError("Unknown calibration event.")
    observed = EVENTS[event][2]
    if event in ("home_win", "draw", "away_win"):
        return (rows["actual_one_x_two"] == observed).astype(int)
    return rows[observed].astype(int)


def calibration_bins(
    probabilities: pd.Series, observations: pd.Series, n_bins: int = 10, min_bin_matches: int = 20
) -> pd.DataFrame:
    """Mean probability, observed event rate and sample size in each bin.

    Pair rows by position. Empty bins receive missing means.
    Mark bins with few observations, but retain them in the evaluation.
    """
    positive_integer(n_bins, "Number of bins", minimum=2)
    positive_integer(min_bin_matches, "Few-observation threshold")
    if len(probabilities) == 0 or len(probabilities) != len(observations):
        raise ValueError("Non-empty probability and event sequences of equal length are required.")
    probabilities = pd.to_numeric(probabilities.reset_index(drop=True), errors="raise")
    observations = pd.to_numeric(observations.reset_index(drop=True), errors="raise")
    if not probabilities.map(lambda value: isfinite(value) and 0 <= value <= 1).all():
        raise ValueError("Calibration requires finite probabilities between zero and one.")
    if not observations.isin((0, 1)).all():
        raise ValueError("Calibration events must be zero or one.")
    assigned = probabilities.map(lambda value: probability_bin(value, n_bins))
    records = []
    for index in range(n_bins):
        mask = assigned == index
        count = int(mask.sum())
        predicted = float(probabilities.loc[mask].mean()) if count else float("nan")
        observed = float(observations.loc[mask].mean()) if count else float("nan")
        records.append({
            "bin": index, "lower": index / n_bins, "upper": (index + 1) / n_bins,
            "matches": count, "positives": int(observations.loc[mask].sum()),
            "mean_probability": predicted, "observed_rate": observed,
            "bias": predicted - observed,
            "sparse": 0 < count < min_bin_matches, "empty": count == 0,
        })
    return pd.DataFrame.from_records(records)


def analyze_calibration(
    predictions: pd.DataFrame, n_bins: int = 10, min_bin_matches: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Examine seven binary events, treating each 1X2 class separately."""
    all_bins = []
    summary = []
    for name in MODEL_NAMES:
        rows = predictions.loc[predictions["model"] == name]
        for event, (_, probability_column, _) in EVENTS.items():
            probabilities = rows[probability_column]
            observations = event_observations(rows, event)
            bins = calibration_bins(probabilities, observations, n_bins, min_bin_matches)
            bins.insert(0, "event", event)
            bins.insert(0, "model", name)
            all_bins.append(bins)
            active = bins.loc[~bins["empty"]]
            predicted = float(probabilities.mean())
            observed = float(observations.mean())
            sparse_matches = int(bins.loc[bins["sparse"], "matches"].sum())
            summary.append({
                "model": name, "event": event, "matches": len(rows),
                "mean_probability": predicted, "observed_rate": observed,
                "bias": predicted - observed,
                "ece": float((active["matches"] * active["bias"].abs()).sum() / len(rows)),
                "empty_bins": int(bins["empty"].sum()),
                "sparse_bins": int(bins["sparse"].sum()),
                "matches_in_sparse_bins": sparse_matches,
            })
    return pd.concat(all_bins, ignore_index=True), pd.DataFrame.from_records(summary)


def history_group(home_matches: int, away_matches: int, minimum_matches: int = 5) -> str:
    positive_integer(minimum_matches, "History threshold")
    for count in (home_matches, away_matches):
        positive_integer(count, "Historical matches", minimum=0)
    if home_matches == 0 or away_matches == 0:
        return "missing_history"
    if min(home_matches, away_matches) < minimum_matches:
        return "limited_history"
    return "enough_history"


def history_group_metrics(predictions: pd.DataFrame, minimum_matches: int = 5) -> pd.DataFrame:
    """Mutually exclusive groups; every match belongs to exactly one group."""
    positive_integer(minimum_matches, "History threshold")
    rows = predictions.copy()
    rows["history_group"] = [
        history_group(int(home), int(away), minimum_matches)
        for home, away in zip(rows["home_history_matches"], rows["away_history_matches"])
    ]
    records = []
    for name in MODEL_NAMES:
        for group in HISTORY_GROUPS:
            selected = rows.loc[(rows["model"] == name) & (rows["history_group"] == group)]
            count = len(selected)
            record = {"model": name, "history_group": group, "matches": count}
            record.update({column: float(selected[column].mean()) if count else float("nan")
                           for column in SCORE_COLUMNS})
            record["one_x_two_accuracy"] = float(selected["one_x_two_hit"].mean()) if count else float("nan")
            for market in ("one_x_two", "btts"):
                column = f"{market}_log_loss_clipped"
                record[column] = int(selected[column].sum())
            records.append(record)
    return pd.DataFrame.from_records(records)


def largest_one_x_two_errors(
    predictions: pd.DataFrame, top_n: int = 5, minimum_history_matches: int = 5
) -> pd.DataFrame:
    """Matches with the largest log loss for each model; probabilities remain unchanged."""
    positive_integer(top_n, "Number of errors to display")
    positive_integer(minimum_history_matches, "History threshold")
    columns = [
        "model", "match_id", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "actual_one_x_two", "p_home", "p_draw", "p_away", "one_x_two_log_loss",
        "home_history_matches", "away_history_matches",
    ]
    results = []
    for name in MODEL_NAMES:
        selected = predictions.loc[predictions["model"] == name].sort_values(
            ["one_x_two_log_loss", "Date", "match_id"], ascending=[False, True, True], kind="stable"
        ).head(top_n)[columns].copy()
        selected.insert(0, "rank", range(1, len(selected) + 1))
        selected["history_group"] = [
            history_group(int(home), int(away), minimum_history_matches)
            for home, away in zip(selected["home_history_matches"], selected["away_history_matches"])
        ]
        results.append(selected)
    return pd.concat(results, ignore_index=True)
