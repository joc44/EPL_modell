"""Data validation for the later season and validation of the development report."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import pandas as pd

from analyze_match import load_clean_matches
from backtest_models import MODEL_NAMES, summarize_predictions
from calibration_analysis import validate_predictions
from main import COLUMNS, clean_matches, load_matches


BASE_SEASONS = ("2021/22", "2022/23", "2023/24")
DEVELOPMENT_SEASON = "2023/24"
VALIDATION_SEASON = "2024/25"
EXPECTED_MATCHES = 380

# Settings for the 2023/24 development run. These values were fixed before
# reviewing the next season's results; they are not tuned here.
LOOKBACK_DAYS = 730
PRIOR_MATCHES = 5.0
HALF_LIFE_DAYS = 180.0
LOG_LOSS_FLOOR = 1e-15
N_BINS = 10
MIN_BIN_MATCHES = 20


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def expected_match_ids(matches: pd.DataFrame) -> pd.Series:
    return (
        "EPL|" + matches["Season"] + "|" + matches["Date"].dt.strftime("%Y-%m-%d")
        + "|" + matches["HomeTeam"] + "|" + matches["AwayTeam"]
    )


def validate_full_season(matches: pd.DataFrame, season: str) -> None:
    """Require a complete twenty-team EPL season with home-and-away fixtures."""
    rows = matches.loc[matches["Season"] == season]
    if len(rows) != EXPECTED_MATCHES:
        raise ValueError(f"{season}: 380 matches expected; found: {len(rows)}.")
    if rows[["match_id", "Season", *COLUMNS]].isna().any().any():
        raise ValueError(f"Missing required data: {season}.")
    if rows["match_id"].duplicated().any() or not rows["match_id"].eq(expected_match_ids(rows)).all():
        raise ValueError(f"Invalid or duplicate match identifier: {season}.")
    for column in ("HomeTeam", "AwayTeam"):
        if not rows[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Empty or invalid team name: {season}.")
    if rows["HomeTeam"].eq(rows["AwayTeam"]).any():
        raise ValueError(f"A team cannot play against itself: {season}.")
    for column in ("FTHG", "FTAG"):
        if not rows[column].map(lambda value: isfinite(value) and value >= 0 and value % 1 == 0).all():
            raise ValueError(f"A non-negative integer goal count is required: {season}, {column}.")

    start_year = int(season[:4])
    start, end = pd.Timestamp(f"{start_year}-07-01"), pd.Timestamp(f"{start_year + 1}-07-01")
    if not ((rows["Date"] >= start) & (rows["Date"] < end)).all():
        raise ValueError(f"The date does not belong to the specified season: {season}.")
    teams = set(rows["HomeTeam"]) | set(rows["AwayTeam"])
    if len(teams) != 20:
        raise ValueError(f"{season}: 20 teams expected; found: {len(teams)}.")
    expected_pairs = {(home, away) for home in teams for away in teams if home != away}
    pairs = list(zip(rows["HomeTeam"], rows["AwayTeam"]))
    if len(set(pairs)) != EXPECTED_MATCHES or set(pairs) != expected_pairs:
        raise ValueError(f"Missing or duplicate home–away pairing: {season}.")


def assert_base_preserved(base: pd.DataFrame, extended: pd.DataFrame) -> None:
    """Preserve every field from the previous three seasons."""
    if set(base["Season"]) != set(BASE_SEASONS):
        raise ValueError("The base file must contain the 2021/22, 2022/23 and 2023/24 seasons.")
    for season in BASE_SEASONS:
        validate_full_season(base, season)
    previous = extended.loc[extended["Season"].isin(BASE_SEASONS)]
    columns = ["match_id", "Season", *COLUMNS]
    left = base[columns].sort_values("match_id").reset_index(drop=True)
    right = previous[columns].sort_values("match_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("The extended data file modifies data from the previous three seasons.") from error


def extend_matches(base: pd.DataFrame, raw_path: Path) -> tuple[pd.DataFrame, int]:
    """Clean only the 2024/25 season and retain the previous data."""
    new_matches, removed = clean_matches(load_matches(raw_path), VALIDATION_SEASON)
    new_matches["match_id"] = expected_match_ids(new_matches)
    validate_full_season(new_matches, VALIDATION_SEASON)
    result = pd.concat([base, new_matches[base.columns]], ignore_index=True)
    result = result.sort_values("Date", kind="stable").reset_index(drop=True)
    assert_base_preserved(base, result)
    validate_extended_matches(result)
    return result, removed


def validate_extended_matches(matches: pd.DataFrame) -> None:
    if set(matches["Season"]) != {*BASE_SEASONS, VALIDATION_SEASON}:
        raise ValueError("The extended data file must contain exactly the 2021/22–2024/25 seasons.")
    for season in (*BASE_SEASONS, VALIDATION_SEASON):
        validate_full_season(matches, season)


def check_frozen_settings(settings: dict) -> None:
    """Prevent unnoticed comparisons made from different development settings."""
    expected = {
        "evaluation_season": DEVELOPMENT_SEASON,
        "evaluation_role": "development_backtest",
        "models": list(MODEL_NAMES),
        "evaluated_matches": EXPECTED_MATCHES,
        "lookback_days": LOOKBACK_DAYS,
        "prior_matches": PRIOR_MATCHES,
        "half_life_days": HALF_LIFE_DAYS,
        "history_rule": "match_date - lookback_days <= Date < match_date",
        "same_day_results_used": False,
        "evaluation_weighting": "equal_weight_per_match",
        "log_loss_probability_floor": LOG_LOSS_FLOOR,
        "one_x_two_brier_scale": "0_to_2",
        "binary_brier_scale": "0_to_1",
        "data_file": "matches_clean.csv",
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"A development setting differs: {key}. Expected: {value!r}; received: {settings.get(key)!r}.")
    if settings["same_day_results_used"] is not False:
        raise ValueError("Same-day results must be disabled.")


def load_development_checkpoint(
    base_path: Path, reports_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate the Step 8 data, detailed report and summary."""
    slug = DEVELOPMENT_SEASON.replace("/", "_")
    predictions_path = reports_dir / f"backtest_predictions_{slug}.csv"
    summary_path = reports_dir / f"backtest_summary_{slug}.csv"
    settings_path = reports_dir / f"backtest_settings_{slug}.json"
    for path in (base_path, predictions_path, summary_path, settings_path):
        if not path.is_file():
            raise FileNotFoundError(f"A previous file is missing: {path}. Results from Steps 2 and 8 are required.")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("The development settings file is unusable.")
    check_frozen_settings(settings)
    if settings.get("data_sha256") != file_sha256(base_path):
        raise ValueError("matches_clean.csv differs from the file used in Step 8. Restore that data file.")

    base = load_clean_matches(base_path)
    assert_base_preserved(base, base)
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
    checked = validate_predictions(predictions)
    if set(checked["Season"]) != {DEVELOPMENT_SEASON}:
        raise ValueError("The development report must contain only the 2023/24 season.")
    expected = base.loc[base["Season"] == DEVELOPMENT_SEASON]
    # A count of 380 is insufficient: fixtures and results must also match.
    for name in MODEL_NAMES:
        recorded = checked.loc[checked["model"] == name]
        columns = ["match_id", "Season", *COLUMNS]
        try:
            pd.testing.assert_frame_equal(
                expected[columns].sort_values("match_id").reset_index(drop=True),
                recorded[columns].sort_values("match_id").reset_index(drop=True),
                check_dtype=False, check_exact=True,
            )
        except AssertionError as error:
            raise ValueError("The development report's match fixtures differ from the base data file.") from error

    calculated = summarize_predictions(predictions).set_index("model").sort_index()
    saved = pd.read_csv(summary_path, encoding="utf-8-sig").set_index("model").sort_index()
    try:
        pd.testing.assert_frame_equal(calculated, saved, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as error:
        raise ValueError("The saved development summary does not match the detailed prediction summary.") from error
    return checked, calculated.reset_index(), settings
