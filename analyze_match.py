from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
COLUMNS = ["match_id", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

# Historical example fixture. Team names must match the names in the cleaned CSV.
MATCH_DATE = "2023-08-12"
HOME_TEAM = "Arsenal"
AWAY_TEAM = "Nottingham Forest"
LOOKBACK_DAYS = 730


def load_clean_matches(csv_path: Path) -> pd.DataFrame:
    """Load the data file produced in Step 2."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Cleaned data file not found: {csv_path}. "
            "Run the main.py program from Step 2 first."
        )

    matches = pd.read_csv(
        csv_path,
        usecols=COLUMNS,
        encoding="utf-8-sig",
        dtype={"FTHG": "Int64", "FTAG": "Int64"},
    )[COLUMNS]
    if matches.empty or matches.isna().any().any():
        raise ValueError("The cleaned data file is empty or incomplete.")
    matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d", errors="raise")
    if (matches[["FTHG", "FTAG"]] < 0).any().any():
        raise ValueError("The data file contains a negative goal count.")
    key = ["Season", "Date", "HomeTeam", "AwayTeam"]
    if matches["match_id"].duplicated().any() or matches.duplicated(subset=key).any():
        raise ValueError("The cleaned data file contains a duplicate match.")

    return matches


def history_before(matches: pd.DataFrame, match_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    """Select matches from the history window strictly before the target date."""
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days <= 0:
        raise ValueError("The history-window length must be a positive integer number of days.")

    match_day = pd.Timestamp(match_date).normalize()
    start_date = match_day - pd.Timedelta(days=lookback_days)
    mask = (matches["Date"] >= start_date) & (matches["Date"] < match_day)
    return matches.loc[mask].sort_values("Date", kind="stable").copy()


def league_stats(history: pd.DataFrame) -> dict:
    """Calculate league averages from the date-filtered matches."""
    if history.empty:
        raise ValueError("No previous matches in the selected history window.")
    return {
        "matches": len(history),
        "home_goal_mean": float(history["FTHG"].mean()),
        "away_goal_mean": float(history["FTAG"].mean()),
    }


def team_stats(history: pd.DataFrame, team: str, venue: str) -> dict:
    """Return home or away goal statistics for a team."""
    venue_columns = {
        "home": ("HomeTeam", "FTHG", "FTAG"),
        "away": ("AwayTeam", "FTAG", "FTHG"),
    }
    if venue not in venue_columns:
        raise ValueError("The venue must be home or away.")
    team_column, goals_for_column, goals_against_column = venue_columns[venue]
    selected = history.loc[history[team_column] == team]
    count = len(selected)

    return {
        "matches": count,
        "goals_for_total": int(selected[goals_for_column].sum()),
        "goals_against_total": int(selected[goals_against_column].sum()),
        "goals_for_mean": float(selected[goals_for_column].mean()) if count else None,
        "goals_against_mean": float(selected[goals_against_column].mean()) if count else None,
    }


def analyze_fixture(matches: pd.DataFrame, home_team: str, away_team: str, match_date: str, lookback_days: int) -> dict:
    """Build the statistical state before a historical fixture."""
    target_date = pd.to_datetime(match_date, format="%Y-%m-%d", errors="raise")
    fixture_mask = (
        (matches["Date"] == target_date)
        & (matches["HomeTeam"] == home_team)
        & (matches["AwayTeam"] == away_team)
    )
    fixture = matches.loc[fixture_mask, ["match_id"]]
    if len(fixture) != 1:
        raise ValueError("The example fixture cannot be uniquely identified. Check the date and team names.")

    history = history_before(matches, target_date, lookback_days)
    league = league_stats(history)
    return {
        "match_id": fixture.iloc[0]["match_id"],
        "match_date": target_date,
        "start_date": target_date - pd.Timedelta(days=lookback_days),
        "last_history_date": history["Date"].max(),
        "league": league,
        "home": team_stats(history, home_team, "home"),
        "away": team_stats(history, away_team, "away"),
    }


def main() -> None:
    matches = load_clean_matches(DATA_PATH)
    analysis = analyze_fixture(matches, HOME_TEAM, AWAY_TEAM, MATCH_DATE, LOOKBACK_DAYS)
    league = analysis["league"]

    print("Step 3: statistics before the selected match")
    print(f"Match: {HOME_TEAM} – {AWAY_TEAM}, {MATCH_DATE}")
    print(f"History-window start: {analysis['start_date'].date()}")
    print(f"Exclusive upper date boundary: {analysis['match_date'].date()}")
    print(f"Latest matchday used: {analysis['last_history_date'].date()}")
    print(f"\nPrevious matches: {league['matches']}")
    print(f"League mean home goals: {league['home_goal_mean']:.3f}")
    print(f"League mean away goals: {league['away_goal_mean']:.3f}")

    for team, venue, stats in [
        (HOME_TEAM, "home", analysis["home"]),
        (AWAY_TEAM, "away", analysis["away"]),
    ]:
        print(f"\n{team} – {venue} matches:")
        print(f"Number of matches: {stats['matches']}")
        if stats["matches"] == 0:
            print("No matching history in the window; goal averages cannot be estimated from this sample.")
        else:
            print(f"Total goals scored: {stats['goals_for_total']}")
            print(f"Total goals conceded: {stats['goals_against_total']}")
            print(f"Mean goals scored: {stats['goals_for_mean']:.3f}")
            print(f"Mean goals conceded: {stats['goals_against_mean']:.3f}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Analysis error: {error}")
