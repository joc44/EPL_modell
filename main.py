from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
MATCH_KEY = ["Season", "Date", "HomeTeam", "AwayTeam"]

SEASON_FILES = {
    "2021/22": "E0_2021_2022.csv",
    "2022/23": "E0_2022_2023.csv",
    "2023/24": "E0_2023_2024.csv",
}

# Explicit name mappings; all other team names remain unchanged.
TEAM_ALIASES = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
}


def load_matches(csv_path: Path) -> pd.DataFrame:
    """Load one season's raw CSV with basic validation."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    matches = pd.read_csv(
        csv_path,
        usecols=COLUMNS,
        encoding="utf-8-sig",
        dtype={"FTHG": "Int64", "FTAG": "Int64"},
    )[COLUMNS]

    if matches.empty:
        raise ValueError(f"No matches in this file: {csv_path.name}")
    if matches.isna().any().any():
        raise ValueError(f"Missing data in required columns: {csv_path.name}")
    if (matches[["FTHG", "FTAG"]] < 0).any().any():
        raise ValueError(f"A negative goal count was found: {csv_path.name}")

    return matches


def clean_matches(matches: pd.DataFrame, season: str) -> tuple[pd.DataFrame, int]:
    """Standardise a season and also return the count of removed duplicates."""
    matches = matches.copy()
    matches["Season"] = season
    matches["Date"] = pd.to_datetime(
        matches["Date"].astype("string").str.strip(),
        format="%d/%m/%Y",
        errors="raise",
    )

    # Validate dates for the three selected completed seasons.
    start_year = int(season[:4])
    start = pd.Timestamp(f"{start_year}-07-01")
    end = pd.Timestamp(f"{start_year + 1}-07-01")
    in_season = (matches["Date"] >= start) & (matches["Date"] < end)
    if not in_season.all():
        raise ValueError(f"Some dates do not belong to season {season}.")

    for column in ["HomeTeam", "AwayTeam"]:
        matches[column] = matches[column].astype("string").str.strip()
        matches[column] = matches[column].replace(TEAM_ALIASES)
        if matches[column].eq("").any():
            raise ValueError(f"Empty team name in season {season}.")
    if matches["HomeTeam"].eq(matches["AwayTeam"]).any():
        raise ValueError(f"A team cannot play against itself: {season}")

    original_count = len(matches)
    matches = matches.drop_duplicates().copy()
    if matches.duplicated(subset=MATCH_KEY).any():
        raise ValueError(f"Conflicting final scores for the same match: {season}")
    removed_count = original_count - len(matches)

    return matches, removed_count


def load_seasons() -> tuple[pd.DataFrame, int]:
    """Load, clean and combine the three seasons chronologically."""
    season_tables = []
    total_removed = 0

    for season, filename in SEASON_FILES.items():
        raw_matches = load_matches(RAW_DIR / filename)
        clean_table, removed = clean_matches(raw_matches, season)
        season_tables.append(clean_table)
        total_removed += removed

    matches = pd.concat(season_tables, ignore_index=True)
    matches = matches.sort_values("Date", kind="stable").reset_index(drop=True)
    matches["match_id"] = (
        "EPL|" + matches["Season"] + "|" + matches["Date"].dt.strftime("%Y-%m-%d")
        + "|" + matches["HomeTeam"] + "|" + matches["AwayTeam"]
    )

    return matches[["match_id", "Season"] + COLUMNS], total_removed


def main() -> None:
    matches, removed_count = load_seasons()
    teams = pd.concat([matches["HomeTeam"], matches["AwayTeam"]])
    total_goals = matches["FTHG"] + matches["FTAG"]

    print("Step 2: combine and clean three Premier League seasons")
    print("\nFirst ten matches in chronological order:")
    print(matches[["Season"] + COLUMNS].head(10).to_string(index=False))
    print("\nMatches per season:")
    print(matches.groupby("Season").size().to_string())

    print(f"\nTotal matches: {len(matches)}")
    print(f"Distinct teams across the three seasons: {teams.nunique()}")
    print(f"Identical duplicates removed: {removed_count}")
    print(f"Total goals: {total_goals.sum()}")
    print(f"Total home goals: {matches['FTHG'].sum()}")
    print(f"Total away goals: {matches['FTAG'].sum()}")
    print(f"Mean home goals: {matches['FTHG'].mean():.3f}")
    print(f"Mean away goals: {matches['FTAG'].mean():.3f}")
    print(f"Mean total goals per match: {total_goals.mean():.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    print(f"\nCleaned data file saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Data-processing error: {error}")
