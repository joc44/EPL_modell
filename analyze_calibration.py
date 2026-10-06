"""Step 9: diagnostics for the backtest predictions."""

from hashlib import sha256
import json
from pathlib import Path
import platform

import pandas as pd

from calibration_analysis import (
    EVENTS, HISTORY_GROUPS, HISTORY_LABELS, MODEL_LABELS, MODEL_NAMES,
    analyze_calibration, history_group_metrics, largest_one_x_two_errors, load_backtest_reports,
)
from calibration_plots import matplotlib, plot_calibration


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATION_SEASON = "2023/24"
REPORTS_DIR = PROJECT_DIR / "data" / "reports"
OUTPUT_DIR = REPORTS_DIR / "diagnostics"
N_BINS = 10
MIN_BIN_MATCHES = 20
MIN_HISTORY_MATCHES = 5
TOP_ERRORS = 5
FOCUS_MODEL = "time_weighted"
CONSOLE_EVENTS = ("home_win", "over_2_5", "btts")


def print_calibration_summary(summary: pd.DataFrame) -> None:
    for event in CONSOLE_EVENTS:
        print(f"\n{EVENTS[event][0]} – season averages and bin-level deviation:")
        print(f"{'Model':<22} {'Mean forecast':>14} {'Observed':>11} {'Difference':>13} {'ECE':>11}")
        for name in MODEL_NAMES:
            row = summary.loc[(summary["event"] == event) & (summary["model"] == name)].iloc[0]
            print(f"{MODEL_LABELS[name]:<22} {row['mean_probability']:>13.2%} {row['observed_rate']:>10.2%} "
                  f"{100 * row['bias']:>+9.2f} pp {100 * row['ece']:>7.2f} pp")
    print("Difference = mean forecast − observed rate; pp = percentage points.")
    print("ECE = match-count-weighted absolute bin-level deviation; a descriptive measure.")


def print_focus_bins(bins: pd.DataFrame) -> None:
    print(f"\nBTTS bins – {MODEL_LABELS[FOCUS_MODEL]}:")
    print(f"{'Probability bin':<14} {'Matches':>6} {'Mean forecast':>14} {'Observed':>11} {'Note':>13}")
    selected = bins.loc[(bins["model"] == FOCUS_MODEL) & (bins["event"] == "btts")]
    for row in selected.to_dict("records"):
        label = f"{100 * row['lower']:.0f}–{100 * row['upper']:.0f}%"
        if row["empty"]:
            print(f"{label:<14} {row['matches']:>6} {'—':>14} {'—':>11} {'empty bin':>13}")
        else:
            note = "few observations" if row["sparse"] else ""
            print(f"{label:<14} {row['matches']:>6} {row['mean_probability']:>13.2%} "
                  f"{row['observed_rate']:>10.2%} {note:>13}")


def print_history_summary(summary: pd.DataFrame) -> None:
    print(f"\nHistory groups – threshold: {MIN_HISTORY_MATCHES} previous venue-specific matches per team")
    for group in HISTORY_GROUPS:
        rows = summary.loc[summary["history_group"] == group]
        print(f"\n{HISTORY_LABELS[group]}:")
        print(f"{'Model':<22} {'Matches':>6} {'1X2 log loss':>14} {'BTTS Brier':>13}")
        for row in rows.to_dict("records"):
            if row["matches"]:
                print(f"{MODEL_LABELS[row['model']]:<22} {row['matches']:>6} "
                      f"{row['one_x_two_log_loss']:>14.4f} {row['btts_brier']:>13.4f}")
            else:
                print(f"{MODEL_LABELS[row['model']]:<22} {0:>6} {'—':>14} {'—':>13}")


def print_largest_errors(errors: pd.DataFrame) -> None:
    print(f"\nLargest 1X2 losses – {MODEL_LABELS[FOCUS_MODEL]}:")
    for row in errors.loc[errors["model"] == FOCUS_MODEL].to_dict("records"):
        actual_column = {"home": "p_home", "draw": "p_draw", "away": "p_away"}[row["actual_one_x_two"]]
        print(f"{row['rank']}. {row['Date'].date()}: {row['HomeTeam']} – {row['AwayTeam']} "
              f"{row['FTHG']}–{row['FTAG']} | probability of the observed outcome: {row[actual_column]:.2%} "
              f"| log loss: {row['one_x_two_log_loss']:.4f}")


def main() -> None:
    if FOCUS_MODEL not in MODEL_NAMES:
        raise ValueError("The highlighted model name is unknown.")
    slug = EVALUATION_SEASON.replace("/", "_")
    predictions_path = REPORTS_DIR / f"backtest_predictions_{slug}.csv"
    settings_path = REPORTS_DIR / f"backtest_settings_{slug}.json"
    predictions, settings = load_backtest_reports(predictions_path, settings_path, EVALUATION_SEASON)
    print("Step 9: calibration and error analysis")
    print(f"Evaluation season: {EVALUATION_SEASON} | Matches: {predictions['match_id'].nunique()}")
    print(f"Models: {len(MODEL_NAMES)} | Events examined: {len(EVENTS)}")
    print(f"Bins: {N_BINS} | Few-observation threshold: fewer than {MIN_BIN_MATCHES} matches")
    print("The analysis uses the probabilities saved in Step 8.")
    bins, summary = analyze_calibration(predictions, N_BINS, MIN_BIN_MATCHES)
    history = history_group_metrics(predictions, MIN_HISTORY_MATCHES)
    errors = largest_one_x_two_errors(predictions, TOP_ERRORS, MIN_HISTORY_MATCHES)
    print_calibration_summary(summary)
    print_focus_bins(bins)
    print_history_summary(history)
    print_largest_errors(errors)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"calibration_bins_{slug}.csv": bins,
        f"calibration_summary_{slug}.csv": summary,
        f"history_group_metrics_{slug}.csv": history,
        f"largest_1x2_errors_{slug}.csv": errors,
    }
    for name, frame in outputs.items():
        export = frame.copy()
        if "Date" in export.columns:
            export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
        export.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")
    plot_calibration(bins, ("home_win", "draw", "away_win"),
                     OUTPUT_DIR / f"calibration_1x2_{slug}.png", "Calibration of the 1X2 outcomes",
                     EVALUATION_SEASON, MIN_BIN_MATCHES, columns=3)
    plot_calibration(bins, ("over_1_5", "over_2_5", "over_3_5", "btts"),
                     OUTPUT_DIR / f"calibration_goals_{slug}.png", "Calibration of goal markets",
                     EVALUATION_SEASON, MIN_BIN_MATCHES, columns=2)
    analysis_settings = {
        "step": 9, "evaluation_season": EVALUATION_SEASON,
        "analysis_role": "development_diagnostics", "n_bins": N_BINS,
        "bin_rule": "left_closed_right_open_except_final_bin_includes_1",
        "minimum_bin_matches_for_filled_marker": MIN_BIN_MATCHES,
        "small_bins_included_in_ece": True,
        "minimum_history_matches": MIN_HISTORY_MATCHES,
        "top_errors_per_model": TOP_ERRORS, "console_focus_model": FOCUS_MODEL,
        "bias_definition": "mean_probability_minus_observed_rate",
        "ece_definition": "sum(bin_matches * abs(bin_bias)) / total_matches",
        "predictions_file": predictions_path.name,
        "predictions_sha256": sha256(predictions_path.read_bytes()).hexdigest(),
        "backtest_settings": settings,
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    (OUTPUT_DIR / f"calibration_settings_{slug}.json").write_text(
        json.dumps(analysis_settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nTables, figures and settings saved in: {OUTPUT_DIR}")
    print("PNG figures: calibration_1x2_" + slug + ".png and calibration_goals_" + slug + ".png")
    print("\nThese are diagnostics on a development season. Deviations in small groups are uncertain.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Calibration analysis error: {error}")
