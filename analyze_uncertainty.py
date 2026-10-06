"""Step 13: exploratory uncertainty analysis of model differences."""

import json
from pathlib import Path
import platform

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "reports" / "uncertainty"
BLOCK_LENGTHS = (3, 5, 10)
PRIMARY_BLOCK_LENGTH = 5
N_RESAMPLES = 5000
CONFIDENCE_LEVEL = 0.95
BASE_SEED = 42
FOCUS_REFERENCE = "time_weighted"
MARKET_LABELS = {
    "one_x_two": "1X2", "over_1_5": "Over/under 1.5", "over_2_5": "Over/under 2.5",
    "over_3_5": "Over/under 3.5", "btts": "BTTS", "exact_score": "Exact score",
}


def plot_intervals(intervals: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    focus = intervals.loc[
        (intervals.reference_model == FOCUS_REFERENCE)
        & (intervals.block_length_matchdays == PRIMARY_BLOCK_LENGTH)
    ]
    seasons = sorted(focus.Season.unique())
    fig, axes = plt.subplots(1, len(seasons), figsize=(12, 6), squeeze=False)
    for axis, season in zip(axes[0], seasons):
        rows = focus.loc[focus.Season == season].set_index("market").loc[list(MARKET_LABELS)]
        for position, row in enumerate(rows.itertuples()):
            color = "#1b7f79" if row.interval_direction == "new_lower" else "#b45231" if row.interval_direction == "new_higher" else "#50677f"
            axis.hlines(position, row.ci_lower, row.ci_upper, color=color, linewidth=2.4)
            axis.plot([row.ci_lower, row.ci_upper], [position, position], "|", color=color, markersize=9)
            axis.plot(row.mean_loss_difference, position, "o", color=color, markersize=6)
        axis.axvline(0, color="#333333", linestyle="--", linewidth=1)
        axis.set_yticks(range(len(rows)), list(MARKET_LABELS.values()))
        axis.invert_yaxis()
        axis.set_title(f"{season} | {PRIMARY_BLOCK_LENGTH}-matchday blocks")
        axis.set_xlabel("Mean log loss difference: new − time-weighted")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Model differences: point estimate and exploratory 95% bootstrap interval", fontsize=12)
    fig.text(0.5, 0.025, "Negative: favours the new model. Intervals are individual.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    try:
        import numpy as np
        import matplotlib
        from backtest_models import LOSS_MARKETS, MODEL_NAMES
        from uncertainty_analysis import (
            DIRECTIONS, bootstrap_loss_intervals, build_paired_losses, summarize_sensitivity,
        )
        from uncertainty_reports import assert_uncertainty_inputs_unchanged, load_uncertainty_inputs
        from validation_data import file_sha256
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in ("numpy", "matplotlib", "scipy"):
            raise SystemExit("A required dependency is missing. Install requirements.txt and requirements_step_11.txt.") from None
        raise
    print("Step 13: paired model differences and uncertainty", flush=True)
    print(f"Block lengths: {', '.join(map(str, BLOCK_LENGTHS))} matchdays | Resamples: {N_RESAMPLES}")
    print(f"Interval: {CONFIDENCE_LEVEL:.0%} | Method: moving block bootstrap, percentile | Base seed: {BASE_SEED}")
    print("Matches are paired by identifier. All matches on a matchday stay together.")
    print("The predictions saved in Step 12 are used; no models are refitted.", flush=True)
    inputs, context = load_uncertainty_inputs(PROJECT_DIR)
    bundles = {}
    interval_tables = []
    # Random seeds follow a rule fixed before reviewing the results.
    for season_index, (season, predictions) in enumerate(sorted(inputs.items())):
        print(f"\nEvaluation season: {season} | Matches: {predictions.match_id.nunique()}", flush=True)
        paired = build_paired_losses(predictions)
        by_length = []
        for block_length in BLOCK_LENGTHS:
            seed = BASE_SEED + 1000 * season_index + block_length
            intervals = bootstrap_loss_intervals(
                paired, block_length, N_RESAMPLES, CONFIDENCE_LEVEL, seed
            )
            by_length.append(intervals)
            print(f"Completed: {block_length}-matchday block, {N_RESAMPLES} resamples", flush=True)
        intervals = pd.concat(by_length, ignore_index=True)
        sensitivity = summarize_sensitivity(intervals, BLOCK_LENGTHS)
        focus = intervals.loc[
            (intervals.reference_model == FOCUS_REFERENCE)
            & (intervals.block_length_matchdays == PRIMARY_BLOCK_LENGTH)
        ].set_index("market")
        print("\nNew model minus time-weighted model – five-matchday blocks:")
        print(f"{'Task':<20} {'Mean difference':>15} {'95% interval':>24}")
        for market in LOSS_MARKETS:
            row = focus.loc[market]
            print(f"{MARKET_LABELS[market]:<20} {row.mean_loss_difference:>+15.4f} "
                  f"[{row.ci_lower:+.4f}; {row.ci_upper:+.4f}]")
        print("A negative difference favours the new model.")
        print("\nCombined indication across the three block lengths – time-weighted reference:")
        for row in sensitivity.loc[sensitivity.reference_model == FOCUS_REFERENCE].to_dict("records"):
            label = DIRECTIONS.get(row["sensitivity_direction"], "the indication depends on the block length")
            print(f"{MARKET_LABELS[row['market']]}: {label}")
        bundles[season] = {"paired": paired, "intervals": intervals, "sensitivity": sensitivity}
        interval_tables.append(intervals)
    assert_uncertainty_inputs_unchanged(PROJECT_DIR, context)
    combined = pd.concat(interval_tables, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for season, bundle in bundles.items():
        slug = season.replace("/", "_")
        for name, key in (("paired_match_losses", "paired"), ("bootstrap_intervals", "intervals"),
                          ("block_sensitivity", "sensitivity")):
            bundle[key].to_csv(
                OUTPUT_DIR / f"{name}_{slug}.csv", index=False, date_format="%Y-%m-%d", encoding="utf-8-sig",
            )
    combined.to_csv(OUTPUT_DIR / "bootstrap_intervals_2023_24_2024_25.csv", index=False, encoding="utf-8-sig")
    plot_intervals(combined, OUTPUT_DIR / "uncertainty_time_weighted_2023_24_2024_25.png")
    settings = {
        "step": 13, "evaluation_role": "exploratory_development_diagnostics",
        "evaluation_seasons": sorted(inputs), "new_model": "fitted_poisson",
        "reference_models": list(MODEL_NAMES), "markets": list(LOSS_MARKETS),
        "evaluated_matches": sum(table.match_id.nunique() for table in inputs.values()),
        "block_lengths_matchdays": list(BLOCK_LENGTHS), "primary_block_length_matchdays": PRIMARY_BLOCK_LENGTH,
        "n_resamples_per_season_and_length": N_RESAMPLES, "confidence_level": CONFIDENCE_LEVEL,
        "base_seed": BASE_SEED, "seed_rule": "base_seed + 1000 * sorted_season_index + block_length",
        "bootstrap_method": "non_circular_overlapping_moving_blocks_of_matchdays",
        "interval_method": "percentile", "quantile_method": "linear",
        "pairing": "match_id", "difference_sign": "new_model_minus_reference",
        "resampling_unit": "complete_matchday", "statistic_weighting": "equal_weight_per_resampled_match",
        "resampled_matchday_count": "same_as_original_season",
        "resampled_match_count": "varies_with_number_of_matches_on_selected_days",
        "last_sampled_block_truncated": True, "blocks_cross_season_boundary": False,
        "same_resample_indices_for_all_comparisons": True,
        "intervals_are_simultaneous": False, "multiple_comparison_adjustment": False,
        "bootstrap_negative_fraction_is_reported_as_probability": False,
        "models_refitted_in_bootstrap": False, "model_parameters_changed": False,
        "conditioning": "fixed_saved_forecasts; excludes model_selection_and_refitting_uncertainty",
        "assumption": "approximately stable loss-difference process with short-range dependence",
        "final_test_season_reserved": "2025/26", "source_context": context,
        "source_sha256": {name: file_sha256(PROJECT_DIR / name) for name in (
            "uncertainty_analysis.py", "uncertainty_reports.py", "analyze_uncertainty.py",
        )},
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "method_sources": [
            "https://stat.ethz.ch/R-manual/R-devel/library/boot/html/tsboot.html",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
            "https://doi.org/10.1214/aos/1176347265",
        ],
    }
    (OUTPUT_DIR / "uncertainty_settings_2023_24_2024_25.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"\nTotal interval rows: {len(combined)} | New reports: {OUTPUT_DIR}")
    print("PNG figure: uncertainty_time_weighted_2023_24_2024_25.png")
    print("\nThis is exploratory analysis. The 95% intervals are individual; there is no multiple-comparison adjustment.")
    print("The calculation examines loss differences for fixed forecasts; it excludes model-selection uncertainty.")
    print("The 2025/26 season remains reserved for the separate final test.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, pd.errors.ParserError) as error:
        raise SystemExit(f"Uncertainty analysis error: {error}")
