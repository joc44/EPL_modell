"""Paired final-test evaluation, uncertainty and complete result reporting."""

import numpy as np
import pandas as pd

from backtest_models import LOSS_MARKETS
from calibration_analysis import EVENTS, calibration_bins, event_observations
from final_test_backtest import FINAL_MODELS, LABELS, REFERENCES
from fitted_backtest import summarize_models
from uncertainty_analysis import MATCH_FIELDS, resample_daily_means, summarize_sensitivity


MARKET_LABELS = {"one_x_two": "1X2", "over_1_5": "Over/under 1.5", "over_2_5": "Over/under 2.5",
                 "over_3_5": "Over/under 3.5", "btts": "BTTS", "exact_score": "Exact score"}


def paired_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [f"{market}_log_loss" for market in LOSS_MARKETS]
    if predictions.empty or not {*MATCH_FIELDS, "model", *columns}.issubset(predictions):
        raise ValueError("Incomplete final pairing input.")
    if set(predictions.model) != set(FINAL_MODELS) or predictions.duplicated(["match_id", "model"]).any():
        raise ValueError("Invalid final model set or duplicate match–model pair.")
    losses = predictions[columns].to_numpy(dtype=float)
    if not (np.isfinite(losses) & (losses >= 0)).all() or predictions.Season.nunique() != 1:
        raise ValueError("Finite, non-negative losses from one season are required.")
    new = predictions.loc[predictions.model == "fitted_poisson"].set_index("match_id").sort_index()
    frames = []
    for reference in REFERENCES:
        old = predictions.loc[predictions.model == reference].set_index("match_id")
        if set(old.index) != set(new.index):
            raise ValueError("Match lists differ in the final pairing.")
        old = old.loc[new.index]
        try:
            pd.testing.assert_frame_equal(new[list(MATCH_FIELDS[1:])], old[list(MATCH_FIELDS[1:])], check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError("Match data differ in the final pairing.") from error
        frame = new[list(MATCH_FIELDS[1:])].reset_index()
        frame["new_model"], frame["reference_model"] = "fitted_poisson", reference
        for market in LOSS_MARKETS:
            frame[f"new_{market}_log_loss"] = new[f"{market}_log_loss"].to_numpy(dtype=float)
            frame[f"reference_{market}_log_loss"] = old[f"{market}_log_loss"].to_numpy(dtype=float)
            frame[f"{market}_loss_difference"] = frame[f"new_{market}_log_loss"] - frame[f"reference_{market}_log_loss"]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["Date"] = pd.to_datetime(result.Date, format="%Y-%m-%d")
    return result.sort_values(["Date", "match_id", "reference_model"]).reset_index(drop=True)


def loss_intervals(predictions: pd.DataFrame, payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = paired_losses(predictions)
    base = paired.loc[paired.reference_model == REFERENCES[0]].set_index("match_id").sort_index()
    dates = pd.DatetimeIndex(sorted(base.Date.unique()))
    counts = base.groupby("Date").size().reindex(dates).to_numpy(dtype=np.int64)
    keys, sums = [], []
    for reference in REFERENCES:
        rows = paired.loc[paired.reference_model == reference].set_index("match_id").loc[base.index]
        for market in LOSS_MARKETS:
            keys.append((reference, market))
            sums.append(rows.groupby("Date")[f"{market}_loss_difference"].sum().reindex(dates).to_numpy(dtype=float))
    sums = np.column_stack(sums)
    policy = payload["uncertainty_policy"]
    records = []
    for length in policy["block_lengths_matchdays"]:
        seed = policy["base_seed"] + length
        draws, sampled_counts = resample_daily_means(sums, counts, length, policy["n_resamples_per_length"], seed)
        tail = (1 - policy["confidence_level"]) / 2
        bounds = np.quantile(draws, [tail, 1 - tail], axis=0, method=policy["quantile_method"])
        point = sums.sum(axis=0) / counts.sum()
        for column, (reference, market) in enumerate(keys):
            lower, upper = bounds[:, column]
            records.append({
                "Season": str(base.Season.iloc[0]), "new_model": "fitted_poisson", "reference_model": reference,
                "market": market, "matches": int(counts.sum()), "matchdays": len(counts),
                "block_length_matchdays": length, "n_resamples": policy["n_resamples_per_length"], "seed": seed,
                "confidence_level": policy["confidence_level"], "mean_loss_difference": float(point[column]),
                "ci_lower": float(lower), "ci_upper": float(upper), "bootstrap_standard_error": float(draws[:, column].std(ddof=1)),
                "bootstrap_mean_difference": float(draws[:, column].mean()),
                "interval_direction": "new_lower" if upper < 0 else "new_higher" if lower > 0 else "includes_zero",
                "degenerate_distribution": bool(np.ptp(draws[:, column]) == 0),
                "resampled_matches_min": int(sampled_counts.min()), "resampled_matches_max": int(sampled_counts.max()),
            })
    intervals = pd.DataFrame(records)
    sensitivity = summarize_sensitivity(intervals, tuple(policy["block_lengths_matchdays"]))
    return paired, intervals, sensitivity


def calibration_report(predictions: pd.DataFrame, payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = payload["evaluation_policy"]
    frames, summaries = [], []
    for name in FINAL_MODELS:
        rows = predictions.loc[predictions.model == name]
        for event, (_, probability_column, _) in EVENTS.items():
            observed = event_observations(rows, event)
            bins = calibration_bins(rows[probability_column], observed, policy["calibration_bins"], policy["min_bin_matches"])
            bins.insert(0, "event", event)
            bins.insert(0, "model", name)
            frames.append(bins)
            active = bins.loc[~bins["empty"]]
            mean, actual = float(rows[probability_column].mean()), float(observed.mean())
            summaries.append({"model": name, "event": event, "matches": len(rows), "mean_probability": mean,
                              "observed_rate": actual, "bias": mean - actual,
                              "ece": float((active.matches * active.bias.abs()).sum() / len(rows)),
                              "empty_bins": int(bins["empty"].sum()), "sparse_bins": int(bins["sparse"].sum()),
                              "matches_in_sparse_bins": int(bins.loc[bins["sparse"], "matches"].sum())})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def primary_assessment(intervals: pd.DataFrame, payload: dict) -> dict:
    policy = payload["evaluation_policy"]
    rows = intervals.loc[(intervals.reference_model == policy["primary_reference"]) & (intervals.market == policy["primary_market"])]
    lengths = payload["uncertainty_policy"]["block_lengths_matchdays"]
    if len(rows) != len(lengths) or set(rows.block_length_matchdays) != set(lengths):
        raise ValueError("Incomplete primary final comparison.")
    values = rows[["mean_loss_difference", "ci_lower", "ci_upper"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or not (rows.ci_lower <= rows.ci_upper).all()
            or not np.allclose(values[:, 0], values[0, 0], rtol=1e-12, atol=1e-12)):
        raise ValueError("Invalid primary loss difference or interval.")
    main = rows.loc[rows.block_length_matchdays == payload["uncertainty_policy"]["primary_block_length_matchdays"]].iloc[0]
    point = float(main.mean_loss_difference)
    return {"primary_market": policy["primary_market"], "primary_metric": policy["primary_metric"],
            "selected_model": payload["selected_model"], "primary_reference": policy["primary_reference"],
            "mean_loss_difference": point, "ci_lower": float(main.ci_lower), "ci_upper": float(main.ci_upper),
            "observed_point_improvement": bool(point < 0), "primary_interval_support": bool(main.ci_upper < 0),
            "all_three_block_lengths_support": bool((rows.ci_upper < 0).all()),
            "model_reselected_from_test": False, "hyperparameters_tuned_from_test": False}


def conclusion_note(summary: pd.DataFrame, intervals: pd.DataFrame, assessment: dict, protocol: dict) -> str:
    header = ["# The separate 2025/26 final test", "", f"Protocol: `{protocol['protocol_id']}`", "",
              "Selected model: jointly fitted Poisson. Primary comparison: 1X2 log loss against the league-average reference.", "",
              "| Model | Matches | 1X2 log loss | 1X2 Brier | 1X2 accuracy |", "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        header.append(f"| {LABELS[row.model]} | {row.matches} | {row.one_x_two_log_loss:.6f} | {row.one_x_two_brier:.6f} | {row.one_x_two_accuracy:.2%} |")
    header.extend(["", f"Primary difference: {assessment['mean_loss_difference']:+.6f}; five-matchday 95% interval: [{assessment['ci_lower']:+.6f}; {assessment['ci_upper']:+.6f}].", "",
                   "Observed point estimate: " + ("favours the selected model." if assessment["observed_point_improvement"] else "shows no improvement for the selected model."),
                   "Primary interval: " + ("supports the direction of lower loss." if assessment["primary_interval_support"] else "does not clearly support lower loss."),
                   "All three block lengths: " + ("support the same direction." if assessment["all_three_block_lengths_support"] else "do not jointly support lower loss."), "",
                   "## All tasks and both references", "",
                   "| Task | Reference | Mean difference | Five-day 95% interval |", "|---|---|---:|---:|"])
    focus = intervals.loc[intervals.block_length_matchdays == 5]
    for row in focus.itertuples(index=False):
        header.append(f"| {MARKET_LABELS[row.market]} | {LABELS[row.reference_model]} | {row.mean_loss_difference:+.6f} | [{row.ci_lower:+.6f}; {row.ci_upper:+.6f}] |")
    header.extend(["", "Negative difference: favours the selected Poisson. Other tasks and the second reference are secondary evaluations.", "",
                   "Every match has equal weight. Same-day results are excluded; earlier completed test matches may enter later history.", "",
                   "Block lengths are 3, 5 and 10 matchdays; 5000 resamples and 95% percentile intervals. Intervals are individual, with no multiple-comparison adjustment.", "",
                   "The calculation is conditional on loss differences for fixed forecasts. It excludes model-selection and refitting uncertainty. An interval containing zero does not establish equivalence.", "",
                   "No model was reselected or hyperparameter tuned from the test. Repeating the season evaluation is reproduction; changes informed by these results begin a new development cycle.", "",
                   "This is a retrospective, chronologically simulated test on a historical season. It is not evidence of forecasts published in real time.", ""])
    return "\n".join(header)


def report_bundle(predictions: pd.DataFrame, diagnostics: pd.DataFrame, protocol: dict) -> tuple[dict, dict, str]:
    payload = protocol["payload"]
    summary = summarize_models(predictions, FINAL_MODELS)
    bins, calibration = calibration_report(predictions, payload)
    paired, intervals, sensitivity = loss_intervals(predictions, payload)
    assessment = primary_assessment(intervals, payload)
    tables = {"final_predictions": predictions, "final_summary": summary, "calibration_bins": bins,
              "calibration_summary": calibration, "paired_match_losses": paired, "bootstrap_intervals": intervals,
              "block_sensitivity": sensitivity, "fitting_diagnostics": diagnostics}
    return tables, assessment, conclusion_note(summary, intervals, assessment, protocol)
