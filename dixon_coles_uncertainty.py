"""DC minus five references: paired matchday block bootstrap."""

from numbers import Real

import numpy as np
import pandas as pd

from backtest_models import LOSS_MARKETS
from dixon_coles_backtest import COMPARISON_MODELS, NEW_MODEL, PREVIOUS_MODELS
from uncertainty_analysis import MATCH_FIELDS, resample_daily_means, summarize_sensitivity


def _match_fields(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(MATCH_FIELDS).issubset(frame.columns):
        raise ValueError("Incomplete or empty match table.")
    result = frame.copy()
    if result[list(MATCH_FIELDS)].isna().any().any():
        raise ValueError("Missing match data.")
    if result.Season.nunique() != 1:
        raise ValueError("Exactly one season can be examined at a time.")
    result["Date"] = pd.to_datetime(result.Date, format="%Y-%m-%d", errors="raise")
    if not result.Date.eq(result.Date.dt.normalize()).all():
        raise ValueError("A matchday must be a date without a time component.")
    goals = result[["FTHG", "FTAG"]].to_numpy(dtype=float)
    if not (np.isfinite(goals) & (goals >= 0) & (goals == np.floor(goals))).all():
        raise ValueError("Non-negative integer goal counts are required.")
    return result


def build_paired_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Each row is one match and one reference; all tasks use the same pairing."""
    columns = [f"{market}_log_loss" for market in LOSS_MARKETS]
    if not {"model", *columns}.issubset(predictions.columns):
        raise ValueError("Missing model or log loss field.")
    result = _match_fields(predictions)
    if result.model.isna().any() or set(result.model) != set(COMPARISON_MODELS):
        raise ValueError("Predictions from all six models are required.")
    if result.duplicated(["match_id", "model"]).any():
        raise ValueError("Duplicate match–model pair.")
    values = result[columns].to_numpy(dtype=float)
    if not (np.isfinite(values) & (values >= 0)).all():
        raise ValueError("Finite, non-negative log loss is required.")
    new = result.loc[result.model == NEW_MODEL].set_index("match_id").sort_index()
    records = []
    for reference in PREVIOUS_MODELS:
        old = result.loc[result.model == reference].set_index("match_id")
        if set(old.index) != set(new.index):
            raise ValueError("The models' match lists differ.")
        old = old.loc[new.index]
        try:
            pd.testing.assert_frame_equal(new[list(MATCH_FIELDS[1:])], old[list(MATCH_FIELDS[1:])],
                                          check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError("Match data differ between the paired models.") from error
        paired = new[list(MATCH_FIELDS[1:])].reset_index()
        paired["new_model"], paired["reference_model"] = NEW_MODEL, reference
        for market in LOSS_MARKETS:
            new_loss = new[f"{market}_log_loss"].to_numpy(dtype=float)
            old_loss = old[f"{market}_log_loss"].to_numpy(dtype=float)
            paired[f"new_{market}_log_loss"] = new_loss
            paired[f"reference_{market}_log_loss"] = old_loss
            paired[f"{market}_loss_difference"] = new_loss - old_loss
        records.append(paired)
    return pd.concat(records, ignore_index=True).sort_values(
        ["Date", "match_id", "reference_model"]
    ).reset_index(drop=True)


def aggregate_matchdays(paired: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Daily sums of differences and match counts for the 30 reference–task pairs."""
    required = {"new_model", "reference_model"}
    for market in LOSS_MARKETS:
        required.update((f"new_{market}_log_loss", f"reference_{market}_log_loss", f"{market}_loss_difference"))
    if not required.issubset(paired.columns):
        raise ValueError("Incomplete paired loss table.")
    result = _match_fields(paired)
    if result[list(required)].isna().any().any():
        raise ValueError("Missing paired value.")
    if set(result.new_model) != {NEW_MODEL} or set(result.reference_model) != set(PREVIOUS_MODELS):
        raise ValueError("Invalid compared model set.")
    if result.duplicated(["match_id", "reference_model"]).any():
        raise ValueError("Duplicate paired match.")
    base = result.loc[result.reference_model == PREVIOUS_MODELS[0]].set_index("match_id").sort_index()
    dates = pd.DatetimeIndex(sorted(base.Date.unique()))
    counts = base.groupby("Date").size().reindex(dates).to_numpy(dtype=np.int64)
    sums, keys = [], []
    for reference in PREVIOUS_MODELS:
        rows = result.loc[result.reference_model == reference].set_index("match_id")
        if set(rows.index) != set(base.index):
            raise ValueError("A match is missing from the paired table.")
        rows = rows.loc[base.index]
        try:
            pd.testing.assert_frame_equal(rows[list(MATCH_FIELDS[1:])], base[list(MATCH_FIELDS[1:])],
                                          check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError("Match data differ between paired rows.") from error
        for market in LOSS_MARKETS:
            new = rows[f"new_{market}_log_loss"].to_numpy(dtype=float)
            old = rows[f"reference_{market}_log_loss"].to_numpy(dtype=float)
            difference = rows[f"{market}_loss_difference"].to_numpy(dtype=float)
            if not (np.isfinite(new) & np.isfinite(old) & np.isfinite(difference) & (new >= 0) & (old >= 0)).all():
                raise ValueError("Invalid paired log loss.")
            if not np.allclose(difference, new - old, rtol=1e-12, atol=1e-12):
                raise ValueError("The difference is not DC minus reference.")
            # For each match, DC loss is identical across all references.
            if not np.allclose(new, base[f"new_{market}_log_loss"], rtol=1e-12, atol=1e-12):
                raise ValueError("DC loss differs between references.")
            daily = rows.groupby("Date")[f"{market}_loss_difference"].sum().reindex(dates)
            sums.append(daily.to_numpy(dtype=float))
            keys.append((reference, market))
    totals = np.column_stack(sums)
    if not np.isfinite(totals).all():
        raise ValueError("Non-finite daily sum of differences.")
    return totals, counts, tuple(keys)


def bootstrap_loss_intervals(
    paired: pd.DataFrame, block_length: int = 5, n_resamples: int = 5000,
    confidence_level: float = 0.95, seed: int = 42,
) -> pd.DataFrame:
    """Use the tested Step 13 resampler for the 30 new comparisons."""
    if (isinstance(confidence_level, bool) or not isinstance(confidence_level, Real)
            or not np.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError("The confidence level must be between zero and one.")
    sums, counts, keys = aggregate_matchdays(paired)
    draws, sampled_counts = resample_daily_means(sums, counts, block_length, n_resamples, seed)
    tail = (1 - confidence_level) / 2
    bounds = np.quantile(draws, [tail, 1 - tail], axis=0, method="linear")
    point, standard_error = sums.sum(axis=0) / counts.sum(), draws.std(axis=0, ddof=1)
    records = []
    for column, (reference, market) in enumerate(keys):
        lower, upper = bounds[:, column]
        direction = "new_lower" if upper < 0 else "new_higher" if lower > 0 else "includes_zero"
        records.append({
            "Season": str(paired.Season.iloc[0]), "new_model": NEW_MODEL,
            "reference_model": reference, "market": market, "matches": int(counts.sum()),
            "matchdays": len(counts), "block_length_matchdays": int(block_length),
            "n_resamples": int(n_resamples), "seed": int(seed), "confidence_level": float(confidence_level),
            "mean_loss_difference": float(point[column]), "ci_lower": float(lower), "ci_upper": float(upper),
            "bootstrap_standard_error": float(standard_error[column]),
            "bootstrap_mean_difference": float(draws[:, column].mean()),
            "interval_direction": direction, "degenerate_distribution": bool(np.ptp(draws[:, column]) == 0),
            "resampled_matches_min": int(sampled_counts.min()), "resampled_matches_max": int(sampled_counts.max()),
        })
    return pd.DataFrame(records)
