"""Matchday block-bootstrap analysis of paired log loss differences."""

from numbers import Integral, Real

import numpy as np
import pandas as pd

from backtest_models import LOSS_MARKETS, MODEL_NAMES


NEW_MODEL = "fitted_poisson"
ALL_MODELS = (*MODEL_NAMES, NEW_MODEL)
MATCH_FIELDS = ("match_id", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
DIRECTIONS = {
    "new_lower": "favours the new model",
    "new_higher": "favours the previous model",
    "includes_zero": "the interval contains zero",
}


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name}: an integer of at least {minimum} is required.")
    return int(value)


def _match_fields(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(MATCH_FIELDS).issubset(frame.columns):
        raise ValueError("Incomplete or empty match table.")
    result = frame.copy()
    if result[list(MATCH_FIELDS)].isna().any().any():
        raise ValueError("Missing match data.")
    if result.Season.nunique() != 1:
        raise ValueError("Exactly one season can be examined at a time.")
    result["Date"] = pd.to_datetime(result.Date, format="%Y-%m-%d", errors="raise")
    if not (result.Date == result.Date.dt.normalize()).all():
        raise ValueError("A matchday must be a date without a time component.")
    for column in ("FTHG", "FTAG"):
        values = result[column].to_numpy(dtype=float)
        if not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all():
            raise ValueError("Non-negative integer goal counts are required.")
    return result


def build_paired_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pair by identifier; every row is one match and one reference model."""
    columns = [f"{market}_log_loss" for market in LOSS_MARKETS]
    required = {"model", *columns}
    if not required.issubset(predictions.columns):
        raise ValueError("Missing model or log loss field.")
    result = _match_fields(predictions)
    if result.model.isna().any() or set(result.model) != set(ALL_MODELS):
        raise ValueError("Predictions from all five models are required.")
    if result.duplicated(["match_id", "model"]).any():
        raise ValueError("Duplicate match–model pair.")
    values = result[columns].to_numpy(dtype=float)
    if not (np.isfinite(values) & (values >= 0)).all():
        raise ValueError("Finite, non-negative log loss is required.")
    new = result.loc[result.model == NEW_MODEL].set_index("match_id").sort_index()
    records = []
    for reference in MODEL_NAMES:
        old = result.loc[result.model == reference].set_index("match_id")
        if set(old.index) != set(new.index):
            raise ValueError("The models' match lists differ.")
        old = old.loc[new.index]
        try:
            pd.testing.assert_frame_equal(
                new[list(MATCH_FIELDS[1:])], old[list(MATCH_FIELDS[1:])],
                check_dtype=False, check_exact=True,
            )
        except AssertionError as error:
            raise ValueError("Match data differ between the paired models.") from error
        paired = new[list(MATCH_FIELDS[1:])].reset_index()
        paired["new_model"] = NEW_MODEL
        paired["reference_model"] = reference
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
    """Daily difference totals and match counts; preserve equal match weights."""
    required = {"new_model", "reference_model"}
    for market in LOSS_MARKETS:
        required.update((f"new_{market}_log_loss", f"reference_{market}_log_loss",
                         f"{market}_loss_difference"))
    if not required.issubset(paired.columns):
        raise ValueError("Incomplete paired loss table.")
    result = _match_fields(paired)
    if result[list(required)].isna().any().any():
        raise ValueError("Missing paired value.")
    if set(result.new_model) != {NEW_MODEL} or set(result.reference_model) != set(MODEL_NAMES):
        raise ValueError("Invalid compared model set.")
    if result.duplicated(["match_id", "reference_model"]).any():
        raise ValueError("Duplicate paired match.")
    base = result.loc[result.reference_model == MODEL_NAMES[0]].set_index("match_id").sort_index()
    dates = pd.DatetimeIndex(sorted(base.Date.unique()))
    counts = base.groupby("Date").size().reindex(dates).to_numpy(dtype=np.int64)
    sums, keys = [], []
    for reference in MODEL_NAMES:
        rows = result.loc[result.reference_model == reference].set_index("match_id")
        if set(rows.index) != set(base.index):
            raise ValueError("A match is missing from the paired table.")
        rows = rows.loc[base.index]
        try:
            pd.testing.assert_frame_equal(
                rows[list(MATCH_FIELDS[1:])], base[list(MATCH_FIELDS[1:])],
                check_dtype=False, check_exact=True,
            )
        except AssertionError as error:
            raise ValueError("Match data differ between paired rows.") from error
        for market in LOSS_MARKETS:
            new = rows[f"new_{market}_log_loss"].to_numpy(dtype=float)
            old = rows[f"reference_{market}_log_loss"].to_numpy(dtype=float)
            difference = rows[f"{market}_loss_difference"].to_numpy(dtype=float)
            if not (np.isfinite(new) & np.isfinite(old) & np.isfinite(difference)
                    & (new >= 0) & (old >= 0)).all():
                raise ValueError("Invalid paired log loss.")
            if not np.allclose(difference, new - old, rtol=1e-12, atol=1e-12):
                raise ValueError("The difference is not new model minus previous model.")
            daily = rows.groupby("Date")[f"{market}_loss_difference"].sum().reindex(dates)
            sums.append(daily.to_numpy(dtype=float))
            keys.append((reference, market))
    totals = np.column_stack(sums)
    if not np.isfinite(totals).all():
        raise ValueError("Non-finite daily sum of differences.")
    return totals, counts, tuple(keys)


def moving_block_indices(
    n_days: int, block_length: int, n_resamples: int, rng: np.random.Generator,
) -> np.ndarray:
    """Overlapping within-season blocks; the last selected block may be truncated."""
    n_days = _positive_integer(n_days, "Number of matchdays", 2)
    block_length = _positive_integer(block_length, "Block length")
    n_resamples = _positive_integer(n_resamples, "Number of resamples")
    if block_length >= n_days:
        raise ValueError("Block length must be less than the number of matchdays.")
    n_blocks = (n_days + block_length - 1) // block_length
    starts = rng.integers(0, n_days - block_length + 1, size=(n_resamples, n_blocks))
    indices = (starts[..., None] + np.arange(block_length)).reshape(n_resamples, -1)
    return indices[:, :n_days]


def resample_daily_means(
    daily_sums: np.ndarray, daily_counts: np.ndarray, block_length: int,
    n_resamples: int = 5000, seed: int = 42, batch_size: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same day indices to resample all 24 model–task pairs."""
    sums = np.asarray(daily_sums, dtype=float)
    counts = np.asarray(daily_counts, dtype=float)
    if (sums.ndim != 2 or counts.ndim != 1 or len(sums) != len(counts)
            or sums.shape[1] == 0 or not np.isfinite(sums).all()
            or not (np.isfinite(counts) & (counts > 0) & (counts == np.floor(counts))).all()):
        raise ValueError("Invalid daily totals or match counts.")
    n_resamples = _positive_integer(n_resamples, "Number of resamples", 2)
    batch_size = _positive_integer(batch_size, "Processing batch")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("A non-negative integer random seed is required.")
    rng = np.random.default_rng(int(seed))
    # Validate parameters before creating an empty output.
    if isinstance(block_length, bool) or not isinstance(block_length, Integral):
        raise ValueError("An integer block length is required.")
    if not 1 <= block_length < len(counts):
        raise ValueError("Block length must be positive and less than the number of matchdays.")
    draws = np.empty((n_resamples, sums.shape[1]), dtype=float)
    resampled_counts = np.empty(n_resamples, dtype=np.int64)
    for begin in range(0, n_resamples, batch_size):
        end = min(begin + batch_size, n_resamples)
        indices = moving_block_indices(len(counts), block_length, end - begin, rng)
        totals = counts[indices].sum(axis=1)
        draws[begin:end] = sums[indices].sum(axis=1) / totals[:, None]
        resampled_counts[begin:end] = totals.astype(np.int64)
    if not np.isfinite(draws).all():
        raise ValueError("Non-finite resampled mean.")
    return draws, resampled_counts


def bootstrap_loss_intervals(
    paired: pd.DataFrame, block_length: int = 5, n_resamples: int = 5000,
    confidence_level: float = 0.95, seed: int = 42,
) -> pd.DataFrame:
    """Individual, exploratory percentile intervals; no p-values."""
    if (isinstance(confidence_level, bool) or not isinstance(confidence_level, Real)
            or not np.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError("The confidence level must be between zero and one.")
    sums, counts, keys = aggregate_matchdays(paired)
    draws, sampled_counts = resample_daily_means(sums, counts, block_length, n_resamples, seed)
    tail = (1 - confidence_level) / 2
    bounds = np.quantile(draws, [tail, 1 - tail], axis=0, method="linear")
    point = sums.sum(axis=0) / counts.sum()
    standard_error = draws.std(axis=0, ddof=1)
    records = []
    for column, (reference, market) in enumerate(keys):
        lower, upper = bounds[:, column]
        direction = "new_lower" if upper < 0 else "new_higher" if lower > 0 else "includes_zero"
        records.append({
            "Season": str(paired.Season.iloc[0]), "new_model": NEW_MODEL,
            "reference_model": reference, "market": market, "matches": int(counts.sum()),
            "matchdays": len(counts), "block_length_matchdays": int(block_length),
            "n_resamples": int(n_resamples), "seed": int(seed),
            "confidence_level": float(confidence_level),
            "mean_loss_difference": float(point[column]),
            "ci_lower": float(lower), "ci_upper": float(upper),
            "bootstrap_standard_error": float(standard_error[column]),
            "bootstrap_mean_difference": float(draws[:, column].mean()),
            "interval_direction": direction,
            "degenerate_distribution": bool(np.ptp(draws[:, column]) == 0),
            "resampled_matches_min": int(sampled_counts.min()),
            "resampled_matches_max": int(sampled_counts.max()),
        })
    return pd.DataFrame(records)


def summarize_sensitivity(intervals: pd.DataFrame, block_lengths: tuple[int, ...]) -> pd.DataFrame:
    """Display all three prespecified block lengths together without selecting a favourable one."""
    required = {"Season", "new_model", "reference_model", "market", "block_length_matchdays",
                "mean_loss_difference", "ci_lower", "ci_upper", "interval_direction"}
    if intervals.empty or not required.issubset(intervals.columns):
        raise ValueError("Incomplete interval table.")
    lengths = tuple(_positive_integer(value, "Block length") for value in block_lengths)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("Distinct block lengths are required.")
    keys = ["Season", "new_model", "reference_model", "market"]
    if intervals.duplicated([*keys, "block_length_matchdays"]).any():
        raise ValueError("Duplicate interval row.")
    records = []
    for group, rows in intervals.groupby(keys, sort=True):
        if set(rows.block_length_matchdays) != set(lengths):
            raise ValueError("A block length is missing from the sensitivity analysis.")
        points = rows.mean_loss_difference.to_numpy(dtype=float)
        if not np.allclose(points, points[0], rtol=1e-12, atol=1e-12):
            raise ValueError("The point estimate differs between block lengths.")
        directions = set(rows.interval_direction)
        if not directions.issubset(DIRECTIONS):
            raise ValueError("Unknown interval indication.")
        sensitivity = next(iter(directions)) if len(directions) == 1 else "block_sensitive"
        record = {**dict(zip(keys, group)), "mean_loss_difference": float(points[0]),
                  "sensitivity_direction": sensitivity}
        for row in rows.itertuples(index=False):
            record[f"ci_lower_block_{row.block_length_matchdays}"] = row.ci_lower
            record[f"ci_upper_block_{row.block_length_matchdays}"] = row.ci_upper
        records.append(record)
    return pd.DataFrame(records)
