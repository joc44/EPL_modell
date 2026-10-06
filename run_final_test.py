"""Step 18/b: fixed three-model final test on the separate season."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path

import pandas as pd

from backtest_models import LOSS_MARKETS
from final_test_backtest import FINAL_MODELS, LABELS, run_final_backtest, validate_final_diagnostics, validate_final_predictions
from final_test_data import DATA_MANIFEST, checked_protocol, load_final_inputs, step18_sources, write_outputs_once
from final_test_protocol import canonical_json
from final_test_reports import MARKET_LABELS, report_bundle
from run_fitted_backtest import print_summary
from uncertainty_reports import _same_table
from validation_data import file_sha256


PROJECT_DIR = Path(__file__).resolve().parent
SETTINGS_FILENAME = "final_test_settings_2025_26.json"
TABLE_KEYS = {
    "final_predictions": ["match_id", "model"], "final_summary": ["model"],
    "calibration_bins": ["model", "event", "bin"], "calibration_summary": ["model", "event"],
    "paired_match_losses": ["match_id", "reference_model"],
    "bootstrap_intervals": ["reference_model", "market", "block_length_matchdays"],
    "block_sensitivity": ["reference_model", "market"], "fitting_diagnostics": ["fit_date"],
}


def show_progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"Processed: {done}/{total} matches", flush=True)


def run_context(project: Path, manifest: dict, protocol: dict) -> dict:
    return {"protocol_id": protocol["protocol_id"], "protocol_file_sha256": manifest["protocol_file_sha256"],
            "data_manifest_sha256": file_sha256(project / "data/processed" / DATA_MANIFEST),
            "processed_data_sha256": manifest["output_sha256"], "raw_data_sha256": manifest["raw_sha256"],
            "step18_source_sha256": step18_sources(project), "runtime": protocol["payload"]["runtime"]}


def load_cached_run(output: Path, context: dict, matches: pd.DataFrame, protocol: dict) -> tuple:
    settings = json.loads((output / SETTINGS_FILENAME).read_text(encoding="utf-8"))
    expected_matches = int(matches.Season.eq(protocol["payload"]["final_test_season"]).sum())
    expected = {"step": "18b", "evaluation_role": "separate_final_test", "evaluation_season": "2025/26",
                "models": list(FINAL_MODELS), "evaluated_matches": expected_matches, "prediction_rows": len(FINAL_MODELS) * expected_matches,
                "failed_matches_skipped": 0, "model_reselected_from_test": False, "hyperparameters_tuned_from_test": False,
                "context": context}
    if any(canonical_json(settings.get(key)) != canonical_json(value) for key, value in expected.items()):
        raise ValueError("The saved final test used different data, source code or protocol.")
    names = {f"{name}_2025_26.csv" for name in TABLE_KEYS} | {"primary_assessment_2025_26.json", "final_test_conclusions_2025_26.md"}
    hashes = settings.get("output_sha256")
    if not isinstance(hashes, dict) or set(hashes) != names:
        raise ValueError("Incomplete final-report fingerprint table.")
    for name, digest in hashes.items():
        if not (output / name).is_file() or file_sha256(output / name) != digest:
            raise ValueError(f"A final report changed or is missing: {name}.")
    predictions = pd.read_csv(output / "final_predictions_2025_26.csv", encoding="utf-8-sig")
    predictions = validate_final_predictions(predictions, matches.loc[matches.Season == "2025/26"])
    diagnostics = pd.read_csv(output / "fitting_diagnostics_2025_26.csv", encoding="utf-8-sig")
    validate_final_diagnostics(diagnostics, matches, protocol["payload"])
    tables, assessment, note = report_bundle(predictions, diagnostics, protocol)
    for name, table in tables.items():
        saved = pd.read_csv(output / f"{name}_2025_26.csv", encoding="utf-8-sig")
        date_fields = ("Date", "history_start", "last_history_date") if name == "final_predictions" else ("Date",) if name == "paired_match_losses" else ()
        for column in date_fields:
            saved[column] = pd.to_datetime(saved[column], format="%Y-%m-%d")
        _same_table(table, saved, TABLE_KEYS[name], f"final {name}")
    saved_assessment = json.loads((output / "primary_assessment_2025_26.json").read_text(encoding="utf-8"))
    if not isinstance(saved_assessment, dict) or set(saved_assessment) != set(assessment):
        raise ValueError("Incomplete saved primary conclusion.")
    for key, value in assessment.items():
        if isinstance(value, float):
            if not isclose(saved_assessment[key], value, rel_tol=0, abs_tol=1e-12):
                raise ValueError("The saved primary conclusion differs.")
        elif canonical_json(saved_assessment[key]) != canonical_json(value):
            raise ValueError("The saved primary conclusion differs.")
    if (output / "final_test_conclusions_2025_26.md").read_text(encoding="utf-8") != note:
        raise ValueError("The saved final conclusion text differs.")
    return tables, assessment


def save_new_run(output: Path, tables: dict, assessment: dict, note: str, context: dict) -> None:
    content = {output / f"{name}_2025_26.csv": table.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8-sig")
               for name, table in tables.items()}
    content[output / "primary_assessment_2025_26.json"] = (json.dumps(assessment, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    content[output / "final_test_conclusions_2025_26.md"] = note.encode("utf-8")
    settings = {"step": "18b", "evaluation_role": "separate_final_test", "evaluation_season": "2025/26",
                "models": list(FINAL_MODELS), "evaluated_matches": int(tables["final_predictions"].match_id.nunique()),
                "prediction_rows": len(tables["final_predictions"]), "failed_matches_skipped": 0,
                "model_reselected_from_test": False, "hyperparameters_tuned_from_test": False, "context": context,
                "output_sha256": {path.name: sha256(data).hexdigest() for path, data in content.items()}}
    content[output / SETTINGS_FILENAME] = (json.dumps(settings, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_outputs_once(content)


def main() -> None:
    print("Step 18/b: final test on the separate 2025/26 season", flush=True)
    matches, manifest, protocol = load_final_inputs(PROJECT_DIR)
    payload = protocol["payload"]
    context = run_context(PROJECT_DIR, manifest, protocol)
    output = PROJECT_DIR / "data/reports/final_test"
    print(f"Validated protocol: {protocol['protocol_id']}")
    print("Models: league average, time-weighted and jointly fitted Poisson")
    print("History window: 730 days | Half-life: 180 days | L2 strength: 1")
    print("Primary evaluation: 1X2 log loss against the league-average reference")
    print("Earlier completed test matches may enter later history; same-day results are excluded.", flush=True)
    if (output / SETTINGS_FILENAME).exists():
        tables, assessment = load_cached_run(output, context, matches, protocol)
        print("The saved final test has been validated; no models were refitted.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Incomplete existing final-test directory; its results will not be overwritten.")
        predictions, diagnostics = run_final_backtest(matches, payload, show_progress)
        tables, assessment, note = report_bundle(predictions, diagnostics, protocol)
        current_matches, current_manifest, current_protocol = load_final_inputs(PROJECT_DIR)
        if canonical_json(run_context(PROJECT_DIR, current_manifest, current_protocol)) != canonical_json(context):
            raise ValueError("The final-test input changed during the run.")
        save_new_run(output, tables, assessment, note, context)
    checked_protocol(PROJECT_DIR)
    summary, diagnostics = tables["final_summary"], tables["fitting_diagnostics"]
    print(f"\nEvaluated matches: {tables['final_predictions'].match_id.nunique()}")
    print(f"Prediction rows: {len(tables['final_predictions'])} (3 models × 380 matches)")
    print(f"Successful matchday fits: {len(diagnostics)}/{len(diagnostics)}")
    print(f"Maximum absolute gradient: {diagnostics.gradient_max_abs.max():.3e}")
    print_summary(summary, LABELS)
    print("\nJointly fitted minus reference – five-matchday 95% intervals:")
    focus = tables["bootstrap_intervals"].loc[tables["bootstrap_intervals"].block_length_matchdays == 5]
    for reference in payload["reference_models"]:
        print(f"\nReference: {LABELS[reference]}")
        for row in focus.loc[focus.reference_model == reference].itertuples(index=False):
            print(f"{MARKET_LABELS[row.market]:<20} {row.mean_loss_difference:+.6f} [{row.ci_lower:+.6f}; {row.ci_upper:+.6f}]")
    print("\nPrimary comparison – Poisson minus league average, 1X2:")
    print(f"Mean log loss difference: {assessment['mean_loss_difference']:+.6f}")
    print(f"95% interval: [{assessment['ci_lower']:+.6f}; {assessment['ci_upper']:+.6f}]")
    print(f"Observed improvement: {assessment['observed_point_improvement']}")
    print(f"Supported by the five-day interval: {assessment['primary_interval_support']}")
    print(f"Supported at all three block lengths: {assessment['all_three_block_lengths_support']}")
    print("\nLog loss probability-floor usage by model:")
    for row in summary.to_dict("records"):
        print(LABELS[row["model"]] + ": " + ", ".join(f"{market}: {row[f'{market}_log_loss_clipped']}" for market in LOSS_MARKETS))
    selected = tables["final_predictions"].loc[tables["final_predictions"].model == "fitted_poisson"]
    print(f"Predictions without full-team history: home {int(selected.home_unknown.sum())}, away {int(selected.away_unknown.sum())}")
    print("\nJointly fitted model – calibration:")
    for row in tables["calibration_summary"].loc[tables["calibration_summary"].model == "fitted_poisson"].itertuples(index=False):
        print(f"{row.event}: predicted {row.mean_probability:.2%} | observed {row.observed_rate:.2%} | difference {100 * row.bias:+.2f} pp | ECE {100 * row.ece:.2f} pp")
    print(f"\nReports saved in: {output}")
    print("No model was reselected or hyperparameter tuned from the test.")
    print("Intervals are individual; they exclude model-selection and refitting uncertainty.")
    print("Further runs on the 2025/26 season are reproduction; changes informed by the results begin a new development cycle.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, pd.errors.ParserError) as error:
        raise SystemExit(f"Final-test error: {error}")
