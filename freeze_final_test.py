"""Step 17: model selection and protocol for the separate 2025/26 test."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "reports" / "model_selection"


def decision_note(protocol: dict, summary: pd.DataFrame) -> str:
    selected = summary.loc[summary.model == "fitted_poisson"].sort_values("Season")
    rows = [f"| {row.Season} | {row.one_x_two_log_loss:.6f} | {row.btts_log_loss:.6f} | {row.exact_score_log_loss:.6f} |"
            for row in selected.itertuples(index=False)]
    return "\n".join([
        "# Model selection and final-test freeze", "",
        "Selected version: jointly fitted, time-weighted, L2-regularised independent Poisson.", "",
        "The primary task is 1X2 and the primary metric is log loss. The main reference is league average; the second reference is time-weighted.", "",
        "Fixed settings: 730-day history window, 180-day half-life and L2 strength 1.", "",
        "Team effects and the goal intercept are re-estimated before every matchday from the history completed by then.", "",
        "Development seasons: 2023/24 and 2024/25. This step does not read the 2025/26 test results.", "",
        "Poisson 1X2 log loss is lower than that of the previous four models in both development seasons. The small additional DC advantage has intervals containing zero in both seasons and at all three block lengths.", "",
        "Poisson is selected for its simpler joint fit and coherent score distribution across all markets. No general Poisson superiority or optimal hyperparameters are claimed.", "",
        "| Development season | 1X2 log loss | BTTS log loss | Exact-score log loss |",
        "|---|---:|---:|---:|", *rows, "",
        "The final test requires the same 380 fixtures for three models. All results on a target matchday are excluded. A failed fit stops the run; no match is skipped or replaced by another model.", "",
        "All goal markets, exact scores, Brier scores, accuracy and calibration are reported. The main conclusion uses the prespecified 1X2 metric and league-average comparison.", "",
        "Uncertainty uses the same matchday block bootstrap: blocks of 3, 5 and 10 days, 5000 resamples and 95% percentile intervals. The main presentation uses five days.", "",
        "No model is reselected or hyperparameter tuned from the test. Subsequent changes begin a new development cycle. Rerunning identical data and protocol is reproduction.", "",
        f"Protocol identifier: `{protocol['protocol_id']}`", "",
        f"Freeze time UTC: `{protocol['created_at_utc']}`", "",
    ])


def main() -> None:
    try:
        from final_test_protocol import (
            PROTOCOL_FILENAME, assert_frozen_context_unchanged, build_protocol_payload, freeze_or_verify_protocol,
        )
        from model_selection_reports import load_selection_inputs, selection_evidence
        from uncertainty_reports import assert_uncertainty_inputs_unchanged
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in ("numpy", "scipy"):
            raise SystemExit("A previous dependency is missing. Use the project's existing virtual environment and requirements.") from None
        raise
    print("Step 17: freeze model selection and the final-test protocol", flush=True)
    print("Use the validated 2023/24 and 2024/25 reports.")
    print("Validate the Step 16 uncertainty reports by recalculation; no models are refitted.")
    print("The 2025/26 test data are not read in this step.", flush=True)
    predictions, intervals, context = load_selection_inputs(PROJECT_DIR)
    summary, focus = selection_evidence(predictions, intervals)
    payload = build_protocol_payload(PROJECT_DIR, context)
    assert_uncertainty_inputs_unchanged(PROJECT_DIR, context)
    assert_frozen_context_unchanged(PROJECT_DIR, payload)
    protocol, created = freeze_or_verify_protocol(OUTPUT_DIR / PROTOCOL_FILENAME, payload)
    summary.to_csv(OUTPUT_DIR / "model_selection_evidence_2023_24_2024_25.csv", index=False, encoding="utf-8-sig")
    focus.to_csv(OUTPUT_DIR / "dc_vs_poisson_uncertainty_2023_24_2024_25.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "selection_decision.md").write_text(decision_note(protocol, summary), encoding="utf-8")
    print("\nSelected model: jointly fitted Poisson")
    print("Fixed parameters: 730 days | half-life 180 days | L2 strength 1")
    print("Primary task and metric: 1X2 log loss")
    print("Main reference: league average | second reference: time-weighted")
    print("\nDevelopment results for the selected model:")
    for row in summary.loc[summary.model == "fitted_poisson"].itertuples(index=False):
        print(f"{row.Season}: 1X2 {row.one_x_two_log_loss:.6f} | BTTS {row.btts_log_loss:.6f} | exact score {row.exact_score_log_loss:.6f}")
    print("\nThe additional DC 1X2 advantage has intervals containing zero in both seasons and at all three block lengths.")
    print("Selection is a development decision that also considers simplicity; it does not establish general model superiority.")
    print("\nFinal test: 2025/26 | expected matches: 380 | models: 3")
    print("Secondary evaluation of the goal markets and exact scores is also fixed.")
    print("Team parameters are re-estimated before every matchday; hyperparameters remain fixed.")
    print("Protocol: " + ("newly created" if created else "previous version validated and unchanged"))
    print(f"Protocol identifier: {protocol['protocol_id']}")
    print(f"Reports saved in: {OUTPUT_DIR}")
    print("\nThe next step prepares and evaluates the separate test season using this protocol.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, pd.errors.ParserError) as error:
        raise SystemExit(f"Model-selection error: {error}")
