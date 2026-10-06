# Premier League Match Probabilities in Python

A Python portfolio project that estimates expected goal counts from historical Premier League final scores, then calculates 1X2, over/under, both-teams-to-score and exact-score probabilities from a coherent score distribution.

The project demonstrates data preparation, chronological backtesting, calibration, paired model comparisons and evaluation on a separate historical season. The selected Poisson model improved 1X2 log loss over the league-average reference on the 2025/26 test. Goal-market results are mixed and are reported alongside the primary result.

This is the complete English edition. It includes all current modules, tests, five raw season CSVs, processed data, generated reports and documentation. Console messages, comments, docstrings, report text and chart labels are in English. Earlier chapter packages are not needed.

## What does it predict?

- Expected home and away goal counts.
- Home win, draw and away win probabilities (1X2).
- Over/under 1.5, 2.5 and 3.5 goals.
- Both teams to score (BTTS).
- Exact-score probabilities and the most probable scores.

The goal estimates use final-score history. They are not shot-based expected-goals (xG) measurements. The examples and evaluations use historical fixtures.

## Final result of version 1

The final test covered all 380 matches of the 2025/26 season. The selected model and two references produced 1140 prediction rows. Each matchday used only results from earlier dates.

| Model | 1X2 log loss ↓ | 1X2 Brier ↓ | Accuracy |
|---|---:|---:|---:|
| League average | 1.0831 | 0.6556 | 42.63% |
| Time-weighted and smoothed | 1.0377 | 0.6235 | 50.00% |
| Jointly fitted Poisson | 1.0344 | 0.6204 | 47.11% |

The prespecified primary comparison is **Poisson minus league average, 1X2 log loss**:

- Mean difference: **−0.048683**.
- Five-matchday block-bootstrap 95% interval: **[−0.082760; −0.010868]**.
- The intervals for three- and ten-matchday blocks also had upper bounds below zero.

Against the time-weighted reference, the smaller difference is −0.003294, with a five-day interval of [−0.019221; +0.019777]. This comparison does not establish a clear additional benefit.

All secondary tasks are also reported:

| Task: log loss ↓ | League average | Time-weighted | Jointly fitted Poisson |
|---|---:|---:|---:|
| Over/under 1.5 | 0.5154 | 0.5213 | 0.5274 |
| Over/under 2.5 | 0.6916 | 0.7004 | 0.7115 |
| Over/under 3.5 | 0.6085 | 0.6122 | 0.6195 |
| BTTS | 0.6911 | 0.6932 | 0.6915 |
| Exact score | 2.9537 | 2.9136 | 2.9223 |

Lower loss is better. Poisson's over/under 2.5 result is worse than the league-average result; the five-day difference interval is [+0.004361; +0.037547]. Its exact-score improvement over league average has an interval containing zero. The conclusions describe both the primary improvement and the limitations on other tasks.

Accuracy considers only the most probable outcome. Log loss evaluates the probability assigned to the observed outcome. Consequently, lower log loss can coexist with lower accuracy.

## Development and selection

Six model variants were compared during development: league average, raw team ratios, ratios smoothed towards league average, time-weighted and smoothed ratios, jointly fitted Poisson, and Dixon–Coles.

The joint model estimates team attack and concede effects while accounting for opponents. The additional DC low-score correction had uncertain incremental 1X2 gains in development, so the simpler joint Poisson model was selected.

| Data or decision | Role |
|---|---|
| 2021/22–2022/23 | Initial historical data |
| 2023/24 and 2024/25 | Development and model comparisons |
| Step 17 | Freeze the model, hyperparameters and final-test rules |
| 2025/26 | Separate historical final test |

The 2024/25 season initially served as later-season validation. Once its results informed further model development, it became development data. The original model selection was fixed before the separate 2025/26 evaluation.

The English edition reproduces this already completed experiment. Translating source files changes their fingerprints, so its reports and protocol were rebuilt together. This does not turn the known 2025/26 results into a new unseen test. See [TRANSLATION_NOTES.md](docs/TRANSLATION_NOTES.md) for provenance and verification.

## Fixed settings

| Setting | Value |
|---|---|
| Selected model | Jointly fitted independent Poisson |
| History window | 730 days |
| Time-weighting half-life | 180 days |
| L2 strength | 1 |
| Primary metric | 1X2 log loss |
| Primary reference | League average |
| Second reference | Time-weighted and smoothed model |

Hyperparameters stay fixed. Learned team effects, the goal intercept and home advantage are re-estimated before each matchday. All same-day results are excluded. Earlier completed test matches may enter later history.

## Quick start

Extract the complete project into its own folder. On Windows, open a terminal in that folder and create a virtual environment:

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m unittest discover -s tests
venv\Scripts\python.exe predict_fitted_poisson.py
```

In PyCharm, select the project's `venv\Scripts\python.exe` as its interpreter. On Linux or macOS, use `python3.12 -m venv venv` and `venv/bin/python` for the subsequent commands.

The bundled frozen report chain was verified with **Python 3.12.14**, Pandas 2.2.3, NumPy 2.3.5 and SciPy 1.15.3. Checking the cached final test requires that recorded numerical environment, including the Python patch version:

```powershell
venv\Scripts\python.exe prepare_final_test_data.py
venv\Scripts\python.exe run_final_test.py
```

The standalone match demonstrations do not require a matching frozen protocol. To rebuild the experiment in a different compatible environment, use the separate-copy procedure in [RUN_GUIDE.md](RUN_GUIDE.md). Do not overwrite the recorded environment or mix English sources with the older Hungarian protocol.

## Data and reports

Source: [football-data.co.uk – English leagues](https://www.football-data.co.uk/englandm.php). Required raw fields are date, home team, away team, home full-time goals and away full-time goals. The archived CSVs cover 2021/22 through 2025/26.

| Location | Contents |
|---|---|
| `data/raw/` | Five original season CSVs and source information |
| `data/processed/` | Cleaned data with stable match identifiers and preparation manifests |
| `data/reports/model_selection/` | Final-test protocol and model-selection evidence |
| `data/reports/final_test/` | Predictions, metrics, calibration, uncertainty and conclusions |
| `tests/` | Automated checks of chronology, mathematics, validation and reproduction |
| `docs/` | Learning guide, translation provenance and verification details |

Key outputs:

- [Final-test conclusions](data/reports/final_test/final_test_conclusions_2025_26.md)
- [Final summary](data/reports/final_test/final_summary_2025_26.csv)
- [Detailed final predictions](data/reports/final_test/final_predictions_2025_26.csv)
- [Bootstrap intervals](data/reports/final_test/bootstrap_intervals_2025_26.csv)

## Verification and Python skills demonstrated

Run the complete suite with `python -m unittest discover -s tests`. The assembled English project has **334 passing tests**, including tests from the earlier chapters. The original Step 19 verification copy contained 217 tests; the larger count comes from collecting the previously separate tests.

The final backtest accepted 114/114 matchday fits. The maximum absolute gradient was 8.902e−05, below the fixed 1e−04 threshold.

The project demonstrates Pandas cleaning and concatenation, date handling, modular functions, probability calculations, NumPy operations, SciPy optimisation, CSV/JSON reporting, Matplotlib figures, error handling and automated tests. File fingerprints connect inputs, source code, protocol and reports.

## Limitations

This is a historical portfolio analysis based on final scores. The pre-match model uses team strength and home advantage; it does not model squads, injuries or shot quality. Goal-market and draw calibration show scope for further development.

The bootstrap examines loss differences for fixed forecasts. It excludes model-selection and refitting uncertainty. Secondary comparisons have no multiple-comparison adjustment.

Once the 2025/26 results have been reviewed, that season cannot serve as an untouched test for a new model selection. Future modelling changes require a new development cycle and an appropriately separated evaluation.

Detailed assumptions: [MODEL_CARD.md](MODEL_CARD.md). Reproduction instructions: [RUN_GUIDE.md](RUN_GUIDE.md). Step-by-step explanations: [LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md).
