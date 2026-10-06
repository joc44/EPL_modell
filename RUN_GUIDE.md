# Running and Reproducing the Project

This guide applies to the complete English release. All current source modules, tests, required raw season CSVs and generated reports are included. There is no need to merge the earlier chapter archives.

## 1. Extract and install

Extract `epl-prediction-model-english.zip` into a new folder, keeping its directory structure. Keep your completed Hungarian project separately.

Verified environment: Python 3.12.14, Pandas 2.2.3, NumPy 2.3.5, SciPy 1.15.3 and Matplotlib 3.10.8. In a Windows terminal opened in the extracted project root:

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m unittest discover -s tests
```

`py -3.12` selects the installed Python 3.12 interpreter. Check the actual patch version with `venv\Scripts\python.exe --version`; cached protocol validation requires the recorded version, 3.12.14.

In PyCharm, select this project's `venv\Scripts\python.exe` as the interpreter and use the project root as the working directory. On Linux or macOS, create the environment with `python3.12 -m venv venv` and replace the Windows executable path with `venv/bin/python`.

The complete requirements file includes the former Step 11 SciPy dependency. `requirements_step_11.txt` remains available for compatibility with the learning scripts' installation messages.

## 2. Try the match demonstrations

The bundled processed data support the historical Arsenal–Nottingham Forest example without building the report chain first:

| Program | Demonstration |
|---|---|
| `analyze_match.py` | Pre-match historical statistics |
| `predict_match.py` | Raw ratio-based expected goal counts |
| `predict_probabilities.py` | Poisson 1X2, goal markets and exact scores |
| `predict_smoothed.py` | Raw versus smoothed estimates |
| `predict_time_weighted.py` | Time-weighted and smoothed estimates |
| `predict_fitted_poisson.py` | Jointly fitted attack–concede Poisson |
| `predict_dixon_coles.py` | Jointly fitted DC and low-score correction |

For example:

```powershell
venv\Scripts\python.exe predict_fitted_poisson.py
```

The demonstration programs do not require an exact match to the final protocol's runtime. Their fixture settings are defined near the start of the relevant scripts; later chapter scripts use their own example settings. They analyse historical fixtures rather than a live fixture feed.

Changing a frozen Python source file invalidates the corresponding report fingerprints. Use a separate copy for model development or changes to frozen modules.

## 3. Check the bundled final results

With the recorded numerical environment, run:

```powershell
venv\Scripts\python.exe prepare_final_test_data.py
venv\Scripts\python.exe run_final_test.py
```

The preparation step validates the protocol and already prepared data. The final runner validates saved predictions and recalculates their reports without refitting the models. It does not overwrite different existing final-test results.

Expected summary:

- 380 evaluated matches and 1140 prediction rows.
- 114 accepted Poisson matchday fits.
- 1X2 log loss: league average 1.0831; time-weighted 1.0377; Poisson 1.0344.
- Primary mean difference −0.048683; five-day 95% interval [−0.082760; −0.010868].

The English edition has its own internally consistent source fingerprints and protocol identifier. Its identifier differs from the Hungarian edition because translated source bytes and regenerated report context differ.

## 4. Rebuild the entire experiment in a separate copy

Use this route for a fresh report build, including in a different compatible environment. Do not regenerate earlier reports inside the already frozen release: some runners recreate settings files, changing inputs expected by later checkpoints.

Create a new empty project folder. Copy the root Python modules, `tests/`, requirements files, documentation and **only `data/raw/`** into it. Do not copy generated files from `data/processed/` or `data/reports/`. Create its own virtual environment and install `requirements.txt`.

Run the following programs in order, using that environment's Python. For example, use `venv\Scripts\python.exe main.py` for the first row.

| Order | Program | Output or prerequisite built |
|---:|---|---|
| 1 | `main.py` | 1140 cleaned matches from the first three seasons |
| 2 | `run_backtest.py` | Four-model development reports for 2023/24 |
| 3 | `prepare_validation_data.py` | 2024/25 added: 1520 matches |
| 4 | `validate_next_season.py` | Four-model reports for 2024/25 |
| 5 | `run_fitted_backtest.py` | Five models on both development seasons |
| 6 | `run_dixon_coles_backtest.py` | Six models on both development seasons |
| 7 | `analyze_dixon_coles_uncertainty.py` | Paired DC uncertainty reports |
| 8 | `freeze_final_test.py` | Reproduced selection and frozen final-test protocol |
| 9 | `prepare_final_test_data.py` | Five seasons and 1900 matches, with final-data manifest |
| 10 | `run_final_test.py` | Final evaluation of 380 fixtures and three models |

Optional learning reports:

- After order 2: `analyze_calibration.py` produces Step 9 calibration tables and PNG figures.
- After order 5: `analyze_uncertainty.py` produces Step 13 comparisons of jointly fitted Poisson with the earlier models.
- The match demonstrations can run after the cleaned data exist. They produce their own outputs, where applicable.

The complete test command is `python -m unittest discover -s tests`. Tests validate behaviour; they do not generate the complete historical report chain for you.

Rebuilding now reproduces a known experiment. A new protocol timestamp does not make the already reviewed 2025/26 results an unseen test.

## 5. Included source data

Source: [football-data.co.uk English leagues](https://www.football-data.co.uk/englandm.php).

| Season | Archived filename | Source CSV |
|---|---|---|
| 2021/22 | `E0_2021_2022.csv` | [2122/E0.csv](https://football-data.co.uk/mmz4281/2122/E0.csv) |
| 2022/23 | `E0_2022_2023.csv` | [2223/E0.csv](https://football-data.co.uk/mmz4281/2223/E0.csv) |
| 2023/24 | `E0_2023_2024.csv` | [2324/E0.csv](https://football-data.co.uk/mmz4281/2324/E0.csv) |
| 2024/25 | `E0_2024_2025.csv` | [2425/E0.csv](https://football-data.co.uk/mmz4281/2425/E0.csv) |
| 2025/26 | `E0_2025_2026.csv` | [2526/E0.csv](https://football-data.co.uk/mmz4281/2526/E0.csv) |

The current model requires `Date`, `HomeTeam`, `AwayTeam`, `FTHG` and `FTAG`. Other provider columns are retained in the raw archive and are not model inputs.

If the local 2025/26 raw file is missing during a fresh build, the final-data preparer downloads it after protocol validation. The distributed release already includes it. Provider corrections can change future downloads, so preserve the archived raw files for exact reproduction.

## 6. Troubleshooting and decision points

| Message or issue | Action |
|---|---|
| Missing Pandas, NumPy, SciPy or Matplotlib | Confirm the project interpreter is selected; install requirements into that environment. |
| Missing raw CSV | Check its filename and the `data/raw/` directory. |
| Missing previous report | Complete the previous row in the rebuild sequence. |
| Runtime differs from the protocol | Use the recorded environment, or rebuild the full chain in a clean separate copy. |
| Source or input fingerprint differs | Restore the release files, or rebuild in a separate copy after documenting the change. |
| Existing final file differs | Keep the recorded output; do not manually change fingerprints to bypass the check. |
| Numerical fit fails | Preserve the error details and inspect inputs; the run must not silently omit a fixture. |

The Python patch version is deliberately part of the recorded numerical environment. Even when package versions match, a different patch version will fail the cached-protocol check. Standalone demonstrations and a clean rebuild are separate from that check.

New model ideas informed by the final results belong to another development cycle. Preserve the published v1 results and protocol.

Project overview: [README.md](README.md). Model assumptions: [MODEL_CARD.md](MODEL_CARD.md). Learning path: [LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md).
