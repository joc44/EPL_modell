# Generated Reports

| Location | Evaluation |
|---|---|
| Root backtest CSVs and settings | Four original models on 2023/24 and 2024/25 |
| `diagnostics/` | Earlier calibration and grouped-error analysis |
| `fitted_poisson/` | One-match joint-Poisson demonstration |
| `fitted_backtest/` | Five-model development comparison |
| `uncertainty/` | Paired uncertainty for the jointly fitted model |
| `dixon_coles/` | One-match DC demonstration and low-score effects |
| `dixon_coles_backtest/` | Six-model comparison and correction decomposition |
| `dixon_coles_uncertainty/` | Paired DC uncertainty |
| `model_selection/` | Selection evidence and reproduced frozen protocol |
| `final_test/` | Three-model final evaluation on 2025/26 |

Predictions and metrics use stable machine-readable model identifiers. Chart labels and explanatory output are in English. CSV reports use UTF-8 with a byte-order mark for spreadsheet compatibility, commas as separators and a decimal point.

Lower log loss and Brier scores are better. Pairwise differences are selected/new model minus reference, so a negative value favours the selected/new model. Intervals are individual; secondary comparisons have no multiple-comparison correction.

The English report chain reproduces the completed experiment. Translation changes source fingerprints, so its manifests and protocol belong to the English edition. Preserve the chain together; do not mix it with Hungarian source files or earlier protocols.
