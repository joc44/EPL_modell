# Model selection and final-test freeze

Selected version: jointly fitted, time-weighted, L2-regularised independent Poisson.

The primary task is 1X2 and the primary metric is log loss. The main reference is league average; the second reference is time-weighted.

Fixed settings: 730-day history window, 180-day half-life and L2 strength 1.

Team effects and the goal intercept are re-estimated before every matchday from the history completed by then.

Development seasons: 2023/24 and 2024/25. This step does not read the 2025/26 test results.

Poisson 1X2 log loss is lower than that of the previous four models in both development seasons. The small additional DC advantage has intervals containing zero in both seasons and at all three block lengths.

Poisson is selected for its simpler joint fit and coherent score distribution across all markets. No general Poisson superiority or optimal hyperparameters are claimed.

| Development season | 1X2 log loss | BTTS log loss | Exact-score log loss |
|---|---:|---:|---:|
| 2023/24 | 0.936754 | 0.676418 | 3.095128 |
| 2024/25 | 0.975870 | 0.692765 | 2.979148 |

The final test requires the same 380 fixtures for three models. All results on a target matchday are excluded. A failed fit stops the run; no match is skipped or replaced by another model.

All goal markets, exact scores, Brier scores, accuracy and calibration are reported. The main conclusion uses the prespecified 1X2 metric and league-average comparison.

Uncertainty uses the same matchday block bootstrap: blocks of 3, 5 and 10 days, 5000 resamples and 95% percentile intervals. The main presentation uses five days.

No model is reselected or hyperparameter tuned from the test. Subsequent changes begin a new development cycle. Rerunning identical data and protocol is reproduction.

Protocol identifier: `2ffa301e859eb9fa19a54b8561b8ff02aa4ef4ae0ba43be05b4898c1df88d3ee`

Freeze time UTC: `2026-10-06T21:19:27+00:00`
