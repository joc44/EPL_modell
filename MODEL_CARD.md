# Model Card — EPL Poisson v1

## Purpose and scope

The model estimates pre-match goal counts and result probabilities for Premier League fixtures from historical final scores. The project demonstrates Python data processing, statistical modelling and chronological evaluation.

Version 1 is evaluated retrospectively. The backtest is not a record of predictions published before the actual matches. The English release reproduces the completed experiment with translated source text and rebuilt fingerprints.

## Inputs and outputs

Required raw inputs: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`. Cleaned data add the season and a stable `match_id`.

Team-name standardisation, match identification, missing-value checks, duplicate handling, goal validation and complete-season checks precede evaluation. The final data file contains 1900 matches from five seasons; 380 matches from 2025/26 are evaluated.

Outputs are expected home and away goal counts, 1X2 probabilities, over/under 1.5/2.5/3.5, BTTS and exact-score probabilities. Every event comes from the same match-level joint score distribution.

## Model structure

The selected model is a jointly fitted, time-weighted, L2-regularised independent Poisson model:

- Log expected home goals = goal intercept + shared home advantage + home attack effect + away concede effect.
- Log expected away goals = goal intercept + away attack effect + home concede effect.

A positive concede effect means more expected goals conceded. Attack log effects and concede log effects each sum to zero, making the parameterisation identifiable. The L2 penalty shrinks team effects towards neutral zero. The goal intercept and shared home advantage receive no L2 penalty.

The optimiser minimises the sum of weighted negative log likelihood and the L2 penalty. Because the objective is a weighted sum, L2 strength 1 is tied to this particular scaling. It is a separate parameter from the earlier ratio models' five-match smoothing weight.

## Fitting rules

| Rule | Value |
|---|---|
| History window | 730 days; lower boundary included |
| Upper date boundary | Target matchday excluded |
| Half-life | 180 days |
| L2 strength | 1 |
| Fit frequency | One new fit before each target matchday |
| Unknown team | Neutral log effects with an explicit unknown-team flag |
| Failed fit | Stop without skipping matches or substituting models |
| Numerical acceptance | Maximum absolute gradient at most 1e−04 |

The fitting team universe comes exclusively from history. Earlier completed test matches may enter the history for later dates. Hyperparameters do not change during the test.

## Probability and evaluation rules

Conditional on the parameters, team goal counts are independent Poisson variables. Event probabilities sum cells in a normalised score table with sufficient coverage. The omitted-tail tolerance is 1e−10 per team, and the computation limit is 200 goals.

The observed exact-score probability is evaluated from the untruncated analytic Poisson product. Log loss uses a probability floor of 1e−15 and reports floor usage separately for every task. None of the three models required clipping for any task in the final run.

The 1X2 Brier score sums three squared errors and has scale 0–2. Binary Brier scores use one squared error and have scale 0–1. Each test match receives equal weight.

Calibration uses ten probability bins and flags bins with fewer than twenty matches. ECE is the match-count-weighted absolute bin-level deviation and serves as a supplementary descriptive measure.

## Development and selection

The 2023/24 and 2024/25 seasons were used for development comparisons. The model family included league average, raw ratios, smoothed ratios, time-weighted ratios, jointly fitted Poisson and Dixon–Coles.

Poisson gave lower 1X2 log loss point estimates than the four original models in both seasons. Intervals for the small additional DC improvement contained zero in both seasons and at all three block lengths. Considering simplicity, Poisson was selected for the separate test. This does not establish optimal hyperparameters or superiority for every task.

Step 17 recorded the original selection and final-test rules. No new model or hyperparameters were selected from the 2025/26 results. The English protocol is a reproduction of those rules after the results were known; its new timestamp is not a new preregistration.

## Main final-test results

| Task: log loss | League average | Time-weighted | Selected Poisson |
|---|---:|---:|---:|
| 1X2 — primary | 1.0831 | 1.0377 | 1.0344 |
| Over/under 1.5 | 0.5154 | 0.5213 | 0.5274 |
| Over/under 2.5 | 0.6916 | 0.7004 | 0.7115 |
| Over/under 3.5 | 0.6085 | 0.6122 | 0.6195 |
| BTTS | 0.6911 | 0.6932 | 0.6915 |
| Exact score | 2.9537 | 2.9136 | 2.9223 |

The primary Poisson-minus-league-average 1X2 difference is −0.048683, with a five-day 95% interval of [−0.082760; −0.010868]. All three block lengths support the favourable direction. Against the time-weighted reference, the difference is −0.003294, with interval [−0.019221; +0.019777].

For over/under 2.5, the difference against league average is +0.019889, with a five-day interval of [+0.004361; +0.037547]. This secondary comparison favours league average. The complete task table and intervals against both references remain in the final reports.

Accepted matchday fits: 114/114. Maximum absolute gradient: 8.902e−05. Poisson predictions without any team history: two home-team cases and zero away-team cases. These full-team flags differ from venue-specific history counts used by the earlier ratio models.

## Calibration findings and limitations

Mean predicted draw probability is 23.84%, against an observed rate of 27.37%: an average underestimate of 3.53 percentage points. Mean over 2.5 probability is 52.80%, against 55.00% observed, while its ECE is 11.33 percentage points. A close season average does not replace bin-level calibration checks.

The current model does not separately track squads, injuries, starting line-ups, red cards or shot quality. Conditional goal independence and common home advantage are simplifying assumptions. Rolling history and time weighting can only partially follow changes in the league.

## Uncertainty and reproduction

Model losses are paired by match identifier. The non-circular overlapping matchday block bootstrap uses block lengths 3, 5 and 10, 5000 resamples, 95% percentile intervals and seed `42 + block_length` for the final season. All matches on a day stay together; the statistic uses equal match weights. Five-day blocks are the primary presentation.

Intervals are conditional on the fixed forecasts. They exclude model-selection and refitting uncertainty. The procedure assumes a roughly stable loss-difference process with short-range dependence. Secondary comparisons have no multiple-comparison adjustment. An interval containing zero does not establish equivalence.

Source, input, protocol and report fingerprints can be checked. Local fingerprints are not an external timestamp or digital signature. Changes informed by the 2025/26 results start a new development cycle; repeating the same experiment is reproduction.

## Technical environment and verification

Verified environment: Python 3.12.14, Pandas 2.2.3, NumPy 2.3.5, SciPy 1.15.3 and Matplotlib 3.10.8. The English project's assembled suite has 334 passing tests. Mathematical and control-flow structure were checked against the original sources with text constants excluded from that comparison.

The English report chain is regenerated with its translated source fingerprints. Reproduction and numerical comparison details are in [TRANSLATION_NOTES.md](docs/TRANSLATION_NOTES.md). Run order: [RUN_GUIDE.md](RUN_GUIDE.md).
