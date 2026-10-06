# Learning Guide: From CSV Files to a Reproduced Final Test

This guide collects the reasoning behind the project's learning steps. The complete release contains the latest module versions, so some early examples now share the later implementations. Start with [RUN_GUIDE.md](../RUN_GUIDE.md) for execution order.

The project asks a probability question: given only information available before a matchday, what distribution of final scores does the model predict? A forecast is a distribution, and its evaluation uses the probability assigned to what actually happened.

## Steps 1–2: Read and clean historical data

The first exercise read one season and checked match counts, team counts and goal averages. The final `main.py` loads three seasons, standardises known team-name aliases, parses day/month/year dates, validates goals and creates stable match identifiers.

Exact duplicate rows can be removed and counted. Conflicting results for the same fixture stop processing, because silently choosing one would make the input ambiguous. The cleaned CSV separates reusable input preparation from later modelling.

Python concepts: paths, dictionaries, functions, Pandas selection, data types, date parsing, concatenation, sorting, validation and CSV output.

Decision: preserve raw data and write processed data separately. This allows cleaning decisions to be inspected and the experiment to be reproduced.

## Step 3: Define what was known before the match

`analyze_match.py` uses the window `target_date - 730 days <= Date < target_date`. The lower date is included; the entire target day is excluded. This avoids relying on match kick-off times that the current workflow does not use.

Compute league goal averages, the home team's home statistics and the away team's away statistics only from this history. A team with no matching history is different from one with observed zero goals. Missing history is flagged explicitly.

Python concepts: filtered DataFrames, timestamps, reusable statistics and structured dictionaries.

Trade-off: excluding the whole day is conservative. Even an earlier completed same-day match is unavailable to this model.

## Step 4: Build a simple expected-goals baseline

`goal_model.py` compares team averages with the correct venue-specific league averages. A team's attacking multiplier is its goals-scored average divided by the corresponding league scoring average. Its concede multiplier measures goals allowed relative to the league rate for that scoring side.

Expected home goals combine the home league average, the home attack multiplier and the away concede multiplier. Expected away goals combine the away league average, the away attack multiplier and the home concede multiplier.

A concede multiplier greater than one means weaker defence in this sample. Missing history uses neutral multipliers. An observed average of zero remains zero in the raw model.

Trade-off: this baseline is easy to explain, but opponent strength and small samples can distort team averages.

## Step 5: Convert expected goals into event probabilities

`score_probabilities.py` builds a Poisson distribution for each team. Multiplying home and away probabilities gives the joint score table under the independence assumption.

Sum the appropriate cells to obtain home win, draw, away win, goal thresholds and BTTS. The most probable exact score is the highest-probability cell. Even the top score is only one possible outcome, not a promise of the result.

The grid expands until its omitted probability is sufficiently small and is then normalised. Exact-score loss is evaluated analytically, so a high observed score outside the finite display table can still be evaluated correctly.

Python concepts: numerical loops, nested structures, probability sums, sorting and precision checks.

Trade-off: independent Poisson is compact and coherent, but does not capture every relationship between the two teams' goals.

## Step 6: Smooth sparse team histories

`shrinkage_model.py` adds league-average prior weight to the observed goal totals and match counts. The initial learning setting is equivalent to five league-average matches.

With many observed matches, team data dominate. With few matches, the league average contributes more. With no history, the estimate becomes the league average. Smoothing also prevents a small sample with zero scored goals from automatically making all scoring events impossible.

Decision: treat prior weight as a model setting, and evaluate its effect rather than assuming smoothing always improves every task.

## Step 7: Give recent results greater weight

`time_weighted_model.py` assigns weight `2 ** (-age_days / half_life_days)`. With a 180-day half-life, a match 180 days old has half the weight of a current-date reference point. The history still excludes the target day.

Weighted goal totals and total weights replace integer totals in smoothing. Total weight is not the actual number of matches. The prior therefore has more influence as old results lose weight.

Trade-off: short half-life follows change more quickly while reducing the usable information in old matches. Half-life and smoothing interact and require evaluation.

## Step 8: Evaluate a full season chronologically

`run_backtest.py` evaluates the four initial models on 2023/24. For each fixture, complete the forecast first, then attach the observed result. Keep the same fixtures and metric definitions for every model.

Log loss rewards probability assigned to the realised event and penalises confident mistakes. Brier scores measure squared probability error. Accuracy considers only the most probable class. These metrics answer related but different questions.

The probability floor applies only during log loss evaluation; it does not change the saved forecast. Every use of the floor is counted. Failed predictions stop the run instead of silently reducing the sample.

Python concepts: iteration, callbacks, report records, summary tables, JSON settings and independent evaluation functions.

## Step 9: Inspect calibration and errors

`analyze_calibration.py` groups saved probabilities into bins and compares their means with observed event rates. A model that assigns 60% probability to a group should observe the event roughly 60% of the time in comparable repeated cases.

Small bins fluctuate more. The analysis flags fewer than twenty observations and keeps empty bins visible. ECE summarises absolute bin-level deviation but depends on binning and sample size.

Inspecting large losses and sparse-history groups helps locate failure modes. A close overall forecast average does not guarantee good calibration in every probability range.

## Step 10: Check a later season

`prepare_validation_data.py` adds 2024/25 while preserving every field from the earlier data. `validate_next_season.py` uses the same model settings on that later season.

Initially this was later-season validation. Once these results informed subsequent model comparisons, both 2023/24 and 2024/25 became development data. The project's documentation records that role change.

Decision: an evaluation set ceases to be untouched once it informs model selection or modification.

## Step 11: Jointly estimate attack and defence

`fitted_poisson_model.py` estimates all team effects together:

- Home log rate: intercept + home advantage + home attack + away concede effect.
- Away log rate: intercept + away attack + home concede effect.

The shared fit accounts for the opponents in each result. Identifiability constraints centre attack and concede effects separately. L2 regularisation shrinks team effects towards neutral zero; the intercept and home advantage are unpenalised.

The objective uses time weights and an analytic gradient. Numerical diagnostics check optimisation and precision. They do not establish predictive quality.

Trade-off: this is more expressive than average ratios and uses matches at both venues, but introduces optimisation, regularisation and numerical validation.

## Step 12: Compare the fitted model on both development seasons

`run_fitted_backtest.py` refits before each target matchday. All forecasts for that day are completed before its results are evaluated. Earlier forecasts are reused only after their data, settings and source fingerprints pass checks.

The joint model's 1X2 point estimates improve on the earlier four models in both development seasons. Gains on other tasks vary.

Decision: compare a new model on identical fixtures and do not equate a small metric difference with a well-established advantage.

## Step 13: Quantify paired differences

`analyze_uncertainty.py` pairs model losses by match identifier. Matchday blocks preserve all matches on each selected day and some short-range dependence between neighbouring days.

The moving block bootstrap resamples blocks of 3, 5 and 10 matchdays. It reports 95% percentile intervals from 5000 resamples. The mean difference is new model minus reference, so negative loss differences favour the new model.

An interval containing zero leaves the direction uncertain; it does not prove equivalence. The calculation is conditional on fixed forecasts and excludes model-selection and refitting uncertainty.

## Step 14: Add the Dixon–Coles low-score correction

`dixon_coles_model.py` fits team effects, home advantage, intercept and rho jointly. The tau correction modifies the 0–0, 0–1, 1–0 and 1–1 cells.

For this parameter convention, negative rho increases 0–0 and 1–1 probabilities and decreases 0–1 and 1–0 probabilities. Positive rho has the opposite effect. Positive tau constraints maintain valid probabilities.

At fixed lambdas the correction preserves total mass and Poisson marginals. It can affect draws, BTTS and over/under 1.5, while leaving over/under 2.5 and 3.5 unchanged. Joint refitting may change those latter markets through changed lambdas.

Decision: distinguish the effect of refitted rates from the effect of the correction itself.

## Steps 15–16: Evaluate DC and its uncertainty

`run_dixon_coles_backtest.py` extends the comparison to six models and records a decomposition into rate refitting and tau correction. `analyze_dixon_coles_uncertainty.py` examines the new paired differences.

The extra DC 1X2 advantage is small, and its intervals include zero in both development seasons at all three block lengths. The correction does not produce uniform improvements across tasks.

Trade-off: additional complexity is useful to study, but this evidence does not require selecting it for the final version.

## Step 17: Freeze the choice and evaluation rules

`freeze_final_test.py` selects the simpler jointly fitted Poisson for v1 and records the model, parameters, references, primary metric, secondary tasks, bootstrap rules, input fingerprints and runtime.

The selected settings are 730 days, a 180-day half-life and L2 strength 1. The primary comparison is 1X2 log loss against league average. Team parameters still update chronologically; hyperparameters stay fixed.

The English release reproduces this selection after the original final results were known. Its newly generated protocol is a consistency checkpoint, not a new preregistration.

## Step 18: Evaluate the separate 2025/26 season

`prepare_final_test_data.py` validates the protocol before processing the final season. `run_final_test.py` evaluates 380 fixtures for three models and reports every prespecified task.

Poisson improves the primary 1X2 comparison against league average; all three block lengths support that direction. Its incremental 1X2 advantage over the time-weighted model is uncertain. Over/under 2.5 is worse than league average in the five-day secondary comparison.

The results remain a historical simulation. Earlier completed test fixtures can enter later history, consistent with the fixed rolling-update policy.

## Step 19: Document the portfolio result

The README presents the concrete problem, pipeline and complete result. The model card states assumptions, evaluation rules and limitations. The run guide makes the work repeatable.

For a portfolio explanation, connect each Python skill to a task: Pandas for cleaning, NumPy for numerical tables, SciPy for fitting, Matplotlib for diagnostics, and CSV/JSON for inspectable reports. Explain both successful comparisons and weaker results.

A future interface can use the frozen v1 outputs. Model changes informed by the final season belong to a new development cycle with a suitably separated evaluation.
