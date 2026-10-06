# The separate 2025/26 final test

Protocol: `2ffa301e859eb9fa19a54b8561b8ff02aa4ef4ae0ba43be05b4898c1df88d3ee`

Selected model: jointly fitted Poisson. Primary comparison: 1X2 log loss against the league-average reference.

| Model | Matches | 1X2 log loss | 1X2 Brier | 1X2 accuracy |
|---|---:|---:|---:|---:|
| League average | 380 | 1.083093 | 0.655575 | 42.63% |
| Time-weighted | 380 | 1.037704 | 0.623528 | 50.00% |
| Jointly fitted | 380 | 1.034410 | 0.620376 | 47.11% |

Primary difference: -0.048683; five-matchday 95% interval: [-0.082760; -0.010868].

Observed point estimate: favours the selected model.
Primary interval: supports the direction of lower loss.
All three block lengths: support the same direction.

## All tasks and both references

| Task | Reference | Mean difference | Five-day 95% interval |
|---|---|---:|---:|
| 1X2 | League average | -0.048683 | [-0.082760; -0.010868] |
| Over/under 1.5 | League average | +0.012014 | [-0.006122; +0.030145] |
| Over/under 2.5 | League average | +0.019889 | [+0.004361; +0.037547] |
| Over/under 3.5 | League average | +0.011047 | [-0.006918; +0.031809] |
| BTTS | League average | +0.000359 | [-0.019541; +0.019528] |
| Exact score | League average | -0.031455 | [-0.068922; +0.010792] |
| 1X2 | Time-weighted | -0.003294 | [-0.019221; +0.019777] |
| Over/under 1.5 | Time-weighted | +0.006054 | [-0.004076; +0.019477] |
| Over/under 2.5 | Time-weighted | +0.011117 | [-0.006720; +0.027650] |
| Over/under 3.5 | Time-weighted | +0.007366 | [-0.014765; +0.027166] |
| BTTS | Time-weighted | -0.001710 | [-0.016426; +0.014054] |
| Exact score | Time-weighted | +0.008680 | [-0.017760; +0.039724] |

Negative difference: favours the selected Poisson. Other tasks and the second reference are secondary evaluations.

Every match has equal weight. Same-day results are excluded; earlier completed test matches may enter later history.

Block lengths are 3, 5 and 10 matchdays; 5000 resamples and 95% percentile intervals. Intervals are individual, with no multiple-comparison adjustment.

The calculation is conditional on loss differences for fixed forecasts. It excludes model-selection and refitting uncertainty. An interval containing zero does not establish equivalence.

No model was reselected or hyperparameter tuned from the test. Repeating the season evaluation is reproduction; changes informed by these results begin a new development cycle.

This is a retrospective, chronologically simulated test on a historical season. It is not evidence of forecasts published in real time.
