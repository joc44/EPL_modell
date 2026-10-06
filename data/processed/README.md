# Processed Data

| File | Matches | Role |
|---|---:|---|
| `matches_clean.csv` | 1140 | Initial three seasons, 2021/22–2023/24 |
| `matches_validation.csv` | 1520 | Four seasons, adding 2024/25 |
| `matches_final_test.csv` | 1900 | Five seasons, adding 2025/26 |

Rows have stable identifiers of the form `EPL|Season|YYYY-MM-DD|HomeTeam|AwayTeam`. Cleaning standardises known team aliases, parses dates, validates non-negative goals and removes exact duplicates while rejecting conflicting fixtures.

The validation and final preparation manifests record input and output fingerprints. The earlier seasons are preserved when extending the data. A target match uses only rows with `target_date - 730 days <= Date < target_date`; all same-day results are excluded.

Preparation programs create these files in a fresh project copy. In the frozen release, the final-data preparer checks existing content and refuses to overwrite different final files.
