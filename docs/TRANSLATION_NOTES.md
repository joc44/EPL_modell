# English Release: Translation and Reproduction Notes

## What is included

This release assembles the latest versions of 44 source modules and 25 test files from the completed learning project. It includes all five raw season CSVs, three processed match files, demonstration outputs, development comparisons, figures, uncertainty reports, model-selection evidence and final-test reports.

The main documentation is [README.md](../README.md), [MODEL_CARD.md](../MODEL_CARD.md) and [RUN_GUIDE.md](../RUN_GUIDE.md). The [learning guide](LEARNING_GUIDE.md) brings the earlier chapter explanations together around the current modules, rather than retaining obsolete instructions to merge chapter ZIPs.

Console output, error messages, comments, docstrings, chart labels and generated explanatory reports have been translated into English. Machine-readable identifiers, score data, team names, calculation settings and formulas are preserved.

Virtual environments, Python caches, editor files and temporary build logs are excluded from the archive. Install dependencies into a local virtual environment as described in the run guide.

## Why the protocol identifier changes

The original protocol records hashes of Python source files. Translating comments and strings changes those bytes. Reusing the old fingerprints with English sources would correctly fail validation.

The complete report chain was therefore rebuilt from the archived raw CSVs using the English sources. Its data-preparation manifests, report context, protocol and final outputs are internally consistent with those sources.

This is a reproduction of the completed v1 experiment after its results were known. The newly generated protocol timestamp does not mean the test was newly preregistered. No model or hyperparameter was selected from the test during translation.

Original verification protocol identifier:

`041d086aa4728fa13d88e95d454b756c18723255df2f2fd712dae59f16b14234`

English reproduction protocol identifier:

`2ffa301e859eb9fa19a54b8561b8ff02aa4ef4ae0ba43be05b4898c1df88d3ee`

The original identifier above belongs to the project verification copy used to build this release. A separate original user installation may have its own identifier because its environment and frozen inputs differ.

## Verification performed

| Check | Result |
|---|---|
| Complete assembled test suite | 334 tests passed |
| Current source and test files | 69 compared with their original versions |
| Non-text syntax-tree structure | Unchanged |
| Full pipeline and demonstration runs | All 19 entry points completed successfully |
| Shared original and English report CSVs | 63 compared; every numerical value matched exactly |
| Maximum observed numerical difference | 0.0 |
| Raw season CSVs | All five byte-identical to their archived sources |
| Processed match CSVs | All three byte-identical to the original processed files |
| Original verification project | All 149 recorded files unchanged |

The syntax-tree comparison preserves names, operators, control flow and numerical constants while excluding text constants from comparison. The tests and numerical report comparison provide additional checks that the translated text does not alter calculations.

The final backtest accepted 114/114 matchday fits and reproduced the 1X2 log loss values 1.0831, 1.0377 and 1.0344 for league average, time-weighted and jointly fitted Poisson respectively.

Detailed per-file hashes, report comparisons, recorded runtime and execution information are stored in [translation_provenance.json](translation_provenance.json). These local records support consistency checks; they are not external signatures or independent timestamps.

## Using the release

Extract it into a separate project folder. Keep the translated sources, processed data and report chain together. Do not copy the English modules over a Hungarian installation while retaining the old frozen protocol.

Use the matching recorded environment to validate bundled final results. In another compatible environment, rebuild the complete chain in a fresh separate copy containing only sources, tests and raw data as inputs.

The model assumptions and limitations remain the same as in the original v1. Future changes informed by the known final results are a new development cycle.
