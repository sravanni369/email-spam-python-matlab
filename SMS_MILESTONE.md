# Project 1: spam detection — SMS milestone

Status: earlier SMS benchmark, kept for reference. The email benchmark in README.md is the main result. No public deployment or production claim.

## Reproduce

Use Python 3.12. From this directory:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python spam.py train
.venv/Scripts/python -m unittest -v
.venv/Scripts/python spam.py predict --text "Are we meeting at noon?"
```

Training downloads the public dataset if absent. Artifacts include the fitted model, split indices and metrics. Only load joblib files you trust. Inference is local and does not send messages to a service.

## Data and decisions

Almeida, T. & Hidalgo, J. (2011), [SMS Spam Collection](https://doi.org/10.24432/C5CC84), UCI Machine Learning Repository, CC BY 4.0. The original download is retained in data/sms.zip. Its SHA-256 is recorded in artifacts/metrics.json. Parsing produces 5,574 records; this is the observed file count, not a manually substituted catalog count.

Case-folded, whitespace-normalized duplicates are removed before stratified 60/20/20 train/validation/test splitting (seed 42). Conflicting duplicate labels cause an error. This removes 415 repetitions, leaving 5,159 unique messages. Normalized duplicate overlap across splits is zero; near-duplicate templates may remain.

TF-IDF unigrams/bigrams and logistic regression are fitted only on training data. C=4 is fixed; the threshold is selected from 0.10–0.90 on validation F0.5 to favor precision. No threshold is selected on test data. The vocabulary is never fitted on validation or test messages. Scores are not externally calibrated probabilities.

## Observed result, September 22, 2026

On 1,032 held-out SMS messages: precision 97.44%, recall 89.06%, F1 93.06%, accuracy 98.35%. There are 3 false positives and 14 false negatives. The always-ham baseline achieves 87.60% accuracy and 0 spam recall. Full validation/test metrics and the threshold are in artifacts/metrics.json.

The error counts matter: legitimate messages can be wrongly flagged, and some spam is missed. The dataset is old, contains SMS rather than email and lacks a time-based deployment evaluation. These results do not establish modern inbox performance.

This is an original local implementation. An initial test invocation used the wrong relative Python path; it was corrected before the recorded test run.
