# Decisions and failure analysis

- Business question: can a small, auditable classifier flag historical spam while making false-positive costs visible?
- Fit preprocessing only on training rows. Exact feature duplicates cannot cross splits. Remove conflicting-label groups rather than resolving them with a guessed label.
- Fix the model family and regularization before examining test metrics. Choose a threshold on validation F0.5, which weights precision more than recall. It is a portfolio design choice, not an established business cost model.
- Keep an always-ham baseline. It exposes why accuracy alone is inadequate: the baseline is 60.17% accurate while catching no spam.
- Lock the Python test at 841 emails. It flags 26 legitimate examples and misses 50 spam examples: false-positive rate 26/506 = 5.14%, false-negative rate 50/335 = 14.93%. There are 285 true positives and 480 true negatives. Do not tune against these observed test mistakes.
- Every held-out prediction and source row index is in `email_artifacts/test_predictions.csv`, with false positives and false negatives labeled. Raw email text is unavailable, so no invented explanation of individual messages is provided. The source-specific features and age of the data limit generalization.
- A bootstrap F1 interval describes uncertainty conditional on this fixed cleaned sample and trained model; it does not measure future inbox performance, source bias, or uncertainty in cleaning decisions.
- Local deployment is a demonstration with input validation. It does not accept raw messages, connect to an inbox, delete mail, or claim production readiness.
- Python's first email run rejected a feature-name parser that assumed one space after a colon. The parser was corrected to strip whitespace; the failure log is preserved. Local MATLAB R2025b failed to start with “File system inconsistency.” MATLAB Online R2026a was used instead.
