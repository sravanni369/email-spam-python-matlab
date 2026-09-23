# Source traceability

Book: **Approaching (Almost) Any Machine Learning Problem**, Abhishek Thakur.

Relevant sections: **Cross-validation**, printed/PDF pages 23 and 25 (stratification and hold-out validation); **Evaluation metrics**, printed/PDF pages 37–39 (precision, recall and threshold tradeoffs). These pages were extracted and visually inspected. Printed page numbers match the PDF page positions in this copy. No book pages are redistributed in the deliverables.

Adaptation: use those validation and metric concepts in a practical email-spam experiment. All project code is newly written; it is not a reproduction of a book listing. The 60/20/20 split, exact-feature duplicate audit, validation F0.5 threshold objective and local service are project design choices. A single hold-out experiment does not establish robust cross-validation performance.

Dataset: Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). [Spambase](https://doi.org/10.24432/C53G6X), UCI Machine Learning Repository. [Dataset page and CC BY 4.0 license](https://archive.ics.uci.edu/dataset/94/spambase). Download ZIP and its SHA-256 are preserved. Data changes: exclude three ambiguous feature groups (six rows); remove 391 redundant same-label rows; retain 4,204 representatives.

Spambase contains 57 precomputed email features, not raw message text. Its documentation warns that personal indicators such as `george` and `650` are specific to its source. The benchmark dates from 1999, so deployment to modern email requires additional validation. Excluding ambiguous groups changes the evaluation population and can make evaluation easier; the excluded counts are disclosed rather than silently dropping them.

Implementation reference: [MathWorks fitclinear](https://www.mathworks.com/help/stats/fitclinear.html). MATLAB uses ridge-regularized logistic regression, LBFGS and train-only standardization. Python uses scikit-learn's logistic regression pipeline. Different solvers/default tolerances can produce differences; matching a split is not a claim of identical numerical optimization.

No novelty or tutorial reproduction is claimed. The earlier SMS milestone remains separately labeled in `SMS_MILESTONE.md` and `artifacts/`.
