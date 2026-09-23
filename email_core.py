"""Analytical core informed by Thakur's validation and evaluation chapters."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import fbeta_score

def fit_email(x, y, seed=42):
    ids = np.arange(len(y))
    train, test = train_test_split(ids, test_size=.2, stratify=y, random_state=seed)
    train, valid = train_test_split(train, test_size=.25, stratify=y[train], random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=seed))
    model.fit(x[train], y[train])
    baseline = DummyClassifier(strategy='most_frequent').fit(x[train], y[train])
    validation_scores = model.predict_proba(x[valid])[:, 1]
    thresholds = np.linspace(.1, .9, 81)
    objective = [fbeta_score(y[valid], validation_scores >= t, beta=.5, zero_division=0) for t in thresholds]
    threshold = float(thresholds[np.argmax(objective)])
    test_scores = model.predict_proba(x[test])[:, 1]
    return {
        'model': model,
        'threshold': threshold,
        'train': train,
        'valid': valid,
        'test': test,
        'validation_scores': validation_scores,
        'test_scores': test_scores,
        'predictions': (test_scores >= threshold).astype(int),
        'baseline_predictions': baseline.predict(x[test]),
    }
