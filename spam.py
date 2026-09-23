"""Reproducible SMS spam benchmark; no email-performance claim."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip'

def normalize(text):
    return ' '.join(text.casefold().split())

def deduplicate(rows):
    groups = {}
    for label, text in rows:
        key = normalize(text)
        if key in groups and groups[key][0] != label:
            raise ValueError('Conflicting labels for normalized duplicate')
        groups.setdefault(key, (label, text))
    return list(groups.values())

def validate_text(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError('text must be a nonempty string of at most 10000 characters')
    return text

def metrics(y, pred):
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                confusion_matrix_tn_fp_fn_tp=confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist())

def train():
    data = ROOT / 'data'; data.mkdir(exist_ok=True)
    archive = data / 'sms.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = z.read('SMSSpamCollection').decode('utf-8').splitlines()
    rows = []
    for line in lines:
        label, text = line.split('\t', 1)
        if label not in ('ham', 'spam'): raise ValueError('Unknown label')
        rows.append((int(label == 'spam'), text))
    unique = deduplicate(rows)
    y, x = zip(*unique)
    train_ids, test_ids = train_test_split(np.arange(len(x)), test_size=.2, stratify=y, random_state=42)
    fit_ids, val_ids = train_test_split(train_ids, test_size=.25, stratify=np.array(y)[train_ids], random_state=42)
    sets = [set(normalize(x[i]) for i in ids) for ids in (fit_ids, val_ids, test_ids)]
    assert all(not sets[a] & sets[b] for a,b in ((0,1),(0,2),(1,2)))
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True), LogisticRegression(C=4, max_iter=1000, random_state=42))
    model.fit([x[i] for i in fit_ids], [y[i] for i in fit_ids])
    val_y = np.array(y)[val_ids]
    scores = model.predict_proba([x[i] for i in val_ids])[:,1]
    # Choose threshold once on validation data, favor precision via F0.5.
    thresholds = np.linspace(.1,.9,81)
    from sklearn.metrics import fbeta_score
    threshold = float(max(thresholds, key=lambda t: fbeta_score(val_y, scores >= t, beta=.5, zero_division=0)))
    test_y = np.array(y)[test_ids]
    test_scores = model.predict_proba([x[i] for i in test_ids])[:,1]
    report = dict(dataset_url=URL, dataset_sha256=hashlib.sha256(raw).hexdigest(), raw_rows=len(rows), unique_rows=len(unique),
                  duplicates_removed=len(rows)-len(unique), split_sizes=dict(train=len(fit_ids), validation=len(val_ids), test=len(test_ids)),
                  normalized_duplicate_overlap=0, seed=42, threshold=threshold,
                  validation=metrics(val_y, scores>=threshold), test=metrics(test_y, test_scores>=threshold),
                  test_average_precision=float(average_precision_score(test_y,test_scores)), majority_baseline=metrics(test_y,np.zeros(len(test_y))),
                  limitations=['SMS benchmark, not email validation', 'Near duplicates and source/time shifts not eliminated', 'Single held-out split; no production claim'])
    artifact = ROOT / 'artifacts'; artifact.mkdir(exist_ok=True)
    joblib.dump(dict(model=model, threshold=threshold), artifact/'model.joblib')
    (artifact/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (artifact/'split_ids.json').write_text(json.dumps(dict(train=fit_ids.tolist(),validation=val_ids.tolist(),test=test_ids.tolist())),encoding='utf-8')
    print(json.dumps(report,indent=2))

def predict(text, bundle=None):
    validate_text(text)
    bundle = bundle or joblib.load(ROOT/'artifacts/model.joblib')
    score = float(bundle['model'].predict_proba([text])[0,1])
    return dict(label='spam' if score >= bundle['threshold'] else 'ham', spam_score=score, threshold=bundle['threshold'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train','predict'])
    parser.add_argument('--text')
    args = parser.parse_args()
    if args.command == 'train': train()
    else: print(json.dumps(predict(args.text)))
