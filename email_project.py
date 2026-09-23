"""Data preparation, audit and local feature-based email inference."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from email_core import fit_email
from spam import metrics

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'email_artifacts'
URL = 'https://archive.ics.uci.edu/static/public/94/spambase.zip'

def prepare(matrix):
    if matrix.ndim != 2 or matrix.shape[1] != 58 or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 57-feature rows plus binary labels')
    if not np.isin(matrix[:,-1], [0,1]).all():
        raise ValueError('Labels must be binary')
    groups = {}
    for i, row in enumerate(matrix):
        groups.setdefault(tuple(row[:-1]), []).append(i)
    conflicting = [ids for ids in groups.values() if len(set(matrix[ids,-1])) > 1]
    good = [ids[0] for ids in groups.values() if len(set(matrix[ids,-1])) == 1]
    return matrix[good,:-1], matrix[good,-1].astype(int), good, {
        'raw_rows':len(matrix), 'unique_feature_groups':len(groups),
        'conflicting_groups_excluded':len(conflicting),
        'conflicting_rows_excluded':sum(map(len,conflicting)),
        'redundant_same_label_rows_removed':len(matrix)-sum(map(len,conflicting))-len(good),
        'usable_rows':len(good)}

def feature_names(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.read('spambase.names').decode()
    return [line.split(':')[0].strip() for line in names.splitlines()
            if ':' in line and line.split(':',1)[1].strip() == 'continuous.']

def validate_features(values):
    if not isinstance(values,list) or len(values) != 57:
        raise ValueError('features must contain exactly 57 numbers in documented order')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values):
        raise ValueError('Every feature must be numeric')
    row = np.asarray(values,dtype=float)
    if not np.isfinite(row).all() or np.any(row<0) or np.any(row[:54]>100):
        raise ValueError('Use finite nonnegative features; frequencies must be <=100')
    if any(float(v).is_integer() is False for v in row[55:]):
        raise ValueError('Longest and total capital runs must be integers')
    if row[54]>row[55] or row[55]>row[56]:
        raise ValueError('Capital-run average <= longest <= total required')
    return row.reshape(1,-1)

def predict(values, bundle=None):
    row=validate_features(values)
    bundle=bundle or joblib.load(ART/'model.joblib')
    score=float(bundle['model'].predict_proba(row)[0,1])
    return {'label':'spam' if score>=bundle['threshold'] else 'ham',
            'spam_score':score,'threshold':bundle['threshold'],
            'scope':'Historical Spambase feature benchmark; not an inbox filter'}

def train():
    ART.mkdir(exist_ok=True)
    path=ROOT/'data/spambase.zip'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(urllib.request.urlopen(URL,timeout=60).read())
    raw=path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        matrix=np.loadtxt(io.BytesIO(z.read('spambase.data')),delimiter=',')
        (ART/'dataset_documentation.txt').write_bytes(z.read('spambase.DOCUMENTATION'))
    x,y,original_ids,audit=prepare(matrix)
    fitted=fit_email(x,y)
    split_sets=[set(map(tuple,x[fitted[s]])) for s in ('train','valid','test')]
    assert all(not split_sets[a]&split_sets[b] for a,b in [(0,1),(0,2),(1,2)])
    test=fitted['test']; truth=y[test]; predicted=fitted['predictions']
    report={'source':URL,'sha256':hashlib.sha256(raw).hexdigest(),'audit':audit,
            'seed':42,'split_sizes':{s:len(fitted[s]) for s in ('train','valid','test')},
            'feature_overlap_between_splits':0,'threshold':fitted['threshold'],
            'validation':metrics(y[fitted['valid']],fitted['validation_scores']>=fitted['threshold']),
            'test':metrics(truth,predicted),'baseline':metrics(truth,fitted['baseline_predictions']),
            'test_average_precision':float(average_precision_score(truth,fitted['test_scores']))}
    # Resample the locked test predictions to describe conditional sampling uncertainty.
    rng=np.random.default_rng(123)
    bootstrap=[]
    for _ in range(1000):
        ids=rng.integers(0,len(test),len(test))
        bootstrap.append(metrics(truth[ids],predicted[ids])['f1'])
    report['f1_bootstrap_95_percentile_interval']=np.quantile(bootstrap,[.025,.975]).tolist()
    report['interval_scope']='Conditional on this fitted model and cleaned test set; not source/time generalization'
    names=feature_names(raw)
    assert len(names)==57
    (ART/'feature_names.json').write_text(json.dumps(names,indent=2))
    (ART/'metrics.json').write_text(json.dumps(report,indent=2))
    joblib.dump({'model':fitted['model'],'threshold':fitted['threshold']},ART/'model.joblib')
    split={s:[original_ids[i] for i in fitted[s]] for s in ('train','valid','test')}
    (ART/'split_source_indices.json').write_text(json.dumps(split,indent=2))
    np.savez(ART/'audit_arrays.npz',train_x=x[fitted['train']],test_y=truth,test_scores=fitted['test_scores'],predictions=predicted)
    with (ART/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['source_row_zero_based','true_label','prediction','spam_score','outcome'])
        for i,t,p,s in zip(test,truth,predicted,fitted['test_scores']):
            writer.writerow([original_ids[i],int(t),int(p),float(s),'correct' if t==p else 'false_positive' if p else 'false_negative'])
    samples=[]
    for label in [0,1]:
        position=next(i for i in test if y[i]==label)
        samples.append({'description':f'Held-out historical sample; true label {label}',
                        'source_row_zero_based':original_ids[position], 'features':x[position].tolist()})
    (ART/'demo_samples.json').write_text(json.dumps(samples,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['train','predict'])
    parser.add_argument('--input',type=Path)
    args=parser.parse_args()
    if args.command=='train': train()
    else: print(json.dumps(predict(json.loads(args.input.read_text())['features'])))
