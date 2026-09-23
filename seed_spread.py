"""Split sensitivity: rerun the same pipeline on 20 different random splits.

The main result uses seed 42. This shows how much the numbers move when only the
split changes; the bootstrap interval in metrics.json does not cover that.
"""
import io
import json
import zipfile
import numpy as np
from email_core import fit_email
from email_project import ROOT, ART, prepare
from spam import metrics

def main(seeds=range(20)):
    with zipfile.ZipFile(ROOT / 'data/spambase.zip') as z:
        matrix = np.loadtxt(io.BytesIO(z.read('spambase.data')), delimiter=',')
    x, y, _, _ = prepare(matrix)
    runs = []
    for seed in seeds:
        fitted = fit_email(x, y, seed=seed)
        m = metrics(y[fitted['test']], fitted['predictions'])
        m['baseline_accuracy'] = metrics(y[fitted['test']], fitted['baseline_predictions'])['accuracy']
        m['threshold'] = fitted['threshold']
        m['seed'] = seed
        runs.append(m)
    summary = {k: {'min': min(r[k] for r in runs), 'max': max(r[k] for r in runs),
                   'mean': float(np.mean([r[k] for r in runs]))}
               for k in ('accuracy', 'precision', 'recall', 'f1', 'threshold', 'baseline_accuracy')}
    report = {'seeds': list(seeds), 'summary': summary, 'runs': runs}
    (ART / 'seed_spread.json').write_text(json.dumps(report, indent=2))
    for k, v in summary.items():
        print(f"{k:18s} min {v['min']:.4f}  mean {v['mean']:.4f}  max {v['max']:.4f}")

if __name__ == '__main__':
    main()
