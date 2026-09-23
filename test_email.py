import json
import threading
import unittest
import urllib.request
import urllib.error
import numpy as np
import joblib
from http.server import ThreadingHTTPServer
import email_project as ep
from serve import Handler

class EmailTests(unittest.TestCase):
    def test_conflicting_and_duplicate_features(self):
        a=np.zeros(58); b=a.copy(); b[-1]=1
        c=np.ones(58)
        x,y,ids,audit=ep.prepare(np.array([a,b,c,c]))
        self.assertEqual(ids,[2])
        self.assertEqual(audit['conflicting_rows_excluded'],2)
        self.assertEqual(audit['redundant_same_label_rows_removed'],1)
    def test_validation(self):
        for value in [None,[],[0]*56,[True]*57,[float('nan')]*57,[-1]*57,[101]*57]:
            with self.assertRaises(ValueError): ep.validate_features(value)
    def test_preprocessing_train_only(self):
        data=np.load(ep.ART/'audit_arrays.npz')
        model=joblib.load(ep.ART/'model.joblib')['model']
        np.testing.assert_allclose(model[0].mean_,data['train_x'].mean(axis=0))
    def test_saved_metrics_independent_counts(self):
        data=np.load(ep.ART/'audit_arrays.npz'); y=data['test_y']; p=data['predictions']
        counts=[int(((y==a)&(p==b)).sum()) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
        self.assertEqual(counts,json.loads((ep.ART/'metrics.json').read_text())['test']['confusion_matrix_tn_fp_fn_tp'])
    def test_source_split_disjoint(self):
        s=json.loads((ep.ART/'split_source_indices.json').read_text())
        self.assertFalse(set(s['train'])&set(s['test']))
        self.assertFalse(set(s['train'])&set(s['valid']))
        self.assertFalse(set(s['valid'])&set(s['test']))
    def test_http(self):
        Handler.bundle=joblib.load(ep.ART/'model.joblib')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            sample=json.loads((ep.ART/'demo_samples.json').read_text())[0]
            req=urllib.request.Request(url+'/predict',data=json.dumps(sample).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req) as r: result=json.load(r)
            self.assertEqual(result,ep.predict(sample['features']))
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(urllib.request.Request(url+'/predict',data=b'{"features":[]}'))
            self.assertEqual(ctx.exception.code,400)
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__': unittest.main()
