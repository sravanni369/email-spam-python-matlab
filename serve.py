"""Local-only demonstration. No inbox access, external calls or message logging."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import joblib
from email_project import ROOT, ART, predict

class Handler(BaseHTTPRequestHandler):
    bundle = None
    def send(self,status,body,kind='application/json'):
        payload=body.encode() if isinstance(body,str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type',kind+'; charset=utf-8')
        self.send_header('Content-Length',str(len(payload)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        if self.path=='/': self.send(200,(ROOT/'demo.html').read_text(encoding='utf-8'),'text/html')
        elif self.path=='/metrics': self.send(200,json.loads((ART/'metrics.json').read_text()))
        elif self.path=='/samples': self.send(200,json.loads((ART/'demo_samples.json').read_text()))
        elif self.path=='/health': self.send(200,{'status':'ok','model':'spambase-logistic-regression'})
        else: self.send(404,{'error':'Not found'})
    def do_POST(self):
        if self.path!='/predict': return self.send(404,{'error':'Not found'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=20000: raise ValueError('Request size must be 1..20000 bytes')
            body=json.loads(self.rfile.read(size))
            if not isinstance(body,dict): raise ValueError('Expected JSON object')
            result=predict(body.get('features'),self.bundle)
            self.send(200,result)
        except (ValueError,TypeError,UnicodeError) as e:
            self.send(400,{'error':str(e)})
    def log_message(self,*args): pass

if __name__=='__main__':
    Handler.bundle=joblib.load(ART/'model.joblib')
    print('Local demo: http://127.0.0.1:8765',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
