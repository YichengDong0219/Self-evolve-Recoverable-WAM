import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from probe_request import request_seeds, attach_probe_request
from build_server_probe import build

class Probe(unittest.TestCase):
    def test_wire_protocol(self):
        for n in (1,8):
            r=attach_probe_request({},list(range(n)))
            self.assertEqual(request_seeds(r,n),list(range(n)))
        for r,n in [({},1),({'probe_seeds':[1,1]},2),({'probe_seeds':[True]},1),({'probe_seeds':[1]},0),({'probe_seeds':[2**63]},1)]:
            with self.assertRaises(ValueError): request_seeds(r,n)
    def test_actual_server_patch(self):
        source=Path(__file__).resolve().parents[1]/'evidence/fact_R91/source_export/FACT/scripts/inference_server.py'
        if not source.exists(): self.skipTest('run source integration test locally against returned export')
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'server'
            build(source,hashlib.sha256(source.read_bytes()).hexdigest(),out,Path(__file__).with_name('probe_request.py'))
            s=(out/'inference_server_probe.py').read_text(); ast.parse(s)
            self.assertNotIn('time.time_ns()',s); self.assertNotIn('_run_pipe(None)',s)
            self.assertIn('torch.argmin(rank_vals)',s); self.assertIn('probe_candidate_actions',s)
            with self.assertRaises(ValueError): build(source,'0'*64,Path(td)/'bad',Path(__file__).with_name('probe_request.py'))

if __name__=='__main__': unittest.main()
