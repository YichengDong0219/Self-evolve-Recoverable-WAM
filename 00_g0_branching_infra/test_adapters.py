import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from client_adapter import probe_client_class
from replay_adapter import replay, action_digest
import bounded_run

class Transport:
    def inference(self,r): return {'probe_seeds':r['probe_seeds']}
class Base:
    def __init__(self): self.client=Transport()
    def _build_request(self,e): return dict(e)
    def reset(self): pass
class Task:
    take_action_cnt=0; eval_success=False; step_lim=10
    def take_action(self,a,action_type):
        assert action_type=='qpos'; self.take_action_cnt+=1
class Adapters(unittest.TestCase):
    def test_bind_and_reset(self):
        c=probe_client_class(Base)(); log=[]
        c.bind_probe('a',42,lambda kind,x:log.append(kind))
        r=c._build_request({}); c.client.inference(r)
        self.assertEqual(log,['request','response'])
        self.assertNotEqual(r['probe_seeds'],c._build_request({})['probe_seeds'])
        c.reset()
        with self.assertRaises(ValueError): c._build_request({})
    def test_replay_prefix_and_corruption(self):
        a=[[0.]*14,[1.]*14]; log=[]; t=Task()
        result=replay(t,a,action_digest(a),lambda t:{'step':t.take_action_cnt},log.append,1)
        self.assertEqual(result['executed'],1); self.assertFalse(result['branch_validated'])
        with self.assertRaises(ValueError): replay(Task(),a,'bad',lambda t:{},log.append)
    def test_foreign_gpu_never_launches(self):
        with tempfile.TemporaryDirectory() as td,patch.object(bounded_run,'gpu_empty',return_value=False),patch.object(bounded_run.subprocess,'Popen') as popen:
            with self.assertRaises(ValueError): bounded_run.run(Path(td),'GPU-test',60,['false'])
            popen.assert_not_called()
    def test_foreground_charge(self):
        import sys,json
        with tempfile.TemporaryDirectory() as td,patch.object(bounded_run,'gpu_empty',return_value=True):
            self.assertEqual(bounded_run.run(Path(td),'GPU-test',60,[sys.executable,'-c','print("fixture")']),0)
            s=json.loads((Path(td)/'ledger.json').read_text());self.assertIsNone(s['active']); self.assertGreater(s['used_gpu_seconds'],0)
if __name__=='__main__':unittest.main()
