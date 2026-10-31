import copy
import math
import tempfile
import subprocess
import unittest
from pathlib import Path
from branch_compare import compare, LIMITS, EXACT
from seed_plan import seed_for
from collect_sources import collect

def state():
    s = {k: {'obj': [0,0,0] if k in ('position_m','linear_velocity_m_s','angular_velocity_rad_s') else [0]} for k in LIMITS}
    s['rotation_wxyz'] = {'obj': [1,0,0,0]}
    s.update({k: {'captured': True, 'value': {'state': 'fixture'}} for k in EXACT})
    for k in ('success_counter', 'action_step'): s[k]['value']=0
    s['remaining_action_queue']['value']=[]
    s['official_success']['value']=False
    return s

class Core(unittest.TestCase):
    def test_identity_and_quaternion_sign(self):
        a=state(); b=copy.deepcopy(a); b['rotation_wxyz']['obj']=[-1,0,0,0]
        self.assertTrue(compare(a,b)['pass'])
    def test_boundaries_and_nan(self):
        for metric in LIMITS:
            a=state(); b=copy.deepcopy(a); b[metric]['obj'][0]=float('nan')
            self.assertFalse(compare(a,b)['pass'])
        a=state(); b=copy.deepcopy(a); b['position_m']['obj'][0]=.001
        self.assertFalse(compare(a,b)['pass'])
    def test_missing_queue_and_objects(self):
        a=state(); b=copy.deepcopy(a); del b['remaining_action_queue']
        self.assertFalse(compare(a,b)['pass'])
        b=copy.deepcopy(a); b['qpos_rad']={}
        self.assertFalse(compare(a,b)['pass'])
    def test_success_and_png(self):
        a=state(); b=copy.deepcopy(a); a['png_sha256']='a'; b['png_sha256']='b'
        self.assertTrue(compare(a,b)['pass']); b['official_success']['value']=True
        self.assertFalse(compare(a,b)['pass'])
    def test_review_fail_open_regressions(self):
        a=state(); b=copy.deepcopy(a)
        for k in EXACT: a[k]={}; b[k]={}
        self.assertFalse(compare(a,b)['pass'])
        a=state(); b=copy.deepcopy(a); a['qpos_m']={'x':[0]}; b['qpos_m']={'x':[.5]}
        self.assertFalse(compare(a,b)['pass'])
        a=state(); b=copy.deepcopy(a); a['rng_state']['value']={'x':float('nan')}; b['rng_state']=a['rng_state']
        self.assertFalse(compare(a,b)['pass'])
        with self.assertRaises(ValueError): seed_for(42,'start','candidate',0,repeat=1,candidate=0)
        self.assertNotEqual(seed_for(42,'start','continuation',1,repeat=0), seed_for(42,'start','continuation',1,repeat=1))
    def test_seed_streams(self):
        seeds=[seed_for(42,'start', 'candidate',0,candidate=i) for i in range(8)]
        self.assertEqual(len(set(seeds)),8)
        self.assertEqual(seed_for(42,'start','continuation',2),seed_for(42,'start','continuation',2))
        self.assertNotEqual(seed_for(42,'start','continuation',2),seed_for(42,'start','continuation',3))
        with self.assertRaises(ValueError): seed_for(42,'start','continuation',2,candidate=1)
    def test_export_current_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'repo'; r.mkdir()
            subprocess.run(['git','init','-q',str(r)],check=True)
            (r/'a.py').write_text('x=1\n')
            subprocess.run(['git','-C',str(r),'add','a.py'],check=True)
            subprocess.run(['git','-C',str(r),'-c','user.name=test','-c','user.email=test@local','commit','-qm','init'],check=True)
            (r/'a.py').write_text('x=2\n'); (r/'secret.env').write_text('excluded')
            out=Path(tmp)/'out'; report=collect({'FACT':r},out)
            self.assertEqual((out/'FACT/a.py').read_text(),'x=2\n')
            self.assertNotIn('FACT/secret.env',report['files'])
            self.assertTrue(report['complete'])
            with self.assertRaises(FileExistsError): collect({'FACT':r},out)

if __name__ == '__main__': unittest.main()
