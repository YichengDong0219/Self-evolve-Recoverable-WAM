import unittest
from onset_timing_codex import detect
class Timing(unittest.TestCase):
    def test_detection_never_precedes_evidence(self):
        rows=[{'step':i,'p':[0,0,1 if i<10 else .95],'tcp_l':[0,0,1],'tcp_r':[0,0,1]} for i in range(11)]
        self.assertIsNone(detect(rows[:10]))
        r=detect(rows);self.assertEqual(r['t_detect'],10);self.assertEqual(r['t_onset_retrospective'],0)
    def test_stall_is_detected_at_window_end(self):
        rows=[{'step':i,'p':[0,0,1],'tcp_l':[0,0,1],'tcp_r':[0,0,1]} for i in range(151)]
        r=detect(rows);self.assertEqual(r['t_detect'],150);self.assertEqual(r['t_onset_retrospective'],54)
    def test_sparse_rejected(self):
        with self.assertRaises(ValueError): detect([{'step':0},{'step':2}])
if __name__=='__main__':unittest.main()
