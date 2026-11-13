import copy
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import torch
    from checkpoint_seed import validate_seed
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires optional training dependencies')
class SeedTests(unittest.TestCase):
    def seed(self):
        return {'plan_hash':'parent','step':42,'best_step':42,
                'model':{'weight':torch.tensor([1.25],dtype=torch.float32)},
                'best_model':{'weight':torch.tensor([1.25],dtype=torch.float16)},
                'optimizer':{'state':{0:{'exp_avg':torch.tensor([0.1])}}}}

    def test_published_best_preserves_full_precision_state(self):
        s=self.seed()
        self.assertIs(validate_seed(s,'parent',42),s)
        self.assertEqual(s['model']['weight'].dtype,torch.float32)

    def test_different_plan_rejected(self):
        with self.assertRaises(RuntimeError):validate_seed(self.seed(),'other',42)

    def test_later_nonbest_optimizer_rejected(self):
        s=self.seed();s['step']=106
        with self.assertRaises(RuntimeError):validate_seed(s,'parent',42)

    def test_model_mismatch_rejected(self):
        s=self.seed();s['model']['weight'][0]=2
        with self.assertRaises(RuntimeError):validate_seed(s,'parent',42)

    def test_nonfinite_optimizer_rejected(self):
        s=self.seed();s['optimizer']['state'][0]['exp_avg'][0]=float('nan')
        with self.assertRaises(RuntimeError):validate_seed(s,'parent',42)


if __name__=='__main__':unittest.main()
