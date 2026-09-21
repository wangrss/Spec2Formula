import unittest
from pathlib import Path
import torch
from spec2formula import Predictor
from spec2formula.io import read_spectra
from spec2formula.predictor import verify_assets

ROOT = Path(__file__).resolve().parents[1]

class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = Predictor(ROOT/'checkpoints/default',candidate_backend='python')
        cls.record = next(read_spectra(ROOT/'examples/synthetic.jsonl'))

    def test_assets_and_end_to_end(self):
        verify_assets(ROOT/'checkpoints/default')
        result = self.model.predict(self.record,top_k=0)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(result['predictions']),result['candidate_count'])
        self.assertGreater(result['candidate_count'],0)
        self.assertEqual(sorted(x['candidate_rank'] for x in result['predictions']),list(range(1,result['candidate_count']+1)))
        for row in result['predictions'][20:]:
            self.assertEqual(row['rank'],row['candidate_rank'])
            self.assertIsNone(row['evidence_score'])

    def test_truth_not_used_and_top_k_only_truncates(self):
        all_rows = self.model.predict(self.record,top_k=0)
        tagged = self.model.predict({**self.record,'formula':'C999H999','target_formula':'Xe123'},top_k=5)
        self.assertEqual(tagged['predictions'],all_rows['predictions'][:5])
        self.assertEqual(tagged['candidate_count'],all_rows['candidate_count'])

    def test_empty_domain_is_explicit(self):
        result = self.model.predict({'precursor_mz':3.007276466621,'peaks':[[1,1]]})
        self.assertEqual(result['status'],'no_candidates')
        self.assertEqual(result['predictions'],[])

if __name__ == '__main__':
    unittest.main()
