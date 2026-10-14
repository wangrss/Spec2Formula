import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from spec2formula import Predictor
from spec2formula.candidates import CandidateEnumerator
from spec2formula.featurize import build_evidence
from spec2formula.io import prepare_spectrum

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures'


class NumericalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model=Predictor(ROOT/'checkpoints/default',candidate_backend='python')
        cls.record=json.loads((FIXTURE/'reference_23.json').read_text())
        cls.spectrum=prepare_spectrum(cls.record)
        with np.load(FIXTURE/'reference_23.npz',allow_pickle=False) as z:
            cls.ref={n:z[n] for n in z.files}

    def test_fixture_identity_and_preprocessing(self):
        provenance=json.loads((FIXTURE/'PROVENANCE.json').read_text())
        for name,digest in provenance['fixture_sha256'].items():
            self.assertEqual(hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest(),digest)
        np.testing.assert_array_equal(self.spectrum.peaks,self.ref['peaks'])
        self.assertEqual(self.spectrum.peak_length,128)

    def test_candidate_features_and_evidence_tokens(self):
        r=self.ref
        features=self.model.features.build(r['counts'],r['masses'],r['dbes'],self.spectrum)
        np.testing.assert_allclose(features,r['features'],atol=1e-7,rtol=1e-7)
        tokens,padding=build_evidence(r['counts'][r['top']],self.spectrum)
        np.testing.assert_allclose(tokens,r['tokens'],atol=1e-7,rtol=1e-7)
        np.testing.assert_array_equal(padding,r['padding'])

    def test_scores_global_standardization_and_chunking(self):
        r=self.ref
        original=self.model.config['inference']['base_candidate_chunk']
        try:
            for size in [512,7]:
                with self.subTest(chunk=size):
                    self.model.config['inference']['base_candidate_chunk']=size
                    base,z,top,evidence,final=self.model._score(self.spectrum,r['counts'],r['features'])
                    for actual,expected in [(base,r['base_scores']),(z,r['base_z']),(evidence,r['evidence_scores'])]:
                        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
                    expected_z=(base.astype(np.float64)-base.mean(dtype=np.float64))/max(float(base.std(dtype=np.float64)),.001)
                    np.testing.assert_allclose(z,expected_z,atol=1e-7,rtol=1e-7)
                    np.testing.assert_array_equal(top,r['top'])
                    np.testing.assert_array_equal(final,r['final'])
        finally:
            self.model.config['inference']['base_candidate_chunk']=original

    def test_complete_inference_has_real_tail(self):
        result=self.model.predict(self.record,top_k=0)
        self.assertEqual(result['candidate_count'],23)
        self.assertEqual(result['reranked_count'],20)
        self.assertEqual([x['formula'] for x in result['predictions']],self.ref['final_formulas'].tolist())
        self.assertEqual(len(result['predictions'][20:]),3)
        for row in result['predictions'][20:]:
            self.assertEqual(row['rank'],row['candidate_rank'])
            self.assertIsNone(row['evidence_score'])

    def test_standardization_sd_floor(self):
        r=self.ref
        with patch.object(self.model.candidate_ranker,'score_packed',side_effect=lambda emb,features,owner:torch.full((len(features),),2.0)):
            _,z,_,_,_=self.model._score(self.spectrum,r['counts'],r['features'])
        np.testing.assert_array_equal(z,np.zeros(23,np.float32))

    def test_native_backend_against_frozen_enumeration(self):
        enum=CandidateEnumerator('auto')
        if enum.backend!='native':
            self.skipTest('Build the optional native backend to run backend parity')
        formulas,counts,masses,dbes=enum.enumerate(self.spectrum.neutral_mass)
        np.testing.assert_array_equal(formulas,self.ref['formulas'])
        np.testing.assert_array_equal(counts,self.ref['counts'])
        np.testing.assert_allclose(masses,self.ref['masses'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(dbes,self.ref['dbes'])


if __name__=='__main__':
    unittest.main()
