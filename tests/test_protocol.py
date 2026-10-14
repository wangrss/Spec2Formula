import copy
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from spec2formula.candidates import CandidateEnumerator
from spec2formula.protocol import validate_frozen_config,expected_native_protocol

ROOT=Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'checkpoints/default/config.json').read_text())

    def test_reference_config_and_operational_chunks(self):
        validate_frozen_config(self.config)
        self.config['inference']['base_candidate_chunk']=7
        validate_frozen_config(self.config)

    def test_scientific_protocol_drift_rejected(self):
        changes=[('candidate_domain','ppm',5),('candidate_domain','floor_da',.002),
                 ('candidate_domain','count_limits',[0]*16),('candidate_domain','elements',['C','H']),
                 ('scope','precursor_mz_exclusive_max',600),('preprocessing','maximum_peaks',64),
                 ('preprocessing','evidence_peaks',16),('inference','rerank_top_k',64),
                 ('inference','evidence_token_dtype','float16'),('inference','base_candidate_chunk',0)]
        for section,key,value in changes:
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):
                c=copy.deepcopy(self.config);c[section][key]=value;validate_frozen_config(c)

    def test_native_protocol_drift_or_legacy_binary_rejected(self):
        binary=ROOT/'native/enumerate.cpp'  # Existing path; process output is mocked.
        wrong=copy.deepcopy(expected_native_protocol());wrong['candidate_domain']['ppm']=5
        with patch('spec2formula.candidates.subprocess.run',return_value=subprocess.CompletedProcess([],0,json.dumps(wrong),'')):
            with self.assertRaisesRegex(ValueError,'differs'):
                CandidateEnumerator('native',binary)
        with patch('spec2formula.candidates.subprocess.run',side_effect=subprocess.CalledProcessError(1,[])):
            with self.assertRaisesRegex(ValueError,'rebuild'):
                CandidateEnumerator('native',binary)


if __name__=='__main__':
    unittest.main()
