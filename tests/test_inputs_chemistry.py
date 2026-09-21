import tempfile
import unittest
from pathlib import Path
import numpy as np
from spec2formula.io import prepare_spectrum, read_spectra
from spec2formula.formula.elements import exact_mass
from spec2formula.formula_ranker.features import PROTON_MASS_DA
from spec2formula.candidates import CandidateEnumerator
from spec2formula.predictor import coverage_preserving_order

ROOT = Path(__file__).resolve().parents[1]

class InputChemistryTests(unittest.TestCase):
    def test_peak_selection_and_scale(self):
        rows = [[i+1, i+1] for i in range(140)] + [[float('nan'),1],[42,-1]]
        s = prepare_spectrum({'precursor_mz':200,'peaks':rows})
        self.assertEqual(s.peak_length,128)
        self.assertEqual(float(s.peaks[0,0]),13)
        self.assertEqual(float(s.peaks[-1,1]),1)
        self.assertEqual(s.collision_energy,0)
        self.assertEqual(len(s.warnings),3)

    def test_invalid_inputs_fail(self):
        valid = {'precursor_mz':100,'peaks':[[50,1]]}
        for update in ({'adduct':'[M+Na]+'},{'precursor_mz':500},{'precursor_mz':float('nan')},
                       {'collision_energy':float('inf')},{'peaks':[[50,0]]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                prepare_spectrum({**valid,**update})

    def test_jsonl_and_mgf_agree(self):
        a = prepare_spectrum(next(read_spectra(ROOT/'examples/synthetic.jsonl')))
        b = prepare_spectrum(next(read_spectra(ROOT/'examples/synthetic.mgf')))
        self.assertEqual(a.id,b.id)
        self.assertEqual(a.precursor_mz,b.precursor_mz)
        self.assertEqual(a.collision_energy,b.collision_energy)
        np.testing.assert_array_equal(a.peaks,b.peaks)

    def test_mgf_ev_is_not_nce(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'x.mgf'
            p.write_text('BEGIN IONS\nPEPMASS=100\nCOLLISIONENERGY=35 eV\n50 1\nEND IONS\n')
            record = next(read_spectra(p))
            s = prepare_spectrum(record,fallback_nce=20)
            self.assertEqual(s.collision_energy,20)
            self.assertTrue(any('not assumed' in note for note in s.warnings))

    def test_known_formula_and_no_candidates(self):
        enum = CandidateEnumerator('python')
        mass = exact_mass({'C':2,'H':6,'O':1})
        formulas, _, masses, _ = enum.enumerate(mass)
        self.assertIn('C2H6O',formulas)
        self.assertTrue(np.all(np.abs(masses-mass)<=.001+1e-10))
        self.assertEqual(enum.enumerate(2.0)[0],[])

    def test_coverage_preserving_ties_and_tail(self):
        base = np.arange(30,0,-1,dtype=np.float32)
        top = np.arange(20)
        final = coverage_preserving_order(base,np.arange(20,dtype=np.float32),top)
        np.testing.assert_array_equal(final[:20],np.arange(19,-1,-1))
        np.testing.assert_array_equal(final[20:],np.arange(20,30))
        np.testing.assert_array_equal(np.sort(final),np.arange(30))
        np.testing.assert_array_equal(coverage_preserving_order(base,np.zeros(20),top),np.arange(30))

if __name__ == '__main__':
    unittest.main()

