import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from spec2formula.massbank import record_input,rebuild,evaluate_predictions,evaluate_frozen_ranks


def write_manifest(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class MassBankTests(unittest.TestCase):
    def synthetic_record(self,folder,cohort='qtof_common'):
        root=Path(folder);p=root/'Synthetic/a.txt';p.parent.mkdir()
        raw=('ACCESSION: a\nLICENSE: CC BY\nMS$FOCUSED_ION: PRECURSOR_M/Z 200\n'
             'MS$FOCUSED_ION: PRECURSOR_TYPE [M+H]+\nPK$NUM_PEAK: 2\n'
             'PK$PEAK: m/z int. rel.int.\n 50 10 999\n 25 5 500\n//\n')
        p.write_text(raw,encoding='utf-8')
        expected=np.zeros((128,2),np.float32);expected[:2]=[[25,.5],[50,1]]
        row={'accession':'a','relative_path':'Synthetic/a.txt','license':'CC BY',
             'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
             'precursor_mz':'200','cohort':cohort,'energy_basis':'explicit_eV','energy_numeric':'10',
             'nce_input':'25.0','retained_peaks':'2','source':'Synthetic','connectivity_group':'g1',
             'fully_disjoint':'True','target_formula':'C8H10O',
             'preprocessed_peaks_sha256':hashlib.sha256(expected.tobytes()).hexdigest()}
        return root,p,row

    def test_rebuild_exact_peaks_energy_and_target_free_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root,p,row=self.synthetic_record(folder)
            manifest=root/'manifest.csv';write_manifest(manifest,[row])
            output=root/'inputs.jsonl'
            self.assertEqual(rebuild(manifest,root,output),1)
            got=json.loads(output.read_text())
            self.assertEqual(got['collision_energy'],25)
            self.assertEqual(got['peaks'],[[50,10],[25,5]])
            self.assertNotIn('target_formula',got);self.assertNotIn('formula',got)
            self.assertFalse(output.with_name('inputs.jsonl.partial').exists())

    def test_changed_source_energy_or_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,p,row=self.synthetic_record(folder)
            for update in [{'nce_input':'10'},{'relative_path':'../outside.txt'}]:
                with self.subTest(update=update),self.assertRaises(ValueError):
                    record_input({**row,**update},root)
            p.write_text(p.read_text()+'changed\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                record_input(row,root)

    def test_missing_failed_predictions_remain_in_both_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'manifest.csv';pred=root/'prediction.jsonl'
            rows=[{'accession':str(i),'source':'s','connectivity_group':g,
                   'fully_disjoint':'False','target_formula':'C2H6O'} for i,g in [(1,'g1'),(2,'g1'),(3,'g2')]]
            write_manifest(manifest,rows)
            pred.write_text('\n'.join(json.dumps({'id':str(i),'status':'ok','predictions':[{'rank':1,'formula':'H6OC2'}]}) for i in [1,3]))
            report=evaluate_predictions(manifest,pred)
            self.assertEqual(report['missing_predictions'],1)
            self.assertAlmostEqual(report['metrics']['all']['spectrum_top1'],2/3)
            self.assertEqual(report['metrics']['all']['molecule_top1'],.75)
            with pred.open('a') as f:f.write('\n'+json.dumps({'id':'2','status':'error','predictions':[]}))
            report=evaluate_predictions(manifest,pred)
            self.assertEqual(report['missing_predictions'],0)
            self.assertEqual(report['metrics']['all']['molecule_top1'],.75)

    def test_duplicate_or_unknown_prediction_ids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'manifest.csv';pred=root/'p.jsonl'
            write_manifest(manifest,[{'accession':'a','source':'s','connectivity_group':'g','fully_disjoint':'False','target_formula':'C2H6O'}])
            for ids in [['a','a'],['unknown']]:
                pred.write_text('\n'.join(json.dumps({'id':i,'status':'no_candidates','predictions':[]}) for i in ids))
                with self.subTest(ids=ids),self.assertRaises(ValueError):
                    evaluate_predictions(manifest,pred)

    def test_archived_ranks_require_complete_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'manifest.csv';ranks=root/'ranks.csv'
            write_manifest(manifest,[{'accession':'a','source':'s','connectivity_group':'g','fully_disjoint':'False','target_formula':'C2H6O'}])
            rank_row={'accession':'a','Spec2Formula_rank':'0','msbuddy_rank':'1','MIST_CF_rank':'nan','SIRIUS_rank':'-1'}
            write_manifest(ranks,[rank_row]);report=evaluate_frozen_ranks(manifest,ranks)
            self.assertEqual(report['methods']['Spec2Formula']['all']['spectrum_top1'],0)
            self.assertEqual(report['methods']['msbuddy']['all']['spectrum_top1'],1)
            write_manifest(ranks,[{**rank_row,'accession':'other'}])
            with self.assertRaises(ValueError):evaluate_frozen_ranks(manifest,ranks)

    def test_frozen_artifact_checksum_rejected_after_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'manifest.csv';pred=root/'p.jsonl'
            write_manifest(manifest,[{'accession':'a','source':'s','connectivity_group':'g','fully_disjoint':'False','target_formula':'C2H6O'}])
            checksum=hashlib.sha256(manifest.read_bytes()).hexdigest()
            (root/'PROVENANCE.json').write_text(json.dumps({'artifacts':{'manifest.csv':{'sha256':checksum}}}))
            pred.write_text('')
            self.assertEqual(evaluate_predictions(manifest,pred)['missing_predictions'],1)
            with manifest.open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'checksum'):evaluate_predictions(manifest,pred)


if __name__=='__main__':
    unittest.main()
