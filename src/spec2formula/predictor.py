"""Portable two-stage inference; no training caches or reference formulas."""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .io import prepare_spectrum, Spectrum
from .candidates import CandidateEnumerator
from .featurize import FeatureBuilder, build_evidence
from .formula_ranker.large_evidence_model import (
    LargeFormulaRanker, LargeFormulaRankerConfig,
    EvidenceResidualTransformer, EvidenceTransformerConfig,
)

def verify_assets(model_dir):
    directory = Path(model_dir).resolve()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    required = {'config.json','candidate_ranker.pt','evidence_reranker.pt',
                'priors/count_log_probabilities.npz','priors/formula_keys.npy','priors/formula_frequencies.npy'}
    if set(manifest.get('files', {})) != required:
        raise ValueError('Model asset manifest has an unexpected file list')
    for name, expected in manifest['files'].items():
        path = directory/name
        if not path.is_file():
            raise ValueError(f'Missing model asset: {name}. Retrieve the complete checkpoint folder.')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2), b''):
                digest.update(block)
        if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Asset mismatch: {name}. If cloned from GitHub, run git lfs pull.')
    return manifest

def stable_order(scores):
    return np.argsort(-np.asarray(scores), kind='stable')

def coverage_preserving_order(base_scores, evidence_scores, top):
    base = stable_order(base_scores)
    if not np.array_equal(np.asarray(top), base[:len(top)]):
        raise ValueError('Evidence must score exactly the leading Candidate-ranker prefix')
    return np.concatenate((base[:len(top)][stable_order(evidence_scores)], base[len(top):]))

class Predictor:
    def __init__(self, model_dir, *, device='cpu', precision='fp32', candidate_backend='auto', enumerator_binary=None):
        self.directory = Path(model_dir).resolve()
        verify_assets(self.directory)
        self.config = json.loads((self.directory/'config.json').read_text(encoding='utf-8'))
        self.device = torch.device(device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('Supported devices are cpu and cuda')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu or install a compatible PyTorch build')
        if precision not in ('fp32','bf16'):
            raise ValueError('precision must be fp32 or bf16')
        if precision == 'bf16':
            if self.device.type != 'cuda':
                raise ValueError('bf16 requires a CUDA device with BF16 support')
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError('The selected CUDA device does not support BF16')
        self.precision = precision
        model_config = self.config['models']
        self.candidate_ranker = LargeFormulaRanker(LargeFormulaRankerConfig(**model_config['candidate_ranker']['config']))
        self.evidence_reranker = EvidenceResidualTransformer(EvidenceTransformerConfig(**model_config['evidence_reranker']['config']))
        for name, model in (('candidate_ranker',self.candidate_ranker),('evidence_reranker',self.evidence_reranker)):
            state = torch.load(self.directory/model_config[name]['file'], map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()
        self.enumerator = CandidateEnumerator(candidate_backend, enumerator_binary)
        self.features = FeatureBuilder(self.directory/'priors')

    def _autocast(self):
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.precision == 'bf16' else nullcontext()

    def _tensor(self, value):
        return torch.from_numpy(np.asarray(value)).to(self.device)

    @torch.no_grad()
    def _score(self, spectrum, counts, features):
        with self._autocast():
            embedding = self.candidate_ranker.encode_spectrum(
                self._tensor(spectrum.peaks[None]),
                self._tensor((np.arange(len(spectrum.peaks)) >= spectrum.peak_length)[None]),
                self._tensor(np.asarray([spectrum.precursor_mz], np.float32)),
                self._tensor(np.asarray([spectrum.collision_energy], np.float32)),
            )
            pieces = []
            chunk = self.config['inference']['base_candidate_chunk']
            for start in range(0,len(features),chunk):
                part = self._tensor(features[start:start+chunk])
                scores = self.candidate_ranker.score_packed(embedding, part, torch.zeros(len(part),dtype=torch.long,device=self.device))
                pieces.append(scores.float().cpu().numpy())
        base = np.concatenate(pieces)
        if not np.isfinite(base).all():
            raise RuntimeError('Candidate ranker returned nonfinite scores')
        order = stable_order(base)
        top = order[:self.config['inference']['rerank_top_k']]
        z = ((base.astype(np.float64)-np.mean(base,dtype=np.float64))/max(float(np.std(base,dtype=np.float64)), .001)).astype(np.float32)
        tokens, padding = build_evidence(counts[top], spectrum)
        with self._autocast():
            evidence = self.evidence_reranker.forward_standardized(
                self._tensor(z[top][None]), embedding, self._tensor(features[top][None]),
                self._tensor(tokens[None]), self._tensor(padding[None]),
                torch.zeros((1,len(top)),dtype=torch.bool,device=self.device))
        evidence = evidence[0].float().cpu().numpy()
        if not np.isfinite(evidence).all():
            raise RuntimeError('Evidence reranker returned nonfinite scores')
        return base, z, top, evidence, coverage_preserving_order(base, evidence, top)

    def predict(self, record, *, top_k=20, fallback_nce=None):
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer; 0 returns the full list')
        spectrum = prepare_spectrum(record, fallback_nce=fallback_nce,
                                    max_peaks=self.config['preprocessing']['maximum_peaks'])
        formulas, counts, masses, dbes = self.enumerator.enumerate(spectrum.neutral_mass)
        result = {'id':spectrum.id, 'status':'ok' if formulas else 'no_candidates',
                  'method':'Spec2Formula', 'model_id':self.config['model_id'],
                  'precursor_mz':spectrum.precursor_mz, 'adduct':'[M+H]+',
                  'nce_used':spectrum.collision_energy,'peaks_used':spectrum.peak_length,
                  'candidate_count':len(formulas), 'reranked_count':min(20,len(formulas)),
                  'device':str(self.device), 'precision':self.precision,'candidate_backend':self.enumerator.backend,
                  'warnings':list(spectrum.warnings), 'predictions':[]}
        if not formulas:
            return result
        features = self.features.build(counts,masses,dbes,spectrum)
        base, z, top, evidence, final = self._score(spectrum, counts, features)
        ranks = np.empty(len(base),dtype=int)
        ranks[stable_order(base)] = np.arange(1,len(base)+1)
        evidence_lookup = dict(zip(map(int,top),map(float,evidence)))
        selected = final if top_k == 0 else final[:top_k]
        for rank, index in enumerate(selected,1):
            result['predictions'].append({
                'rank':rank,'formula':formulas[index], 'neutral_exact_mass':float(masses[index]),
                'precursor_error_ppm':float((spectrum.precursor_mz-(masses[index]+1.007276466621))/(masses[index]+1.007276466621)*1e6),
                'candidate_rank':int(ranks[index]), 'candidate_score':float(base[index]),
                'candidate_score_z':float(z[index]),'evidence_score':evidence_lookup.get(int(index))})
        return result
