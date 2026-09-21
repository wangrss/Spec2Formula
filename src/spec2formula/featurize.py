"""Frozen candidate priors and exact non-isotope evidence."""
from pathlib import Path
import numpy as np
from .formula_ranker.features import pack_formula_counts, nominal_bottom_up_evidence, neural_candidate_features
from .formula_evidence.exact_matcher import ExactMatcherConfig, match_candidate_spectrum
from .formula_evidence.tokens import exact_evidence_tokens

NOMINAL_MASSES = (12, 1, 11, 14, 16, 19, 28, 31, 32, 35, 75, 80, 79, 120, 121, 127)

def nominal_bitset(counts):
    bits = 1
    for count, mass in zip(counts, NOMINAL_MASSES):
        remaining, chunk = int(count), 1
        while remaining:
            take = min(chunk, remaining)
            bits |= bits << (take * mass)
            bits &= (1 << 512) - 1
            remaining -= take
            chunk *= 2
    return np.frombuffer(bits.to_bytes(64, 'little'), dtype='<u8').copy()

class FeatureBuilder:
    def __init__(self, directory):
        directory = Path(directory)
        with np.load(directory/'count_log_probabilities.npz', allow_pickle=False) as prior:
            self.logp = [prior[f'element_{i}'].astype(np.float32) for i in range(16)]
        self.keys = np.load(directory/'formula_keys.npy', allow_pickle=False)
        self.frequencies = np.load(directory/'formula_frequencies.npy', allow_pickle=False)

    def build(self, counts, masses, dbes, spectrum):
        if not len(counts):
            return np.empty((0, 25), np.float32)
        count_prior = np.zeros(len(counts), np.float32)
        for index in range(16):
            count_prior += self.logp[index][counts[:, index]]
        keys = pack_formula_counts(counts)
        positions = np.searchsorted(self.keys, keys)
        valid = np.flatnonzero(positions < len(self.keys))
        hits = valid[self.keys[positions[valid]] == keys[valid]]
        frequency = np.zeros(len(counts), np.float32)
        frequency[hits] = self.frequencies[positions[hits]]
        priors = np.stack((count_prior, np.log1p(frequency)), axis=1)
        bitsets = np.stack([nominal_bitset(row) for row in counts])
        nominal = nominal_bottom_up_evidence(bitsets, spectrum.peaks, spectrum.peak_length, spectrum.precursor_mz)
        return neural_candidate_features(counts, masses, dbes, spectrum.neutral_mass, nominal, priors)

def build_evidence(counts, spectrum):
    tokens = np.zeros((len(counts), 32, 40), np.float32)
    padding = np.ones((len(counts), 32), bool)
    peaks = spectrum.peaks[:spectrum.peak_length].astype(np.float64)
    config = ExactMatcherConfig(top_peaks=32, enable_isotope=False)
    for index, parent in enumerate(counts):
        match = match_candidate_spectrum(parent, peaks, spectrum.precursor_mz, config)
        batch = exact_evidence_tokens(match, peaks, spectrum.precursor_mz, maximum_tokens=32)
        tokens[index] = batch.tokens
        padding[index] = batch.padding_mask
    return tokens, padding

