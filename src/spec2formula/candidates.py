"""Target-free candidate enumeration with optional native acceleration."""
from pathlib import Path
import os
import subprocess
import warnings
import numpy as np
from .formula.elements import EXPANDED_ELEMENTS, hill_formula
from .formula.enumerator import FormulaDomainConfig, enumerate_formula_domain
from .formula_ranker.features import COUNT_LIMITS

class CandidateEnumerator:
    def __init__(self, backend='auto', binary=None):
        if backend not in ('auto', 'python', 'native'):
            raise ValueError('backend must be auto, python or native')
        self.binary = Path(binary).resolve() if binary else None
        if self.binary is None:
            name = 'enumerate.exe' if os.name == 'nt' else 'enumerate'
            roots = (Path.cwd(), Path(__file__).resolve().parents[2])
            self.binary = next((root/'_bin'/name for root in roots if (root/'_bin'/name).is_file()), None)
        if self.binary is not None and not self.binary.is_file():
            raise ValueError(f'Enumerator binary not found: {self.binary}')
        if backend == 'native' and self.binary is None:
            raise ValueError('Run python tools/build_enumerator.py or provide --enumerator-binary')
        self.backend = 'native' if backend != 'python' and self.binary else 'python'
        self.warned = False

    def enumerate(self, neutral_mass):
        if self.backend == 'native':
            proc = subprocess.run([str(self.binary), format(neutral_mass, '.17g')], capture_output=True, text=True, check=True)
            rows = [line.split('\t') for line in proc.stdout.splitlines() if line.strip()]
            counts = np.asarray([[int(x) for x in row[2:]] for row in rows], dtype=np.int16).reshape(-1, 16)
            masses = np.asarray([float(row[0]) for row in rows], dtype=np.float64)
            dbes = np.asarray([float(row[1]) for row in rows], dtype=np.float32)
        else:
            if neutral_mass > 200 and not self.warned:
                warnings.warn('Pure Python exhaustive enumeration may be slow at high mass. Build the optional native enumerator.', RuntimeWarning)
                self.warned = True
            config = FormulaDomainConfig(elements=EXPANDED_ELEMENTS, fixed_max_counts=dict(zip(EXPANDED_ELEMENTS, map(int, COUNT_LIMITS))))
            half_width = max(.001, neutral_mass * 3e-6)
            candidates = enumerate_formula_domain((neutral_mass-half_width, neutral_mass+half_width), config)
            candidates.sort(key=lambda x:(x.exact_mass, tuple(dict(x.counts).get(e, 0) for e in EXPANDED_ELEMENTS)))
            counts = np.asarray([[dict(x.counts).get(e, 0) for e in EXPANDED_ELEMENTS] for x in candidates], dtype=np.int16).reshape(-1, 16)
            masses = np.asarray([x.exact_mass for x in candidates], dtype=np.float64)
            dbes = np.asarray([x.dbe for x in candidates], dtype=np.float32)
        formulas = [hill_formula(dict(zip(EXPANDED_ELEMENTS, map(int, row)))) for row in counts]
        return formulas, counts, masses, dbes

