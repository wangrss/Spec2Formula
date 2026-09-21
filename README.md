# Spec2Formula

Standalone molecular-formula inference from high-resolution MS/MS spectra,
using a **Candidate ranker** followed by an **Evidence reranker**.

[Input/output reference](docs/INPUT_OUTPUT.md) ·
[Model card](docs/MODEL_CARD.md)

This repository contains inference code and a frozen pretrained model pair.
No training or evaluation dataset is included. The example spectra are
artificial inputs for checking installation, not experimental observations.

## Requirements

| Component | Version / purpose | Required? |
|---|---|---|
| Python | 3.11 recommended and tested; package accepts 3.10-3.12 | Yes |
| PyTorch | 2.5.1 reference version | Yes |
| NumPy | 1.26.4 reference version | Yes |
| pip | Package installation | Yes for the pip workflow |
| setuptools / wheel | setuptools >=68; resolved automatically by pip during package installation | Installation only |
| C++ compiler | C++17-capable g++ or clang++; faster candidate enumeration | Optional |
| NVIDIA GPU and driver | Driver compatible with the chosen PyTorch CUDA wheel | GPU execution only |
| Git LFS | Retrieve checkpoint files when cloning an LFS-enabled repository | Conditional |

CPU inference requires no GPU or CUDA installation. Official PyTorch CUDA
wheels include their runtime libraries; a separately installed CUDA Toolkit
or nvcc is not required for this inference package. The optional C++ enumerator
is a CPU program and does not depend on CUDA. RDKit, pandas, SciPy, SIRIUS and
msbuddy are not dependencies of this package. Tests use Python's standard
`unittest` module. CPU inference and native compilation were tested on Linux;
CUDA execution has not yet been validated for this release.

## Quick start

Python 3.11 is recommended. Create a new environment, then run these commands
from the repository root. The CPU installation works without a GPU:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

spec2formula verify --model-dir checkpoints/default
spec2formula predict --input examples/synthetic.jsonl --output predictions/demo.jsonl
```

If using Conda, `conda env create -f environment.yml` provides an alternative
environment definition. The reference versions are Python 3.11, NumPy 1.26.4
and PyTorch 2.5.1. Do not install into an existing research environment unless
you intend to change it.

### Checkpoint files

The local release folder already contains the two `.pt` files and frozen priors
under `checkpoints/default/` (about 89 MB combined). Git LFS tracking is configured
for `.pt` files. After cloning a published GitHub repository with LFS weights:

```bash
git lfs install
git lfs pull
spec2formula verify --model-dir checkpoints/default
```

If using a separate release asset, extract its `checkpoints/default/` folder to
the repository root. There is no automatic download or remote service call.
The small frozen prior files are required model assets, not optional datasets.

### Faster candidate enumeration (recommended)

The portable Python enumerator works without a compiler. Exhaustive search at
higher precursor masses is considerably faster with the optional C++17 backend:

```bash
python tools/build_enumerator.py
# macOS / other clang installations: python tools/build_enumerator.py --cxx clang++
```

This creates `_bin/enumerate` (or `enumerate.exe`) in this checkout. The CLI
automatically detects it when run from the repository root. To use it from
another directory, provide `--enumerator-binary /path/to/_bin/enumerate`.
Both backends enumerate the same candidate domain. No prebuilt binary is
required or distributed. Linux needs a C++17 compiler such as g++; Windows can
use MinGW-w64 g++.

## Predict your own spectra

Supported formats are JSONL and MGF. The default model supports singly
protonated **[M+H]+**, recorded **precursor m/z < 500**, and finite positive peaks.
Precursor-error-based cohort filtering is not applied. Candidate search uses
`max(0.001 Da, neutral_mass * 3 ppm)` with frozen elemental and chemical rules.

```bash
spec2formula predict --input your_spectra.mgf --output predictions/formulas.jsonl --top-k 20
```

For a missing normalized collision energy (NCE), supply `--nce 30` when that is
the appropriate experimental NCE. Otherwise missing NCE becomes zero with an
explicit warning in each affected result. Collision energy in eV is not NCE;
the parser does not silently equate the two.

`--top-k 0` writes every enumerated candidate. This option changes output length,
not the fixed Evidence budget of 20. Outputs are never automatically overwritten.
An invalid input stops the run with a nonzero exit code; completed records remain
in a clearly named `.partial` file rather than a success-named output file.

### Python API

```python
from spec2formula import Predictor

predictor = Predictor("checkpoints/default", device="cpu")
result = predictor.predict({
    "id": "example",
    "precursor_mz": 123.080441,
    "adduct": "[M+H]+",
    "collision_energy": 30,
    "peaks": [[79.0542, 100], [95.0491, 55], [123.0804, 8]],
}, top_k=20)
print(result["predictions"])
```

### Optional GPU execution

Install the PyTorch 2.5.1 wheel appropriate for your CUDA/driver combination in
the new environment, then pass `--device cuda:0`. `--precision bf16` additionally
requires a CUDA device supporting BF16. The default is CPU FP32.
Different precision, hardware or batching can change nearly tied ranks;
CPU FP32 output should not be described as bitwise reproduction of historical
CUDA BF16 evaluation.

## How inference works

1. Retain at most 128 finite positive peaks by intensity, normalize by the largest
   retained intensity, and order by m/z. Use precursor m/z and NCE as metadata.
2. Enumerate the complete frozen 16-element candidate domain inside the mass window.
3. Construct the 25 candidate features, including the supplied frozen training priors.
4. Score all candidates with the Candidate ranker; standardize scores over the whole
   candidate list using population SD with a floor of 0.001.
5. Build up to 32 exact fragment/neutral-loss tokens for each of the leading 20
   candidates. Apply the Evidence reranker and reorder only that prefix.
6. Append the Candidate-ranker tail unchanged. Reference formulas are never inputs
   and are never injected into the candidate list.

Scores are ranking scores, not calibrated probabilities. `evidence_score` is null
for tail candidates; do not sort the whole output by this partially populated field.
Use `rank` as the final order. `candidate_rank` preserves the first-stage result.

## Repository contents

```text
src/spec2formula/       models, chemistry, preprocessing, API and CLI
checkpoints/default/   inference-only weights, configs, priors and SHA256 manifest
native/                optional fast candidate enumerator source
tools/                 optional native build helper
examples/              one artificial spectrum in JSONL and MGF
tests/                 chemistry, input and end-to-end regression checks
docs/                  model card and file formats
```

The released reference pair is the validation-selected Candidate ranker (epoch
14) and Evidence reranker (epoch 7), both seed 520, used for the six-source
external evaluation. It is a single model pair, not a five-fold ensemble or a
new model fitted to all samples. Export removes optimizer states; all model
tensors are exactly preserved. Details are in the model card.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests use artificial spectra and the included checkpoints. Training and dataset preparation
are outside this inference release. Public repository URL, software/weight
license terms and publication citation will be completed before publication.
