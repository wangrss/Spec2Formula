# Model card: spec2formula-reference-v1

## Identity

| Component | Parameters | Selection |
|---|---:|---|
| Candidate ranker | 19,221,044 | Epoch 14, seed 520 |
| Evidence reranker | 2,982,019 | Epoch 7, seed 520; Candidate ranker frozen |

This is the frozen reference pair used in the six-source external evaluation.
It corresponds to one cross-validation fold's validation-selected pair. It is
not an ensemble and was not retrained on all data. Five-fold mean performance
describes five independently evaluated pairs, not a guaranteed score for this
single released pair on arbitrary user data.

## Architecture and inputs

The Candidate ranker uses a six-layer, 512-dimensional spectrum Transformer,
a 25-feature candidate encoder and cosine-plus-interaction scoring. The
Evidence reranker uses four 256-dimensional Transformer layers and 40-dimensional
candidate-conditioned fragment/neutral-loss tokens. Isotope evidence is disabled
for this model path. Complete architecture hyperparameters are in `config.json`.

Candidate elements, in feature order, are C, H, B, N, O, F, Si, P, S, Cl, As, Se,
Br, Sn, Sb and I. Count limits are 43,77,1,13,19,17,5,5,8,8,1,2,6,1,1,5.
Carbon is required; nonnegative integer DBE and frozen valence/Senior constraints
apply. Mass constants use the same reference isotopes as training, including
Se-80 and Sn-120. These constants must not be silently replaced by a chemistry
library's different isotope convention.

## Supported use

- High-resolution, positive-ion `[M+H]+` MS/MS with precursor m/z <500.
- At most 128 intensity-selected peaks for the spectrum encoder, and 32 peaks
  for exact-evidence construction.
- NCE metadata where available; missing-NCE zero fallback is explicitly reported.
- Full candidate search with a 3-ppm tolerance and 0.001-Da floor.
- Fixed top-20 evidence reranking, preserving the complete candidate tail.

No `<1 ppm` precursor-error cohort filter is applied. Search tolerance is a
candidate-generation parameter and is different from an input cohort filter.
Formula identification does not identify a unique molecular structure.
Unsupported adducts/elements, chemistry outside the frozen domain or inaccurate
precursor measurements can exclude the correct formula.

## Artifacts and numerical behavior

Exported checkpoints contain state dictionaries only. All tensors are checked
against the frozen source checkpoints; optimizer states, training data, paths
and training caches are omitted. `manifest.json` contains the checksums needed
to verify all six runtime assets when loading the model.

The three prior files contain aggregate formula frequencies and per-element
count probabilities estimated on 25,965 training connectivity groups. They are
necessary for these weights and do not contain spectra, structures, molecule
identifiers or per-sample records. Replacing them with zeros changes the model.

Live inference constructs evidence in FP32 without an on-disk feature cache,
as in the external-evaluation interface. Historical internal evaluation also
used precomputed FP16 evidence/embedding caches and CUDA BF16 arithmetic.
The default public CPU FP32 path does not claim bitwise equality to that cached
GPU path. Exported weights and CPU FP32 reference parity were verified before
release. CUDA execution has not yet been validated for this release.

## Distribution

The local release contains the actual `.pt` files, not LFS pointer stubs.
`.gitattributes` configures Git LFS for eventual GitHub publication. A separate
checkpoint release archive can alternatively be used. Code and checkpoint
license terms, authorship metadata, publication citation and public URLs remain
to be supplied before public distribution; this local handoff adds no license grant.
