# MassBank paper cohorts

These files fix the exact MassBank membership used in Section S8 of the paper.
They contain record identifiers, source and preprocessing checksums, attribution,
record-level licenses, evaluation labels and archived target ranks. Source
record text and peaks are obtained from the official MassBank checkout.

| Manifest | Spectra | Unique connectivity groups |
|---|---:|---:|
| orbitrap_manifest.csv | 18,583 | 1,760 |
| qtof_manifest.csv | 22,573 | 2,162 |

`PROVENANCE.json` gives source sizes, licenses, artifact hashes and processing
rules. Source files are pinned to MassBank-data release 2026.03, commit
`705afb7bccc3b2c42410a744eef73674716a60ef`. Group counts are union counts;
groups shared by several sources are counted once in the combined view.

## Rebuild exact inputs

Install Spec2Formula as described in the repository README. Obtain an official
MassBank-data checkout and select the pinned commit. Preserve original LF bytes
on Windows so that record checksums agree:

```bash
git -c core.autocrlf=false clone https://github.com/MassBank/MassBank-data.git ../MassBank-data
git -C ../MassBank-data -c core.autocrlf=false checkout --detach 705afb7bccc3b2c42410a744eef73674716a60ef

python tools/rebuild_massbank.py --manifest reproducibility/massbank/orbitrap_manifest.csv --source-root ../MassBank-data --output predictions/orbitrap_inputs.jsonl
python tools/rebuild_massbank.py --manifest reproducibility/massbank/qtof_manifest.csv --source-root ../MassBank-data --output predictions/qtof_inputs.jsonl
```

The manifest is the frozen membership and deduplication decision. Rebuilding
selects those records and verifies every source checksum, accession, license,
precursor and processed peak array. It does not rerun the historical full-source
eligibility, identifier-resolution or internal-connectivity-overlap audits.
The recorded disjointness labels preserve that prior audit's result.

Inputs retain only finite positive peaks, at most 128 peaks selected stably by
descending intensity, normalized by the largest intensity and ordered stably
by m/z, then represented as float32. Recorded precursor m/z must be below 500
and the precursor type must be [M+H]+. There is no additional precursor-error
quality filter. No evaluation formula is written to the inference JSONL.

Orbitrap uses the frozen Option-B scalar-energy compatibility view. The initial
conversion's narrower NCE-only column is not interchangeable with that view.
Both its source scalar and actual float32 model input are preserved.

QTOF uses `NCE_proxy = 500 * E_eV / precursor_mz`. Explicit scalar V is assigned
the same numerical energy under the singly charged-ion assumption; `title_V`
identifies a voltage unit recovered from the original record title. Missing,
unresolved or multivalued energies are not imputed. The recorded float32 input
and the model's existing `clamp(NCE / 200, 0, 2)` transformation are retained.
This proxy is an experimental input adaptation, not manufacturer-reported NCE.

## Recalculate archived tables

```bash
python tools/evaluate_massbank.py --manifest reproducibility/massbank/orbitrap_manifest.csv --frozen-ranks reproducibility/massbank/orbitrap_frozen_ranks.csv --output predictions/orbitrap_archived_metrics.json
python tools/evaluate_massbank.py --manifest reproducibility/massbank/qtof_manifest.csv --frozen-ranks reproducibility/massbank/qtof_frozen_ranks.csv --output predictions/qtof_archived_metrics.json
```

These commands recalculate the saved four-method Top-1 results. They perform
no inference. The supplied metric JSON files match the paper-facing source
tables. Target rank zero denotes an unsuccessful identification or absent
target; every rank other than one contributes zero to Top-1 accuracy.

Spectrum weighting assigns equal weight to each retained spectrum. Molecule
weighting first averages hits within each source/cohort-specific connectivity
group and then averages groups equally. Source and internal-disjoint views use
their own denominators. No failed or empty outputs are removed.

## Evaluate newly generated Spec2Formula predictions

```bash
spec2formula predict --input predictions/qtof_inputs.jsonl --output predictions/qtof_predictions.jsonl --top-k 1 --candidate-backend native
python tools/evaluate_massbank.py --manifest reproducibility/massbank/qtof_manifest.csv --predictions predictions/qtof_predictions.jsonl --output predictions/qtof_new_metrics.json
```

The evaluator uses accession IDs and reads reference formulas only after
prediction. Missing, empty and unsuccessful outputs remain failures. Duplicate
or unknown IDs are rejected. Repeat with the Orbitrap manifest for that cohort.
CPU FP32 and historical CUDA BF16 arithmetic can differ; newly generated
predictions are evaluated separately from archived results. This material
reproduces the external cohorts and saved results, not the five-fold training
or comparator software execution.

## Source licenses

Original license strings, authors, copyright notices, source URLs and publication
fields are retained per record. The source licenses cover those source-derived
fields and records; project MIT software licensing does not relicense them.

| Original license string | Orbitrap records | QTOF records |
|---|---:|---:|
| CC BY | 10,408 | 4,246 |
| CC BY-SA | 4,678 | 1,602 |
| CC0 | 3,497 | 299 |
| dl-de/by-2-0 | 0 | 12,813 |
| CC BY-NC-SA | 0 | 3,613 |

The 3,613 CC BY-NC-SA QTOF records retain their noncommercial and share-alike
conditions. Obtain and use source records under their original terms. License
strings without a version are preserved verbatim rather than assigned an
invented version. No NIST23 source spectra are included.
