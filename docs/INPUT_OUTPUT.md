# Input and output

## JSONL

One JSON object per line:

```json
{"id":"sample-1","precursor_mz":123.080441,"adduct":"[M+H]+","collision_energy":30,"peaks":[[79.0542,100],[95.0491,55]]}
```

| Field | Meaning |
|---|---|
| `id` | Optional identifier, copied into output; line number used when omitted by CLI |
| `precursor_mz` | Required measured protonated precursor m/z in Da, <500 |
| `adduct` | `[M+H]+`; this is the assumed value if omitted |
| `collision_energy` or `nce` | Normalized collision energy, not eV; optional |
| `peaks` | Required list of [m/z, intensity] pairs |

Other fields, including any reference-formula field, do not enter inference.
Each spectrum is independent. Duplicate identifiers are preserved; output row
order corresponds to input order. Nonfinite/nonpositive peaks are discarded
with a record-level warning, and entirely invalid spectra fail explicitly.

## MGF

```text
BEGIN IONS
TITLE=sample-1
PEPMASS=123.080441
CHARGE=1+
ADDUCT=[M+H]+
NCE=30
79.0542 100
95.0491 55
END IONS
```

`PEPMASS` uses its first number. `CHARGE`, if supplied, must be +1. `ADDUCT`
or `ION`, if supplied, must be `[M+H]+`. The `NCE` field accepts a number,
optionally followed by `%` or `NCE`. Generic `CE`/`COLLISIONENERGY` values
are not automatically interpreted as NCE. Use `--nce` for records missing NCE
only when a suitable normalized value is known.

## JSONL output

One output object per input spectrum. Metadata includes model version, precursor,
NCE used, peak count, candidate count, reranked count, device, precision,
enumeration backend and warnings. `status` is `ok` or `no_candidates`.
No candidate-domain matches gives a valid empty result, not an injected formula.

Each row in `predictions` contains:

| Field | Meaning |
|---|---|
| `rank` | Spec2Formula rank, starting at 1 |
| `formula` | Neutral elemental formula in Hill notation |
| `neutral_exact_mass` | Neutral mass calculated using the frozen reference-isotope masses |
| `precursor_error_ppm` | Signed `(observed - predicted protonated m/z) / predicted * 1e6` |
| `candidate_rank` | Rank before Evidence reranking |
| `candidate_score` | Candidate ranker raw score |
| `candidate_score_z` | Score standardized across the full candidate list |
| `evidence_score` | Local residual score, or null outside the reranked prefix |

The algorithm concatenates the reordered first 20 candidates and the unchanged
tail. Local evidence scores are not a globally comparable score for tail entries.
None of these score fields is a calibrated identification probability.

