# Reviewed release changes

Prepared on 2026-10-14 from GitHub commit
`3496f81ba57a209b61a7fd5b479c5d5b11061d4f`.

- CLI output is reserved before model loading and published atomically without
  replacing an existing target. Partial results remain available on failure.
- Artificial regression fixtures freeze candidate features, exact-evidence
  tokens, scores and a complete 23-candidate final ranking after independent
  reference-code parity. Tests cover chunking, complete-domain standardization,
  native enumeration and competing output writers.
- A GitHub Actions workflow runs reference CPU checks with LFS checkpoint
  retrieval. Its first hosted run occurs after publication of the workflow.
- Software licensing is MIT. The author has expressly deferred model asset
  licensing; these assets are excluded from the project MIT License, as
  recorded in `checkpoints/default/LICENSE`.
- MassBank manifests identify all 18,583 Orbitrap and 22,573 common QTOF
  records with original licenses and attribution. Rebuilding verifies source
  and processed peak checksums; evaluation retains all failures and reproduces
  the archived four-method Top-1 tables.
- Git preserves the checksum-bearing MassBank CSV bytes across platforms.
- Frozen scientific configuration is checked on loading. The C++ enumerator
  exposes its compiled domain, isotope masses and nominal valences for Python
  compatibility checks. Unused CSV code was removed.

Validation: 30 tests passed without skips on Python 3.11.16, NumPy 1.26.4 and
PyTorch 2.5.1 CPU FP32. JSONL/MGF CLI outputs match; a built wheel passes isolated
inference. Five artificial neutral masses, including a 43,873-candidate case,
exactly match the original C++ enumeration. All seven pre-existing model asset
files, including configuration and manifest, retain their original SHA256.

No training, model tuning, new MassBank model inference, CUDA validation or
five-fold metric replay was performed. MassBank input reconstruction and
archived metric arithmetic were verified. Raw MassBank and NIST23 source
spectra are excluded from this release. GitHub publication and the first hosted
CI run are separate from these local and remote validations.
