"""Candidate formula features and bounded nominal subformula evidence."""

from __future__ import annotations

import numpy as np


PROTON_MASS_DA = 1.007276466621
COUNT_LIMITS = np.asarray([43, 77, 1, 13, 19, 17, 5, 5, 8, 8, 1, 2, 6, 1, 1, 5], dtype=np.float32)
COUNT_BIT_WIDTHS = np.asarray([6, 7, 1, 4, 5, 5, 3, 3, 4, 4, 1, 2, 3, 1, 1, 3], dtype=np.uint64)
COUNT_BIT_SHIFTS = np.concatenate(
    (np.asarray([0], dtype=np.uint64), np.cumsum(COUNT_BIT_WIDTHS[:-1], dtype=np.uint64))
)


def pack_formula_counts(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.uint64)
    if values.shape[-1] != 16:
        raise ValueError("formula counts must have 16 elements")
    return np.bitwise_or.reduce(values << COUNT_BIT_SHIFTS, axis=-1)


def _bit_membership(bitsets: np.ndarray, nominal_masses: np.ndarray) -> np.ndarray:
    masses = np.asarray(nominal_masses, dtype=np.int64)
    valid = (masses >= 0) & (masses < 512)
    clipped = np.clip(masses, 0, 511)
    words = clipped // 64
    shifts = (clipped % 64).astype(np.uint64)
    selected = np.asarray(bitsets, dtype=np.uint64)[:, words]
    hits = ((selected >> shifts[None, :]) & np.uint64(1)).astype(np.float32)
    hits[:, ~valid] = 0.0
    return hits


def nominal_bottom_up_evidence(
    bitsets: np.ndarray,
    peaks: np.ndarray,
    peak_length: int,
    precursor_mz: float,
    *,
    top_peaks: int = 32,
) -> np.ndarray:
    """Return four candidate-specific nominal fragment/loss feasibility features."""

    candidate_count = int(bitsets.shape[0])
    if candidate_count == 0:
        return np.empty((0, 4), dtype=np.float32)
    values = np.asarray(peaks[: int(peak_length)], dtype=np.float32)
    if len(values) == 0:
        return np.zeros((candidate_count, 4), dtype=np.float32)
    order = np.argsort(values[:, 1], kind="stable")[::-1][:top_peaks]
    values = values[order]
    intensity = values[:, 1].astype(np.float32)
    intensity_sum = float(intensity.sum()) or 1.0
    fragment_nominal = np.rint(np.maximum(values[:, 0] - PROTON_MASS_DA, 0.0)).astype(np.int64)
    loss_nominal = np.rint(np.maximum(float(precursor_mz) - values[:, 0], 0.0)).astype(np.int64)
    fragment_hits = _bit_membership(bitsets, fragment_nominal)
    loss_hits = _bit_membership(bitsets, loss_nominal)
    return np.stack(
        (
            fragment_hits.mean(axis=1),
            (fragment_hits * intensity[None, :]).sum(axis=1) / intensity_sum,
            loss_hits.mean(axis=1),
            (loss_hits * intensity[None, :]).sum(axis=1) / intensity_sum,
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def normalized_candidate_features(
    counts: np.ndarray,
    exact_mass: np.ndarray,
    dbe: np.ndarray,
    neutral_mass: float,
    evidence: np.ndarray,
) -> np.ndarray:
    counts_scaled = np.asarray(counts, dtype=np.float32) / COUNT_LIMITS[None, :]
    mass_scaled = np.asarray(exact_mass, dtype=np.float32)[:, None] / 500.0
    dbe_scaled = np.asarray(dbe, dtype=np.float32)[:, None] / 40.0
    half_width = max(0.001, float(neutral_mass) * 3.0e-6)
    mass_error = (
        np.abs(np.asarray(exact_mass, dtype=np.float64) - float(neutral_mass)) / half_width
    ).astype(np.float32)[:, None]
    return np.concatenate((counts_scaled, mass_scaled, dbe_scaled, mass_error, evidence), axis=1)


def neural_candidate_features(
    counts: np.ndarray,
    exact_mass: np.ndarray,
    dbe: np.ndarray,
    neutral_mass: float,
    evidence: np.ndarray,
    prior_features: np.ndarray,
) -> np.ndarray:
    physical = normalized_candidate_features(counts, exact_mass, dbe, neutral_mass, evidence)
    priors = np.asarray(prior_features, dtype=np.float32).copy()
    priors[:, 0] /= 50.0
    priors[:, 1] /= 5.0
    return np.concatenate((physical, priors), axis=1)
