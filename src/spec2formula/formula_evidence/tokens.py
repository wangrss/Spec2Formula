"""Candidate-specific, non-isotope exact fragment/loss evidence tokens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spec2formula.formula_evidence.exact_matcher import ExactMatcherResult
from spec2formula.formula_ranker.features import COUNT_LIMITS


EVIDENCE_TOKEN_NAMES = (
    "observed_mz_scaled",
    "sqrt_intensity_fraction",
    "neutral_loss_scaled",
    "fragment_hit",
    "loss_hit",
    "either_hit",
    "fragment_normalized_error",
    "loss_normalized_error",
) + tuple(f"fragment_{index}_count_scaled" for index in range(16)) + tuple(
    f"loss_{index}_count_scaled" for index in range(16)
)


@dataclass(frozen=True)
class EvidenceTokenBatch:
    tokens: np.ndarray
    padding_mask: np.ndarray


def exact_evidence_tokens(
    result: ExactMatcherResult,
    peaks: np.ndarray,
    precursor_mz: float,
    *,
    maximum_tokens: int = 32,
    maximum_mz: float = 500.0,
) -> EvidenceTokenBatch:
    """Convert exact assignments into one fixed-width token per selected peak."""

    if maximum_tokens < 1:
        raise ValueError("maximum_tokens must be positive")
    values = np.asarray(peaks, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("peaks must have shape (n, 2)")
    source_indices = np.asarray(result.selected_source_peak_indices, dtype=np.int64)
    if len(source_indices) > maximum_tokens:
        raise ValueError("matcher selected more peaks than the token budget")
    if len(source_indices) and (
        int(source_indices.min()) < 0 or int(source_indices.max()) >= len(values)
    ):
        raise ValueError("selected peak index is outside the input spectrum")
    tokens = np.zeros((maximum_tokens, len(EVIDENCE_TOKEN_NAMES)), dtype=np.float32)
    padding_mask = np.ones(maximum_tokens, dtype=np.bool_)
    if not len(source_indices):
        return EvidenceTokenBatch(tokens=tokens, padding_mask=padding_mask)

    selected = values[source_indices]
    intensity = np.maximum(selected[:, 1], 0.0)
    intensity_total = float(intensity.sum()) or 1.0
    valid_count = len(selected)
    tokens[:valid_count, 0] = (selected[:, 0] / maximum_mz).astype(np.float32)
    tokens[:valid_count, 1] = np.sqrt(intensity / intensity_total).astype(np.float32)
    tokens[:valid_count, 2] = (
        np.maximum(float(precursor_mz) - selected[:, 0], 0.0) / maximum_mz
    ).astype(np.float32)
    padding_mask[:valid_count] = False

    fragment_by_peak = {
        assignment.selected_peak_index: assignment
        for assignment in result.fragment_assignments
    }
    loss_by_peak = {
        assignment.selected_peak_index: assignment for assignment in result.loss_assignments
    }
    for index in range(valid_count):
        fragment = fragment_by_peak.get(index)
        loss = loss_by_peak.get(index)
        fragment_hit = fragment is not None
        loss_hit = loss is not None
        tokens[index, 3] = float(fragment_hit)
        tokens[index, 4] = float(loss_hit)
        tokens[index, 5] = float(fragment_hit or loss_hit)
        if fragment is not None:
            tokens[index, 6] = min(
                float(fragment.match.normalized_absolute_error), 2.0
            ) / 2.0
            tokens[index, 8:24] = (
                np.asarray(fragment.match.counts, dtype=np.float32) / COUNT_LIMITS
            )
        if loss is not None:
            tokens[index, 7] = min(float(loss.match.normalized_absolute_error), 2.0) / 2.0
            tokens[index, 24:40] = (
                np.asarray(loss.match.counts, dtype=np.float32) / COUNT_LIMITS
            )
    return EvidenceTokenBatch(tokens=tokens, padding_mask=padding_mask)
