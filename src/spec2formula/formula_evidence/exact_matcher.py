"""Non-learning candidate-conditioned exact fragment/loss/isotope matcher."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from spec2formula.formula.elements import ELEMENTS, EXPANDED_ELEMENTS
from spec2formula.formula_evidence.isotope import ISOTOPE_SHIFTS, nearest_shift_matches


PROTON_MASS_DA = 1.007276466621
ELEMENT_ORDER = EXPANDED_ELEMENTS
ELEMENT_INDEX = {element: index for index, element in enumerate(ELEMENT_ORDER)}
ELEMENT_MASSES = np.asarray(
    [ELEMENTS[element].monoisotopic_mass for element in ELEMENT_ORDER], dtype=np.float64
)


@dataclass(frozen=True)
class ExactMatcherConfig:
    ppm: float = 3.0
    floor_da: float = 0.001
    top_peaks: int = 32
    isotope_ppm: float = 3.0
    isotope_floor_da: float = 0.001
    precursor_slack_da: float = 0.01
    enable_isotope: bool = True


@dataclass(frozen=True)
class SubformulaMatch:
    counts: tuple[int, ...]
    exact_mass: float
    target_mass: float
    signed_error_da: float
    tolerance_da: float

    @property
    def normalized_absolute_error(self) -> float:
        return abs(self.signed_error_da) / self.tolerance_da


@dataclass(frozen=True)
class PeakAssignment:
    selected_peak_index: int
    source_peak_index: int
    peak_mz: float
    intensity: float
    kind: str
    match: SubformulaMatch


@dataclass(frozen=True)
class IsotopeAssignment:
    shift: str
    element: str
    base_selected_peak_index: int
    partner_selected_peak_index: int
    base_source_peak_index: int
    partner_source_peak_index: int
    spacing_signed_error_da: float
    base_fragment: SubformulaMatch
    chain_order: int


EXACT_FEATURE_NAMES = (
    "fragment_peak_fraction",
    "fragment_intensity_fraction",
    "loss_peak_fraction",
    "loss_intensity_fraction",
    "either_peak_fraction",
    "either_intensity_fraction",
    "fragment_mean_normalized_absolute_error",
    "loss_mean_normalized_absolute_error",
    "c13_pair_count_log1p",
    "c13_exact_supported_fraction",
    "c13_exact_supported_partner_intensity_fraction",
    "all_isotope_exact_supported_count_log1p",
)


@dataclass(frozen=True)
class ExactMatcherResult:
    features: tuple[float, ...]
    fragment_assignments: tuple[PeakAssignment, ...]
    loss_assignments: tuple[PeakAssignment, ...]
    isotope_assignments: tuple[IsotopeAssignment, ...]
    selected_source_peak_indices: tuple[int, ...]

    def feature_dict(self) -> dict[str, float]:
        return dict(zip(EXACT_FEATURE_NAMES, self.features, strict=True))

    def to_dict(self) -> dict[str, object]:
        return {
            "features": self.feature_dict(),
            "fragment_assignments": [asdict(value) for value in self.fragment_assignments],
            "loss_assignments": [asdict(value) for value in self.loss_assignments],
            "isotope_assignments": [asdict(value) for value in self.isotope_assignments],
            "selected_source_peak_indices": list(self.selected_source_peak_indices),
        }


def mass_tolerance_da(target_mass: float, ppm: float, floor_da: float) -> float:
    return max(float(floor_da), float(target_mass) * float(ppm) * 1.0e-6)


def counts_array(counts: Mapping[str, int] | Sequence[int] | np.ndarray) -> np.ndarray:
    if isinstance(counts, Mapping):
        unknown = sorted(set(counts) - set(ELEMENT_ORDER))
        if unknown:
            raise ValueError(f"unsupported elements: {unknown}")
        values = np.asarray([counts.get(element, 0) for element in ELEMENT_ORDER], dtype=np.int16)
    else:
        values = np.asarray(counts, dtype=np.int16)
    if values.shape != (len(ELEMENT_ORDER),):
        raise ValueError(f"counts must have shape ({len(ELEMENT_ORDER)},)")
    if bool(np.any(values < 0)):
        raise ValueError("negative element count")
    return values


class BoundedSubformulaMatcher:
    """Exact mass lookup over every subformula bounded by one parent formula."""

    def __init__(self, parent_counts: Mapping[str, int] | Sequence[int] | np.ndarray):
        self.parent_counts = counts_array(parent_counts)
        active = [index for index, count in enumerate(self.parent_counts) if count > 0]
        left: list[int] = []
        right: list[int] = []
        log_sizes = [0.0, 0.0]
        for index in sorted(active, key=lambda item: int(self.parent_counts[item]), reverse=True):
            side = 0 if log_sizes[0] <= log_sizes[1] else 1
            (left if side == 0 else right).append(index)
            log_sizes[side] += math.log(int(self.parent_counts[index]) + 1)
        self.left_elements = tuple(left)
        self.right_elements = tuple(right)
        self.left_masses, self.left_counts = self._enumerate_half(self.left_elements)
        right_masses, right_counts = self._enumerate_half(self.right_elements)
        order = np.argsort(right_masses, kind="stable")
        self.right_masses = right_masses[order]
        self.right_counts = right_counts[order]

    def _enumerate_half(self, indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if not indices:
            return np.asarray([0.0], dtype=np.float64), np.zeros((1, 0), dtype=np.int16)
        ranges = [range(int(self.parent_counts[index]) + 1) for index in indices]
        rows = np.asarray(list(itertools.product(*ranges)), dtype=np.int16)
        masses = rows.astype(np.float64) @ ELEMENT_MASSES[np.asarray(indices, dtype=np.int64)]
        return masses, rows

    @property
    def partial_state_count(self) -> int:
        return int(len(self.left_masses) + len(self.right_masses))

    def _minimum_mask(self, side: str, minimum: np.ndarray) -> np.ndarray:
        indices = self.left_elements if side == "left" else self.right_elements
        rows = self.left_counts if side == "left" else self.right_counts
        if not indices:
            return np.ones(len(rows), dtype=bool)
        required = minimum[np.asarray(indices, dtype=np.int64)]
        return np.all(rows >= required[None, :], axis=1)

    def best_match(
        self,
        target_mass: float,
        *,
        ppm: float = 3.0,
        floor_da: float = 0.001,
        minimum_counts: Mapping[str, int] | Sequence[int] | np.ndarray | None = None,
    ) -> SubformulaMatch | None:
        target_mass = float(target_mass)
        if not math.isfinite(target_mass) or target_mass <= 0.0:
            return None
        if minimum_counts is None:
            left_indices = None
            right_indices = None
            left_mass = self.left_masses
            right_mass = self.right_masses
        else:
            minimum = counts_array(minimum_counts)
            if bool(np.any(minimum > self.parent_counts)):
                return None
            left_indices = np.flatnonzero(self._minimum_mask("left", minimum))
            right_indices = np.flatnonzero(self._minimum_mask("right", minimum))
            if not len(left_indices) or not len(right_indices):
                return None
            left_mass = self.left_masses[left_indices]
            right_mass = self.right_masses[right_indices]
        desired = target_mass - left_mass
        positions = np.searchsorted(right_mass, desired, side="left")
        right_choice = np.clip(positions, 0, len(right_mass) - 1)
        right_before = np.clip(positions - 1, 0, len(right_mass) - 1)
        error_choice = np.abs(left_mass + right_mass[right_choice] - target_mass)
        error_before = np.abs(left_mass + right_mass[right_before] - target_mass)
        use_before = error_before <= error_choice
        selected_right_local = np.where(use_before, right_before, right_choice)
        errors = np.where(use_before, error_before, error_choice)
        best_local = int(np.argmin(errors))
        tolerance = mass_tolerance_da(target_mass, ppm, floor_da)
        if float(errors[best_local]) > tolerance:
            return None
        left_row = (
            best_local if left_indices is None else int(left_indices[best_local])
        )
        chosen_right = int(selected_right_local[best_local])
        right_row = (
            chosen_right if right_indices is None else int(right_indices[chosen_right])
        )
        result_counts = np.zeros(len(ELEMENT_ORDER), dtype=np.int16)
        if self.left_elements:
            result_counts[np.asarray(self.left_elements)] = self.left_counts[left_row]
        if self.right_elements:
            result_counts[np.asarray(self.right_elements)] = self.right_counts[right_row]
        if not bool(np.any(result_counts)):
            return None
        exact_mass = float(result_counts.astype(np.float64) @ ELEMENT_MASSES)
        return SubformulaMatch(
            counts=tuple(int(value) for value in result_counts),
            exact_mass=exact_mass,
            target_mass=target_mass,
            signed_error_da=exact_mass - target_mass,
            tolerance_da=tolerance,
        )


def _select_peaks(
    peaks: np.ndarray,
    precursor_mz: float,
    top_peaks: int,
    precursor_slack_da: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(peaks, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("peaks must have shape (n, 2)")
    valid = (
        np.isfinite(values[:, 0])
        & np.isfinite(values[:, 1])
        & (values[:, 0] > 0.0)
        & (values[:, 1] > 0.0)
    )
    indices = np.flatnonzero(valid)
    if len(indices) > top_peaks:
        ranked = np.argsort(values[indices, 1], kind="stable")[-top_peaks:]
        indices = indices[ranked]
    order = np.argsort(values[indices, 0], kind="stable")
    indices = indices[order]
    return values[indices], indices


def _minimum_for_element(element: str, count: int = 1) -> np.ndarray:
    values = np.zeros(len(ELEMENT_ORDER), dtype=np.int16)
    values[ELEMENT_INDEX[element]] = count
    return values


def match_candidate_spectrum(
    parent_counts: Mapping[str, int] | Sequence[int] | np.ndarray,
    peaks: np.ndarray,
    precursor_mz: float,
    config: ExactMatcherConfig = ExactMatcherConfig(),
) -> ExactMatcherResult:
    """Return exact local assignments and fixed candidate-level evidence features."""

    matcher = BoundedSubformulaMatcher(parent_counts)
    selected, source_indices = _select_peaks(
        peaks, precursor_mz, config.top_peaks, config.precursor_slack_da
    )
    if not len(selected):
        return ExactMatcherResult(
            features=tuple(0.0 for _ in EXACT_FEATURE_NAMES),
            fragment_assignments=(),
            loss_assignments=(),
            isotope_assignments=(),
            selected_source_peak_indices=(),
        )
    intensity = selected[:, 1]
    intensity_sum = float(intensity.sum()) or 1.0
    fragment_assignments: list[PeakAssignment] = []
    loss_assignments: list[PeakAssignment] = []
    fragment_errors: list[float] = []
    loss_errors: list[float] = []
    fragment_hit = np.zeros(len(selected), dtype=bool)
    loss_hit = np.zeros(len(selected), dtype=bool)
    for index, (peak_mz, peak_intensity) in enumerate(selected):
        fragment = matcher.best_match(
            float(peak_mz) - PROTON_MASS_DA,
            ppm=config.ppm,
            floor_da=config.floor_da,
        )
        if fragment is not None:
            fragment_hit[index] = True
            fragment_errors.append(fragment.normalized_absolute_error)
            fragment_assignments.append(PeakAssignment(
                selected_peak_index=index,
                source_peak_index=int(source_indices[index]),
                peak_mz=float(peak_mz),
                intensity=float(peak_intensity),
                kind="protonated_fragment",
                match=fragment,
            ))
        loss = matcher.best_match(
            float(precursor_mz) - float(peak_mz),
            ppm=config.ppm,
            floor_da=config.floor_da,
        )
        if loss is not None:
            loss_hit[index] = True
            loss_errors.append(loss.normalized_absolute_error)
            loss_assignments.append(PeakAssignment(
                selected_peak_index=index,
                source_peak_index=int(source_indices[index]),
                peak_mz=float(peak_mz),
                intensity=float(peak_intensity),
                kind="neutral_loss",
                match=loss,
            ))

    isotope_assignments: list[IsotopeAssignment] = []
    c13_pair_count = 0
    c13_supported = 0
    c13_supported_partner_intensity = 0.0
    all_supported = 0
    isotope_shifts = ISOTOPE_SHIFTS if config.enable_isotope else ()
    for shift in isotope_shifts:
        if int(matcher.parent_counts[ELEMENT_INDEX[shift.element]]) == 0:
            continue
        spacing = nearest_shift_matches(
            selected[:, 0],
            selected[:, 1],
            shift.delta_da,
            ppm=config.isotope_ppm,
            floor_da=config.isotope_floor_da,
        )
        eligible_base = (
            selected[spacing["base_index"], 0]
            <= float(precursor_mz) + config.precursor_slack_da
        )
        spacing = {key: value[eligible_base] for key, value in spacing.items()}
        edge_set = set(zip(spacing["base_index"].tolist(), spacing["partner_index"].tolist()))
        base_set = set(spacing["base_index"].tolist())
        if shift.name == "C13_minus_C12":
            c13_pair_count = len(edge_set)
        minimum_one = _minimum_for_element(shift.element, 1)
        minimum_two = _minimum_for_element(shift.element, 2)
        for edge_index, (base, partner) in enumerate(
            zip(spacing["base_index"], spacing["partner_index"], strict=True)
        ):
            base_index = int(base)
            partner_index = int(partner)
            chain_order = 2 if partner_index in base_set else 1
            minimum = minimum_two if chain_order == 2 else minimum_one
            base_match = matcher.best_match(
                float(selected[base_index, 0]) - PROTON_MASS_DA,
                ppm=config.ppm,
                floor_da=config.floor_da,
                minimum_counts=minimum,
            )
            if base_match is None:
                continue
            all_supported += 1
            if shift.name == "C13_minus_C12":
                c13_supported += 1
                c13_supported_partner_intensity += float(selected[partner_index, 1])
            isotope_assignments.append(IsotopeAssignment(
                shift=shift.name,
                element=shift.element,
                base_selected_peak_index=base_index,
                partner_selected_peak_index=partner_index,
                base_source_peak_index=int(source_indices[base_index]),
                partner_source_peak_index=int(source_indices[partner_index]),
                spacing_signed_error_da=float(spacing["signed_residual_da"][edge_index]),
                base_fragment=base_match,
                chain_order=chain_order,
            ))

    either_hit = fragment_hit | loss_hit
    features = (
        float(np.mean(fragment_hit)),
        float(intensity[fragment_hit].sum() / intensity_sum),
        float(np.mean(loss_hit)),
        float(intensity[loss_hit].sum() / intensity_sum),
        float(np.mean(either_hit)),
        float(intensity[either_hit].sum() / intensity_sum),
        float(np.mean(fragment_errors)) if fragment_errors else 1.0,
        float(np.mean(loss_errors)) if loss_errors else 1.0,
        math.log1p(c13_pair_count),
        float(c13_supported / c13_pair_count) if c13_pair_count else 0.0,
        float(c13_supported_partner_intensity / intensity_sum),
        math.log1p(all_supported),
    )
    return ExactMatcherResult(
        features=features,
        fragment_assignments=tuple(fragment_assignments),
        loss_assignments=tuple(loss_assignments),
        isotope_assignments=tuple(isotope_assignments),
        selected_source_peak_indices=tuple(int(value) for value in source_indices),
    )
