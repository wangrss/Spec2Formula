"""Pure MS2 isotope-spacing utilities with no abundance hard constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class IsotopeShift:
    name: str
    element: str
    delta_da: float
    nominal_order: int


ISOTOPE_SHIFTS = (
    IsotopeShift("B11_minus_B10", "B", 0.9963684, 1),
    IsotopeShift("N15_minus_N14", "N", 0.9970349, 1),
    IsotopeShift("Si29_minus_Si28", "Si", 0.99956817, 1),
    IsotopeShift("C13_minus_C12", "C", 1.00335484, 1),
    IsotopeShift("S34_minus_S32", "S", 1.9957959, 2),
    IsotopeShift("Si30_minus_Si28", "Si", 1.99684364, 2),
    IsotopeShift("Cl37_minus_Cl35", "Cl", 1.99704991, 2),
    IsotopeShift("Br81_minus_Br79", "Br", 1.9979535, 2),
    IsotopeShift("Se82_minus_Se80", "Se", 2.0001781, 2),
    IsotopeShift("O18_minus_O16", "O", 2.00424638, 2),
)

FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula_counts(formula: str) -> dict[str, int]:
    position = 0
    result: dict[str, int] = {}
    for match in FORMULA_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"invalid formula: {formula!r}")
        element, count = match.groups()
        result[element] = result.get(element, 0) + (int(count) if count else 1)
        position = match.end()
    if position != len(formula) or not result:
        raise ValueError(f"invalid formula: {formula!r}")
    return result


def tolerance_da(target_mz: np.ndarray | float, ppm: float, floor_da: float):
    return np.maximum(float(floor_da), np.asarray(target_mz) * float(ppm) * 1.0e-6)


def nearest_shift_matches(
    mz: np.ndarray,
    intensity: np.ndarray,
    shift_da: float,
    *,
    ppm: float,
    floor_da: float,
) -> dict[str, np.ndarray]:
    """Match each base peak to its nearest observed partner near base+shift."""

    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.ndim != 1 or intensity.shape != mz.shape:
        raise ValueError("mz and intensity must be same-length vectors")
    if len(mz) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "base_index": empty_i,
            "partner_index": empty_i.copy(),
            "residual_da": empty_f,
            "signed_residual_da": empty_f.copy(),
            "target_mz": empty_f.copy(),
            "log10_intensity_ratio": empty_f.copy(),
        }
    if not bool(np.all(np.diff(mz) >= 0)):
        raise ValueError("mz must be sorted")

    target = mz + float(shift_da)
    positions = np.searchsorted(mz, target, side="left")
    right = np.clip(positions, 0, len(mz) - 1)
    left = np.clip(positions - 1, 0, len(mz) - 1)
    right_error = np.abs(mz[right] - target)
    left_error = np.abs(mz[left] - target)
    use_right = right_error < left_error
    partner = np.where(use_right, right, left)
    residual = np.where(use_right, right_error, left_error)
    signed_residual = mz[partner] - target
    valid = residual <= tolerance_da(target, ppm, floor_da)
    base = np.flatnonzero(valid)
    partner = partner[valid]
    residual = residual[valid]
    signed_residual = signed_residual[valid]
    target = target[valid]
    ratio = np.log10(np.maximum(intensity[partner], 1.0e-30) / np.maximum(intensity[base], 1.0e-30))
    return {
        "base_index": base.astype(np.int64, copy=False),
        "partner_index": partner.astype(np.int64, copy=False),
        "residual_da": residual,
        "signed_residual_da": signed_residual,
        "target_mz": target,
        "log10_intensity_ratio": ratio,
    }


def choose_frozen_tolerance(rows: Iterable[Mapping[str, float]]) -> dict[str, float | str | bool]:
    """Apply the preregistered deterministic tolerance-selection rule."""

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("no tolerance rows")
    eligible = [
        row
        for row in candidates
        if float(row["precision_proxy"]) >= 0.80
        and float(row["retention_proxy"]) >= 0.80
    ]
    used_fallback = not eligible
    if eligible:
        pool = eligible
        reason = "precision_and_retention_threshold"
    else:
        for row in candidates:
            precision = max(float(row["precision_proxy"]), 0.0)
            retention = max(float(row["retention_proxy"]), 0.0)
            row["harmonic_proxy"] = (
                2.0 * precision * retention / (precision + retention)
                if precision + retention > 0
                else 0.0
            )
        maximum = max(float(row["harmonic_proxy"]) for row in candidates)
        pool = [row for row in candidates if abs(float(row["harmonic_proxy"]) - maximum) <= 1.0e-15]
        reason = "maximum_harmonic_proxy_fallback"
    selected = min(
        pool,
        key=lambda row: (
            max(float(row["floor_da"]), 250.0 * float(row["ppm"]) * 1.0e-6),
            float(row["floor_da"]),
            float(row["ppm"]),
        ),
    )
    return {
        "ppm": float(selected["ppm"]),
        "floor_da": float(selected["floor_da"]),
        "precision_proxy": float(selected["precision_proxy"]),
        "retention_proxy": float(selected["retention_proxy"]),
        "selection_reason": reason,
        "used_fallback": used_fallback,
    }
