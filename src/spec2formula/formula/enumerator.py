"""Target-independent exhaustive enumeration inside a neutral-mass window."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .elements import CORE_ELEMENTS, ELEMENTS, exact_mass, hill_formula
from .rules import evaluate_formula_rules


@dataclass(frozen=True)
class FormulaDomainConfig:
    elements: tuple[str, ...] = CORE_ELEMENTS
    fixed_max_counts: Mapping[str, int] = field(default_factory=dict)
    require_carbon: bool = True
    minimum_dbe: float = 0.0
    enforce_integer_dbe: bool = True
    enforce_senior: bool = True

    def __post_init__(self) -> None:
        if "H" not in self.elements:
            raise ValueError("hydrogen must be enabled")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("elements must be unique")
        unknown = sorted(set(self.elements) - set(ELEMENTS))
        if unknown:
            raise ValueError(f"unsupported elements: {unknown}")
        if any(value < 0 for value in self.fixed_max_counts.values()):
            raise ValueError("fixed maximum counts must be nonnegative")


@dataclass(frozen=True)
class FormulaCandidate:
    formula: str
    exact_mass: float
    dbe: float
    counts: tuple[tuple[str, int], ...]


def _physical_maximum(element: str, high_mass: float) -> int:
    return max(0, math.floor(high_mass / ELEMENTS[element].monoisotopic_mass + 1.0e-12))


def enumerate_formula_domain(
    mass_window: tuple[float, float],
    config: FormulaDomainConfig,
) -> list[FormulaCandidate]:
    """Enumerate every configured formula satisfying mass and frozen rules.

    Hydrogen is resolved last from the residual mass interval. No target
    formula, learned probability, or database-membership predicate is accepted.
    """

    low_mass, high_mass = (float(mass_window[0]), float(mass_window[1]))
    if low_mass <= 0.0 or high_mass < low_mass:
        raise ValueError("mass_window must be a positive ordered interval")

    heavy_elements = [element for element in config.elements if element != "H"]
    # Large-mass elements first sharply reduce branching while preserving the
    # same mathematical domain. Formula output is sorted canonically later.
    heavy_elements.sort(key=lambda element: ELEMENTS[element].monoisotopic_mass, reverse=True)
    maxima: dict[str, int] = {}
    for element in config.elements:
        physical = _physical_maximum(element, high_mass)
        configured = config.fixed_max_counts.get(element)
        maxima[element] = physical if configured is None else min(physical, int(configured))

    suffix_max_mass = [0.0] * (len(heavy_elements) + 1)
    for index in range(len(heavy_elements) - 1, -1, -1):
        element = heavy_elements[index]
        suffix_max_mass[index] = (
            suffix_max_mass[index + 1]
            + maxima[element] * ELEMENTS[element].monoisotopic_mass
        )
    hydrogen_mass = ELEMENTS["H"].monoisotopic_mass
    hydrogen_global_max = maxima["H"]
    candidates: list[FormulaCandidate] = []
    counts: dict[str, int] = {}

    def recurse(index: int, partial_mass: float) -> None:
        if index == len(heavy_elements):
            hydrogen_low = max(0, math.ceil((low_mass - partial_mass) / hydrogen_mass - 1.0e-10))
            hydrogen_high = min(
                hydrogen_global_max,
                math.floor((high_mass - partial_mass) / hydrogen_mass + 1.0e-10),
            )
            if hydrogen_high < hydrogen_low:
                return
            for hydrogen_count in range(hydrogen_low, hydrogen_high + 1):
                counts["H"] = hydrogen_count
                materialized = {element: count for element, count in counts.items() if count > 0}
                rules = evaluate_formula_rules(
                    materialized,
                    require_carbon=config.require_carbon,
                    minimum_dbe=config.minimum_dbe,
                    enforce_integer_dbe=config.enforce_integer_dbe,
                    enforce_senior=config.enforce_senior,
                )
                if not rules.passes:
                    continue
                mass = partial_mass + hydrogen_count * hydrogen_mass
                if low_mass - 1.0e-10 <= mass <= high_mass + 1.0e-10:
                    candidates.append(
                        FormulaCandidate(
                            formula=hill_formula(materialized),
                            exact_mass=mass,
                            dbe=rules.dbe,
                            counts=tuple(sorted(materialized.items())),
                        )
                    )
            counts.pop("H", None)
            return

        element = heavy_elements[index]
        element_mass = ELEMENTS[element].monoisotopic_mass
        minimum_count = 1 if config.require_carbon and element == "C" else 0
        maximum_count = min(maxima[element], math.floor((high_mass - partial_mass) / element_mass + 1.0e-12))
        remaining_max = suffix_max_mass[index + 1] + hydrogen_global_max * hydrogen_mass
        for count in range(minimum_count, maximum_count + 1):
            next_mass = partial_mass + count * element_mass
            if next_mass > high_mass + 1.0e-10:
                break
            if next_mass + remaining_max < low_mass - 1.0e-10:
                continue
            counts[element] = count
            recurse(index + 1, next_mass)
        counts.pop(element, None)

    recurse(0, 0.0)
    candidates.sort(key=lambda item: (item.exact_mass, item.formula))
    return candidates
