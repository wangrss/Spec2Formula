"""Frozen monoisotopic masses and nominal valences for Stage 1D."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementSpec:
    symbol: str
    monoisotopic_mass: float
    nominal_valence: int


# Masses use the most abundant naturally occurring isotope, consistent with
# ordinary neutral-formula monoisotopic mass calculations.
ELEMENTS: dict[str, ElementSpec] = {
    "H": ElementSpec("H", 1.00782503223, 1),
    "B": ElementSpec("B", 11.00930536, 3),
    "C": ElementSpec("C", 12.0, 4),
    "N": ElementSpec("N", 14.00307400443, 3),
    "O": ElementSpec("O", 15.99491461957, 2),
    "F": ElementSpec("F", 18.99840316273, 1),
    "Si": ElementSpec("Si", 27.97692653465, 4),
    "P": ElementSpec("P", 30.97376199842, 3),
    "S": ElementSpec("S", 31.9720711744, 2),
    "Cl": ElementSpec("Cl", 34.968852682, 1),
    "As": ElementSpec("As", 74.92159457, 3),
    "Se": ElementSpec("Se", 79.9165218, 2),
    "Br": ElementSpec("Br", 78.9183376, 1),
    "Sn": ElementSpec("Sn", 119.90220163, 4),
    "Sb": ElementSpec("Sb", 120.9038120, 3),
    "I": ElementSpec("I", 126.904468, 1),
}

CORE_ELEMENTS: tuple[str, ...] = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
EXPANDED_ELEMENTS: tuple[str, ...] = (
    "C",
    "H",
    "B",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "As",
    "Se",
    "Br",
    "Sn",
    "Sb",
    "I",
)


def exact_mass(counts: dict[str, int]) -> float:
    unknown = sorted(set(counts) - set(ELEMENTS))
    if unknown:
        raise ValueError(f"unsupported elements: {unknown}")
    return sum(ELEMENTS[element].monoisotopic_mass * count for element, count in counts.items())


def hill_formula(counts: dict[str, int]) -> str:
    present = {element: count for element, count in counts.items() if count > 0}
    order: list[str] = []
    if "C" in present:
        order.append("C")
        if "H" in present:
            order.append("H")
    order.extend(sorted(element for element in present if element not in order))
    return "".join(
        element + (str(present[element]) if present[element] != 1 else "")
        for element in order
    )
