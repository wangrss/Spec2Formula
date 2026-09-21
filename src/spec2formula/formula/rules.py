"""DBE, valence parity, and SENIOR checks for neutral formulas."""

from __future__ import annotations

from dataclasses import dataclass

from .elements import ELEMENTS


@dataclass(frozen=True)
class FormulaRuleResult:
    passes: bool
    dbe: float
    atom_count: int
    valence_sum: int
    reasons: tuple[str, ...]


def generalized_dbe(counts: dict[str, int]) -> float:
    """Return 1 + 1/2 Σ n_i(v_i-2) using frozen nominal valences."""

    unknown = sorted(set(counts) - set(ELEMENTS))
    if unknown:
        raise ValueError(f"unsupported elements: {unknown}")
    return 1.0 + 0.5 * sum(
        count * (ELEMENTS[element].nominal_valence - 2)
        for element, count in counts.items()
    )


def evaluate_formula_rules(
    counts: dict[str, int],
    *,
    require_carbon: bool = True,
    minimum_dbe: float = 0.0,
    enforce_integer_dbe: bool = True,
    enforce_senior: bool = True,
) -> FormulaRuleResult:
    reasons: list[str] = []
    unknown = sorted(set(counts) - set(ELEMENTS))
    if unknown:
        return FormulaRuleResult(False, float("nan"), 0, 0, ("unsupported_elements:" + ";".join(unknown),))
    if any(not isinstance(count, int) or count < 0 for count in counts.values()):
        return FormulaRuleResult(False, float("nan"), 0, 0, ("noninteger_or_negative_count",))
    if require_carbon and counts.get("C", 0) < 1:
        reasons.append("carbon_required")
    atom_count = sum(counts.values())
    if atom_count < 1:
        reasons.append("empty_formula")
    valences = [ELEMENTS[element].nominal_valence for element, count in counts.items() if count > 0]
    valence_sum = sum(ELEMENTS[element].nominal_valence * count for element, count in counts.items())
    dbe = generalized_dbe(counts)
    if dbe < minimum_dbe - 1.0e-9:
        reasons.append("dbe_below_minimum")
    if enforce_integer_dbe and abs(dbe - round(dbe)) > 1.0e-9:
        reasons.append("noninteger_dbe_or_valence_parity")
    if enforce_senior and atom_count > 1:
        maximum_valence = max(valences, default=0)
        if valence_sum < 2 * maximum_valence:
            reasons.append("senior_max_valence")
        if valence_sum < 2 * (atom_count - 1):
            reasons.append("senior_connected_graph")
    return FormulaRuleResult(
        passes=not reasons,
        dbe=dbe,
        atom_count=atom_count,
        valence_sum=valence_sum,
        reasons=tuple(reasons),
    )
