"""Funciones para calcular valores esperados en escenarios probabilísticos."""


def calculate_expected_value(probability: float, benefit: float) -> float:
    """Devuelve el valor esperado de un beneficio."""
    if not isinstance(probability, (int, float)):
        raise TypeError("La probabilidad debe ser numérica.")
    if not 0 <= probability <= 1:
        raise ValueError("La probabilidad debe estar entre 0 y 1.")

    if not isinstance(benefit, (int, float)):
        raise TypeError("El beneficio debe ser numérico.")
    if benefit < 0:
        raise ValueError("El beneficio no puede ser negativo.")

    return float(probability) * float(benefit)
