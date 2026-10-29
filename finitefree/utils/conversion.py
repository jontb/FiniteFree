from typing import Any

import numpy as np
import sympy as sp


def flint_to_float(val: Any) -> float:
    r"""
    Convert a rational/numeric value to a Python float.
    Values with integer numerator ``p`` and denominator ``q`` attributes are
    rounded once, to nearest with ties to even, from the exact ratio, including
    huge numerator/denominator integers. Results beyond the float range become
    signed infinity and tiny results may become subnormal or signed zero. Other
    values use ``float(val)``. This helper does not provide a finiteness
    guarantee or an interval bound.
    """
    if hasattr(val, "p") and hasattr(val, "q"):
        numerator = int(val.p)
        denominator = int(val.q)
        try:
            # Python integer true division is correctly rounded.
            return numerator / denominator
        except OverflowError:
            negative = (numerator < 0) != (denominator < 0)
            return float("-inf") if negative else float("inf")
    return float(val)


def sympy_to_fmpq(val: Any) -> Any:
    r"""
    Convert an integer/rational or finite Python/NumPy float to FLINT fmpq.
    Floating scalars use their own as_integer_ratio(), preserving supplied binary
    precision rather than recovering decimal intent. Other values are converted
    through SymPy Rational and must be rational-convertible.
    """
    import flint

    if isinstance(val, flint.fmpq):
        return val
    if isinstance(val, flint.fmpz):
        return flint.fmpq(val, 1)
    if isinstance(val, (int, np.integer)):
        return flint.fmpq(int(val), 1)
    if isinstance(val, sp.Rational):
        return flint.fmpq(int(val.p), int(val.q))
    if isinstance(val, (float, np.floating)):
        num, den = val.as_integer_ratio()
        return flint.fmpq(num, den)
    val_sym = sp.Rational(sp.sympify(val))
    return flint.fmpq(int(val_sym.p), int(val_sym.q))


def fmpq_poly_to_sympy_coeffs(poly: Any) -> list[sp.Rational]:
    """Converts flint.fmpq_poly (ascending) to SymPy coefficients (descending)."""
    coeffs_asc = poly.coeffs()
    coeffs = []
    for val in reversed(coeffs_asc):
        coeffs.append(sp.Rational(int(val.p), int(val.q)))
    return coeffs
