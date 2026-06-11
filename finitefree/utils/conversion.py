from typing import Any

import numpy as np
import sympy as sp


def flint_to_float(val: Any) -> float:
    """Safely converts a Flint fmpq/fmpz, SymPy Rational, or other numeric type to a float."""
    if hasattr(val, "p") and hasattr(val, "q"):
        try:
            return float(int(val.p)) / float(int(val.q))
        except OverflowError:
            from decimal import Decimal
            try:
                return float(Decimal(int(val.p)) / Decimal(int(val.q)))
            except Exception:
                return float("inf") if (int(val.p) * int(val.q)) > 0 else float("-inf")
    return float(val)


def sympy_to_fmpq(val: Any) -> Any:
    """Converts a SymPy rational value exactly to a Flint fmpq."""
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
        num, den = float(val).as_integer_ratio()
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
