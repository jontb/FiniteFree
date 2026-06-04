from typing import Any, List, Optional

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .core import RealRootedPolynomial


def FiniteCauchyTransform(p: RealRootedPolynomial) -> sp.Expr:
    """
    Computes the Finite Cauchy Transform G_p^{(d)}(z) = \frac{1}{d} \frac{p'(z)}{p(z)}
    Returns a SymPy expression.
    """
    z = sp.Symbol("z")
    d = p.degree

    poly = sp.Poly(list(p.coeffs), z)
    expr = poly.as_expr()

    dp_coeffs = [p.coeffs[i] * (d - i) for i in range(d)]
    dp_poly = sp.Poly(dp_coeffs, z)
    p_prime = dp_poly.as_expr()

    return (1 / d) * (p_prime / expr)


def FiniteSTransform(p: RealRootedPolynomial) -> NDArray[np.object_]:
    """
    Computes the finite S-Transform discretely on {-k/d}.
    Returns a dense array of length d, where index k-1 maps to -k/d.
    Raises ValueError if strict positivity constraint is violated.
    """
    d = p.degree
    e_k = p.normalized_coeffs()

    s_transform = np.zeros(d, dtype=object)

    for k in range(1, d + 1):
        if e_k[k - 1] == 0 or e_k[k] == 0:
            raise ValueError(
                "Strict positivity constraint violated: a zero coefficient "
                f"was encountered at index {k - 1} or {k}."
            )

        val_num = e_k[k - 1]
        val_den = e_k[k]
        # Maintain exact rational/integer representation to avoid float truncation
        if isinstance(val_num, (int, np.integer)) and isinstance(
            val_den, (int, np.integer)
        ):
            if val_num % val_den == 0:
                s_transform[k - 1] = val_num // val_den
            else:
                s_transform[k - 1] = sp.Rational(val_num, val_den)
        else:
            s_transform[k - 1] = val_num / val_den

    return s_transform


def FiniteRTransform(
    p: RealRootedPolynomial, order: int = 5, d: Optional[int] = None
) -> List[Any]:
    """
    Extracts finite free cumulants κ_n^{(d)}(p) exactly using the classical
    cumulant-moment recurrence (O(n²)), which is equivalent to Möbius
    inversion over the partition lattice but avoids exponential partition
    enumeration.
    Returns the first `order` finite free cumulants (which strictly
    linearize ⊞_d).
    """
    import math

    if d is None:
        d = p.degree
    e_k = p.normalized_coeffs(d)

    # c_n = classical cumulant of the sequence (e_1, e_2, ..., e_n)
    # Using the recurrence: c_n = e_n - sum_{k=1}^{n-1} C(n-1, k-1) * c_k * e_{n-k}
    # Then: kappa_n^{(d)} = c_n * (n-1)! * (-d)^{n-1}
    c = []  # c[0] = c_1, c[1] = c_2, etc.
    cumulants = []
    for n in range(1, order + 1):
        if n > d:
            cumulants.append(0)
            c.append(0)
            continue

        # c_n = e_n - sum_{k=1}^{n-1} C(n-1, k-1) * c_k * e_{n-k}
        cn = e_k[n]  # e_n (0-indexed, e_k[0] = e_0 = 1)
        for k in range(1, n):
            cn -= math.comb(n - 1, k - 1) * c[k - 1] * e_k[n - k]
        c.append(cn)

        kappa_n = cn * math.factorial(n - 1) * ((-d) ** (n - 1))
        cumulants.append(kappa_n)

    return cumulants
