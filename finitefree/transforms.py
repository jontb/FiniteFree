from typing import Any, List, Optional

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .core import RealRootedPolynomial, UnitaryPolynomial


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
    if not isinstance(p, UnitaryPolynomial) and not p.has_strictly_positive_roots:
        raise ValueError(
            "Strict positivity constraint violated: roots must be strictly positive."
        )

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


class FiniteTTransform:
    """
    Definition 6.3 (Finite T-transform).

    Given a polynomial p in P_d(R_>=0), the finite T-transform T_d(p)(t)
    is the right-continuous step function on (0, 1) defined in terms of
    the coefficients of p.
    """

    def __init__(self, p: RealRootedPolynomial) -> None:
        if not p.has_non_negative_roots:
            raise ValueError(
                "Finite T-transform is only defined for polynomials with "
                "non-negative roots."
            )

        self.p = p
        self.d = p.degree
        self.e_k = p.normalized_coeffs()

        # Multiplicity r of the root 0 of p is trailing zeros in coeffs
        self.r = 0
        while self.r < self.d and p.coeffs[self.d - self.r] == 0:
            self.r += 1

    def __call__(self, t: Any) -> Any:
        """
        Evaluates the finite T-transform at t in (0, 1).
        """
        t_val = float(t)
        if t_val <= 0 or t_val >= 1:
            raise ValueError("t must be in the open interval (0, 1).")

        import math

        if isinstance(t, (int, float)):
            k = int(math.floor(t_val * self.d)) + 1
        else:
            t_sym = sp.sympify(t)
            k = int(sp.floor(t_sym * self.d)) + 1

        if k <= self.r:
            return 0

        if k > self.d:
            k = self.d

        val_num = self.e_k[self.d - k + 1]
        val_den = self.e_k[self.d - k]

        if val_den == 0:
            raise ValueError(
                f"Zero division encountered: e_tilde_{self.d - k} is zero."
            )

        if isinstance(val_num, (int, np.integer)) and isinstance(
            val_den, (int, np.integer)
        ):
            if val_num % val_den == 0:
                return val_num // val_den
            return sp.Rational(val_num, val_den)
        return val_num / val_den


def SymmetricFiniteSTransform(p: RealRootedPolynomial) -> NDArray[np.object_]:
    """
    Computes the symmetric finite S-Transform discretely on {-k/d}.
    p must be symmetric of even degree 2d.
    Returns an array of length d-r, where index k-1 maps to -k/d.
    """
    if p.degree % 2 != 0:
        raise ValueError(
            "Polynomial degree must be even (2d) for symmetric S-transform."
        )
    if not p.is_symmetric():
        raise ValueError("Polynomial must be symmetric.")

    d = p.degree // 2
    e_k = p.normalized_coeffs()

    # Multiplicity 2r of the root 0
    zero_mult = 0
    while zero_mult < p.degree and p.coeffs[p.degree - zero_mult] == 0:
        zero_mult += 1
    r = zero_mult // 2

    s_transform = np.zeros(d - r, dtype=object)

    for k in range(1, d - r + 1):
        val_num = e_k[2 * (k - 1)]
        val_den = e_k[2 * k]

        if val_den == 0:
            raise ValueError(f"Zero division encountered: e_tilde_{2 * k} is zero.")

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
