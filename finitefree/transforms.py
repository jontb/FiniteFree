from typing import Any, List, Optional

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .core import RealRootedPolynomial, UnitaryPolynomial
from .utils.conversion import flint_to_float, sympy_to_fmpq


def FiniteCauchyTransform(p: RealRootedPolynomial) -> sp.Expr:
    r"""
    Computes the Finite Cauchy Transform $G_p^{(d)}(z) = \frac{1}{d} \frac{p'(z)}{p(z)}$
    Returns a SymPy expression.
    """
    z = sp.Symbol("z")
    d = p.degree
    if d == 0:
        raise ValueError("Finite Cauchy transform requires a positive degree.")

    poly = sp.Poly(list(p.coeffs), z)
    expr = poly.as_expr()

    dp_coeffs = [p.coeffs[i] * (d - i) for i in range(d)]
    dp_poly = sp.Poly(dp_coeffs, z)
    p_prime = dp_poly.as_expr()

    return sp.Rational(1, d) * (p_prime / expr)


def FiniteSTransform(p: RealRootedPolynomial, exact: bool = True) -> NDArray[Any]:
    r"""
    Computes the finite S-Transform discretely on $\{-k/d\}$.
    Returns a dense array of length $d$, where index $k-1$ maps to $-k/d$.
    Raises ValueError if strict positivity constraint is violated.
    """
    if not isinstance(p, UnitaryPolynomial) and not p.has_strictly_positive_roots:
        raise ValueError(
            "Strict positivity constraint violated: roots must be strictly positive."
        )

    d = p.degree
    e_k = p._normalized_coeffs_flint()

    s_transform = np.zeros(d, dtype=object if exact else np.float64)

    for k in range(1, d + 1):
        val_num = e_k[k - 1]
        val_den = e_k[k]
        if val_num == 0 or val_den == 0:
            raise ValueError(
                "Strict positivity constraint violated: a zero coefficient "
                f"was encountered at index {k - 1} or {k}."
            )

        res = val_num / val_den
        if exact:
            s_transform[k - 1] = sp.Rational(int(res.p), int(res.q))
        else:
            s_transform[k - 1] = flint_to_float(res)

    return s_transform


def FiniteRTransform(
    p: RealRootedPolynomial,
    order: int = 5,
    d: Optional[int] = None,
    numerical: bool = False,
    prec: int = 256,
) -> List[Any]:
    r"""
    Extracts finite free cumulants $\kappa_n^{(d)}(p)$ exactly using the classical
    cumulant-moment recurrence ($O(n^2)$), which is equivalent to Möbius
    inversion over the partition lattice but avoids exponential partition
    enumeration.
    Returns the first `order` finite free cumulants (which strictly
    linearize $\boxplus_d$).
    Uses Definition 2.14 of Arizmendi et al., arXiv:2408.09337:
    the classical cumulant of the normalized coefficients is scaled by
    $(-d)^{n-1}/(n-1)!$. Orders above d are returned as zero.
    The numerical path centers the requested coefficient prefix exactly before
    using Arb at `prec` bits, then restores the first cumulant (the mean).
    Higher cumulants are invariant under translation. Numerical results remain
    precision-dependent approximations, particularly at high orders.
    """
    import math

    import flint

    if d is None:
        d = p.degree
    e_k = p._normalized_coeffs_flint(d)

    if numerical:
        count = max(0, min(order, d))
        mean = e_k[1] if count else flint.fmpq(0)
        if mean != 0:
            # For a shift by -mean, normalized coefficients obey a binomial
            # transform. Only the prefix used by the recurrence is needed;
            # shifting the entire polynomial would do unnecessary exact work.
            powers = [(-mean) ** n for n in range(count + 1)]
            e_k = [
                sum(
                    (math.comb(n, k) * e_k[k] * powers[n - k] for k in range(n + 1)),
                    flint.fmpq(0),
                )
                for n in range(count + 1)
            ]
        else:
            e_k = e_k[: count + 1]
        # Use controlled high-precision Arb floats to avoid rational arithmetic blowup
        old_prec = flint.ctx.prec
        flint.ctx.prec = prec
        try:
            arb_e = []
            for v in e_k:
                val = flint.arb(v)
                arb_e.append(val)

            c_arb = []
            cumulants = []
            for n in range(1, order + 1):
                if n > d:
                    cumulants.append(0.0)
                    c_arb.append(flint.arb(0))
                    continue

                cn = arb_e[n]
                for k in range(1, n):
                    cn -= math.comb(n - 1, k - 1) * c_arb[k - 1] * arb_e[n - k]
                c_arb.append(cn)

                # Use exact operations on the arb/fmpz before floating to avoid loss of precision
                kappa_n = cn * ((-d) ** (n - 1)) / math.factorial(n - 1)
                cumulants.append(float(kappa_n))

            if count:
                cumulants[0] = float(flint.arb(mean))
            return cumulants
        finally:
            flint.ctx.prec = old_prec

    # c_n = classical cumulant of the sequence (e_1, e_2, ..., e_n)
    # Using the recurrence: c_n = e_n - sum_{k=1}^{n-1} C(n-1, k-1) * c_k * e_{n-k}
    # Then: kappa_n^{(d)} = c_n * (-d)^{n-1} / (n-1)!
    c = []  # c[0] = c_1, c[1] = c_2, etc.
    cumulants = []
    for n in range(1, order + 1):
        if n > d:
            cumulants.append(0)
            c.append(flint.fmpq(0))
            continue

        cn = e_k[n]
        for k in range(1, n):
            cn -= math.comb(n - 1, k - 1) * c[k - 1] * e_k[n - k]
        c.append(cn)

        # Maintain exact representation
        kappa_n = cn * ((-d) ** (n - 1)) / math.factorial(n - 1)
        cumulants.append(sp.Rational(int(kappa_n.p), int(kappa_n.q)))

    return cumulants


class FiniteTTransform:
    r"""
    Definition 6.3 (Finite T-transform).

    Given a polynomial $p \in P_d(\mathbb{R}_{\ge 0})$, the finite T-transform $T_d(p)(t)$
    is the right-continuous step function on $(0, 1)$ defined in terms of
    the coefficients of $p$.
    """

    def __init__(self, p: RealRootedPolynomial) -> None:
        if p.degree == 0:
            raise ValueError("Finite T-transform requires a positive degree.")
        if not p.has_non_negative_roots:
            raise ValueError(
                "Finite T-transform is only defined for polynomials with "
                "non-negative roots."
            )

        self.p = p
        self.d = p.degree
        self.e_k = p._normalized_coeffs_flint()

        # Multiplicity r of the root 0 of p is trailing zeros in coeffs
        self.r = 0
        if p._is_flint:
            while self.r < self.d and p._fmpq_poly[self.r] == 0:
                self.r += 1
        else:
            while self.r < self.d and p.coeffs[self.d - self.r] == 0:
                self.r += 1

    def __call__(self, t: Any) -> Any:
        """
        Evaluate the right-continuous step function at a finite real t in (0, 1).
        Rational values and stored binary float values select intervals exactly;
        a float near a rational boundary may lie on either side of that boundary.
        """
        message = "t must be in the open interval (0, 1) and be a finite real value."
        try:
            if isinstance(t, (float, np.floating)):
                # Retain NumPy extended precision instead of narrowing to float64.
                t_rat = sp.Rational(*t.as_integer_ratio())
            else:
                t_rat = sympy_to_fmpq(t)
        except (TypeError, ValueError, OverflowError):
            # Preserve evaluation at real symbolic constants such as sqrt(2)/2.
            try:
                t_sym = sp.sympify(t)
                if not isinstance(t_sym, sp.Expr) or t_sym.is_real is not True:
                    raise ValueError(message)
                if not (0 < t_sym < 1):
                    raise ValueError(message)
                k = int(sp.floor(t_sym * self.d)) + 1
            except (TypeError, ValueError, sp.SympifyError) as error:
                raise ValueError(message) from error
        else:
            if not (0 < t_rat < 1):
                raise ValueError(message)
            k = (int(t_rat.p) * self.d) // int(t_rat.q) + 1

        if k <= self.r:
            return 0

        val_num = self.e_k[self.d - k + 1]
        val_den = self.e_k[self.d - k]

        if val_den == 0:
            raise ValueError(
                f"Zero division encountered: e_tilde_{self.d - k} is zero."
            )

        res = val_num / val_den
        return sp.Rational(int(res.p), int(res.q))


def SymmetricFiniteSTransform(
    p: RealRootedPolynomial, exact: bool = True
) -> NDArray[Any]:
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
    e_k = p._normalized_coeffs_flint()

    # Multiplicity 2r of the root 0
    zero_mult = 0
    if p._is_flint:
        while zero_mult < p.degree and p._fmpq_poly[zero_mult] == 0:
            zero_mult += 1
    else:
        while zero_mult < p.degree and p.coeffs[p.degree - zero_mult] == 0:
            zero_mult += 1
    r = zero_mult // 2

    s_transform = np.zeros(d - r, dtype=object if exact else np.float64)

    for k in range(1, d - r + 1):
        val_num = e_k[2 * (k - 1)]
        val_den = e_k[2 * k]

        if val_den == 0:
            raise ValueError(f"Zero division encountered: e_tilde_{2 * k} is zero.")

        res = val_num / val_den
        if exact:
            s_transform[k - 1] = sp.Rational(int(res.p), int(res.q))
        else:
            s_transform[k - 1] = flint_to_float(res)

    return s_transform
