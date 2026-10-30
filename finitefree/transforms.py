import math
import numbers
import operator
from typing import Any, List, Literal, Optional, Sequence, SupportsIndex, Union

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

    coeffs = p.coeffs
    poly = sp.Poly(list(coeffs), z)
    expr = poly.as_expr()

    dp_coeffs = [coeffs[i] * (d - i) for i in range(d)]
    dp_poly = sp.Poly(dp_coeffs, z)
    p_prime = dp_poly.as_expr()

    return sp.Rational(1, d) * (p_prime / expr)


def FiniteSTransform(p: RealRootedPolynomial, exact: bool = True) -> NDArray[Any]:
    r"""
    Return coefficient-ratio S values at the nodes -k/d, k=1,...,d.
    Index k-1 contains e_{k-1}/e_k. The ordinary real-root API requires strictly
    positive roots; zero roots raise ValueError. Unit-circle objects bypass that
    positivity gate, but this coefficient path still requires rational inputs and
    nonzero neighboring coefficients. exact=True returns SymPy rationals;
    exact=False returns float64 values without a final-finiteness guarantee.
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
    Only the requested normalized-coefficient prefix is extracted when possible.
    A bounded per-polynomial prefix cache is separate from complete coefficients;
    ambient dimensions retain their own exact binomial normalization.
    """
    import math

    import flint

    if d is None:
        d = p.degree
    count = max(0, min(order, d))
    e_k = p._normalized_coeffs_flint_prefix(count, d)

    if numerical:
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
    # Then: kappa_n^{(d)} = c_n * (-d)^{n-1} / (n-1)!
    c = _classical_cumulants(e_k, count)
    cumulants = []
    for n in range(1, order + 1):
        if n > d:
            cumulants.append(0)
            continue

        # Maintain exact representation
        kappa_n = c[n - 1] * ((-d) ** (n - 1)) / math.factorial(n - 1)
        cumulants.append(sp.Rational(int(kappa_n.p), int(kappa_n.q)))

    return cumulants


def _egf_series(moments: List[Any], count: int) -> Any:
    r"""Return $\sum_{k \le count} m_k z^k / k!$ as a FLINT series of that length."""
    import flint

    factorial = 1
    terms = [flint.fmpq(1)]
    for k in range(1, count + 1):
        factorial *= k
        terms.append(moments[k] / factorial)
    return flint.fmpq_series(terms, prec=count + 1)


class _SeriesCap:
    """Temporarily raise FLINT's process-wide series length cap."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __enter__(self) -> None:
        import flint

        self.saved = flint.ctx.cap
        flint.ctx.cap = max(self.saved, self.length)

    def __exit__(self, *exc: Any) -> None:
        import flint

        flint.ctx.cap = self.saved


def _classical_cumulants(moments: List[Any], count: int) -> List[Any]:
    r"""
    Return classical cumulants $c_1, \ldots, c_{count}$ of moments with $m_0 = 1$.
    They are $c_n = n! [z^n] \log \sum_k m_k z^k / k!$, which equals the recurrence
    $c_n = m_n - \sum_{k<n} \binom{n-1}{k-1} c_k m_{n-k}$ in exact arithmetic;
    FLINT's series logarithm replaces the quadratic Python-level loop.
    """
    import flint

    if count == 0:
        return []
    with _SeriesCap(count + 1):
        log_series = _egf_series(moments, count).log()
    coefficients = log_series.coeffs()
    cumulants = []
    factorial = 1
    for n in range(1, count + 1):
        factorial *= n
        value = coefficients[n] if n < len(coefficients) else 0
        cumulants.append(flint.fmpq(value) * factorial)
    return cumulants


def _moments_of_scaled_cumulants(moments: List[Any], count: int, t: Any) -> List[Any]:
    r"""
    Return $m_0, \ldots, m_{count}$ whose classical cumulants are t times those of
    the given moments: $n! [z^n] \exp(t \log \sum_k m_k z^k / k!)$.
    """
    import flint

    if count == 0:
        return [flint.fmpq(1)]
    with _SeriesCap(count + 1):
        powered = (_egf_series(moments, count).log() * t).exp()
    coefficients = powered.coeffs()
    result = [flint.fmpq(1)]
    factorial = 1
    for n in range(1, count + 1):
        factorial *= n
        value = coefficients[n] if n < len(coefficients) else 0
        result.append(flint.fmpq(value) * factorial)
    return result


def _moment_index(value: Any, name: str, *, positive: bool = False) -> int:
    """Return a moment order or dimension accepted by the integer-index protocol."""
    bound = "positive" if positive else "nonnegative"
    message = f"{name} must be a {bound} integer."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        index = operator.index(value)
    except TypeError as error:
        raise ValueError(message) from error
    if index < (1 if positive else 0):
        raise ValueError(message)
    return index


def root_moments(
    p: RealRootedPolynomial,
    order: SupportsIndex,
    *,
    d: Optional[SupportsIndex] = None,
) -> List[sp.Rational]:
    r"""
    Return normalized root moments $m_0,\ldots,m_{\mathrm{order}}$ exactly.
    For degree n and ambient dimension d >= n (default n), d-n zero roots are
    embedded: $m_0=1$ and $m_k=\frac{1}{d}\sum_{j=1}^{n}\lambda_j^k$ for k >= 1.
    Power sums come from Newton identities on the rational coefficients, so the
    result is invariant under a nonzero common coefficient scalar and orders
    above the degree remain meaningful. No roots are extracted and neither
    real-rootedness nor positivity is checked or certified: for a polynomial
    with nonreal roots the values are power sums over its complex roots, not
    moments of a spectral measure. A nonzero constant requires an explicit d.
    Rational coefficients are required. Returns a new list of SymPy rationals.
    """
    import flint

    count = _moment_index(order, "order")
    if not p._is_flint:
        raise ValueError("Root moments require rational coefficients.")
    n = p.degree
    if d is None:
        if n == 0:
            raise ValueError(
                "A constant polynomial requires an explicit positive dimension d."
            )
        dimension = n
    else:
        dimension = _moment_index(d, "d", positive=True)
        if dimension < n:
            raise ValueError(
                f"Ambient dimension d ({dimension}) cannot be less than polynomial "
                f"degree ({n})."
            )

    # e[k] is the k-th elementary symmetric polynomial of the roots.
    poly = p._fmpq_poly
    leading = poly[n]
    e = [(-1) ** k * poly[n - k] / leading for k in range(min(count, n) + 1)]

    # Newton: s_k = sum_{i=1}^{min(k-1,n)} (-1)^(i-1) e_i s_{k-i} + (-1)^(k-1) k e_k.
    sums: List[Any] = [flint.fmpq(n)]
    for k in range(1, count + 1):
        total = k * e[k] if k <= n else flint.fmpq(0)
        if k <= n and k % 2 == 0:
            total = -total
        for i in range(1, min(k - 1, n) + 1):
            term = e[i] * sums[k - i]
            total = total + term if i % 2 == 1 else total - term
        sums.append(total)

    moments: List[sp.Rational] = [sp.Integer(1)]
    for k in range(1, count + 1):
        value = sums[k] / dimension
        moments.append(sp.Rational(int(value.p), int(value.q)))
    return moments


def _normalized_from_cumulants(cumulants: List[Any], d: int, one: Any) -> List[Any]:
    """Invert kappa_n = (-d)^(n-1) c_n/(n-1)! and the moment-cumulant recurrence."""
    classical = []
    factor = one  # (n-1)!/(-d)^(n-1)
    for n, kappa in enumerate(cumulants, start=1):
        if n > 1:
            factor = factor * (n - 1) / (-d)
        classical.append(kappa * factor)
    # m_n = sum_{k=1}^{n} C(n-1, k-1) c_k m_{n-k}, with m_0 = 1.
    moments = [one]
    for n in range(1, d + 1):
        total = one * 0
        for k in range(1, n + 1):
            total = total + math.comb(n - 1, k - 1) * classical[k - 1] * moments[n - k]
        moments.append(total)
    return moments


def inverse_finite_r(
    cumulants: Union[Sequence[Any], NDArray[Any]],
    *,
    d: Optional[SupportsIndex] = None,
    numerical: bool = False,
) -> Any:
    r"""
    Reconstruct the monic degree-d polynomial with finite cumulants
    $\kappa_1,\ldots,\kappa_d$, the inverse of FiniteRTransform(p, order=d, d=d).
    The complete sequence is required: d defaults to its length and, if given,
    must equal it; missing cumulants are never filled with zeros. A degree-n
    source in ambient dimension d reconstructs $x^{d-n}p(x)$ monic.
    Exact mode requires rational values (integers, rationals, FLINT values) and
    rejects floats, so observations are not silently promoted to exact data.
    It returns a new RealRootedPolynomial with lazy validation: arbitrary
    cumulants need not give a real-rooted polynomial, and none is asserted.
    numerical=True takes finite real values and returns a new float64 array of
    descending coefficients, without a conditioning bound; nonfinite
    coefficients raise RuntimeError.
    """
    if not isinstance(numerical, (bool, np.bool_)):
        raise ValueError("numerical must be a boolean.")
    if isinstance(cumulants, (str, bytes)):
        raise ValueError("cumulants must be a sequence of values.")
    values = list(cumulants)
    if not values:
        raise ValueError("cumulants must contain at least one value.")
    dimension = len(values)
    if d is not None and _moment_index(d, "d", positive=True) != dimension:
        raise ValueError(
            f"Full reconstruction requires exactly d cumulants; got {dimension} "
            f"for d={operator.index(d)}."
        )

    if numerical:
        real = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Real
            ):
                raise ValueError("Numerical cumulants must be finite real numbers.")
            real.append(float(value))
        if not all(math.isfinite(value) for value in real):
            raise ValueError("Numerical cumulants must be finite real numbers.")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = _normalized_from_cumulants(real, dimension, 1.0)
                result = np.array(
                    [
                        (-1) ** k * math.comb(dimension, k) * normalized[k]
                        for k in range(dimension + 1)
                    ],
                    dtype=np.float64,
                )
        except OverflowError:
            result = np.array([np.inf])
        if not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Reconstructed coefficients exceed the finite float64 range."
            )
        return result

    import flint

    exact = []
    for value in values:
        if isinstance(value, (bool, np.bool_, float, np.floating, sp.Float)):
            raise ValueError(
                "Exact reconstruction requires rational cumulants; convert floats "
                "explicitly or use numerical=True."
            )
        try:
            exact.append(sympy_to_fmpq(value))
        except (TypeError, ValueError, sp.SympifyError) as error:
            raise ValueError(
                "Exact reconstruction requires rational cumulants."
            ) from error
    normalized = _normalized_from_cumulants(exact, dimension, flint.fmpq(1))
    return RealRootedPolynomial.from_normalized_coeffs(normalized)


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
        self.e_k = list(p._normalized_coeffs_flint())

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
    p: RealRootedPolynomial,
    exact: bool = True,
    *,
    convention: Literal["ratio", "standard"] = "ratio",
) -> NDArray[Any]:
    r"""
    Return the legacy ratio or standard symmetric finite S-transform.
    For degree 2d and zero multiplicity 2r, the array has length d-r; index k-1
    corresponds to -k/d and contains e_{2(k-1)} / e_{2k}.
    The default convention="ratio" preserves the existing squared-transform
    output and parity-only validation. convention="standard" implements the
    positive-imaginary square root in Definition 8.1 of Arizmendi et al.,
    arXiv:2408.09337v2. It requires positive degree and checks real-rootedness
    using the polynomial's validation contract, including explicit assumptions.
    Both paths require rational coefficients. exact=True returns SymPy values;
    exact=False returns float64 ratios or finite nonzero complex128 values.
    """
    if convention not in ("ratio", "standard"):
        raise ValueError("convention must be 'ratio' or 'standard'")
    if not p._is_flint:
        raise ValueError("Symmetric finite S-transform requires rational coefficients")
    if convention == "standard" and p.degree == 0:
        raise ValueError("Standard symmetric S-transform requires positive degree")
    if p.degree % 2 != 0:
        raise ValueError(
            "Polynomial degree must be even (2d) for symmetric S-transform."
        )
    if not p.is_symmetric():
        raise ValueError("Polynomial must be symmetric.")
    if convention == "standard":
        p.verify_real_rootedness()

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

    dtype = (
        object if exact else (np.complex128 if convention == "standard" else np.float64)
    )
    s_transform = np.zeros(d - r, dtype=dtype)

    for k in range(1, d - r + 1):
        val_num = e_k[2 * (k - 1)]
        val_den = e_k[2 * k]

        if val_den == 0:
            raise ValueError(f"Zero division encountered: e_tilde_{2 * k} is zero.")

        res = val_num / val_den
        if convention == "standard":
            if res >= 0:
                raise ValueError(
                    "Standard symmetric S-transform requires real symmetric roots"
                )
            magnitude = sp.sqrt(-sp.Rational(int(res.p), int(res.q)))
            if exact:
                s_transform[k - 1] = sp.I * magnitude
            else:
                # Take the square root before narrowing: its square may overflow
                # or underflow float64 while the transform remains representable.
                value = float(magnitude)
                if not np.isfinite(value) or value == 0:
                    raise RuntimeError(
                        "Standard symmetric S-transform exceeds finite complex128 range"
                    )
                s_transform[k - 1] = complex(0, value)
        elif exact:
            s_transform[k - 1] = sp.Rational(int(res.p), int(res.q))
        else:
            s_transform[k - 1] = flint_to_float(res)

    return s_transform
