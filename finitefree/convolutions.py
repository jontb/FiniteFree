import math
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .core import RealRootedPolynomial


def symmetric_additive(
    p: RealRootedPolynomial, q: RealRootedPolynomial, d: int
) -> RealRootedPolynomial:
    r"""
    Compute the symmetric finite additive coefficient convolution.
    For ambient d>=max(p.degree,q.degree), use monic normalized coefficients
    with zero padding: $e_k(r)=\sum_{i=0}^k\binom{k}{i}e_i(p)e_{k-i}(q)$.
    An EGF polynomial product and O(d) scaling updates use exact FLINT rationals.
    Arithmetic/bit cost depends on polynomial multiplication and coefficient sizes.
    Real-rooted inputs give a real-rooted result, but formal construction does not
    eagerly verify input/output geometry.
    """
    if p.degree > d or q.degree > d:
        raise ValueError("Polynomial degrees cannot exceed dimension d.")

    import flint

    e_p = p._normalized_coeffs_flint(d)
    e_q = q._normalized_coeffs_flint(d)

    # Compute U[k] = 1 / k! sequentially in O(d)
    U: list[Any] = [None] * (d + 1)
    U[0] = flint.fmpq(1)
    for k in range(1, d + 1):
        U[k] = U[k - 1] / k

    A_coeffs = []
    B_coeffs = []
    for k in range(d + 1):
        A_coeffs.append(e_p[k] * U[k])
        B_coeffs.append(e_q[k] * U[k])

    A_poly = flint.fmpq_poly(A_coeffs)
    B_poly = flint.fmpq_poly(B_coeffs)
    C_poly = A_poly * B_poly

    e_res = []
    curr_fact = flint.fmpz(1)
    for k in range(d + 1):
        if k > 0:
            curr_fact *= k
        val_res = C_poly[k] * curr_fact
        e_res.append(val_res)

    result = RealRootedPolynomial.from_normalized_coeffs(e_res)
    if (
        p.degree == q.degree == d
        and p._hermite_variance is not None
        and q._hermite_variance is not None
    ):
        # Shifted/dilated Hermite polynomials are closed under boxplus_d.
        # Keep exact provenance rather than inferring a family from rounded coefficients.
        from .utils.roots import _gaussian_roots

        result = _gaussian_roots(
            result,
            p._hermite_variance + q._hermite_variance,
            p._hermite_center + q._hermite_center,
        )
        result._is_verified = True
    return result


def multiplicative(
    p: RealRootedPolynomial, q: RealRootedPolynomial, d: int
) -> RealRootedPolynomial:
    r"""
    Compute the monic finite multiplicative coefficient convolution.
    Ambient d must cover both degrees; normalized sequences are zero padded and
    $e_k(r)=e_k(p)e_k(q)$. Rational arithmetic is exact and scale-invariant.
    Real-root preservation requires real-rooted inputs with at least one having
    non-negative roots. The coefficient API does not enforce that domain and its
    formal result is validated lazily.
    """
    if p.degree > d or q.degree > d:
        raise ValueError("Polynomial degrees cannot exceed dimension d.")

    e_p = p._normalized_coeffs_flint(d)
    e_q = q._normalized_coeffs_flint(d)

    e_res = []
    for k in range(d + 1):
        e_res.append(e_p[k] * e_q[k])

    return RealRootedPolynomial.from_normalized_coeffs(e_res)


def asymmetric_additive(
    p: RealRootedPolynomial,
    q: RealRootedPolynomial,
    d: int,
    weights: Optional[Sequence[Any]] = None,
    *,
    columns: Optional[int] = None,
) -> RealRootedPolynomial:
    r"""
    Compute the asymmetric (rectangular) additive convolution exactly.
    For $d\times m$ matrices A and B with $p=\chi(AA^*)$, $q=\chi(BB^*)$ and
    independent Haar rotations Q, R, the result is the expected characteristic
    polynomial of $(A+QBR)(A+QBR)^*$. `columns` is m, the number of columns,
    with m >= d; the default m = d is the square asymmetric convolution of
    Marcus, Spielman and Srivastava, and m > d is the rectangular convolution of
    Gribinski and Marcus. In normalized coefficients
    $e_k(r)=\sum_{i+j=k}\binom{k}{i}\frac{(m-i)!(m-j)!}{m!(m-k)!}e_i(p)e_j(q)$.
    Ambient d must cover both input degrees. A scaled FLINT polynomial Cauchy
    product constructs the monic output; bit cost depends on coefficient sizes.
    When weights is supplied, weights[0] and weights[1] dilate the respective
    input roots before convolution; they are not rank or dimension parameters.
    The real-root preservation theorem uses non-negative-root inputs; this formal
    coefficient implementation does not enforce that theorem's domain eagerly.
    """
    if p.degree > d or q.degree > d:
        raise ValueError("Polynomial degrees cannot exceed dimension d.")
    if columns is None:
        m = d
    else:
        if isinstance(columns, bool) or not isinstance(columns, int):
            raise ValueError("columns must be an integer.")
        if columns < d:
            raise ValueError("columns cannot be smaller than dimension d.")
        m = columns

    if weights is not None:
        p = p.dilation(weights[0])
        q = q.dilation(weights[1])

    import flint

    e_p = p._normalized_coeffs_flint(d)
    e_q = q._normalized_coeffs_flint(d)

    # W[i] = (m - i)! / i! sequentially in O(d); m = d is the square case.
    W: list[Any] = [None] * (d + 1)
    W[0] = flint.fmpq(math.factorial(m))
    for i in range(1, d + 1):
        W[i] = W[i - 1] / (i * (m - i + 1))

    A_coeffs = []
    B_coeffs = []
    for i in range(d + 1):
        A_coeffs.append(e_p[i] * W[i])
        B_coeffs.append(e_q[i] * W[i])

    # Cauchy product (polynomial multiplication) of scaled sequences in O(d log d)
    A_poly = flint.fmpq_poly(A_coeffs)
    B_poly = flint.fmpq_poly(B_coeffs)
    C_poly = A_poly * B_poly

    # e_k = k! / (m! (m - k)!) * C_k, built from 1 / (W[0] W'[k]) with
    # W'[k] = (m - k)! / k! = W[k].
    e_res = []
    for k in range(d + 1):
        val_res = C_poly[k] / (W[0] * W[k])
        e_res.append(val_res)

    return RealRootedPolynomial.from_normalized_coeffs(e_res)


_CONVOLUTION_KINDS = ("multiplicative", "additive")


def _padded_float_roots(p: RealRootedPolynomial, d: int) -> NDArray[np.float64]:
    """Float64 roots of p padded with ambient zero roots to length d."""
    roots = np.asarray(p.evaluate_roots_float64(), dtype=np.float64)
    return np.sort(np.concatenate((roots, np.zeros(d - len(roots)))))


def _haar_spectrum(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    kind: str,
    draws: int,
    rng: Optional[np.random.Generator],
) -> NDArray[np.float64]:
    from .ensembles import sample_haar_orthogonal

    d = len(a)
    if kind == "multiplicative":
        if np.any(a < 0):
            a, b = b, a
        if np.any(a < 0):
            raise ValueError(
                "Multiplicative sampling needs one input with non-negative roots."
            )
        half = np.sqrt(a)
    total = np.zeros(d)
    for _ in range(draws):
        rotation = np.asarray(sample_haar_orthogonal(d, rng=rng), dtype=np.float64)
        conjugated = (rotation * b) @ rotation.T
        if kind == "multiplicative":
            matrix = half[:, None] * conjugated * half[None, :]
        else:
            matrix = np.diag(a) + conjugated
        total += np.linalg.eigvalsh((matrix + matrix.T) / 2)
    return np.asarray(total / draws, dtype=np.float64)


def sample_convolution_spectrum(
    p: RealRootedPolynomial,
    q: RealRootedPolynomial,
    d: int,
    *,
    kind: str = "multiplicative",
    draws: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    r"""
    Estimate the roots of a finite free convolution from random matrices.
    With A and B diagonal matrices of the roots of p and q (padded with zeros to
    ambient d) and O Haar orthogonal, the expected characteristic polynomial of
    $A^{1/2} O B O^T A^{1/2}$ is `multiplicative(p, q, d)` and that of
    $A + O B O^T$ is `symmetric_additive(p, q, d)`. This returns the rank-wise
    mean of the sorted eigenvalues over `draws` samples, in ascending order.
    It is a statistical estimate, not the convolution's roots: use it as the
    `initial` argument of `evaluate_roots_float64`, which certifies the result.
    kind is "multiplicative" (one input must have non-negative roots) or
    "additive". rng=None uses NumPy's legacy global stream.
    """
    if kind not in _CONVOLUTION_KINDS:
        raise ValueError('kind must be "multiplicative" or "additive".')
    if p.degree > d or q.degree > d:
        raise ValueError("Polynomial degrees cannot exceed dimension d.")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer.")
    return _haar_spectrum(
        _padded_float_roots(p, d), _padded_float_roots(q, d), kind, draws, rng
    )


class PreparedConvolution:
    r"""
    Repeated finite free convolution against one fixed polynomial, with roots.

    Prepare q once for ambient dimension d and `kind` ("multiplicative" or
    "additive", meaning `multiplicative` and `symmetric_additive`). `convolve`
    returns the same exact polynomial as those functions. `roots` returns the
    float64 roots of the convolution with a supplied polynomial or root array,
    choosing starting points in this order: the caller's `initial`; the roots
    returned by the previous call (a warm start for iterative use); a sampled
    spectrum (`seed_sampler`, default Haar samples as in
    `sample_convolution_spectrum`). Each start is refined in Arb arithmetic
    and accepted only with the sign-change certificate of
    `evaluate_roots_float64(initial=...)`; if every start fails, the exact
    isolation path of `evaluate_roots_float64()` is used. The answer therefore
    does not depend on the route beyond float64 rounding.

    A root array of length d is read as exact binary values. Its convolution
    coefficients are enclosed in Arb balls rather than formed as exact
    rationals, so the certificate covers the exact polynomial without
    constructing it. `counts` records how many calls ended on each route.
    seed_sampler, when given, maps (roots, rng) to d starting values and
    replaces the Haar sampler (for example weighted Wishart draws whose expected
    characteristic polynomial is the same convolution). The object owns its
    prepared data; returned arrays are read-only.
    """

    def __init__(
        self,
        q: RealRootedPolynomial,
        d: int,
        *,
        kind: str = "multiplicative",
        draws: int = 4,
        rng: Optional[np.random.Generator] = None,
        seed_sampler: Optional[Callable[[NDArray[np.float64], Any], Any]] = None,
    ) -> None:
        if kind not in _CONVOLUTION_KINDS:
            raise ValueError('kind must be "multiplicative" or "additive".')
        if isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise ValueError("d must be a positive integer.")
        if q.degree > d:
            raise ValueError("Polynomial degrees cannot exceed dimension d.")
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive integer.")
        self.q = q
        self.d = d
        self.kind = kind
        self.draws = draws
        self.rng = rng
        self.seed_sampler = seed_sampler
        self._e_q = list(q._normalized_coeffs_flint(d))
        self._q_roots: Optional[NDArray[np.float64]] = None
        self.last_roots: Optional[NDArray[np.float64]] = None
        self.counts = {"initial": 0, "warm": 0, "seeded": 0, "isolated": 0}

    def convolve(self, p: RealRootedPolynomial) -> RealRootedPolynomial:
        """Return the exact convolution polynomial of p with the prepared q."""
        if self.kind == "multiplicative":
            return multiplicative(p, self.q, self.d)
        return symmetric_additive(p, self.q, self.d)

    def _enclosure(self, roots: NDArray[np.float64]) -> Any:
        """Arb enclosure of the convolution with prod(x - root) at current precision."""
        import flint

        d = self.d
        ascending = flint.arb_poly.from_roots(
            [flint.arb(float(r)) for r in roots]
        ).coeffs()
        ascending = ascending + [flint.arb(0)] * (d + 1 - len(ascending))
        if self.kind == "multiplicative":
            # Coefficient of x^(d-k) is (-1)^k C(d,k) e_k(p) e_k(q) = c_p[d-k] e_k(q).
            out = [ascending[d - k] * flint.arb(self._e_q[k]) for k in range(d + 1)]
            return flint.arb_poly(out[::-1])
        binomial = flint.fmpz(1)
        factorial = flint.fmpz(1)
        left = []
        right = []
        binomials = []
        factorials = []
        for k in range(d + 1):
            if k > 0:
                binomial = (binomial * (d - k + 1)) // k
                factorial = factorial * k
            binomials.append(binomial)
            factorials.append(factorial)
            e_p = ascending[d - k] / flint.arb(binomial)
            if k % 2:
                e_p = -e_p
            left.append(e_p / flint.arb(factorial))
            right.append(flint.arb(self._e_q[k]) / flint.arb(factorial))
        product = (flint.arb_poly(left) * flint.arb_poly(right)).coeffs()
        product = product + [flint.arb(0)] * (d + 1 - len(product))
        descending: list[Any] = [None] * (d + 1)
        for k in range(d + 1):
            value = product[k] * flint.arb(factorials[k]) * flint.arb(binomials[k])
            descending[d - k] = -value if k % 2 else value
        return flint.arb_poly(descending)

    def _seed(self, roots: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        try:
            if self.seed_sampler is not None:
                guess = np.asarray(self.seed_sampler(roots, self.rng), dtype=np.float64)
            else:
                if self._q_roots is None:
                    self._q_roots = _padded_float_roots(self.q, self.d)
                guess = _haar_spectrum(
                    roots, self._q_roots, self.kind, self.draws, self.rng
                )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return None
        if guess.shape != (self.d,) or not np.all(np.isfinite(guess)):
            return None
        return guess

    def roots(
        self,
        p: Union[RealRootedPolynomial, Sequence[Any], NDArray[Any]],
        *,
        initial: Union[Sequence[Any], NDArray[Any], None] = None,
    ) -> NDArray[np.float64]:
        """Return read-only float64 roots of the convolution of p with q.

        p is a polynomial of degree at most d or an array of exactly d finite
        real roots. See the class description for starting points, certification
        and the exact fallback.
        """
        from .utils.refine import certified_real_roots_from

        d = self.d
        if isinstance(p, RealRootedPolynomial):
            if p.degree > d:
                raise ValueError("Polynomial degrees cannot exceed dimension d.")
            polynomial: Optional[RealRootedPolynomial] = self.convolve(p)
            p_roots: Optional[NDArray[np.float64]] = None
        else:
            p_roots = np.sort(np.asarray(p, dtype=np.float64).ravel())
            if p_roots.shape != (d,) or not np.all(np.isfinite(p_roots)):
                raise ValueError("A root array must hold exactly d finite values.")
            polynomial = None
        if initial is not None and len(initial) != d:
            raise ValueError("initial must supply one starting value per root.")

        def attempt(start: Any) -> Optional[NDArray[np.float64]]:
            if polynomial is not None:
                if polynomial.degree != d:
                    return None
                return certified_real_roots_from(polynomial._fmpq_poly, start)
            assert p_roots is not None
            array_roots = p_roots
            return certified_real_roots_from(
                lambda: self._enclosure(array_roots), start
            )

        result = None
        route = "isolated"
        for name in ("initial", "warm", "seeded"):
            if name == "initial":
                start = initial
            elif name == "warm":
                start = self.last_roots
            else:
                if p_roots is None:
                    assert polynomial is not None
                    try:
                        seed_roots = _padded_float_roots(p, d)  # type: ignore[arg-type]
                    except (RuntimeError, ValueError):
                        seed_roots = None
                else:
                    seed_roots = p_roots
                start = None if seed_roots is None else self._seed(seed_roots)
            if start is None:
                continue
            result = attempt(start)
            if result is not None:
                route = name
                break
        if result is None:
            if polynomial is None:
                assert p_roots is not None
                polynomial = self.convolve(RealRootedPolynomial.from_roots(p_roots))
            result = np.asarray(polynomial.evaluate_roots_float64(), dtype=np.float64)
        elif polynomial is not None:
            polynomial._is_verified = True
        out = np.array(result, dtype=np.float64)
        out.setflags(write=False)
        self.counts[route] += 1
        self.last_roots = out
        return out
