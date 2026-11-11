import math
from typing import Any, Optional, Sequence

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
