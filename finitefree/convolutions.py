import functools
import math

from .core import RealRootedPolynomial


@functools.lru_cache(maxsize=None)
def _get_factorial_list(d: int) -> list[int]:
    return [math.factorial(i) for i in range(d + 1)]


def symmetric_additive(
    p: RealRootedPolynomial, q: RealRootedPolynomial, d: int
) -> RealRootedPolynomial:
    r"""
    Computes the finite free symmetric additive convolution (p \boxplus_d q).
    Optimized via exponential generating function (EGF) polynomial multiplication
    using python-flint's GMP-backed fmpq_poly in O(d log d) time in C.
    """
    if p.degree > d or q.degree > d:
        raise ValueError("Polynomial degrees cannot exceed dimension d.")

    import flint

    e_p = p._normalized_coeffs_flint(d)
    e_q = q._normalized_coeffs_flint(d)

    # Precompute factorials via cache
    factorials = _get_factorial_list(d)

    A_coeffs = []
    B_coeffs = []

    for k in range(d + 1):
        fact = factorials[k]
        A_coeffs.append(e_p[k] / fact)
        B_coeffs.append(e_q[k] / fact)

    A_poly = flint.fmpq_poly(A_coeffs)
    B_poly = flint.fmpq_poly(B_coeffs)
    C_poly = A_poly * B_poly

    e_res = []
    for k in range(d + 1):
        val_res = C_poly[k] * factorials[k]
        e_res.append(val_res)

    return RealRootedPolynomial.from_normalized_coeffs(e_res)


def multiplicative(
    p: RealRootedPolynomial, q: RealRootedPolynomial, d: int
) -> RealRootedPolynomial:
    r"""
    Computes the finite free multiplicative convolution (p \boxtimes_d q).
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
    p: RealRootedPolynomial, q: RealRootedPolynomial, d: int
) -> RealRootedPolynomial:
    r"""
    Computes the finite free asymmetric additive convolution (p \uplus_d q)
    exactly. Optimized to run in O(d log d) time via a Cauchy product of
    scaled coefficient sequences using python-flint's compiled C-level fmpq_poly
    multiplication.
    """
    if p.degree > d or q.degree > d:
        raise ValueError("Polynomial degrees cannot exceed dimension d.")

    import flint

    e_p = p._normalized_coeffs_flint(d)
    e_q = q._normalized_coeffs_flint(d)

    # Precompute factorials via cache
    factorials = _get_factorial_list(d)

    A_coeffs = []
    B_coeffs = []

    for i in range(d + 1):
        fact_ratio = factorials[d - i]
        fact_i = factorials[i]
        A_coeffs.append(e_p[i] * flint.fmpq(fact_ratio, fact_i))
        B_coeffs.append(e_q[i] * flint.fmpq(fact_ratio, fact_i))

    # Cauchy product (polynomial multiplication) of scaled sequences in O(d log d)
    A_poly = flint.fmpq_poly(A_coeffs)
    B_poly = flint.fmpq_poly(B_coeffs)
    C_poly = A_poly * B_poly

    e_res = []
    fact_d = factorials[d]
    for k in range(d + 1):
        val_res = C_poly[k] * flint.fmpq(factorials[k], fact_d * factorials[d - k])
        e_res.append(val_res)

    return RealRootedPolynomial.from_normalized_coeffs(e_res)
