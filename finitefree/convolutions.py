import math
from typing import Any, Optional, Sequence

from .core import RealRootedPolynomial


def symmetric_additive(
    p: RealRootedPolynomial, q: RealRootedPolynomial, d: int
) -> RealRootedPolynomial:
    r"""
    Computes the finite free symmetric additive convolution $(p \boxplus_d q)$.
    Optimized via exponential generating function (EGF) polynomial multiplication
    using python-flint's GMP-backed fmpq_poly in $O(d \log d)$ time in C.
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

    return RealRootedPolynomial.from_normalized_coeffs(e_res)


def multiplicative(
    p: RealRootedPolynomial, q: RealRootedPolynomial, d: int
) -> RealRootedPolynomial:
    r"""
    Computes the finite free multiplicative convolution $(p \boxtimes_d q)$.
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
) -> RealRootedPolynomial:
    r"""
    Computes the finite free asymmetric additive convolution $(p \uplus_d q)$
    exactly. Optimized to run in $O(d \log d)$ time via a Cauchy product of
    scaled coefficient sequences using python-flint's compiled C-level fmpq_poly
    multiplication.
    """
    if p.degree > d or q.degree > d:
        raise ValueError("Polynomial degrees cannot exceed dimension d.")

    if weights is not None:
        p = p.dilation(weights[0])
        q = q.dilation(weights[1])

    import flint

    e_p = p._normalized_coeffs_flint(d)
    e_q = q._normalized_coeffs_flint(d)

    # Compute W[i] = (d - i)! / i! sequentially in O(d)
    W: list[Any] = [None] * (d + 1)
    W[0] = flint.fmpq(math.factorial(d))
    for i in range(1, d + 1):
        W[i] = W[i - 1] / (i * (d - i + 1))

    A_coeffs = []
    B_coeffs = []
    for i in range(d + 1):
        A_coeffs.append(e_p[i] * W[i])
        B_coeffs.append(e_q[i] * W[i])

    # Cauchy product (polynomial multiplication) of scaled sequences in O(d log d)
    A_poly = flint.fmpq_poly(A_coeffs)
    B_poly = flint.fmpq_poly(B_coeffs)
    C_poly = A_poly * B_poly

    e_res = []
    W_d = W[d]
    for k in range(d + 1):
        val_res = C_poly[k] * (W_d / W[k])
        e_res.append(val_res)

    return RealRootedPolynomial.from_normalized_coeffs(e_res)
