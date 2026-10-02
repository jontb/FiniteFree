"""Jacobi matrices for stable numerical roots of classical orthogonal families.

The monic recurrence p_{k+1}=(x-a_k)p_k-b_k p_{k-1} gives a symmetric
tridiagonal matrix with diagonal a_k and off-diagonal sqrt(b_k).
See NIST DLMF 18.2(vi) and 18.9. Parameter arithmetic precedes float conversion
to preserve small differences such as alpha+1 near the Laguerre hard edge.
"""

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import sympy as sp

from .conversion import flint_to_float, sympy_to_fmpq

if TYPE_CHECKING:
    from ..core import RealRootedPolynomial


def _attach_recurrence(
    p: "RealRootedPolynomial", diagonal: Sequence[Any], squared_off: Sequence[Any]
) -> "RealRootedPolynomial":
    if p.degree < 2:
        return p
    diag_exact = [sympy_to_fmpq(x) for x in diagonal]
    diag = np.array([flint_to_float(x) for x in diag_exact])
    squared = np.array([flint_to_float(sympy_to_fmpq(x)) for x in squared_off])
    if (
        np.all(np.isfinite(diag))
        and np.all(np.isfinite(squared))
        and np.all(squared > 0)
        and all(x == 0 or value != 0 for x, value in zip(diag_exact, diag))
    ):
        p._root_recurrence = (diag, np.sqrt(squared))
    return p


def _hermite_roots(
    p: "RealRootedPolynomial", physicist: bool
) -> "RealRootedPolynomial":
    divisor = 2 if physicist else 1
    p._hermite_variance = sp.Rational(1, divisor)
    return _attach_recurrence(
        p, [0] * p.degree, [sp.Rational(k, divisor) for k in range(1, p.degree)]
    )


def _laguerre_roots(p: "RealRootedPolynomial", alpha: Any) -> "RealRootedPolynomial":
    if alpha <= -1:
        return p
    p = _attach_recurrence(
        p,
        [2 * k + alpha + 1 for k in range(p.degree)],
        [k * (k + alpha) for k in range(1, p.degree)],
    )
    p._root_recurrence_driver = "sterf"
    p._root_recurrence_positive = True
    return p


def _gaussian_roots(
    p: "RealRootedPolynomial", variance: Any, center: Any = 0
) -> "RealRootedPolynomial":
    p = _hermite_roots(p, False)
    p._hermite_variance = variance
    p._hermite_center = center
    variance_float = flint_to_float(variance)
    center_float = flint_to_float(center)
    if (
        variance_float <= 0
        or not np.isfinite(variance_float)
        or not np.isfinite(center_float)
    ):
        p._root_recurrence = None
    else:
        p._root_scale = float(np.sqrt(variance_float))
        p._root_shift = sympy_to_fmpq(center)
    return p


def _jacobi_roots(
    p: "RealRootedPolynomial", alpha: Any, beta: Any
) -> "RealRootedPolynomial":
    if alpha <= -1 or beta <= -1 or p.degree < 2:
        return p
    s = alpha + beta
    diagonal = [(beta - alpha) / (s + 2)]
    diagonal.extend(
        (beta - alpha) * s / ((2 * k + s) * (2 * k + s + 2)) for k in range(1, p.degree)
    )
    # The k=1 cancellation also handles the removable singularity s=-1.
    squared = [4 * (1 + alpha) * (1 + beta) / ((s + 2) ** 2 * (s + 3))]
    squared.extend(
        4
        * k
        * (k + alpha)
        * (k + beta)
        * (k + s)
        / ((2 * k + s) ** 2 * (2 * k + s + 1) * (2 * k + s - 1))
        for k in range(2, p.degree)
    )
    return _attach_recurrence(p, diagonal, squared)


def _legendre_roots(p: "RealRootedPolynomial") -> "RealRootedPolynomial":
    return _attach_recurrence(
        p,
        [0] * p.degree,
        [sp.Rational(k * k, 4 * k * k - 1) for k in range(1, p.degree)],
    )


def _chebyshev_roots(
    p: "RealRootedPolynomial", first_kind: bool
) -> "RealRootedPolynomial":
    squared = [sp.Rational(1, 4)] * (p.degree - 1)
    if first_kind and squared:
        squared[0] = sp.Rational(1, 2)
    return _attach_recurrence(p, [0] * p.degree, squared)
