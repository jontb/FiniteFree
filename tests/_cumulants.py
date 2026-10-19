"""Independent cumulant definitions and shared exact/numerical assertions."""

import math
from typing import Any, Iterator

import numpy as np
import sympy as sp

from finitefree import FiniteRTransform, RealRootedPolynomial


def normalized_coefficients(p: RealRootedPolynomial, order: int, d: int) -> list[Any]:
    coefficients = list(p.coeffs)
    return [
        (-1) ** k
        * sp.cancel(sp.sympify(coefficients[k]) / sp.sympify(coefficients[0]))
        / math.comb(d, k)
        if k <= p.degree
        else 0
        for k in range(min(order, d) + 1)
    ]


def logarithm_reference(p: RealRootedPolynomial, order: int, d: int) -> list[Any]:
    """Truncated formal logarithm, independent of the production recurrence."""
    z = sp.Symbol("z")
    moments = normalized_coefficients(p, order, d)
    u = sum(moments[k] * z**k / math.factorial(k) for k in range(1, len(moments)))
    log = sp.Poly(0, z)
    power = sp.Poly(1, z)
    for k in range(1, min(order, d) + 1):
        power = sp.Poly(power.as_expr() * u, z)
        power = sp.Poly(
            sum(c * z ** n[0] for n, c in power.terms() if n[0] <= order), z
        )
        log += power * sp.Rational((-1) ** (k + 1), k)
    return [
        sp.cancel(n * (-d) ** (n - 1) * log.nth(n)) if n <= d else 0
        for n in range(1, order + 1)
    ]


def partitions(elements: list[int]) -> Iterator[list[list[int]]]:
    if not elements:
        yield []
        return
    first = elements[0]
    for sub in partitions(elements[1:]):
        for i, block in enumerate(sub):
            yield sub[:i] + [[first] + block] + sub[i + 1 :]
        yield sub + [[first]]


def partition_reference(p: RealRootedPolynomial, order: int, d: int) -> list[Any]:
    """Direct partition-lattice definition, used at small orders."""
    moments = normalized_coefficients(p, order, d)
    result = []
    for n in range(1, order + 1):
        total = (
            sum(
                (-1) ** (len(partition) - 1)
                * math.factorial(len(partition) - 1)
                * math.prod(moments[len(block)] for block in partition)
                for partition in partitions(list(range(1, n + 1)))
            )
            if n <= d
            else 0
        )
        result.append(
            sp.cancel(total * sp.Rational((-d) ** (n - 1), math.factorial(n - 1)))
        )
    return result


def assert_cumulants(
    p: RealRootedPolynomial,
    order: int,
    *,
    d: int | None = None,
    expected: list[Any] | None = None,
    prec: int = 256,
    rtol: float = 1e-14,
    atol: float = 1e-14,
) -> list[Any]:
    dimension = p.degree if d is None else d
    reference = logarithm_reference(p, order, dimension)
    if expected is not None:
        assert reference == expected
    assert FiniteRTransform(p, order=order, d=dimension) == reference
    actual = FiniteRTransform(p, order=order, d=dimension, numerical=True, prec=prec)
    assert len(actual) == order
    np.testing.assert_allclose(
        actual, [float(v) for v in reference], rtol=rtol, atol=atol
    )
    return reference
