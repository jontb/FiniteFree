"""Independent determinant oracles across scales, blocks and cache histories."""

import numpy as np
import sympy as sp

from finitefree.hyperbolic import PreparedMatrixPencil
from tests._cases import case


def test_prepared_derivatives_against_symbolic_determinants_across_scales() -> None:
    matrices = np.array(
        [
            [[2, 1, 0], [0, 1, 1], [1, 0, 3]],
            [[1, 0, 1], [-1, 2, 0], [0, 1, 1]],
            [[0, 1, -1], [1, 0, 2], [2, -1, 1]],
        ],
        dtype=float,
    )
    variables = sp.symbols("x:3")
    symbolic = sum(
        (x * sp.Matrix(a.astype(int)) for x, a in zip(variables, matrices)), sp.zeros(3)
    ).det()
    first = sp.Matrix([sp.diff(symbolic, x) for x in variables])
    second = sp.hessian(symbolic, variables)
    points = np.array([[2, 0.25, -0.5], [-1, 0.5, 2], [3, -0.25, 1.5]])[::-1]
    directions = np.array([[0.25, -0.5, 1], [1, 0, -1], [-0.5, 1, 0.25]])[::-1]
    values, gradients, hessians = [], [], []
    for point in points:
        substitutions = dict(zip(variables, map(sp.Rational, point)))
        value = float(symbolic.subs(substitutions))
        gradient = np.array(first.subs(substitutions), dtype=float).ravel() / value
        hessian = np.array(second.subs(substitutions), dtype=float) / value
        values.append(value)
        gradients.append(gradient)
        hessians.append(hessian - np.outer(gradient, gradient))
    for scale in [1e-150, 1.0, 1e150]:
        for point_block, coefficient_block in [(1, 1), (2, 2), (64, 4)]:
            with case(
                scale=scale,
                point_block=point_block,
                coefficient_block=coefficient_block,
            ):
                source = matrices * scale
                prepared = PreparedMatrixPencil(
                    source,
                    point_block_size=point_block,
                    coefficient_block_size=coefficient_block,
                )
                factor = prepared.factor(points)
                source[:] = 999  # Construction and factorization must own their data.
                expected_hvp = np.einsum("bij,bj->bi", hessians, directions)
                # Request HVP before gradient, then alternate directions/cache hits.
                np.testing.assert_allclose(
                    factor.logabsdet_hvp(directions),
                    expected_hvp,
                    rtol=3e-12,
                    atol=3e-12,
                )
                np.testing.assert_allclose(
                    factor.logabsdet_gradient(), gradients, rtol=3e-12, atol=3e-12
                )
                np.testing.assert_allclose(
                    factor.logabsdet_hvp(-2 * directions),
                    -2 * expected_hvp,
                    rtol=3e-12,
                    atol=3e-12,
                )
                sign, logabsdet = factor.slogdet()
                np.testing.assert_array_equal(sign, np.sign(values))
                np.testing.assert_allclose(
                    logabsdet,
                    np.log(np.abs(values)) + 3 * np.log(scale),
                    rtol=0,
                    atol=5e-13,
                )
                np.testing.assert_allclose(
                    prepared.evaluate(points) / scale,
                    np.einsum("bi,ijk->bjk", points, matrices),
                    rtol=2e-15,
                    atol=2e-15,
                )
