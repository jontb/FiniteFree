"""Independent numeric and exact contracts across large and strided batches."""

from typing import Any

import numpy as np
import pytest
import sympy as sp

from finitefree.multivariate import MultivariatePolynomial


def _points(layout: str) -> Any:
    rng = np.random.default_rng(49)
    if layout == "single_strided":
        return np.array([0.2, 0.3, 0.4])[::-1]
    points = rng.uniform(-0.6, 0.6, (5, 2117, 3))
    if layout == "contiguous":
        return points
    if layout == "fortran":
        return np.asfortranarray(points)
    if layout == "transpose":
        return points.transpose(1, 0, 2)
    if layout == "negative_strides":
        return points[::-1, ::-1, ::-1]
    if layout == "broadcast":
        return np.broadcast_to(points[0, :1], (5, 2117, 3))
    if layout == "empty":
        return points[:, :0]
    if layout == "slice":
        return rng.uniform(-0.6, 0.6, (17001, 6))[::2, ::2]
    raise AssertionError(layout)


@pytest.mark.parametrize(
    "layout",
    [
        "single_strided",
        "contiguous",
        "fortran",
        "transpose",
        "negative_strides",
        "broadcast",
        "empty",
        "slice",
    ],
)
@pytest.mark.parametrize("operation", ["evaluate", "gradient", "hessian"])
def test_large_and_strided_batches_match_analytic_polynomial(
    layout: str, operation: str
) -> None:
    x, y, z = sp.symbols("x y z")
    p = MultivariatePolynomial(x**4 * y + x * y * z / 3 + z**2 - 2 * x + 5, [x, y, z])
    points = _points(layout)
    snapshot = points.copy()
    a, b, c = (points[..., i] for i in range(3))
    if operation == "evaluate":
        expected = a**4 * b + a * b * c / 3 + c**2 - 2 * a + 5
    elif operation == "gradient":
        expected = np.stack(
            [4 * a**3 * b + b * c / 3 - 2, a**4 + a * c / 3, a * b / 3 + 2 * c], axis=-1
        )
    else:
        expected = np.empty((*points.shape[:-1], 3, 3))
        expected[..., 0, 0] = 12 * a**2 * b
        expected[..., 0, 1] = expected[..., 1, 0] = 4 * a**3 + c / 3
        expected[..., 0, 2] = expected[..., 2, 0] = b / 3
        expected[..., 1, 1] = 0
        expected[..., 1, 2] = expected[..., 2, 1] = a / 3
        expected[..., 2, 2] = 2
    method = getattr(p, operation + "_float64")
    actual = method(points)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(points, snapshot)
    if isinstance(actual, np.ndarray):
        assert actual.dtype == np.float64 and actual.flags.owndata
        assert not np.shares_memory(actual, points)
        actual.fill(999)
        np.testing.assert_allclose(method(points), expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("operation", ["evaluate", "gradient", "hessian"])
@pytest.mark.parametrize("bad", ["input", "intermediate"])
def test_nonfinite_values_in_later_blocks_are_rejected(
    operation: str, bad: str
) -> None:
    x = sp.Symbol("x")
    p = MultivariatePolynomial(x**5, [x])
    points = np.zeros((8211, 1))
    points[-1, 0] = np.inf if bad == "input" else 1e150
    with pytest.raises(ValueError if bad == "input" else RuntimeError, match="finite"):
        getattr(p, operation + "_float64")(points)


@pytest.mark.parametrize("operation", ["evaluate", "gradient", "hessian"])
def test_more_than_sixty_four_powers_match_separable_formula(operation: str) -> None:
    variables = sp.symbols("x0:70")
    terms = {}
    for i in range(70):
        alpha = [0] * 70
        alpha[i] = 5
        terms[tuple(alpha)] = i + 1
    p = MultivariatePolynomial.from_coefficients(terms, variables)
    points = np.linspace(-0.6, 0.6, 210).reshape(3, 70)
    weights = np.arange(1, 71)
    if operation == "evaluate":
        expected = np.sum(weights * points**5, axis=-1)
    elif operation == "gradient":
        expected = 5 * weights * points**4
    else:
        expected = np.zeros((3, 70, 70))
        diagonal = np.arange(70)
        expected[:, diagonal, diagonal] = 20 * weights * points**3
    np.testing.assert_allclose(
        getattr(p, operation + "_float64")(points), expected, rtol=3e-14, atol=3e-14
    )


@pytest.mark.parametrize("target", ["same", "renamed", "reordered"])
@pytest.mark.parametrize("merge", [False, True])
def test_symbol_composition_preserves_maps_in_equal_and_different_contexts(
    target: str, merge: bool
) -> None:
    x, y, z = source = sp.symbols("x y z", real=True)
    p = MultivariatePolynomial(x**3 / 7 + 2 * x * y + 3 * z**2 + y, source)
    variables = {
        "same": source,
        "renamed": sp.symbols("u v w", real=True),
        "reordered": source[::-1],
    }[target]
    a, b, c = variables
    substitutions = {x: c, y: c if merge else b, z: a}
    q = p.substitute(substitutions, variables=variables)
    assert (
        q.coefficients()
        == sp.Poly(p.expr.subs(substitutions, simultaneous=True), variables).as_dict()
    )
    assert q.variables == list(variables) and q is not p
    np.testing.assert_allclose(
        q.gradient_float64([0.2, 0.3, 0.4]),
        [
            float(sp.diff(q.expr, v).subs(dict(zip(variables, [0.2, 0.3, 0.4]))))
            for v in variables
        ],
    )
    assert p.expr == x**3 / 7 + 2 * x * y + 3 * z**2 + y


def test_native_reordering_retains_large_exact_exponents_and_assumptions() -> None:
    x, y = sp.symbols("x y", real=True)
    terms: dict[tuple[int, ...], Any] = {
        (2**65 + 7, 3): sp.Rational(2**270 + 11, 2**255 + 19),
        (0, 0): -5,
    }
    p = MultivariatePolynomial.from_coefficients(terms, [x, y])
    q = p.reorder_variables([y, x])
    assert q.coefficients() == {(b, a): c for (a, b), c in terms.items()}
    assert q.reorder_variables([x, y]).coefficients() == terms
    assert q.variables == [y, x] and p.coefficients() == terms


@pytest.mark.parametrize("family", ["constant", "linear", "quadratic"])
@pytest.mark.parametrize("operation", ["evaluate", "gradient", "hessian"])
def test_constant_results_keep_large_owned_shapes_and_validate_coordinates(
    family: str, operation: str
) -> None:
    x, y, z = variables = sp.symbols("x y z")
    expr = {
        "constant": sp.Rational(7, 3),
        "linear": x / 3 - 2 * y + z,
        "quadratic": x**2 / 3 - y * z + 2 * z**2,
    }[family]
    p = MultivariatePolynomial(expr, variables)
    points = np.asfortranarray(np.ones((5, 2117, 3)))
    symbolic = {
        "evaluate": expr,
        "gradient": [sp.diff(expr, v) for v in variables],
        "hessian": sp.hessian(expr, variables),
    }[operation]
    expected = np.asarray(sp.lambdify(variables, symbolic)(1, 1, 1), dtype=float)
    method = getattr(p, operation + "_float64")
    actual = method(points)
    np.testing.assert_array_equal(actual, np.broadcast_to(expected, actual.shape))
    assert actual.flags.owndata and not np.shares_memory(actual, points)
    actual.fill(999)
    np.testing.assert_array_equal(
        method(points), np.broadcast_to(expected, actual.shape)
    )
    points[-1, -1, -1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        method(points)
