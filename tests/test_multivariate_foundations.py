from typing import Any, Callable

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree.multivariate import MultivariatePolynomial


def _rational(value: Any) -> sp.Rational:
    # SymPy 1.12 does not implement exact mixed equality with FLINT rationals.
    return sp.Rational(int(value.p), int(value.q))


def test_exact_evaluation_against_symbolic_substitution() -> None:
    x, y = sp.symbols("x y")
    expression = sp.Rational(2, 3) * x**3 * y - x + 7
    p = MultivariatePolynomial(expression, [x, y])
    for point in ([1, 2], [sp.Rational(1, 3), -2], [0.1, np.float64(0.3)]):
        exact = [sp.Rational(v) for v in point]
        assert _rational(p.evaluate(point)) == expression.subs(dict(zip((x, y), exact)))


def test_native_context_does_not_silently_relabel_variables() -> None:
    x, y = sp.symbols("x y")
    ctx = flint.fmpq_mpoly_ctx.get(("x", "y"))
    native = ctx.from_dict({(1, 0): 2, (0, 1): 3})
    with pytest.raises(ValueError, match="context|variable"):
        MultivariatePolynomial(native, [y, x])


def test_polynomial_owns_native_input_and_exports() -> None:
    x, y = sp.symbols("x y")
    ctx = flint.fmpq_mpoly_ctx.get(("x", "y"))
    native = ctx.from_dict({(1, 0): 2})
    p = MultivariatePolynomial(native, [x, y])
    native += ctx.gen(1)
    assert p.expr == 2 * x
    exported = p.to_fmpq_mpoly()
    exported += ctx.gen(1)
    variables = p.variables
    variables[0] = sp.Symbol("z")
    assert p.expr == 2 * x and p.variables == [x, y]


def test_negative_derivative_order_is_rejected() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x**2 * y + x, [x, y])
    with pytest.raises(ValueError, match="nonnegative"):
        p.mixed_partial_derivative([-1, 0])


def test_normalized_coefficients_require_homogeneity() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x**2 * y + x, [x, y])
    with pytest.raises(ValueError, match="homogeneous"):
        p.normalized_coefficients()


@pytest.mark.parametrize(
    "variables",
    [
        [],
        ["x"],
        [1],
        [sp.Symbol("x"), sp.Symbol("x")],
        [sp.Symbol("x"), sp.Symbol("x", real=True)],
    ],
)
def test_invalid_variable_contexts(variables: list[Any]) -> None:
    with pytest.raises(ValueError, match="Variables|Variable"):
        MultivariatePolynomial(0, variables)


@pytest.mark.parametrize("order", [-1, 0.5, True, np.bool_(True)])
def test_invalid_derivative_counts(order: Any) -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x**2 + y, [x, y])
    with pytest.raises((ValueError, TypeError)):
        p.mixed_partial_derivative([order, 0])
    with pytest.raises((ValueError, TypeError)):
        p.partial_derivative(x, order)


def test_derivatives_and_euler_identity() -> None:
    x, y, z = sp.symbols("x y z")
    expression = 2 * x**3 + sp.Rational(1, 3) * x * y * z - y**2 * z
    p = MultivariatePolynomial(expression, [x, y, z])
    grad = p.gradient()
    hess = p.hessian()
    assert sp.expand(sum(v * g.expr for v, g in zip((x, y, z), grad))) == 3 * expression
    for i, a in enumerate((x, y, z)):
        assert grad[i].expr == sp.diff(expression, a)
        for j, b in enumerate((x, y, z)):
            assert hess[i][j].expr == sp.diff(expression, a, b)
    assert p.mixed_partial_derivative([np.int64(2), 0, 0]).expr == 12 * x
    assert p.mixed_partial_derivative([100, 0, 0]).expr == 0
    assert p.mixed_partial_derivative([10**20, 0, 0]).expr == 0
    assert p.partial_derivative(x, 0).expr == expression
    with pytest.raises(ValueError):
        p.partial_derivative(sp.Symbol("w"))
    with pytest.raises(ValueError):
        p.mixed_partial_derivative([1])
    with pytest.raises(ValueError):
        p.directional_derivative([1])
    direction = [sp.Rational(1, 3), -2, 5]
    assert sp.expand(p.directional_derivative(direction).expr) == sp.expand(
        sum(v * sp.diff(expression, a) for a, v in zip((x, y, z), direction))
    )


@pytest.mark.parametrize("seed", range(8))
def test_rational_ring_identities_against_sympy(seed: int) -> None:
    x, y = sp.symbols("x y")
    rng = np.random.default_rng(seed)

    def expression() -> sp.Expr:
        return sp.expand(
            sum(
                sp.Rational(int(rng.integers(-5, 6)), int(rng.integers(1, 6)))
                * x ** int(rng.integers(0, 4))
                * y ** int(rng.integers(0, 4))
                for _ in range(6)
            )
        )

    a, b = expression(), expression()
    p, q = MultivariatePolynomial(a, [x, y]), MultivariatePolynomial(b, [x, y])
    for actual, expected in (
        (p + q, a + b),
        (p - q, a - b),
        (p * q, a * b),
        (p ** np.int64(2), a**2),
        (-p, -a),
        (sp.Rational(2, 3) + p, sp.Rational(2, 3) + a),
        (sp.Rational(2, 3) - p, sp.Rational(2, 3) - a),
        (sp.Rational(2, 3) * p, sp.Rational(2, 3) * a),
    ):
        assert actual.expr == sp.expand(expected)
        point = [sp.Rational(1, 3), sp.Rational(-2, 5)]
        assert _rational(actual.evaluate(point)) == expected.subs(
            dict(zip((x, y), point))
        )
    assert ((p + q) * p).expr == (p * p + q * p).expr
    assert (p * 0).expr == 0 and (p * 0).degree() == -1
    assert (p**0).expr == 1


def test_algebra_rejects_implicit_variable_reordering() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x, [x, y])
    q = MultivariatePolynomial(y, [y, x])
    operations: list[Callable[[], MultivariatePolynomial]] = [
        lambda: p + q,
        lambda: p - q,
        lambda: p * q,
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="variable sequences"):
            operation()
    powers: list[Any] = [-1, 0.5, True]
    for power in powers:
        with pytest.raises((TypeError, ValueError)):
            p**power


def test_sparse_coefficient_roundtrip_and_ownership() -> None:
    x, y = sp.symbols("x y")
    terms: dict[tuple[int, ...], Any] = {
        (2, 0): sp.Rational(1, 3),
        (0, 1): -4,
        (0, 0): 2,
    }
    p = MultivariatePolynomial.from_coefficients(terms, [x, y])
    terms[(2, 0)] = 99
    actual = p.coefficients()
    assert actual == {(2, 0): sp.Rational(1, 3), (0, 1): -4, (0, 0): 2}
    actual[(2, 0)] = 100
    q = MultivariatePolynomial.from_coefficients(p.coefficients(), p.variables)
    assert p.expr == q.expr == x**2 / 3 - 4 * y + 2
    assert MultivariatePolynomial.from_coefficients({}, [x, y]).expr == 0
    for exponents in ((-1, 0), (0.5, 0), (True, 0), (1,)):
        with pytest.raises((ValueError, TypeError)):
            invalid: Any = {exponents: 1}
            MultivariatePolynomial.from_coefficients(invalid, [x, y])


def test_batched_float_evaluation_against_exact_reference() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(sp.Rational(2, 3) * x**3 * y - x + 7, [x, y])
    points = np.random.default_rng(51).uniform(-2, 2, (3, 4, 2))
    result = p.evaluate_float64(points)
    assert isinstance(result, np.ndarray)
    expected = np.array(
        [
            float(p.expr.subs({x: sp.Rational(a), y: sp.Rational(b)}))
            for a, b in points.reshape(-1, 2)
        ]
    ).reshape(3, 4)
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=2e-14)
    assert isinstance(p.evaluate_float64([1, 2]), float)
    empty = p.evaluate_float64(np.empty((0, 2)))
    assert isinstance(empty, np.ndarray) and empty.shape == (0,)
    result[0, 0] = 100
    np.testing.assert_allclose(
        p.evaluate_float64(points), expected, rtol=2e-14, atol=2e-14
    )


@pytest.mark.parametrize(
    "point",
    [
        1,
        [1],
        [1, 2, 3],
        [float("inf"), 1],
        [float("nan"), 1],
        [1 + 1j, 2],
        [10**500, 1],
    ],
)
def test_invalid_float_points(point: object) -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError):
        MultivariatePolynomial(x + y, [x, y]).evaluate_float64(point)


def test_numerical_range_and_zero_constants() -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(RuntimeError, match="coefficients"):
        MultivariatePolynomial(10**500 * x, [x, y]).evaluate_float64([1, 0])
    with pytest.raises(RuntimeError, match="nonfinite"):
        MultivariatePolynomial(x**100, [x, y]).evaluate_float64([1e10, 0])
    assert (
        MultivariatePolynomial(sp.Rational(1, 10**500) * x, [x, y]).evaluate_float64(
            [1, 0]
        )
        == 0.0
    )
    assert MultivariatePolynomial(0, [x, y]).evaluate_float64([1, 2]) == 0.0
    np.testing.assert_array_equal(
        MultivariatePolynomial(3, [x, y]).evaluate_float64([[1, 2], [3, 4]]), [3, 3]
    )


def test_exact_line_restriction_preserves_scalar_and_polynomial_identity() -> None:
    x, y, t = sp.symbols("x y t")
    expression = 3 * x**2 * y + sp.Rational(1, 3) * x + 7
    p = MultivariatePolynomial(expression, [x, y])
    base = [sp.Rational(1, 2), -2]
    direction = [3, sp.Rational(-2, 3)]
    q = p.restrict_line(base, direction)
    expected = expression.subs(
        {x: base[0] + t * direction[0], y: base[1] + t * direction[1]}
    )
    for value in (sp.Rational(1, 3), -2, 0):
        assert _rational(q.evaluate(value)) == expected.subs(t, value)
    assert q.coeffs[0] == sp.Poly(expected, t).LC()
    assert (
        MultivariatePolynomial(3, [x, y]).restrict_line([1, 2], [0, 0]).coeffs[0] == 3
    )
    with pytest.raises(ValueError, match="identically zero"):
        MultivariatePolynomial(expression - 7, [x, y]).restrict_line([0, 0], [0, 0])


def test_line_restriction_does_not_infer_real_rootedness() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x**2 + y**2, [x, y])
    q = p.restrict_line([1, 0], [0, 1])
    with pytest.raises(ValueError, match="real"):
        q.verify_real_rootedness()
    with pytest.raises(ValueError, match="identically zero"):
        p.restrict_line([0, 0], [0, 0])
    with pytest.raises(ValueError):
        p.restrict_line([1], [0, 1])


def test_homogeneous_normalization_reconstruction_and_zero() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(2 * x**3 + sp.Rational(3, 2) * x * y**2, [x, y])
    normalized = p.normalized_coefficients()
    assert all(type(k) is int for alpha in normalized for k in alpha)
    reconstructed = sum(
        c
        * sp.factorial(3)
        / sp.prod(sp.factorial(k) for k in alpha)
        * x ** alpha[0]
        * y ** alpha[1]
        for alpha, c in normalized.items()
    )
    assert sp.expand(reconstructed) == p.expr
    assert MultivariatePolynomial(0, [x, y]).normalized_coefficients() == {}


def test_polynomial_derivatives_work_at_singular_determinant_point() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x**2 - y**2, [x, y])
    assert [g.evaluate([0, 0]) for g in p.gradient()] == [0, 0]
    assert [[v.evaluate([0, 0]) for v in row] for row in p.hessian()] == [
        [2, 0],
        [0, -2],
    ]


@pytest.mark.parametrize("method", ["direct", "interpolated", "sparse"])
def test_public_evaluation_of_matrix_polynomials_matches_exact_determinant(
    method: str,
) -> None:
    from finitefree.hyperbolic import SymmetricMatrixPencil

    matrices = [
        sp.Matrix([[sp.Rational(1, 3), 2], [2, -1]]),
        sp.Matrix([[3, sp.Rational(-1, 2)], [sp.Rational(-1, 2), 1]]),
    ]
    pencil = SymmetricMatrixPencil([m.tolist() for m in matrices])
    factories = {
        "direct": MultivariatePolynomial.from_symmetric_matrix_pencil,
        "interpolated": MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated,
        "sparse": MultivariatePolynomial.from_symmetric_matrix_pencil_sparse,
    }
    p = factories[method](pencil)
    for a, b in ((1, 2), (sp.Rational(1, 3), sp.Rational(-2, 5)), (0, 0)):
        assert (
            _rational(p.evaluate([a, b])) == (a * matrices[0] + b * matrices[1]).det()
        )
