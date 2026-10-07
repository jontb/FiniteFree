"""Independent contracts for context changes and batched polynomial derivatives."""

import itertools
from fractions import Fraction
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree.multivariate import MultivariatePolynomial


def _exact(value: Any) -> sp.Rational:
    return sp.Rational(int(value.p), int(value.q))


@pytest.mark.parametrize("expression", ["zero", "constant", "mixed"])
def test_reorder_preserves_symbol_meanings_and_sparse_coefficients(
    expression: str,
) -> None:
    x, y, z = sp.symbols("x y z")
    expr = {"zero": 0, "constant": 7, "mixed": x**3 * y / 3 - 5 * z + x * y * z + 2}[
        expression
    ]
    p = MultivariatePolynomial(expr, [x, y, z])
    values = {x: Fraction(1, 2), y: -3, z: 4}
    for order in itertools.permutations((x, y, z)):
        q = p.reorder_variables(order)
        assert q.variables == list(order)
        assert q.expr == p.expr
        expected = {
            tuple(alpha[(x, y, z).index(v)] for v in order): c
            for alpha, c in p.coefficients().items()
        }
        assert q.coefficients() == expected
        assert q.to_fmpq_mpoly().context().names() == tuple(v.name for v in order)
        assert q.evaluate([values[v] for v in order]) == p.evaluate(
            list(values.values())
        )
        assert q is not p


@pytest.mark.parametrize(
    "variables",
    [
        [],
        ["y", "x"],
        [sp.Symbol("x")],
        [sp.Symbol("x"), sp.Symbol("x")],
        [sp.Symbol("x"), sp.Symbol("z")],
        [sp.Symbol("x", real=True), sp.Symbol("y")],
    ],
)
def test_reorder_rejects_renaming_dropped_symbols_and_changed_assumptions(
    variables: Any,
) -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError, match="Variables|Variable|permutation"):
        MultivariatePolynomial(x + 2 * y, [x, y]).reorder_variables(variables)


def test_reorder_and_substitution_own_inputs_exports_and_numeric_caches() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x * x + y, [x, y])
    replacement = MultivariatePolynomial(x + y, [x, y])
    target = [y, x]
    reordered = p.reorder_variables(target)
    substituted = p.substitute({x: replacement})
    p.gradient_float64([2, 3])
    p.hessian_float64([2, 3])
    target[:] = [x, y]
    for obj in (p, replacement, reordered, substituted):
        exported = obj.to_fmpq_mpoly()
        exported += exported.context().constant(999)
        obj.variables[:] = [sp.Symbol("a"), sp.Symbol("b")]
    assert reordered.variables == [y, x] and reordered.expr == p.expr == x * x + y
    assert substituted.expr == sp.expand((x + y) ** 2 + y)
    assert replacement.expr == x + y
    np.testing.assert_array_equal(reordered.gradient_float64([3, 2]), [1, 4])
    np.testing.assert_array_equal(p.hessian_float64([2, 3]), [[2, 0], [0, 0]])


@pytest.mark.parametrize("native_replacements", [False, True])
def test_substitution_is_simultaneous_and_order_independent(
    native_replacements: bool,
) -> None:
    x, y = sp.symbols("x y")
    expression = x * x + 3 * y + x * y
    replacements: dict[sp.Symbol, Any] = {x: y, y: x + 1}
    if native_replacements:
        replacements = {
            v: MultivariatePolynomial(e, [x, y]) for v, e in replacements.items()
        }
    p = MultivariatePolynomial(expression, [x, y])
    q = p.substitute(replacements)
    expected = sp.expand(expression.subs({x: y, y: x + 1}, simultaneous=True))
    assert q.expr == expected
    assert p.substitute(dict(reversed(list(replacements.items())))).expr == expected
    assert q.expr != sp.expand(expression.subs({x: y}).subs({y: x + 1}))
    assert p.substitute({}).expr == expression and p.substitute({}) is not p


def test_explicit_target_context_supports_renaming_projection_and_embedding() -> None:
    x, y, u, v, w = sp.symbols("x y u v w")
    p = MultivariatePolynomial(x * x + x * y + y, [x, y])
    renamed = p.substitute({x: u, y: v}, variables=[v, u])
    assert renamed.variables == [v, u]
    assert renamed.expr == u * u + u * v + v
    assert _exact(renamed.evaluate([3, 2])) == 13
    projected = p.substitute({x: 2}, variables=[y])
    assert projected.variables == [y] and projected.expr == 3 * y + 4
    embedded = p.substitute({x: u + v, y: u * v}, variables=[w, u, v])
    assert embedded.variables == [w, u, v]
    assert embedded.expr == sp.expand((u + v) ** 2 + (u + v) * u * v + u * v)
    assert embedded.partial_derivative(w).expr == 0
    scalar = p.substitute({x: 2, y: Fraction(1, 3)}, variables=[w])
    assert isinstance(scalar, MultivariatePolynomial)
    assert scalar.variables == [w] and scalar.expr == 5
    assert p.substitute({x: 2, y: 3}).variables == [x, y]


@pytest.mark.parametrize(
    "value",
    [
        Fraction(2, 3),
        sp.Rational(2, 3),
        flint.fmpq(2, 3),
        np.int64(7),
        0.1,
        np.float32(0.1),
    ],
)
def test_substitution_scalars_keep_the_supplied_exact_value(value: Any) -> None:
    x = sp.Symbol("x")
    p = MultivariatePolynomial(x * x + 2 * x, [x])
    if isinstance(value, (float, np.floating)):
        expected = sp.Rational(*value.as_integer_ratio())
    elif isinstance(value, flint.fmpq):
        expected = _exact(value)
    else:
        expected = sp.Rational(value)
    q = p.substitute({x: value})
    assert q.expr == expected**2 + 2 * expected
    assert q.variables == [x] and q.degree() == 0


def test_substitution_preserves_extended_scalar_precision() -> None:
    if np.finfo(np.longdouble).nmant <= 52:
        pytest.skip("longdouble has no extra precision on this platform")
    x = sp.Symbol("x")
    value = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
    q = MultivariatePolynomial(x, [x]).substitute({x: value})
    assert q.expr == sp.Rational(2**60 + 1, 2**60)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        "x",
        "1/3",
        [1, 2],
        np.array(2),
        np.array([2]),
        1 + 0j,
        float("inf"),
        float("nan"),
        sp.oo,
        sp.sqrt(2),
        sp.I,
        sp.pi,
    ],
)
def test_substitution_rejects_nonrational_scalar_domains(value: Any) -> None:
    x = sp.Symbol("x")
    with pytest.raises(ValueError, match="rational|finite"):
        MultivariatePolynomial(x, [x]).substitute({x: value})


@pytest.mark.parametrize(
    "replacement",
    [
        sp.Symbol("z"),
        sp.Symbol("x", real=True),
        sp.sin(sp.Symbol("x")),
        1 / sp.Symbol("x"),
        sp.sqrt(2) * sp.Symbol("x"),
        sp.I * sp.Symbol("x"),
    ],
)
def test_substitution_rejects_unknown_symbols_and_nonpolynomial_domains(
    replacement: sp.Expr,
) -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError, match="target|rational"):
        MultivariatePolynomial(x + y, [x, y]).substitute({x: replacement})


def test_substitution_requires_explicit_matching_contexts_and_stored_keys() -> None:
    x, y, z = sp.symbols("x y z")
    p = MultivariatePolynomial(x + y, [x, y])
    q = MultivariatePolynomial(y + x * x, [y, x])
    with pytest.raises(ValueError, match="variable sequences"):
        p.substitute({x: q})
    assert p.substitute({x: q.reorder_variables([x, y])}).expr == x * x + 2 * y
    for mapping in ({z: 1}, {"x": 1}, {sp.Symbol("x", real=True): 1}):
        with pytest.raises(ValueError, match="keys"):
            p.substitute(mapping)
    with pytest.raises(TypeError, match="mapping"):
        p.substitute([(x, 1)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target"):
        p.substitute({x: 1}, variables=[z])
    with pytest.raises(ValueError, match="nonempty"):
        p.substitute({x: 1, y: 2}, variables=[])


@pytest.mark.parametrize("seed", range(4))
def test_batch_derivatives_against_independent_symbolic_references(seed: int) -> None:
    variables = sp.symbols("x y z")
    rng = np.random.default_rng(seed)
    expression = sp.expand(
        sum(
            sp.Rational(int(rng.integers(-4, 5)), int(rng.integers(1, 5)))
            * sp.prod(v ** int(rng.integers(0, 4)) for v in variables)
            for _ in range(10)
        )
    )
    p = MultivariatePolynomial(expression, variables)
    points = rng.uniform(-1, 1, (2, 3, 3))
    expected_g, expected_h = [], []
    for point in points.reshape(-1, 3):
        mapping = dict(zip(variables, [sp.Rational(float(v)) for v in point]))
        expected_g.append(
            [float(sp.diff(expression, v).subs(mapping)) for v in variables]
        )
        expected_h.append(
            [
                [float(sp.diff(expression, v, w).subs(mapping)) for w in variables]
                for v in variables
            ]
        )
    gradient, hessian = p.gradient_float64(points), p.hessian_float64(points)
    assert gradient.shape == (2, 3, 3) and hessian.shape == (2, 3, 3, 3)
    assert gradient.dtype == hessian.dtype == np.float64
    np.testing.assert_allclose(
        gradient, np.array(expected_g).reshape(2, 3, 3), rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        hessian, np.array(expected_h).reshape(2, 3, 3, 3), rtol=2e-13, atol=2e-13
    )
    np.testing.assert_array_equal(hessian, np.swapaxes(hessian, -1, -2))


@pytest.mark.parametrize(
    "dtype", [np.int64, np.float16, np.float32, np.float64, np.longdouble]
)
def test_derivative_batch_shapes_dtypes_broadcast_and_ownership(dtype: Any) -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x * x * y + y * y, [x, y])
    point = np.array([2, 3], dtype=dtype)
    points = np.broadcast_to(point, (2, 1, 4, 2))
    gradient, hessian = p.gradient_float64(points), p.hessian_float64(points)
    assert gradient.shape == (2, 1, 4, 2)
    assert hessian.shape == (2, 1, 4, 2, 2)
    assert gradient.dtype == hessian.dtype == np.float64
    np.testing.assert_array_equal(gradient, np.broadcast_to([12, 10], gradient.shape))
    np.testing.assert_array_equal(
        hessian, np.broadcast_to([[6, 4], [4, 2]], hessian.shape)
    )
    gradient[:] = 99
    hessian[..., 0, 1] = 88
    point[:] = 0
    np.testing.assert_array_equal(p.gradient_float64([2, 3]), [12, 10])
    np.testing.assert_array_equal(p.hessian_float64([2, 3]), [[6, 4], [4, 2]])
    assert not np.shares_memory(gradient, points)
    assert not np.shares_memory(hessian, points)


@pytest.mark.parametrize("expression", [0, 7, 10**500])
@pytest.mark.parametrize("shape", [(2,), (0, 2), (2, 0, 3, 2)])
def test_constant_zero_and_empty_derivative_batches(
    expression: Any, shape: tuple[int, ...]
) -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(expression, [x, y])
    points = np.zeros(shape)
    gradient, hessian = p.gradient_float64(points), p.hessian_float64(points)
    assert gradient.shape == shape and hessian.shape == (*shape, 2)
    np.testing.assert_array_equal(gradient, np.zeros(shape))
    np.testing.assert_array_equal(hessian, np.zeros((*shape, 2)))


@pytest.mark.parametrize(
    "points",
    [
        1,
        [1],
        [1, 2, 3],
        [[1], [1, 2]],
        [float("nan"), 0],
        [0, float("inf")],
        [1 + 0j, 2],
        np.array([1j, 2], dtype=object),
        [10**500, 0],
    ],
)
@pytest.mark.parametrize(
    "method", ["evaluate_float64", "gradient_float64", "hessian_float64"]
)
def test_derivative_batches_reject_invalid_coordinates(
    points: Any, method: str
) -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError):
        getattr(MultivariatePolynomial(x + y, [x, y]), method)(points)


def test_single_variable_and_singular_points_need_no_coordinate_division() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x * x * y + y * y, [x, y])
    np.testing.assert_array_equal(p.gradient_float64([0, 0]), [0, 0])
    np.testing.assert_array_equal(p.hessian_float64([0, 0]), [[0, 0], [0, 2]])
    determinant = MultivariatePolynomial(x * y, [x, y])
    np.testing.assert_array_equal(determinant.hessian_float64([0, 0]), [[0, 1], [1, 0]])
    univariate = MultivariatePolynomial(3 * x**3, [x])
    np.testing.assert_array_equal(univariate.gradient_float64([2]), [36])
    np.testing.assert_array_equal(univariate.hessian_float64([2]), [[36]])
    assert univariate.gradient_float64(np.empty((0, 1))).shape == (0, 1)


@pytest.mark.parametrize(
    "method", ["evaluate_float64", "gradient_float64", "hessian_float64"]
)
def test_object_arrays_cannot_hide_numpy_complex_coordinates(method: str) -> None:
    x = sp.Symbol("x")
    points = np.array([np.complex128(1j)], dtype=object)
    with pytest.raises(ValueError, match="real"):
        getattr(MultivariatePolynomial(x * x, [x]), method)(points)


def test_many_distinct_powers_against_exact_derivative_sums() -> None:
    x = sp.Symbol("x")
    expression = sum(x**k for k in range(100))
    p = MultivariatePolynomial(expression, [x])
    points = np.array([[0], [0.5], [1]])
    for method, order in ((p.gradient_float64, 1), (p.hessian_float64, 2)):
        actual = method(points).reshape(-1)
        expected = [
            float(sp.diff(expression, x, order).subs(x, sp.Rational(float(value))))
            for value in points[:, 0]
        ]
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


def test_exact_differentiation_precedes_float_conversion_and_ignores_unused_constants() -> (
    None
):
    x = sp.Symbol("x")
    p = MultivariatePolynomial.from_coefficients(
        {(1_000_000,): sp.Rational(1, 10**324)}, [x]
    )
    assert p.evaluate_float64([1]) == 0.0
    assert p.gradient_float64([1])[0] == float(sp.Rational(1_000_000, 10**324))
    assert p.hessian_float64([1])[0, 0] == float(
        sp.Rational(1_000_000 * 999_999, 10**324)
    )
    p = MultivariatePolynomial(10**500 + x * x, [x])
    np.testing.assert_array_equal(p.gradient_float64([3]), [6])
    np.testing.assert_array_equal(p.hessian_float64([3]), [[2]])
    with pytest.raises(RuntimeError, match="coefficients"):
        p.evaluate_float64([3])


@pytest.mark.parametrize(
    "method, expression, point, match",
    [
        ("gradient_float64", 10**500 * sp.Symbol("x"), [0], "coefficients"),
        ("hessian_float64", 10**500 * sp.Symbol("x") ** 2, [0], "coefficients"),
        ("gradient_float64", sp.Symbol("x") ** 3, [1e200], "nonfinite"),
        ("hessian_float64", sp.Symbol("x") ** 4, [1e200], "nonfinite"),
        ("gradient_float64", 10**308 * sp.Symbol("x") ** 2, [1], "coefficients"),
    ],
)
def test_derivative_range_failures_are_explicit_and_do_not_poison_exact_algebra(
    method: str, expression: Any, point: Any, match: str
) -> None:
    x = sp.Symbol("x")
    p = MultivariatePolynomial(expression, [x])
    with pytest.raises(RuntimeError, match=match):
        getattr(p, method)(point)
    assert p.expr == expression
    assert p.partial_derivative(x).expr == sp.diff(expression, x)
    if match == "nonfinite":
        assert np.all(np.isfinite(getattr(p, method)([0])))


def test_derivatives_can_be_finite_when_value_overflows() -> None:
    x = sp.Symbol("x")
    p = MultivariatePolynomial(x * x, [x])
    with pytest.raises(RuntimeError, match="nonfinite"):
        p.evaluate_float64([1e200])
    np.testing.assert_array_equal(p.gradient_float64([1e200]), [2e200])
    np.testing.assert_array_equal(p.hessian_float64([1e200]), [[2]])


def test_batch_derivative_float64_narrowing_and_underflow_are_explicit() -> None:
    x = sp.Symbol("x")
    supplied = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
    p = MultivariatePolynomial(x * x, [x])
    np.testing.assert_array_equal(p.gradient_float64([supplied]), [2.0])
    assert p.expr == x * x
    tiny = MultivariatePolynomial(sp.Rational(1, 10**500) * x * x, [x])
    np.testing.assert_array_equal(tiny.gradient_float64([1]), [0])
    np.testing.assert_array_equal(tiny.hessian_float64([1]), [[0]])


def test_simultaneous_composition_exact_and_numeric_chain_rules() -> None:
    x, y, u, v = sp.symbols("x y u v")
    expression = x**3 * y + x * y * y + sp.Rational(2, 3) * x
    maps = [u * u + v, u * v + 2 * v * v]
    p = MultivariatePolynomial(expression, [x, y])
    composed = p.substitute({x: maps[0], y: maps[1]}, variables=[u, v])
    expected = sp.expand(expression.subs(dict(zip((x, y), maps)), simultaneous=True))
    assert composed.expr == expected
    outer_g = sp.Matrix([sp.diff(expression, a) for a in (x, y)])
    outer_h = sp.hessian(expression, (x, y))
    jacobian = sp.Matrix(maps).jacobian((u, v))
    outer_map = dict(zip((x, y), maps))
    chain_g = jacobian.T * outer_g.subs(outer_map, simultaneous=True)
    chain_h = jacobian.T * outer_h.subs(outer_map, simultaneous=True) * jacobian
    chain_h += sum(
        (
            outer_g[i].subs(outer_map, simultaneous=True) * sp.hessian(maps[i], (u, v))
            for i in range(2)
        ),
        sp.zeros(2),
    )
    for i in range(2):
        assert sp.expand(composed.gradient()[i].expr - chain_g[i]) == 0
        for j in range(2):
            assert sp.expand(composed.hessian()[i][j].expr - chain_h[i, j]) == 0
    points = np.array([[0, 0], [0.2, -0.3], [-0.5, 0.8]])
    expected_g = np.array(
        [
            np.array(chain_g.subs(dict(zip((u, v), point))), dtype=float).ravel()
            for point in points
        ]
    )
    expected_h = np.array(
        [
            np.array(chain_h.subs(dict(zip((u, v), point))), dtype=float)
            for point in points
        ]
    )
    np.testing.assert_allclose(
        composed.gradient_float64(points), expected_g, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        composed.hessian_float64(points), expected_h, rtol=2e-13, atol=2e-13
    )


def test_derivatives_against_finite_differences_of_polynomial_values() -> None:
    x, y, z = sp.symbols("x y z")
    p = MultivariatePolynomial(x**4 + 2 * x * y * z + y * y * z * z + 3 * z, [x, y, z])
    point = np.array([0.3, -0.4, 0.6])
    step = 1e-5
    basis = step * np.eye(3)
    expected_g = np.array(
        [
            (p.evaluate_float64(point + e) - p.evaluate_float64(point - e)) / (2 * step)
            for e in basis
        ]
    )
    expected_h = np.stack(
        [
            (p.gradient_float64(point + e) - p.gradient_float64(point - e)) / (2 * step)
            for e in basis
        ],
        axis=-1,
    )
    np.testing.assert_allclose(
        p.gradient_float64(point), expected_g, rtol=2e-8, atol=2e-8
    )
    np.testing.assert_allclose(
        p.hessian_float64(point), expected_h, rtol=2e-8, atol=2e-8
    )
