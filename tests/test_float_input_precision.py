from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import RealRootedPolynomial
from finitefree.hyperbolic import MultiplicativeMatrixPencil, SymmetricMatrixPencil
from finitefree.utils.conversion import sympy_to_fmpq


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("sign", [-1, 1])
def test_numpy_scalar_conversion_preserves_the_supplied_dyadic_value(
    dtype: Any, sign: int
) -> None:
    exponent = min(np.finfo(dtype).nmant, 60)
    value = dtype(sign) * (dtype(1) + np.ldexp(dtype(1), -exponent))
    expected = flint.fmpq(sign * (2**exponent + 1), 2**exponent)
    assert sympy_to_fmpq(value) == expected


def extended_value() -> tuple[Any, sp.Rational]:
    if np.finfo(np.longdouble).nmant <= 52:
        pytest.skip("longdouble has no extra precision on this platform")
    value = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
    return value, sp.Rational(2**60 + 1, 2**60)


@pytest.mark.parametrize("construction", ["coefficients", "roots"])
def test_polynomial_exact_inputs_preserve_extended_precision(construction: str) -> None:
    value, expected = extended_value()
    p = (
        RealRootedPolynomial([1, -value])
        if construction == "coefficients"
        else RealRootedPolynomial.from_roots([value])
    )
    assert list(p.coeffs) == [1, -expected]
    assert p.evaluate(expected) == 0
    assert list(p.normalized_coeffs()) == [1, expected]
    # Final float64 output still rounds; that must not alter the exact input.
    np.testing.assert_array_equal(p.evaluate_roots_float64(), [1.0])
    assert p.coeffs[1] == -expected


@pytest.mark.parametrize("operation", ["shift", "dilation"])
def test_affine_parameters_use_the_original_extended_precision(operation: str) -> None:
    value, expected = extended_value()
    p = RealRootedPolynomial.from_roots([1, 2])
    actual = p.shift(value) if operation == "shift" else p.dilation(value)
    x = sp.Symbol("x")
    roots = [r + expected if operation == "shift" else r * expected for r in [1, 2]]
    reference = sp.Poly(sp.prod(x - r for r in roots), x).all_coeffs()
    assert list(actual.coeffs) == reference


@pytest.mark.parametrize(
    "pencil_type", [SymmetricMatrixPencil, MultiplicativeMatrixPencil]
)
def test_exact_matrix_entries_and_parameters_keep_extended_precision(
    pencil_type: Any,
) -> None:
    value, expected = extended_value()
    entry = pencil_type([np.array([[value]], dtype=np.longdouble)])
    parameter = pencil_type([[[1]]])
    for polynomial in (
        entry.characteristic_polynomial([1]),
        parameter.characteristic_polynomial([value]),
    ):
        assert [sp.Rational(int(c.p), int(c.q)) for c in polynomial] == [-expected, 1]
    assert entry.evaluate([1])[0, 0] == 1.0


def test_exact_symmetry_check_rejects_extended_precision_asymmetry() -> None:
    value, _ = extended_value()
    source = np.array([[1, value], [1, 1]], dtype=np.longdouble)
    with pytest.raises(ValueError, match="symmetric"):
        SymmetricMatrixPencil([source])
