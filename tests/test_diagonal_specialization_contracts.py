import itertools
import math
from fractions import Fraction
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.utils.modular import prime_generator


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_modular_agreement_cannot_hide_specialization_coefficients(
    parallel: bool, sign: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("finitefree.utils.parallel.os.cpu_count", lambda: 4)
    coefficient = sign * math.prod(itertools.islice(prime_generator(1000000007), 8))
    pencil = SymmetricMatrixPencil([np.eye(4), np.diag([coefficient, 1, 2, 3])])
    actual = pencil.diagonal_specialization([1, 0], [0, -1], parallel=parallel)
    z = sp.Symbol("z")
    expected = sp.Poly((z - coefficient) * (z - 1) * (z - 2) * (z - 3), z)
    assert list(actual.coeffs) == expected.all_coeffs()


@pytest.mark.parametrize("dtype", [float, np.float32, np.float64, np.longdouble])
def test_floating_specialization_parameters_preserve_binary_values(dtype: Any) -> None:
    weight, bias = dtype(0.1), dtype(0.3)
    expected = sp.Rational(*bias.as_integer_ratio()) / sp.Rational(
        *weight.as_integer_ratio()
    )
    pencil = SymmetricMatrixPencil([[[1]]])
    actual = pencil.diagonal_specialization([weight], [bias])
    assert list(actual.coeffs) == [1, expected]


@pytest.mark.parametrize("third", [Fraction(1, 3), sp.Rational(1, 3), flint.fmpq(1, 3)])
def test_rational_specialization_matches_an_independent_determinant(third: Any) -> None:
    matrices = [sp.Matrix([[2, 1], [1, 3]]), sp.Matrix([[1, 0], [0, -2]])]
    pencil = SymmetricMatrixPencil([matrix.tolist() for matrix in matrices])
    actual = pencil.diagonal_specialization([third, -2], [1, third])
    z = sp.Symbol("z")
    expected = sp.Poly(
        ((z / 3 + 1) * matrices[0] + (-2 * z + sp.Rational(1, 3)) * matrices[1]).det(),
        z,
    ).monic()
    assert list(actual.coeffs) == expected.all_coeffs()


def test_specialization_preserves_extended_precision_offset() -> None:
    if np.finfo(np.longdouble).nmant <= 52:
        pytest.skip("longdouble has no extra precision on this platform")
    value = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
    pencil = SymmetricMatrixPencil([[[1]]])
    actual = pencil.diagonal_specialization([1], [value])
    assert list(actual.coeffs) == [1, sp.Rational(2**60 + 1, 2**60)]


@pytest.mark.parametrize(
    "invalid",
    [
        np.nan,
        np.inf,
        1j,
        complex(0.5),
        np.complex64(0.5),
        np.complex128(0.5),
        sp.sqrt(2),
    ],
)
@pytest.mark.parametrize("parameter", ["weight", "bias"])
def test_specialization_rejects_nonrational_parameters(
    invalid: Any, parameter: str
) -> None:
    pencil = SymmetricMatrixPencil([[[1]]])
    with pytest.raises(ValueError, match="finite real rational"):
        pencil.diagonal_specialization(
            [invalid if parameter == "weight" else 1],
            [invalid if parameter == "bias" else 1],
        )
