"""Exact utility contracts checked against references outside the library.

Determinants use SymPy's integer ``Matrix.det()``; CRT results are checked by
their congruences and symmetric range; float conversion is checked against the
exact half-ulp rounding interval; real-root counts use SymPy's ``count_roots``.
"""

import math
import random
from fractions import Fraction
from typing import Iterator

import flint
import pytest
import sympy as sp

from finitefree import PrecisionContext
from finitefree.utils import modular
from finitefree.utils.conversion import flint_to_float
from tests._cases import case

# Largest modulus whose residue products and sums stay below 2**63.
INT64_SAFE_MODULUS = math.isqrt(2**63 - 1)


@pytest.fixture
def restore_flint_precision() -> Iterator[None]:
    original = flint.ctx.prec
    try:
        yield
    finally:
        flint.ctx.prec = original


@pytest.mark.usefixtures("restore_flint_precision")
def test_precision_context_restores_the_precision_seen_on_entry() -> None:
    flint.ctx.prec = 53
    context = PrecisionContext(degree=10, prec=256)
    flint.ctx.prec = 128  # changed after construction, before entry
    with context:
        assert flint.ctx.prec == 256
    assert flint.ctx.prec == 128

    # Sequential reuse restores the value current at the second entry.
    flint.ctx.prec = 90
    with context:
        assert flint.ctx.prec == 256
    assert flint.ctx.prec == 90

    # Nested reuse of one object restores each entry's own precision.
    flint.ctx.prec = 53
    with context:
        flint.ctx.prec = 77
        with context:
            assert flint.ctx.prec == 256
        assert flint.ctx.prec == 77
    assert flint.ctx.prec == 53

    # Nested distinct contexts and exceptions.
    outer = PrecisionContext(degree=10, prec=300)
    inner = PrecisionContext(degree=10, prec=400)
    with outer:
        flint.ctx.prec = 64
        with pytest.raises(ZeroDivisionError):
            with inner:
                assert flint.ctx.prec == 400
                raise ZeroDivisionError
        assert flint.ctx.prec == 64
    assert flint.ctx.prec == 53

    # Documented default max(53, int(2.5 * degree)).
    for degree, expected in [(1, 53), (21, 53), (22, 55), (100, 250)]:
        with case(degree=degree):
            with PrecisionContext(degree=degree):
                assert flint.ctx.prec == expected
            assert flint.ctx.prec == 53


def _is_correctly_rounded(value: float, exact: Fraction) -> bool:
    """Check round-to-nearest-even against both float64 neighbours exactly."""
    if not math.isfinite(value):
        return False
    error = abs(Fraction(value) - exact)
    for neighbour in (
        math.nextafter(value, -math.inf),
        math.nextafter(value, math.inf),
    ):
        if not math.isfinite(neighbour):
            continue
        other = abs(Fraction(neighbour) - exact)
        if other < error:
            return False
        if other == error:
            # Ties go to the even significand (in units of the last place).
            significand = Fraction(value) / Fraction(math.ulp(value))
            if significand.denominator != 1 or significand.numerator % 2:
                return False
    return True


def test_flint_to_float_rounds_rationals_correctly() -> None:
    p, q = 916312224894977032587213, 48946970203956156237443
    assert flint_to_float(flint.fmpq(p, q)) == 18.720509585717235
    assert _is_correctly_rounded(18.720509585717235, Fraction(p, q))

    rng = random.Random(20261029)
    cases = [(p, q), (2**60 + 1, 3 * 2**58 + 7), (10**400 + 1, 3 * 10**399)]
    for _ in range(300):
        numerator = rng.getrandbits(rng.choice([60, 80, 120, 300]))
        denominator = rng.getrandbits(rng.choice([55, 76, 110, 290])) + 1
        cases.append((rng.choice([-1, 1]) * numerator, denominator))
    # Tiny subnormal results are still rounded once.
    cases.extend([(1, 3 * 2**1060), (-7, 5 * 2**1070)])
    for numerator, denominator in cases:
        exact = Fraction(numerator, denominator)
        for value in (
            flint.fmpq(numerator, denominator),
            sp.Rational(numerator, denominator),
        ):
            with case(numerator=numerator, denominator=denominator, kind=type(value)):
                result = flint_to_float(value)
                assert type(result) is float
                assert _is_correctly_rounded(result, exact)


def test_flint_to_float_keeps_documented_range_limits() -> None:
    assert flint_to_float(flint.fmpq(10**400, 3)) == math.inf
    assert flint_to_float(flint.fmpq(-(10**400), 3)) == -math.inf
    assert flint_to_float(sp.Rational(-(10**400), 7)) == -math.inf
    tiny = flint_to_float(flint.fmpq(1, 10**400))
    assert tiny == 0.0 and math.copysign(1.0, tiny) == 1.0
    negative_tiny = flint_to_float(flint.fmpq(-1, 10**400))
    assert negative_tiny == 0.0 and math.copysign(1.0, negative_tiny) == -1.0
    assert flint_to_float(flint.fmpz(-12)) == -12.0
    assert flint_to_float(flint.arb(0.25)) == 0.25


def _sympy_det_mod(matrix: list[list[int]], p: int) -> int:
    return int(sp.Matrix(matrix).det()) % p if matrix else 1 % p


def test_modular_det_reduces_signed_and_large_entries() -> None:
    rng = random.Random(7)
    matrices = [
        [[-2, 1], [1, 1]],
        [[-1]],
        [[3, -5], [2, 4]],
        [[0, -1], [-2, 3]],
        [[2**70, 1], [1, 1]],
        [[-(2**64) - 3, 5], [2**63, -(2**80)]],
        [[1, 2], [2, 4]],
    ]
    for size in (3, 4, 5):
        matrices.append(
            [
                [
                    rng.choice([-1, 1]) * rng.getrandbits(rng.choice([4, 40, 90]))
                    for _ in range(size)
                ]
                for _ in range(size)
            ]
        )
    moduli = [7, 1000000007, 3037000493, 3037000507, 2**61 - 1, 2**89 - 1]
    for p in moduli:
        for matrix in matrices:
            with case(p=p, matrix=matrix):
                expected = _sympy_det_mod(matrix, p)
                result = modular.modular_det([list(row) for row in matrix], p)
                assert result == expected
                assert 0 <= result < p
    assert modular.modular_det([[-2, 1], [1, 1]], 7) == 4
    assert modular.modular_det([[-1]], 7) == 6
    assert modular.modular_det([[2**70, 1], [1, 1]], 1000000007) == 270016252


def test_compiled_modular_kernels_reduce_and_validate_inputs() -> None:
    fast = pytest.importorskip("finitefree.utils.modular_fast")
    assert fast.modular_det([[-2, 1], [1, 1]], 7) == 4
    assert fast.modular_det([[-1]], 7) == 6
    assert fast.modular_det([[2**70, 1], [1, 1]], 1000000007) == 270016252
    for p in (1, 0, -7, INT64_SAFE_MODULUS + 1):
        with case(p=p):
            with pytest.raises(ValueError):
                fast.modular_det([[1]], p)
    import numpy as np

    primes = np.array([3037000493, 1000000007], dtype=np.int64)
    x_true = -123456789012345678
    residues = np.array([x_true, x_true], dtype=np.int64)  # unreduced, negative
    assert int(fast.crt(residues, primes)) == x_true
    residues = np.array([-(2**63), 2**63 - 1], dtype=np.int64)
    result = int(fast.crt(residues, primes))
    assert (result - -(2**63)) % 3037000493 == 0
    assert (result - (2**63 - 1)) % 1000000007 == 0
    with pytest.raises(ValueError):
        fast.crt(
            np.array([1, 2], dtype=np.int64),
            np.array([3037000507, 1000000007], dtype=np.int64),
        )


def test_modular_det_honors_the_fallback_flag() -> None:
    import os

    if os.environ.get("PYFFP_DISABLE_CYTHON") == "1":
        assert modular._modular_det_fast is None
    else:
        fast = pytest.importorskip("finitefree.utils.modular_fast")
        assert modular._modular_det_fast is fast.modular_det


def test_crt_reconstructs_symmetric_values_for_large_primes() -> None:
    prime_sets = [
        [3, 5, 7],
        [1000000007, 1000000009, 1000000021],
        [3037000493, 1000000007],
        [3037000507, 1000000007],
        [4294967311, 4294967357, 4294967371],
        [2**61 - 1, 2**89 - 1],
    ]
    rng = random.Random(11)
    for primes in prime_sets:
        modulus = math.prod(primes)
        values = [0, 1, -1, modulus // 2, -(modulus // 2)]
        values += [rng.randrange(-(modulus // 2), modulus // 2 + 1) for _ in range(20)]
        for x_true in values:
            for residues in (
                [x_true % p for p in primes],
                [x_true for _ in primes],  # unreduced and possibly negative
                [x_true + (2**80) * p for p in primes],
            ):
                with case(primes=primes, x_true=x_true, residues=residues):
                    result = modular.crt(residues, primes)
                    assert all((result - r) % p == 0 for r, p in zip(residues, primes))
                    assert -(modulus // 2) <= result <= modulus // 2
                    assert result == x_true
