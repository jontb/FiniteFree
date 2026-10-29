"""Exact utility contracts checked against references outside the library.

Determinants use SymPy's integer ``Matrix.det()``; CRT results are checked by
their congruences and symmetric range; float conversion is checked against the
exact half-ulp rounding interval; real-root counts use SymPy's ``count_roots``.
"""

import math
from typing import Iterator

import flint
import pytest

from finitefree import PrecisionContext
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
