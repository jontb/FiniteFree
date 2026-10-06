from typing import Any

import flint
import numpy as np
import pytest

from finitefree import RealRootedPolynomial


@pytest.mark.parametrize("roots", [[1, 2], [-2, 1, 3]])
@pytest.mark.parametrize(
    "scale",
    [flint.fmpq(2), flint.fmpq(-3), flint.fmpq(1, 10**400), flint.fmpq(10**400)],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_generic_numerical_roots_are_invariant_under_nonmonic_scalar_multiples(
    roots: list[int], scale: Any, parallel: bool
) -> None:
    monic = RealRootedPolynomial.from_roots(roots)._fmpq_poly
    p = RealRootedPolynomial(monic * scale, monic=False)
    original = flint.fmpq_poly(p._fmpq_poly)
    actual = p.evaluate_roots_float64(exact=False, parallel=parallel)
    np.testing.assert_allclose(actual, roots, rtol=1e-12, atol=1e-12)
    assert p._fmpq_poly == original
    assert p._fmpq_poly[p.degree] == scale
    assert not actual.flags.writeable
    # Independently isolated roots use the original nonmonic coefficients.
    reference = p.evaluate_roots_float64(exact=True)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
