from typing import Any

import flint
import numpy as np

from finitefree import RealRootedPolynomial
from tests._cases import case


def test_generic_numerical_roots_are_invariant_under_nonmonic_scalar_multiples() -> (
    None
):
    roots: list[int]
    scale: Any
    parallel: bool
    for parallel in [False, True]:
        for scale in [
            flint.fmpq(2),
            flint.fmpq(-3),
            flint.fmpq(1, 10**400),
            flint.fmpq(10**400),
        ]:
            for roots in [[1, 2], [-2, 1, 3]]:
                with case(roots=roots, scale=scale, parallel=parallel):
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
                    np.testing.assert_allclose(
                        actual, reference, rtol=1e-12, atol=1e-12
                    )
