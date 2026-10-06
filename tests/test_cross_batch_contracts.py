from typing import Any

import flint
import numpy as np
import pytest

from finitefree import (
    FiniteRTransform,
    PrecisionContext,
    RealRootedPolynomial,
    gue_expected_poly,
)


@pytest.mark.parametrize("exact", [True, False])
def test_arb_validation_cache_obeys_the_root_array_ownership_contract(
    exact: bool,
) -> None:
    source = flint.fmpq_poly([-1, 1]) ** 32
    p = RealRootedPolynomial(source)
    with PrecisionContext(degree=p.degree, prec=192):
        assert p.verify_real_rootedness()
        roots = p.evaluate_roots_float64(exact=exact)
        assert not roots.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            roots[0] = 100
        np.testing.assert_array_equal(
            p.evaluate_roots_float64(exact=exact), np.ones(32)
        )


def test_arb_precision_upgrade_retains_readonly_cache_results() -> None:
    p = RealRootedPolynomial(flint.fmpq_poly([-1, 1]) ** 32)
    with PrecisionContext(degree=p.degree, prec=96):
        p.verify_real_rootedness()
        low = p.evaluate_roots_float64()
        assert not low.flags.writeable
    with PrecisionContext(degree=p.degree, prec=192):
        high = p.evaluate_roots_float64()
        assert high is not low
        assert not high.flags.writeable
        assert p._roots_cached_prec >= 192
        np.testing.assert_array_equal(high, np.ones(32))


@pytest.mark.parametrize(
    "scale", [flint.fmpq(-3), flint.fmpq(1, 10**400), flint.fmpq(10**400)]
)
@pytest.mark.parametrize("dimension", [8, 4])
def test_nonmonic_snapshots_projections_and_shifted_numerical_cumulants_agree(
    scale: Any, dimension: int
) -> None:
    source = gue_expected_poly(8).shift(10**50)._fmpq_poly * scale
    p = RealRootedPolynomial(source, monic=False)
    before = list(p.coeffs)
    source[0] = 0
    assert list(p.coeffs) == before
    projected = p.projection(dimension)
    coefficients = projected.normalized_coeffs()
    assert coefficients[0] == 1
    assert not coefficients.flags.writeable
    actual = FiniteRTransform(projected, order=4, numerical=True, prec=128)
    assert actual[0] == 1e50
    # Hermite finite cumulants: mean, variance, then zeros. Projection multiplies
    # the variance by j/d; multiplying all coefficients changes none of them.
    np.testing.assert_allclose(actual[1:], [dimension / 8, 0, 0], rtol=0, atol=1e-28)
