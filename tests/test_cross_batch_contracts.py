from typing import Any

import flint
import numpy as np

from finitefree import (
    FiniteRTransform,
    RealRootedPolynomial,
    gue_expected_poly,
)
from tests._cases import case


def test_nonmonic_snapshots_projections_and_shifted_numerical_cumulants_agree() -> None:
    scale: Any
    dimension: int
    for dimension in [8, 4]:
        for scale in [flint.fmpq(-3), flint.fmpq(1, 10**400), flint.fmpq(10**400)]:
            with case(scale=scale, dimension=dimension):
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
                np.testing.assert_allclose(
                    actual[1:], [dimension / 8, 0, 0], rtol=0, atol=1e-28
                )
