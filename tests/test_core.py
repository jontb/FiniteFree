import flint
import numpy as np
import pytest

from finitefree.core import PrecisionContext, RealRootedPolynomial


def test_precision_context() -> None:
    initial_prec = flint.ctx.prec
    with PrecisionContext(degree=100):
        assert flint.ctx.prec >= 53
        assert flint.ctx.prec > initial_prec
    assert flint.ctx.prec == initial_prec


def test_real_rooted_polynomial_monic() -> None:
    # 2x^2 - 8 -> x^2 - 4
    poly = RealRootedPolynomial([2, 0, -8])
    assert np.allclose(list(poly.coeffs), [1, 0, -4])
    assert poly.degree == 2


def test_verify_real_rootedness_success() -> None:
    # x^2 - 3x + 2 = (x-1)(x-2)
    poly = RealRootedPolynomial([1, -3, 2])
    assert poly.verify_real_rootedness()


def test_verify_real_rootedness_failure() -> None:
    # x^2 + 1
    poly = RealRootedPolynomial([1, 0, 1])
    with pytest.raises(ValueError, match="not real-rooted"):
        poly.verify_real_rootedness()


def test_normalized_coeffs() -> None:
    # p(x) = x^2 - 2x + 1 = (x-1)^2
    # c_0 = 1, c_1 = -2, c_2 = 1
    # c_k = (-1)^k \binom{2}{k} e_k
    # k=0: c_0 = 1 * 1 * e_0 => e_0 = 1
    # k=1: c_1 = -1 * 2 * e_1 => -2 = -2 e_1 => e_1 = 1
    # k=2: c_2 = 1 * 1 * e_2 => 1 = 1 e_2 => e_2 = 1
    poly = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)
    e_k = poly.normalized_coeffs()
    assert np.allclose(list(e_k), [1, 1, 1])


def test_from_normalized_coeffs() -> None:
    e_k = [1, 1, 1]
    poly = RealRootedPolynomial.from_normalized_coeffs(e_k)
    assert np.allclose(list(poly.coeffs), [1, -2, 1])
    assert poly._is_verified


def test_verify_root_interlacing() -> None:
    # p(x) = x^2 - 3x + 2, roots: 1, 2
    # p'(x) = 2x - 3, root: 1.5
    # 1 < 1.5 < 2
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    assert p.verify_root_interlacing(strict=True)

    # Double root: p(x) = x^2 - 2x + 1 = (x-1)^2
    # p'(x) = 2x - 2, root: 1
    p2 = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)
    # Should pass non-strict
    assert p2.verify_root_interlacing(strict=False)

    # Should fail strict
    with pytest.raises(ValueError, match="Strict root interlacing failed"):
        p2.verify_root_interlacing(strict=True)
