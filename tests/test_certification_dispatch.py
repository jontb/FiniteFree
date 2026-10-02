from typing import Any

import flint
import numpy as np
import pytest

from finitefree import PrecisionContext, RealRootedPolynomial


@pytest.mark.parametrize("degree", [15, 16, 20, 30])
def test_medium_squarefree_factors_reuse_a_certified_arb_root_cache(
    degree: int,
) -> None:
    expected = np.arange(-degree // 2, -degree // 2 + degree, dtype=float)
    coefficients = RealRootedPolynomial.from_roots(expected)._fmpq_poly
    p = RealRootedPolynomial(coefficients)
    with PrecisionContext(degree=degree, prec=128):
        initial_precision = flint.ctx.prec
        assert p.verify_real_rootedness()
        assert flint.ctx.prec == initial_precision
        assert p._roots_cached_exact
        assert p._roots_cached_prec == initial_precision
        np.testing.assert_array_equal(p.evaluate_roots_float64(), expected)
        assert not p._roots_cached.flags.writeable


@pytest.mark.parametrize("degree", [15, 16, 20, 30])
def test_medium_degree_nearly_real_conjugate_pairs_are_rejected(degree: int) -> None:
    real = RealRootedPolynomial.from_roots(range(1, degree - 1))._fmpq_poly
    # The imaginary parts are 10^-40, much smaller than a float tolerance.
    coefficients = real * flint.fmpq_poly([flint.fmpq(1, 10**80), 0, 1])
    p = RealRootedPolynomial(coefficients)
    with pytest.raises(ValueError, match="not real-rooted"):
        p.verify_real_rootedness()
    assert not p._is_verified
    assert p._roots_cached is None


@pytest.mark.parametrize("complex_pair", [False, True])
def test_unavailable_medium_degree_arb_certificate_falls_back_to_exact_sturm(
    complex_pair: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_roots = range(1, 15) if complex_pair else range(1, 17)
    coefficients = RealRootedPolynomial.from_roots(real_roots)._fmpq_poly
    if complex_pair:
        coefficients *= flint.fmpq_poly([1, 0, 1])
    p = RealRootedPolynomial(coefficients)
    calls = []

    def unavailable() -> bool:
        calls.append(True)
        raise RuntimeError("Backend could not isolate roots")

    monkeypatch.setattr(p, "_verify_real_rootedness_arb", unavailable)
    if complex_pair:
        with pytest.raises(ValueError, match="not real-rooted"):
            p.verify_real_rootedness()
        assert not p._is_verified
    else:
        assert p.verify_real_rootedness()
    assert calls == [True]
    assert p._roots_cached is None


def test_repeated_small_factors_keep_exact_sturm_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = RealRootedPolynomial(flint.fmpq_poly([-3, 1]) ** 30)

    def forbidden() -> Any:
        raise AssertionError("A small squarefree factor did not need Arb")

    monkeypatch.setattr(p, "_verify_real_rootedness_arb", forbidden)
    assert p.verify_real_rootedness()
    assert p._roots_cached is None
    np.testing.assert_array_equal(p.evaluate_roots_float64(), np.full(30, 3.0))


def test_assumed_real_roots_still_bypass_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = RealRootedPolynomial(flint.fmpq_poly([-1, 1]) ** 30, assume_real_rooted=True)

    def forbidden() -> Any:
        raise AssertionError("The explicit assumption should bypass certification")

    monkeypatch.setattr(p, "_verify_real_rootedness_arb", forbidden)
    assert p.verify_real_rootedness()
