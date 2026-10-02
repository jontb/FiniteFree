from typing import Any

import flint
import numpy as np
import pytest

from finitefree import PrecisionContext, RealRootedPolynomial


@pytest.mark.parametrize("degree", [1, 2, 16, 32])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("exact", [False, True])
def test_certified_out_of_range_roots_raise_without_poisoning_cache(
    degree: int, sign: int, exact: bool
) -> None:
    p = RealRootedPolynomial(flint.fmpq_poly([-sign * 10**400, 1]) ** degree)
    with PrecisionContext(degree=degree, prec=192):
        assert p.verify_real_rootedness()
        assert p._roots_cached is None
        for _ in range(2):
            with pytest.raises(RuntimeError, match="finite float64 range"):
                p.evaluate_roots_float64(exact=exact)
            assert p._is_verified
            assert p._roots_cached is None


@pytest.mark.parametrize("degree", [1, 2, 32])
@pytest.mark.parametrize("exact", [False, True])
def test_exact_from_roots_construction_does_not_require_float_representability(
    degree: int, exact: bool
) -> None:
    root = 10**400
    p = RealRootedPolynomial.from_roots([root] * degree)
    assert p.verify_real_rootedness()
    assert p._fmpq_poly == flint.fmpq_poly([-root, 1]) ** degree
    assert p._roots_cached is None
    with pytest.raises(RuntimeError, match="finite float64 range"):
        p.evaluate_roots_float64(exact=exact)
    assert p._roots_cached is None


@pytest.mark.parametrize("degree", [2, 32])
def test_parallel_out_of_range_roots_use_the_same_error_contract(degree: int) -> None:
    p = RealRootedPolynomial(flint.fmpq_poly([-(10**400), 1]) ** degree)
    with pytest.raises(RuntimeError, match="finite float64 range"):
        p.evaluate_roots_float64(exact=False, parallel=True)
    assert p._roots_cached is None


@pytest.mark.parametrize("degree", [2, 32])
@pytest.mark.parametrize("exact", [False, True])
def test_representable_large_roots_remain_finite_and_readonly(
    degree: int, exact: bool
) -> None:
    root = flint.fmpq(10**300)
    p = RealRootedPolynomial(flint.fmpq_poly([-root, 1]) ** degree)
    with PrecisionContext(degree=degree, prec=192):
        roots = p.evaluate_roots_float64(exact=exact)
    np.testing.assert_allclose(roots, np.full(degree, 1e300), rtol=1e-14)
    assert np.all(np.isfinite(roots))
    assert not roots.flags.writeable


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_backend_result_is_rejected_and_a_later_result_can_succeed(
    bad: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = RealRootedPolynomial([1, -3, 2])
    assert p.verify_real_rootedness()
    monkeypatch.setattr(
        p,
        "_evaluate_roots_float64_uncached",
        lambda **kwargs: np.array([1.0, bad]),
    )
    with pytest.raises(RuntimeError, match="finite float64 range"):
        p.evaluate_roots_float64()
    assert p._roots_cached is None

    def finite_result(**kwargs: Any) -> Any:
        return np.array([1.0, 2.0])

    monkeypatch.setattr(p, "_evaluate_roots_float64_uncached", finite_result)
    roots = p.evaluate_roots_float64()
    np.testing.assert_array_equal(roots, [1.0, 2.0])
    assert not roots.flags.writeable


@pytest.mark.parametrize("exact", [False, True])
def test_constant_polynomial_returns_an_empty_finite_root_array(exact: bool) -> None:
    roots = RealRootedPolynomial([7]).evaluate_roots_float64(exact=exact)
    assert roots.dtype == np.float64
    assert len(roots) == 0
    assert not roots.flags.writeable
