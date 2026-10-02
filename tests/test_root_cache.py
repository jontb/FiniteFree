from typing import Any, cast

import flint
import numpy as np

from finitefree import PrecisionContext, RealRootedPolynomial


class CountingPolynomial:
    def __init__(self, polynomial: Any) -> None:
        self.polynomial = polynomial
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.polynomial, name)

    def __getitem__(self, index: int) -> Any:
        return self.polynomial[index]

    def complex_roots(self) -> Any:
        self.calls += 1
        return self.polynomial.complex_roots()


def test_arb_validation_reuses_roots_and_multiplicities() -> None:
    expected = [1.0] * 16 + [3.0] * 20
    original = RealRootedPolynomial.from_roots(expected)
    p = RealRootedPolynomial(original._fmpq_poly)
    probe = CountingPolynomial(p._fmpq_poly)
    cast(Any, p)._fmpq_poly = probe
    actual = p.evaluate_roots_float64(exact=False)
    assert probe.calls == 1
    np.testing.assert_array_equal(actual, expected)
    p.evaluate_roots_float64(exact=True)
    assert probe.calls == 1


def test_higher_precision_request_refreshes_arb_cache() -> None:
    # A known real-rooted degree-32 polynomial with no family metadata.
    p = RealRootedPolynomial(RealRootedPolynomial.from_roots(range(1, 33))._fmpq_poly)
    probe = CountingPolynomial(p._fmpq_poly)
    cast(Any, p)._fmpq_poly = probe
    with PrecisionContext(degree=p.degree, prec=96):
        p.verify_real_rootedness()
        assert probe.calls == 1
    with PrecisionContext(degree=p.degree, prec=192):
        roots = p.evaluate_roots_float64(exact=True)
        assert probe.calls == 2
        np.testing.assert_array_equal(roots, np.arange(1, 33))
        p.evaluate_roots_float64(exact=True)
        assert probe.calls == 2


def test_rejected_complex_roots_do_not_fill_cache() -> None:
    polynomial = flint.fmpq_poly([1, 0, 1]) * flint.fmpq_poly([0, 1]) ** 32
    p = RealRootedPolynomial(polynomial)
    try:
        p.verify_real_rootedness()
    except ValueError:
        pass
    else:
        raise AssertionError("Complex roots were accepted")
    assert p._roots_cached is None
    assert not p._roots_cached_exact
