from fractions import Fraction
from typing import Any, cast

import flint
import numpy as np
import pytest

from finitefree import PrecisionContext, RealRootedPolynomial, legendre_polynomial
from tests._cases import case


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


def assert_owned_root_cache(p: RealRootedPolynomial, expected: Any) -> Any:
    for exact in (False, True):
        with case(exact=exact):
            roots = p.evaluate_roots_float64(exact=exact)
            assert not roots.flags.writeable
            np.testing.assert_allclose(roots, expected, rtol=2e-15, atol=2e-15)
            with pytest.raises(ValueError, match="read-only"):
                roots[0] = 100
            editable = roots.copy()
            editable[0] = 100
            assert roots is p.evaluate_roots_float64(exact=exact)
            np.testing.assert_allclose(roots, expected, rtol=2e-15, atol=2e-15)
    return roots


def test_root_cache_ownership_reuse_multiplicity_and_precision_lifecycle() -> None:
    for expected in ([1.0] * 16 + [3.0] * 20, list(range(1, 33)), [1] * 32):
        with case(expected=expected):
            # Strip from_roots metadata so validation must isolate roots via Arb.
            p = RealRootedPolynomial(
                RealRootedPolynomial.from_roots(expected)._fmpq_poly
            )
            probe = CountingPolynomial(p._fmpq_poly)
            cast(Any, p)._fmpq_poly = probe
            with PrecisionContext(degree=p.degree, prec=96):
                assert p.verify_real_rootedness()
                assert probe.calls == 1
                low = assert_owned_root_cache(p, expected)
                assert probe.calls == 1
            with PrecisionContext(degree=p.degree, prec=192):
                high = assert_owned_root_cache(p, expected)
                assert high is not low
                assert p._roots_cached_prec >= 192
                assert probe.calls == 2
            with PrecisionContext(degree=p.degree, prec=96):
                assert assert_owned_root_cache(p, expected) is high
                assert probe.calls == 2
            assert not low.flags.writeable
            np.testing.assert_array_equal(low, expected)

    # The same ownership contract also applies before Arb cache initialization.
    for constructor in ("coefficients", "roots", "recurrence"):
        with case(constructor=constructor):
            if constructor == "recurrence":
                p = legendre_polynomial(2)
                expected_roots = np.array([-1, 1]) / np.sqrt(3)
            else:
                p = (
                    RealRootedPolynomial.from_roots([-1, 1])
                    if constructor == "roots"
                    else RealRootedPolynomial([1, 0, -1])
                )
                expected_roots = np.array([-1, 1])
            assert_owned_root_cache(p, expected_roots)


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


def test_from_roots_cache_is_correctly_rounded_without_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [
        Fraction(1, 3),
        Fraction(-7, 3),
        Fraction(10**20 + 1, 7),
        Fraction(1, 10**30),
        Fraction(1, 3),
    ]
    p = RealRootedPolynomial.from_roots(roots)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("known exact roots should not be isolated again")

    monkeypatch.setattr(p, "_evaluate_roots_float64_uncached", forbidden)
    expected = sorted(float(root) for root in roots)  # Fraction rounds correctly
    for prec in [53, 4096]:
        with case(prec=prec):
            with PrecisionContext(degree=p.degree, prec=prec):
                actual = p.evaluate_roots_float64(exact=True)
            assert actual.tolist() == expected
            assert not actual.flags.writeable
