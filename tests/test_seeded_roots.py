"""Certified roots from a starting point, sampled seeds and prepared convolutions.

The reference throughout is independent Arb isolation of the exact rational
polynomial (`complex_roots` at 192 bits), never a second run of the new path.
"""

import sys
import warnings
from typing import Any

import flint
import numpy as np
import pytest

from finitefree import (
    PrecisionContext,
    RealRootedPolynomial,
    unitary_hermite_polynomial,
)
from finitefree.convolutions import (
    PreparedConvolution,
    multiplicative,
    sample_convolution_spectrum,
    symmetric_additive,
)
from finitefree.utils.refine import certified_real_roots_from
from tests._cases import case


def reference_roots(p: RealRootedPolynomial) -> Any:
    with PrecisionContext(degree=p.degree, prec=192):
        pairs: Any = p._fmpq_poly.complex_roots()
        assert all(root.imag.is_zero() for root, _ in pairs)
        return np.sort([float(root.real) for root, mult in pairs for _ in range(mult)])


def close(actual: Any, expected: Any, rtol: float = 4e-15) -> None:
    scale = np.maximum(np.abs(expected), 1e-300)
    assert np.max(np.abs(np.asarray(actual) - expected) / scale) <= rtol


def derivative_of_lognormal(d: int, seed: int) -> tuple[RealRootedPolynomial, Any]:
    roots = np.sort(np.random.default_rng(seed).lognormal(0.0, 0.8, d + 1))
    midpoints = 0.5 * (roots[1:] + roots[:-1])
    # Rolle: the derivative's roots interlace `roots`; the midpoints are a start.
    return RealRootedPolynomial.from_roots(roots).derivative(), midpoints


def test_initial_guess_contract_matches_independent_isolation() -> None:
    for d, seed in ((1, 0), (2, 1), (7, 2), (40, 3), (120, 4)):
        with case(d=d, seed=seed):
            p, start = derivative_of_lognormal(d, seed)
            fresh, _ = derivative_of_lognormal(d, seed)
            precision = flint.ctx.prec
            got = p.evaluate_roots_float64(initial=start)
            assert flint.ctx.prec == precision
            close(got, reference_roots(fresh))
            assert not got.flags.writeable
            assert p._is_verified
            # A satisfied cache is returned without consulting a later guess.
            assert p.evaluate_roots_float64(initial=np.zeros(d)) is got


def test_initial_guess_is_ill_conditioned_case_precision_adaptive() -> None:
    # Roots 1..48: monomial evaluation cancels far more than 2.5*d bits resolve.
    d = 48
    p = RealRootedPolynomial.from_roots(np.arange(1, d + 2)).derivative()
    start = np.arange(1, d + 1) + 0.5
    close(p.evaluate_roots_float64(initial=start), reference_roots(p))


def test_initial_guess_failures_fall_back_without_changing_results() -> None:
    cases: list[tuple[str, RealRootedPolynomial, Any]] = [
        ("repeated root", RealRootedPolynomial([1, -2, 1]), [0.9, 1.1]),
        ("far start", derivative_of_lognormal(12, 5)[0], np.linspace(1e6, 2e6, 12)),
        ("tied start", derivative_of_lognormal(9, 6)[0], np.ones(9)),
    ]
    for name, p, start in cases:
        with case(name=name):
            close(p.evaluate_roots_float64(initial=start), reference_roots(p), 1e-12)

    not_real = RealRootedPolynomial([1, 0, 1])
    assert certified_real_roots_from(not_real._fmpq_poly, [-0.5, 0.5]) is None
    with pytest.raises(ValueError):
        not_real.evaluate_roots_float64(initial=[-0.5, 0.5])
    with pytest.raises(ValueError, match="one starting value per root"):
        derivative_of_lognormal(5, 0)[0].evaluate_roots_float64(initial=[1.0, 2.0])
    unitary = unitary_hermite_polynomial(3, 1)
    with pytest.raises(ValueError, match="not supported for unitary"):
        unitary.evaluate_roots_float64(initial=[0.0, 0.1, 0.2])
    assert len(unitary.evaluate_roots_float64()) == 3


def test_certificate_rejects_wrong_candidates() -> None:
    # A start that converges to a strict subset of the roots must not certify.
    p = RealRootedPolynomial.from_roots([1, 2, 3, 4])
    assert certified_real_roots_from(p._fmpq_poly, [1.0, 2.0, 3.0]) is None
    assert certified_real_roots_from(p._fmpq_poly, [np.nan, 2.0, 3.0, 4.0]) is None
    got = certified_real_roots_from(p._fmpq_poly, [0.7, 2.2, 2.9, 4.4])
    assert got is not None
    close(got, np.array([1.0, 2.0, 3.0, 4.0]))


def convolution_inputs(d: int, seed: int) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    positive = np.sort(rng.lognormal(0.0, 1.0, d))
    noise = np.sort(rng.lognormal(0.0, 0.5, d))
    signed = np.sort(rng.normal(0.0, 1.0, d))
    return positive, noise, signed


def test_sampled_spectrum_seeds_certified_convolution_roots() -> None:
    for d, seed in ((6, 0), (30, 1), (80, 2)):
        positive, noise, signed = convolution_inputs(d, seed)
        q = RealRootedPolynomial.from_roots(noise)
        for kind, operation, roots in (
            ("multiplicative", multiplicative, positive),
            ("additive", symmetric_additive, signed),
        ):
            with case(d=d, kind=kind):
                p = RealRootedPolynomial.from_roots(roots)
                rng = np.random.default_rng(seed + 100)
                guess = sample_convolution_spectrum(p, q, d, kind=kind, rng=rng)
                assert guess.shape == (d,) and np.all(np.diff(guess) >= 0)
                expected = reference_roots(operation(p, q, d))
                # The sample is an estimate of the same spectrum, not the roots.
                assert np.max(np.abs(guess - expected)) < 0.5 * np.ptp(expected)
                got = operation(p, q, d).evaluate_roots_float64(initial=guess)
                close(got, expected)


def test_sampled_spectrum_mean_matches_expected_polynomial_low_degree() -> None:
    # d = 2: the mean of the sorted eigenvalues is not the roots, but the mean
    # trace and determinant are the convolution's e_1 and e_2 exactly.
    p = RealRootedPolynomial.from_roots([1, 3])
    q = RealRootedPolynomial.from_roots([2, 5])
    rng = np.random.default_rng(7)
    draws = np.array(
        [
            sample_convolution_spectrum(p, q, 2, kind="additive", draws=1, rng=rng)
            for _ in range(4000)
        ]
    )
    expected = symmetric_additive(p, q, 2)
    e1 = -float(expected.coeffs[1])
    e2 = float(expected.coeffs[2])
    assert abs(draws.sum(axis=1).mean() - e1) < 1e-9  # trace is deterministic
    assert abs(draws.prod(axis=1).mean() - e2) < 0.08


def test_sampled_spectrum_validation() -> None:
    p = RealRootedPolynomial.from_roots([-1, 2])
    q = RealRootedPolynomial.from_roots([-3, 1])
    with pytest.raises(ValueError, match="non-negative roots"):
        sample_convolution_spectrum(p, q, 2)
    with pytest.raises(ValueError, match="kind"):
        sample_convolution_spectrum(p, q, 2, kind="free")
    with pytest.raises(ValueError, match="draws"):
        sample_convolution_spectrum(p, q, 2, kind="additive", draws=0)
    with pytest.raises(ValueError, match="dimension"):
        sample_convolution_spectrum(p, q, 1, kind="additive")
    with pytest.raises(TypeError):
        sample_convolution_spectrum(p, q, 2, kind="additive", rng=0)  # type: ignore[arg-type]


def test_prepared_convolution_routes_match_exact_convolution() -> None:
    for d, seed in ((5, 0), (30, 1), (90, 2)):
        positive, noise, signed = convolution_inputs(d, seed)
        q = RealRootedPolynomial.from_roots(noise)
        for kind, operation, roots in (
            ("multiplicative", multiplicative, positive),
            ("additive", symmetric_additive, signed),
        ):
            with case(d=d, kind=kind):
                prepared = PreparedConvolution(
                    q, d, kind=kind, rng=np.random.default_rng(seed)
                )
                p = RealRootedPolynomial.from_roots(roots)
                exact = operation(p, q, d)
                assert prepared.convolve(p)._fmpq_poly == exact._fmpq_poly
                expected = reference_roots(exact)

                from_array = prepared.roots(roots)  # no previous roots: seeded
                close(from_array, expected)
                assert not from_array.flags.writeable
                from_polynomial = prepared.roots(p)  # warm start from last call
                close(from_polynomial, expected)
                moved = roots * 1.01 if kind == "multiplicative" else roots + 0.01
                close(
                    prepared.roots(moved),
                    reference_roots(
                        operation(RealRootedPolynomial.from_roots(moved), q, d)
                    ),
                )
                assert prepared.counts == {
                    "initial": 0,
                    "warm": 2,
                    "seeded": 1,
                    "isolated": 0,
                }
                close(prepared.roots(roots, initial=expected), expected)
                assert prepared.counts["initial"] == 1
                assert prepared.last_roots is not None


def test_prepared_convolution_custom_sampler_and_exact_fallback() -> None:
    d = 12
    positive, noise, _ = convolution_inputs(d, 3)
    q = RealRootedPolynomial.from_roots(noise)
    expected = reference_roots(
        multiplicative(RealRootedPolynomial.from_roots(positive), q, d)
    )
    calls: list[int] = []

    def bad_sampler(roots: Any, rng: Any) -> Any:
        calls.append(len(roots))
        return np.linspace(1e8, 2e8, len(roots))  # useless start: must fall back

    prepared = PreparedConvolution(q, d, seed_sampler=bad_sampler)
    close(prepared.roots(positive), expected, 1e-12)
    assert calls == [d] and prepared.counts["isolated"] == 1

    def wrong_shape(roots: Any, rng: Any) -> Any:
        return np.ones(3)

    prepared = PreparedConvolution(q, d, seed_sampler=wrong_shape)
    close(prepared.roots(positive), expected, 1e-12)
    assert prepared.counts["isolated"] == 1

    # Lower-degree q: the ambient-d result has repeated zero roots, which the
    # sign-change certificate cannot cover, so the exact path answers.
    low_q = RealRootedPolynomial.from_roots(noise[:8])
    low_p = RealRootedPolynomial.from_roots(positive[:10])
    prepared = PreparedConvolution(low_q, d, rng=np.random.default_rng(0))
    low_expected = reference_roots(multiplicative(low_p, low_q, d))
    assert np.count_nonzero(low_expected == 0) == 4
    got = prepared.roots(low_p)
    assert np.allclose(got, low_expected, rtol=1e-12, atol=1e-300)
    assert prepared.counts["isolated"] == 1


def test_prepared_convolution_validation() -> None:
    q = RealRootedPolynomial.from_roots([1, 2, 3])
    with pytest.raises(ValueError, match="kind"):
        PreparedConvolution(q, 3, kind="free")
    with pytest.raises(ValueError, match="dimension"):
        PreparedConvolution(q, 2)
    with pytest.raises(ValueError, match="positive integer"):
        PreparedConvolution(q, 0)
    with pytest.raises(ValueError, match="draws"):
        PreparedConvolution(q, 3, draws=True)
    prepared = PreparedConvolution(q, 3)
    with pytest.raises(ValueError, match="exactly d finite"):
        prepared.roots([1.0, 2.0])
    with pytest.raises(ValueError, match="exactly d finite"):
        prepared.roots([1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match="one starting value"):
        prepared.roots([1.0, 2.0, 3.0], initial=[1.0])
    with pytest.raises(ValueError, match="dimension"):
        prepared.roots(RealRootedPolynomial.from_roots([1, 2, 3, 4]))


def test_issue_67_parallel_seeker_results_are_certified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reproducer of issue #67: derivatives of polynomials with lognormal roots.
    # 0.4.1 returned errors up to 4.8e+02 here without any signal. Rolle's
    # theorem gives an independent check beside the Arb reference.
    monkeypatch.setitem(sys.modules, "cupy", None)
    for d in (20, 40, 60, 100):
        with case(d=d):
            roots = np.sort(np.random.default_rng(0).lognormal(0.0, 0.8, d + 1))
            p = RealRootedPolynomial.from_roots(roots).derivative()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                got = p.evaluate_roots_float64(parallel=True, exact=False)
            assert np.all((roots[:-1] < got) & (got < roots[1:]))
            close(got, reference_roots(p), 1e-13)


def test_issue_67_converged_but_wrong_seeker_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import finitefree.core as core

    monkeypatch.setitem(sys.modules, "cupy", None)
    expected = np.arange(1.0, 9.0)
    calls: list[int] = []

    def plausible_but_wrong(xp: Any, scaled: Any) -> Any:
        calls.append(len(scaled) - 1)
        return np.linspace(40.0, 50.0, len(scaled) - 1)  # finite, real, sorted

    monkeypatch.setattr(core, "_aberth_ehrlich_real_roots", plausible_but_wrong)
    poly = flint.fmpq_poly([1])
    for root in expected:
        poly *= flint.fmpq_poly([-int(root), 1])
    p = RealRootedPolynomial(poly, assume_real_rooted=True)
    got = p.evaluate_roots_float64(parallel=True, exact=False)
    assert calls == [8]
    np.testing.assert_allclose(got, expected, rtol=1e-10)
