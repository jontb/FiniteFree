"""Certified real roots from a starting point (Arb Aberth-Ehrlich refinement)."""

from typing import Any, Optional

import flint
import numpy as np
from numpy.typing import NDArray


def _arb_float(value: Any) -> float:
    return float(value.mid()) if hasattr(value, "mid") else float(value)


def _sign_change_certificate(poly: Any, roots: NDArray[np.float64]) -> Optional[bool]:
    """Check d strict sign changes of an Arb polynomial over d + 1 bracket points.

    The brackets are the midpoints between consecutive candidates plus one point
    beyond each end. A degree-d real polynomial with d sign changes has exactly
    one simple real root in each bracket. Return True on success, False when
    the candidates are not strictly increasing or a sign pattern fails, and
    None when a value's ball contains zero (working precision too low).
    """
    d = len(roots)
    if d == 1:
        span = max(abs(float(roots[0])), 1.0)
        points = np.array([roots[0] - span, roots[0] + span])
    else:
        if np.any(np.diff(roots) <= 0):
            return False
        points = np.concatenate(
            (
                [roots[0] - (roots[1] - roots[0])],
                0.5 * (roots[1:] + roots[:-1]),
                [roots[-1] + (roots[-1] - roots[-2])],
            )
        )
        if np.any(np.diff(points) <= 0) or not np.all(np.isfinite(points)):
            return False
    previous = None
    for point in points:
        value = poly(flint.arb(float(point)))
        if value.contains(0) or not value.is_finite():
            return None
        positive = bool(value > 0)
        if previous is not None and positive == previous:
            return False
        previous = positive
    return True


def _refine_at_precision(
    poly: Any,
    initial: NDArray[np.float64],
    max_sweeps: int,
    tol: float,
) -> tuple[Optional[NDArray[np.float64]], bool]:
    """One Aberth-Ehrlich run at the current precision: (roots or None, precision_failure)."""
    derivative = poly.derivative()
    points = [flint.arb(float(v)) for v in initial]
    for _ in range(max_sweeps):
        newton = [(poly(x) / derivative(x)).mid() for x in points]
        current = np.array([_arb_float(x) for x in points])
        corrections = np.array([_arb_float(w) for w in newton])
        if not np.all(np.isfinite(corrections)):
            return None, True
        gaps = current[:, None] - current[None, :]
        np.fill_diagonal(gaps, np.inf)
        if np.any(gaps == 0):
            return None, False
        repulsion = (1.0 / gaps).sum(axis=1)
        updated = []
        largest = 0.0
        for x, w, rep in zip(points, newton, repulsion):
            step = w / (1 - w * flint.arb(float(rep)))
            new = (x - step).mid()
            if not new.is_finite():
                return None, True
            updated.append(new)
            scale = max(abs(_arb_float(x)), 1e-300)
            largest = max(largest, abs(_arb_float(step)) / scale)
        points = updated
        if largest < tol:
            break
    else:
        # Aberth stalled above tol: polish each candidate with plain Newton steps,
        # which converge quadratically once the candidates are separated.
        for _ in range(8):
            largest = 0.0
            polished = []
            for x in points:
                step = (poly(x) / derivative(x)).mid()
                if not step.is_finite():
                    return None, True
                polished.append((x - step).mid())
                scale = max(abs(_arb_float(x)), 1e-300)
                largest = max(largest, abs(_arb_float(step)) / scale)
            points = polished
            if largest < tol:
                break
        else:
            return None, True
    roots = np.sort(np.array([_arb_float(x) for x in points], dtype=np.float64))
    if not np.all(np.isfinite(roots)):
        return None, True
    certificate = _sign_change_certificate(poly, roots)
    if certificate is None:
        return None, True
    return (roots, False) if certificate else (None, False)


def _extra_bits_needed(poly: Any, points: NDArray[np.float64], tol: float) -> int:
    """Bits to add so a Newton correction's ball radius stays below tol*|x|.

    Monomial evaluation near clustered or widely scaled roots cancels many
    digits; the radius of p(x)/p'(x) at the starting points measures that loss.
    """
    derivative = poly.derivative()
    worst = 0.0
    for value in points:
        x = flint.arb(float(value))
        step = poly(x) / derivative(x)
        if not step.is_finite():
            return -1
        radius = _arb_float(step.rad())
        target = tol * max(abs(float(value)), 1e-300) / 16.0
        if radius > target:
            worst = max(worst, float(np.log2(radius / target)))
    return int(np.ceil(worst))


def certified_real_roots_from(
    poly: Any,
    initial: Any,
    *,
    prec: Optional[int] = None,
    max_sweeps: int = 60,
    tol: float = 1e-14,
) -> Optional[NDArray[np.float64]]:
    """Refine `initial` to the real roots of a real polynomial and certify them.

    `poly` is a `flint.fmpq_poly`, a `flint.arb_poly` enclosing the intended
    real polynomial, or a zero-argument callable returning such an enclosure at
    the precision current when it is called (so a retry can tighten it).
    `initial` holds d finite starting values. Aberth-Ehrlich
    iteration (simultaneous Newton with repulsion) starts at `prec` bits, default
    max(64, int(2.5*d)), raised when the Newton corrections at the starting
    points are not resolved to `tol`, and runs once more at twice that
    precision if a ball contained zero or the iteration stalled. The result is accepted only when the polynomial changes
    sign strictly across the d + 1 points bracketing the sorted candidates,
    which proves one simple real root per bracket for every polynomial inside
    the enclosure. Returned values are float64 candidates inside those
    brackets, not error bounds. Return None when the iteration or certificate
    fails (including repeated roots). The process-wide FLINT precision is
    restored on exit.
    """
    start = np.sort(np.asarray(initial, dtype=np.float64).ravel())

    def enclosure() -> Any:
        if isinstance(poly, flint.fmpq_poly):
            return flint.arb_poly(poly)
        if isinstance(poly, flint.arb_poly):
            return poly
        return poly()

    d = len(start)
    if d < 1 or not np.all(np.isfinite(start)):
        return None
    if d > 1 and np.any(np.diff(start) <= 0):
        # Aberth needs distinct starting points; separate exact ties slightly.
        spread = max(float(np.max(np.abs(start))), 1.0)
        start = start + spread * 1e-9 * np.arange(d)
    working = prec if prec is not None else max(64, int(2.5 * d))
    saved = flint.ctx.prec
    try:
        flint.ctx.prec = working
        first = enclosure()
        if int(first.degree()) != d:
            return None
        extra = _extra_bits_needed(first, start, tol)
        if extra < 0:
            extra = working
        working += extra
        for attempt_prec in (working, 2 * working):
            flint.ctx.prec = attempt_prec
            arb_polynomial = enclosure()
            roots, precision_failure = _refine_at_precision(
                arb_polynomial, start, max_sweeps, tol
            )
            if roots is not None:
                return roots
            if not precision_failure:
                return None
        return None
    finally:
        flint.ctx.prec = saved
