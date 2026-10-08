"""Generate offline dashboards for FiniteFree's existing visualization families."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid
from scipy.special import airy

from finitefree import FiniteRTransform, FiniteTTransform, RealRootedPolynomial
from finitefree.convolutions import multiplicative, symmetric_additive
from finitefree.ensembles import (
    gue_expected_poly,
    sample_goe,
    sample_gse,
    sample_gue,
    sample_wishart,
    wishart_expected_poly,
)
from finitefree.orthogonal import legendre_polynomial, unitary_hermite_polynomial

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "visuals" / "interactive"
DEGREES = [8, 16, 32, 64, 128, 256]
RATIOS = [sp.Rational(1, 4), sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)]
SEEDS = [1701, 2903, 4517]


def limit_measure(kind: str, gamma: float = 1.0) -> dict[str, Any]:
    """CDFs include atoms; density arrays describe only the continuous part."""
    if kind == "semicircle":
        x = np.linspace(-2, 2, 1025)
        density = np.sqrt(np.maximum(0, 4 - x * x)) / (2 * np.pi)
        cdf = (
            0.5 + (x * np.sqrt(np.maximum(0, 4 - x * x)) / 4 + np.arcsin(x / 2)) / np.pi
        )
        atom = 0.0
    elif kind == "arcsine":
        theta = np.linspace(0, np.pi, 1025)
        x = -np.cos(theta)
        cdf = theta / np.pi
        density = np.zeros_like(x)
        density[1:-1] = 1 / (np.pi * np.sin(theta[1:-1]))
        atom = 0.0
    elif kind == "mp":
        theta = np.linspace(0, np.pi, 4097)
        x = 1 + gamma - 2 * np.sqrt(gamma) * np.cos(theta)
        integrand = np.zeros_like(x)
        np.divide(2 * np.sin(theta) ** 2, np.pi * x, out=integrand, where=x > 0)
        if gamma == 1:
            integrand[0] = 2 / np.pi
        atom = max(0.0, 1 - 1 / gamma)
        cdf = atom + cumulative_trapezoid(integrand, theta, initial=0)
        # Trapezoidal quadrature in angle resolves the hard edge without x-grid
        # singularities. Normalize its tiny integration error to known mass.
        cdf = atom + (cdf - atom) * ((1 - atom) / (cdf[-1] - atom))
        density = np.zeros_like(x)
        np.divide(
            np.sqrt(np.maximum(0, (x[-1] - x) * (x - x[0]))),
            2 * np.pi * gamma * x,
            out=density,
            where=x > 0,
        )
    else:
        raise ValueError(kind)
    return {
        "x": x.tolist(),
        "density": density.tolist(),
        "cdf": cdf.tolist(),
        "atom": atom,
        "support": [float(x[0]), float(x[-1])],
    }


def wishart_roots(d: int, n: int) -> list[float]:
    """Use the supported Laguerre recurrence and covariance duality for n<d."""
    if n >= d:
        roots = wishart_expected_poly(d, n).evaluate_roots_float64(exact=False)
        return [float(x) for x in roots]
    positive = wishart_expected_poly(n, d).evaluate_roots_float64(exact=False) * (d / n)
    return [0.0] * (d - n) + [float(x) for x in positive]


def sample_entries(d: int, n: int, family: str, distribution: str) -> Any:
    """Independent centered, variance-one coordinates for bounded-entry models."""
    if distribution not in {"rademacher", "uniform"}:
        raise ValueError(distribution)

    def coordinates(shape: tuple[int, ...]) -> Any:
        if distribution == "rademacher":
            return np.random.choice([-1.0, 1.0], size=shape)
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=shape)

    if family in {"goe", "gue"}:
        raw = coordinates((d, d))
        if family == "gue":
            raw = (raw + 1j * coordinates((d, d))) / np.sqrt(2)
        upper = np.triu(raw, 1)
        result = (upper + upper.conj().T) / np.sqrt(d)
        diagonal_scale = np.sqrt(2 / d) if family == "goe" else 1 / np.sqrt(d)
        np.fill_diagonal(result, coordinates((d,)) * diagonal_scale)
        return result
    beta = int(family[-1])
    if beta == 1:
        x = coordinates((d, n))
    else:
        a = (coordinates((d, n)) + 1j * coordinates((d, n))) / np.sqrt(2)
        if beta == 2:
            x = a
        else:
            b = (coordinates((d, n)) + 1j * coordinates((d, n))) / np.sqrt(2)
            x = np.block([[a, b], [-b.conj(), a.conj()]])
    return (x @ x.conj().T) / (2 * n if beta == 4 else n)


def sampled_spectra(degrees: list[int], distribution: str) -> dict[str, Any]:
    spectra: dict[str, Any] = {}
    state = np.random.get_state()
    try:
        for name, sampler in [
            ("gue", sample_gue),
            ("goe", sample_goe),
            ("gse", sample_gse),
        ]:
            if name == "gse" and distribution != "gaussian":
                continue
            rows: list[Any] = []
            for d in degrees:
                samples: list[Any] = []
                for seed in SEEDS:
                    np.random.seed(seed + d)
                    matrix = (
                        sampler(d)
                        if distribution == "gaussian"
                        else sample_entries(d, d, name, distribution)
                    )
                    eigenvalues = np.linalg.eigvalsh(matrix)
                    if name == "gse":
                        assert np.allclose(
                            eigenvalues[::2], eigenvalues[1::2], atol=1e-10
                        )
                        eigenvalues = eigenvalues[::2]
                    samples.append(eigenvalues.tolist())
                rows.append({"d": d, "samples": samples})
            spectra[name] = rows
        for beta in [1, 2, 4]:
            rows = []
            for ratio in RATIOS:
                by_degree = []
                for d in degrees:
                    n = int(d / ratio)
                    samples = []
                    for seed in SEEDS:
                        np.random.seed(seed + d + n)
                        matrix = (
                            sample_wishart(d, n, beta=beta)
                            if distribution == "gaussian"
                            else sample_entries(d, n, f"wishart{beta}", distribution)
                        )
                        eigenvalues = np.linalg.eigvalsh(matrix)
                        if beta == 4:
                            assert np.allclose(
                                eigenvalues[::2], eigenvalues[1::2], atol=1e-10
                            )
                            eigenvalues = eigenvalues[::2]
                        # Structural nullity is known from matrix dimensions;
                        # don't infer the atom by thresholding positive roots.
                        eigenvalues[: max(0, d - n)] = 0
                        samples.append(eigenvalues.tolist())
                    by_degree.append({"d": d, "n": n, "samples": samples})
                rows.append(by_degree)
            spectra[f"wishart{beta}"] = rows
    finally:
        np.random.set_state(state)
    return spectra


def ensemble_data(degrees: list[int] = DEGREES) -> dict[str, Any]:
    spectra = sampled_spectra(degrees, "gaussian")
    spectra["hermite"] = [
        {
            "d": d,
            "roots": gue_expected_poly(d).evaluate_roots_float64(exact=False).tolist(),
        }
        for d in degrees
    ]
    spectra["laguerre"] = [
        [
            {"d": d, "n": int(d / ratio), "roots": wishart_roots(d, int(d / ratio))}
            for d in degrees
        ]
        for ratio in RATIOS
    ]
    spectra["legendre"] = [
        {
            "d": d,
            "roots": legendre_polynomial(d)
            .evaluate_roots_float64(exact=False)
            .tolist(),
        }
        for d in degrees
    ]
    return {
        "degrees": degrees,
        "ratios": [float(r) for r in RATIOS],
        "seeds": SEEDS,
        "spectra": spectra,
        "entry_spectra": {
            name: sampled_spectra(degrees, name) for name in ["rademacher", "uniform"]
        },
        "limits": {
            "semicircle": limit_measure("semicircle"),
            "arcsine": limit_measure("arcsine"),
            "mp": [limit_measure("mp", float(r)) for r in RATIOS],
        },
    }


def transform_data(degrees: list[int] = DEGREES) -> dict[str, Any]:
    rows = []
    for ratio in RATIOS:
        by_degree = []
        for d in degrees:
            n = int(d / ratio)
            # Integer negative Laguerre parameter is permitted here; no roots
            # are requested from its unsupported recurrence metadata.
            polynomial = wishart_expected_poly(d, n)
            transform = FiniteTTransform(polynomial)
            values = [str(transform(sp.Rational(2 * k + 1, 2 * d))) for k in range(d)]
            by_degree.append(
                {"d": d, "n": n, "steps": values, "nullity": max(0, d - n)}
            )
        rows.append(by_degree)
    cumulants = []
    # Even squares make the CLT dilation exactly rational, with variance d/(d-1).
    clt_degrees = [4, 16, 36, 64, 100, 144, 196, 256]
    for d in clt_degrees:
        p = RealRootedPolynomial.from_roots([-1, 1] * (d // 2))
        power = p.additive_power(d).dilation(sp.Rational(1, math.isqrt(d)))
        cumulants.append([str(c) for c in FiniteRTransform(power, order=8)])
    return {
        "degrees": degrees,
        "ratios": [float(r) for r in RATIOS],
        "rows": rows,
        "clt_degrees": clt_degrees,
        "cumulants": cumulants,
    }


def interlacing_data() -> dict[str, Any]:
    d = 6
    p = RealRootedPolynomial.from_roots([1, 3, 5, 7, 9, 11])
    rows = []
    for j in range(21):
        shift = sp.Rational(j, 10)
        q = RealRootedPolynomial.from_roots(
            [sp.Integer(v) + shift for v in [1, 3, 5, 7, 9, 11]]
        )
        operand = RealRootedPolynomial.from_roots(
            [sp.Rational(k, 2) for k in [1, 3, 5, 7, 9, 11]]
        )
        results: dict[str, Any] = {}
        for name, operation in [
            ("additive", symmetric_additive),
            ("multiplicative", multiplicative),
        ]:
            a, b = operation(p, operand, d), operation(q, operand, d)
            results[name] = {
                "p": a.evaluate_roots_float64().tolist(),
                "q": b.evaluate_roots_float64().tolist(),
            }
        rows.append(
            {
                "shift": str(shift),
                "input": {
                    "p": [1, 3, 5, 7, 9, 11],
                    "q": [float(v) + float(shift) for v in [1, 3, 5, 7, 9, 11]],
                },
                **results,
            }
        )
    return {"rows": rows, "operand": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]}


def hermite_functions(d: int, xs: Any) -> Any:
    phi = np.empty((d, len(xs)))
    phi[0] = np.exp(-(xs**2) / 2) / np.pi**0.25
    if d > 1:
        phi[1] = np.sqrt(2) * xs * phi[0]
    for k in range(1, d - 1):
        phi[k + 1] = (
            np.sqrt(2 / (k + 1)) * xs * phi[k] - np.sqrt(k / (k + 1)) * phi[k - 1]
        )
    return phi


def kernel_data(degrees: list[int] = DEGREES) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for region, bounds in [("bulk", (-3, 3)), ("edge", (-4, 2))]:
        xs = np.linspace(*bounds, 301)
        if region == "bulk":
            limit = np.sinc(xs)
        else:
            ai, aip, _, _ = airy(xs)
            ai0, aip0, _, _ = airy(0)
            limit = np.empty_like(xs)
            np.divide(ai * aip0 - aip * ai0, xs, out=limit, where=xs != 0)
            limit[xs == 0] = aip0**2
        curves = []
        for d in degrees:
            scale = (
                np.pi / np.sqrt(2 * d)
                if region == "bulk"
                else 1 / (np.sqrt(2) * d ** (1 / 6))
            )
            center = 0 if region == "bulk" else np.sqrt(2 * d)
            phi = hermite_functions(d, np.append(center + xs * scale, center))
            values = np.sum(phi[:, :-1] * phi[:, -1:], axis=0) * scale
            curves.append(values.tolist())
        rows[region] = {"x": xs.tolist(), "limit": limit.tolist(), "curves": curves}
    return {"degrees": degrees, "rows": rows}


def unitary_data() -> dict[str, Any]:
    degrees = [4, 8, 12]
    times = [sp.Rational(k, 10) for k in range(51)]
    rows = []
    for d in degrees:
        frames = []
        for t in times:
            p = unitary_hermite_polynomial(d, t)
            coefficients = [float(sp.N(c, 30)) for c in p.coeffs]
            # The coalesced t=0 roots are known exactly; a companion solve there
            # amplifies rounding. Positive times retain raw numerical roots.
            roots = np.ones(d, dtype=complex) if t == 0 else p.evaluate_roots_float64()
            assert np.max(np.abs(np.abs(roots) - 1)) < 1e-6
            frames.append(
                {
                    "coefficients": coefficients,
                    "roots": [[float(z.real), float(z.imag)] for z in roots],
                }
            )
        rows.append(frames)
    return {"degrees": degrees, "times": [float(t) for t in times], "rows": rows}


def generate(output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    models = {
        "ensemble-convergence": ensemble_data(),
        "finite-transforms": transform_data(),
        "convolution-interlacing": interlacing_data(),
        "hermite-kernels": kernel_data(),
        "unitary-root-flow": unitary_data(),
    }
    template = (ASSETS / "dashboard.html").read_text(encoding="utf-8")
    common = (ASSETS / "common.js").read_text(encoding="utf-8")
    style = (ASSETS / "theme.css").read_text(encoding="utf-8") + (
        ASSETS / "dashboard.css"
    ).read_text(encoding="utf-8")
    app = (ASSETS / "dashboard.js").read_text(encoding="utf-8")
    paths = []
    for name, model in models.items():
        encoded = json.dumps(model, separators=(",", ":"), allow_nan=False).replace(
            "</", "<\\/"
        )
        html = (
            template.replace("<!--STYLE-->", style)
            .replace("<!--COMMON-->", common)
            .replace("<!--APP-->", app)
            .replace("<!--DATA-->", encoded)
            .replace("<!--NAME-->", name)
        )
        path = output / f"{name}.html"
        path.write_text(html, encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "visuals" / "generated")
    for path in generate(parser.parse_args().output):
        print(path)


if __name__ == "__main__":
    main()
