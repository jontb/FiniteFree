"""Generate two self-contained HTML geometry explorers from exact FiniteFree data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from finitefree.hyperbolic.pencils import SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "visuals" / "interactive"


def named_determinant(
    matrices: list[Any], names: tuple[sp.Symbol, ...]
) -> MultivariatePolynomial:
    """Use the public exact pencil constructor and explicit context substitution."""
    result = MultivariatePolynomial.from_symmetric_matrix_pencil(
        SymmetricMatrixPencil(matrices)
    )
    return result.substitute(dict(zip(result.variables, names)), variables=names)


def sparse_terms(polynomial: MultivariatePolynomial) -> list[dict[str, Any]]:
    return [
        {"powers": list(alpha), "coefficient": str(coefficient)}
        for alpha, coefficient in sorted(polynomial.coefficients().items())
    ]


def cone_model() -> tuple[MultivariatePolynomial, MultivariatePolynomial]:
    t, x, y, z = sp.symbols("t x y z")
    matrices: list[Any] = [
        np.eye(3, dtype=int),
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    ]
    polynomial = named_determinant(matrices, (t, x, y, z))
    section = polynomial.substitute({t: 1}, variables=[x, y, z])
    return polynomial, section


def moving_model() -> MultivariatePolynomial:
    x, y, z = sp.symbols("x y z")
    diagonal = np.diag([-3, -1, 1, 3])
    path = np.diag(np.ones(3, dtype=int), 1) + np.diag(np.ones(3, dtype=int), -1)
    return named_determinant([np.eye(4, dtype=int), diagonal, path], (x, y, z))


def moving_data() -> dict[str, Any]:
    polynomial = moving_model()
    x, _, _ = polynomial.variables
    derivative = polynomial.partial_derivative(x)
    # Browser controls index this exact rational grid. No float is passed to the
    # exact backend; strings preserve rational coefficients in the exported data.
    ys = [sp.Rational(i, 20) for i in range(-40, 41)]
    zs = [sp.Rational(i, 40) for i in range(41)]
    lines: list[list[dict[str, Any]]] = []
    for z in zs:
        row = []
        for y in ys:
            line = polynomial.restrict_line([0, y, z], [1, 0, 0])
            dline = line.derivative(monic=False)
            # Independent public multivariate derivative restricted to the same
            # line must agree with the public univariate derivative, including scale.
            reference = derivative.restrict_line([0, y, z], [1, 0, 0])
            assert list(reference.coeffs) == list(dline.coeffs)
            row.append(
                {
                    "p": [str(c) for c in line.coeffs],
                    "d": [str(c) for c in dline.coeffs],
                }
            )
        lines.append(row)
    return {
        "terms": sparse_terms(polynomial),
        "derivative_terms": sparse_terms(derivative),
        "ys": [str(y) for y in ys],
        "zs": [str(z) for z in zs],
        "lines": lines,
        "coefficient_order": "descending",
        "matrix_diagonal": [-3, -1, 1, 3],
        "matrix_edges": [[0, 1], [1, 2], [2, 3]],
    }


def generate(output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    _, section = cone_model()
    models = {
        "hyperbolicity-cone": {"terms": sparse_terms(section)},
        "moving-line-roots": moving_data(),
    }
    common = (ASSETS / "common.js").read_text(encoding="utf-8")
    style = (ASSETS / "theme.css").read_text(encoding="utf-8")
    paths = []
    for name, model in models.items():
        template = (ASSETS / f"{name}.html").read_text(encoding="utf-8")
        encoded = json.dumps(model, separators=(",", ":")).replace("</", "<\\/")
        html = (
            template.replace("<!--STYLE-->", style)
            .replace("<!--DATA-->", encoded)
            .replace("<!--COMMON-->", common)
            .replace("<!--APP-->", (ASSETS / f"{name}.js").read_text(encoding="utf-8"))
        )
        path = output / f"{name}.html"
        path.write_text(html, encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "visuals" / "generated")
    args = parser.parse_args()
    for path in generate(args.output):
        print(path)


if __name__ == "__main__":
    main()
