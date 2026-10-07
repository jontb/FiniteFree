"""Independent determinant, PSD-boundary and real-root/interlacing example checks."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sympy as sp

from examples.interactive_geometry import cone_model, generate, moving_model


@pytest.mark.parametrize(
    "point,psd,rank",
    [
        ([0, 0, 0], True, 3),
        ([1, sp.Rational(1, 2), sp.Rational(1, 2)], True, 2),
        ([1, 1, 1], True, 1),
        ([sp.Rational(6, 5)] * 3, False, 3),
        ([0, 0, sp.Rational(11, 10)], False, 3),
        ([sp.Rational(-1, 2), sp.Rational(-1, 2), 1], True, 2),
    ],
)
def test_cone_exact_section_and_psd_chambers(
    point: list[Any], psd: bool, rank: int
) -> None:
    polynomial, section = cone_model()
    t, x, y, z = polynomial.variables
    independent = sp.Matrix([[t, x, y], [x, t, z], [y, z, t]])
    assert polynomial.expr == sp.expand(independent.det())
    a, b, c = point
    matrix = sp.Matrix([[1, a, b], [a, 1, c], [b, c, 1]])
    expected = matrix.det()
    value = section.evaluate(point)
    assert sp.Rational(int(value.p), int(value.q)) == expected
    assert matrix.rank() == rank
    assert (all(v >= 0 for v in [1 - a * a, 1 - b * b, 1 - c * c, expected])) == psd
    eigenvalues = np.linalg.eigvalsh(np.array(matrix, dtype=float))
    assert bool(eigenvalues.min() >= -1e-12) == psd
    if point == [sp.Rational(6, 5)] * 3:
        assert expected > 0 and np.sum(eigenvalues < 0) == 2


@pytest.mark.parametrize(
    "y,z", [(0, 0), (1, 0), (0, 1), (sp.Rational(1, 20), sp.Rational(1, 40)), (-2, 1)]
)
def test_moving_slices_against_symbolic_determinants_and_matrix_spectrum(
    y: Any, z: Any
) -> None:
    polynomial = moving_model()
    x, sy, sz = polynomial.variables
    diagonal = sp.diag(-3, -1, 1, 3)
    adjacency = sp.Matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    independent = x * sp.eye(4) + sy * diagonal + sz * adjacency
    assert polynomial.expr == sp.expand(independent.det())
    line = polynomial.restrict_line([0, y, z], [1, 0, 0])
    expected = sp.Poly(independent.subs({sy: y, sz: z}).det(), x)
    assert list(line.coeffs) == expected.all_coeffs()
    derivative = line.derivative(monic=False)
    assert list(derivative.coeffs) == expected.diff().all_coeffs()
    assert line.evaluate(0) == expected.eval(0)
    roots = -np.linalg.eigvalsh(np.array(y * diagonal + z * adjacency, dtype=float))[
        ::-1
    ]
    for root in roots:
        scale = 1 + sum(
            abs(float(c) * root ** (4 - i)) for i, c in enumerate(line.coeffs)
        )
        assert abs(line.evaluate(float(root))) / scale < 1e-12
    if z == 0 and y == 0:
        assert list(line.coeffs) == [1, 0, 0, 0, 0]
        assert list(derivative.coeffs) == [4, 0, 0, 0]
    elif z != 0:
        assert np.all(np.diff(roots) > 0)
        derivative_roots = np.sort(np.roots(np.array(derivative.coeffs, dtype=float)))
        assert np.all(derivative_roots >= roots[:-1] - 1e-10)
        assert np.all(derivative_roots <= roots[1:] + 1e-10)


def test_export_is_portable_and_every_control_point_matches_exact_backend(
    tmp_path: Path,
) -> None:
    pages = generate(tmp_path)
    assert len(pages) == 2
    models = {}
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert '<html lang="en">' in html and '<meta name="viewport"' in html
        assert "<!--DATA-->" not in html and "<!--APP-->" not in html
        assert "<script src=" not in html and "<link " not in html
        encoded = html.split('<script id="model" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
        models[page.stem] = json.loads(encoded)
    model = models["moving-line-roots"]
    assert len(model["ys"]) == 81 and len(model["zs"]) == 41
    diagonal = np.diag([-3, -1, 1, 3])
    adjacency = np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1)
    max_root = 0.0
    for zi, zvalue in enumerate(model["zs"]):
        z = float(sp.Rational(zvalue))
        for yi, yvalue in enumerate(model["ys"]):
            y = float(sp.Rational(yvalue))
            entry = model["lines"][zi][yi]
            coefficients = np.array([float(sp.Rational(c)) for c in entry["p"]])
            assert [sp.Rational(c) for c in entry["d"]] == [
                sp.Rational(c) * (4 - i) for i, c in enumerate(entry["p"][:-1])
            ]
            roots = -np.linalg.eigvalsh(y * diagonal + z * adjacency)
            # Independent numerical eigenvalues must satisfy the exported polynomial.
            scale = 1 + sum(
                abs(c) * np.abs(roots) ** (4 - i) for i, c in enumerate(coefficients)
            )
            assert np.max(np.abs(np.polyval(coefficients, roots)) / scale) < 1e-12
            max_root = max(max_root, float(np.max(np.abs(roots))))
    assert max_root < 6.7  # The complete exported root family fits the displayed axes.
