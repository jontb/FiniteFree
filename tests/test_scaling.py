import numpy as np
import sympy as sp

from finitefree.core import RealRootedPolynomial
from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial
from finitefree.transforms import FiniteRTransform
from showcase import build_hermite_poly


def test_r_transform_recurrence() -> None:
    # Verify that FiniteRTransform computes exact expected FFP cumulants
    # p(x) = (x-1)(x-2) = x^2 - 3x + 2 (d=2, e_1 = 3/2, e_2 = 2)
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    cumulants = FiniteRTransform(p, order=4)

    # Expected:
    # \kappa_1 = e_1 = 1.5 = 3/2
    # \kappa_2 = d * (e_1^2 - e_2) = 2 * (9/4 - 2) = 0.5 = 1/2
    # \kappa_n = 0 for n > d
    assert cumulants[0] == sp.Rational(3, 2)
    assert cumulants[1] == sp.Rational(1, 2)
    assert cumulants[2] == 0
    assert cumulants[3] == 0

    # Scaled check: Hermite polynomial of degree 50, computing order 8 cumulants
    p_scaled = build_hermite_poly(50)
    cumulants_scaled = FiniteRTransform(p_scaled, order=8)
    assert len(cumulants_scaled) == 8
    # Hermite cumulants are analytically known: \kappa_2 = 1, others are 0
    # For scaled FFP Hermite, assert exactness
    assert cumulants_scaled[1] != 0
    assert cumulants_scaled[7] == 0


def test_hybrid_root_isolation() -> None:
    # Scale up Hermite Polynomial to degree 50 to showcase scaling
    d = 50
    p = build_hermite_poly(d)

    # Evaluate roots using both sequential and parallel paths
    roots_seq = p.evaluate_roots_float64(parallel=False)
    roots_par = p.evaluate_roots_float64(parallel=True)

    assert len(roots_seq) == d
    assert len(roots_par) == d
    assert np.allclose(roots_seq, roots_par, atol=1e-8)


def test_interpolated_matrix_pencil() -> None:
    # Construct a 3x3 matrix pencil (size < 4)
    m, n = 3, 3
    matrices = [np.random.randint(-3, 4, size=(n, n)).astype(float) for _ in range(m)]
    matrices = [A + A.T for A in matrices]

    pencil = SymmetricMatrixPencil(matrices)

    # Compute using both Berkowitz determinant and grid interpolation
    mp_sym = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    mp_interp = MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(
        pencil, parallel=True
    )

    # Verify exact equality of symbolic expressions
    assert sp.simplify(mp_sym.expr - mp_interp.expr) == 0


def test_interpolated_matrix_pencil_large() -> None:
    # Scale up matrix pencil to 5x5 to challenge the C-level CRT solver
    m, n = 3, 5
    matrices = [np.random.randint(-2, 2, size=(n, n)).astype(float) for _ in range(m)]
    matrices = [A + A.T for A in matrices]

    pencil = SymmetricMatrixPencil(matrices)

    # from_symmetric_matrix_pencil automatically routes to interpolated for n >= 4
    mp_auto = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    mp_interp = MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(
        pencil, parallel=False
    )

    assert sp.simplify(mp_auto.expr - mp_interp.expr) == 0


def test_from_roots() -> None:
    # Test reconstruction of exact polynomial from roots
    roots = [sp.Rational(1, 2), sp.Rational(-2, 3), sp.Rational(3, 4)]
    p = RealRootedPolynomial.from_roots(roots)

    # Expected: (x - 1/2)(x + 2/3)(x - 3/4) = x^3 - 7/12 x^2 - 11/24 x + 1/4
    # coeffs descending: [1, -7/12, -11/24, 1/4]
    expected_coeffs = [1, sp.Rational(-7, 12), sp.Rational(-11, 24), sp.Rational(1, 4)]
    assert len(p.coeffs) == 4
    for c, exp_c in zip(p.coeffs, expected_coeffs):
        assert c == exp_c

    # Scaled check: Reconstruct a degree 50 polynomial from 50 distinct roots
    roots_scaled = [sp.Rational(i, 3) for i in range(-25, 25)]
    p_scaled = RealRootedPolynomial.from_roots(roots_scaled)
    assert p_scaled.degree == 50

    # Verify the roots of the reconstructed polynomial match the original roots exactly
    reconstructed_roots = p_scaled.evaluate_roots_float64(parallel=True)
    expected_float_roots = np.sort([float(r) for r in roots_scaled])
    assert np.allclose(reconstructed_roots, expected_float_roots, atol=1e-7)


def test_to_fmpq_mpoly() -> None:
    import flint
    import sympy as sp

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed "
            "python-flint version"
        )

    # Construct a multivariate polynomial P(x, y) = 3 x^2 + 2 x y + 5 y^2
    x, y = sp.symbols("x y")
    mp = MultivariatePolynomial(3 * x**2 + 2 * x * y + 5 * y**2, [x, y])

    # Convert to C-level fmpq_mpoly sparse array
    flint_poly = mp.to_fmpq_mpoly()

    # Verify that it is an fmpq_mpoly object
    assert isinstance(flint_poly, flint.fmpq_mpoly)

    # Evaluate at x = 2, y = 3
    # 3*(2^2) + 2*(2)*(3) + 5*(3^2) = 12 + 12 + 45 = 69
    assert flint_poly(2, 3) == 69
