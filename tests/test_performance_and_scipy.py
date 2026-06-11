import numpy as np

from finitefree.core import RealRootedPolynomial, UnitaryPolynomial
from finitefree.utils.modular import modular_det


def test_modular_det() -> None:
    # Test modular determinant on a simple 3x3 matrix
    p = 1000000007
    A = [
        [2, 1, 3],
        [1, 5, 2],
        [3, 2, 4]
    ]
    # Classical determinant is 2*(20-4) - 1*(4-6) + 3*(2-15) = 32 + 2 - 39 = -5 = 1000000002 mod p
    det = modular_det(A, p)
    assert det == (p - 5) % p


def test_sturm_subresultant_prs() -> None:
    # Construct a degree-15 polynomial to trigger subresultant PRS
    roots = list(range(1, 16))
    p = RealRootedPolynomial.from_roots(roots)
    assert p.degree == 15
    # Verify rootedness (this will route to subresultant PRS internally)
    assert p.verify_real_rootedness() is True


def test_unitary_polynomial_gpu_fallback() -> None:
    # Unitary polynomial x^2 - 2exp(-1/4)z + 1
    u = UnitaryPolynomial([1, -2, 1])
    # Run with gpu=True, should fallback to CPU safely if CuPy is not present or run on GPU if present
    roots = u.evaluate_roots_float64(gpu=True)
    assert len(roots) == 2
    for r in roots:
        assert np.isclose(np.abs(r), 1.0)


def test_to_scipy_dist() -> None:
    # Polynomial roots: 1, 2, 3, 4
    p = RealRootedPolynomial.from_roots([1.0, 2.0, 3.0, 4.0])
    dist = p.to_scipy_dist()

    # Test support
    assert np.isclose(dist.a, 1.0)
    assert np.isclose(dist.b, 4.0)

    # Test CDF evaluations
    assert np.isclose(dist.cdf(1.0), 0.0)
    assert np.isclose(dist.cdf(4.0), 1.0)
    assert np.isclose(dist.cdf(2.5), 0.5) # middle of roots [1, 2, 3, 4]

    # Test PDF evaluations (density is positive inside support)
    assert dist.pdf(2.0) > 0
    assert dist.pdf(0.5) == 0.0
    assert dist.pdf(4.5) == 0.0

    # Test sampling
    samples = dist.rvs(size=10)
    assert len(samples) == 10
    assert np.all(samples >= 1.0)
    assert np.all(samples <= 4.0)


def test_cython_pencil_evals() -> None:
    from finitefree.utils.modular_fast import (  # type: ignore[import-untyped]
        eval_diagonal_specialization_mod_p,
        eval_points_grid_mod_p,
    )

    # 1. Test eval_diagonal_specialization_mod_p
    A = np.array([[2, 1], [1, 3]], dtype=np.int64)
    B = np.array([[1, 0], [0, 2]], dtype=np.int64)
    p = 1000000007
    deg = 2
    # det(z * A + B) = (2z + 1)(3z + 2) - z^2 = 6z^2 + 7z + 2 - z^2 = 5z^2 + 7z + 2
    # z=0: 2
    # z=1: 14
    # z=2: 5*4 + 14 + 2 = 36
    res_diag = list(eval_diagonal_specialization_mod_p(A, B, p, deg))
    assert res_diag == [2, 14, 36]

    # 2. Test eval_points_grid_mod_p
    matrices = np.array([[[2, 1], [1, 3]], [[1, 0], [0, 2]]], dtype=np.int64)
    grid_pts = np.array([[0, 1], [1, 1]], dtype=np.int64)
    # pt [0, 1] -> 0*A + 1*B = B -> det is 2
    # pt [1, 1] -> 1*A + 1*B = A + B -> det(A+B) = det([[3, 1], [1, 5]]) = 15 - 1 = 14
    res_grid = list(eval_points_grid_mod_p(matrices, grid_pts, p))
    assert res_grid == [2, 14]


def test_pure_python_fallback() -> None:
    import os
    import subprocess
    import sys

    # Run pytest on all other tests with PYFFP_DISABLE_CYTHON=1
    env = os.environ.copy()
    env["PYFFP_DISABLE_CYTHON"] = "1"

    # Run the entire test suite with fallback enabled, excluding this performance file
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "--ignore=tests/test_performance_and_scipy.py",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Fallback tests failed:\n{res.stdout}\n{res.stderr}"



