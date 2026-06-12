import flint
import numpy as np

from finitefree import (
    DiscreteFiniteKernel,
    OrthogonalPolynomialKernel,
    gap_probability_continuous,
    gap_probability_discrete,
    hermite_polynomial,
    sample_discrete,
)


def test_discrete_finite_kernel() -> None:
    # 3x3 identity matrix as a projection kernel of trace 2 (projection on 2 coordinates)
    K = [
        [flint.fmpq(2, 3), flint.fmpq(1, 3), flint.fmpq(0)],
        [flint.fmpq(1, 3), flint.fmpq(2, 3), flint.fmpq(0)],
        [flint.fmpq(0), flint.fmpq(0), flint.fmpq(1)],
    ]

    kernel = DiscreteFiniteKernel(K)

    # Check evaluation
    assert kernel(0, 0) == flint.fmpq(2, 3)
    assert kernel(0, 1) == flint.fmpq(1, 3)
    assert kernel(2, 2) == flint.fmpq(1)

    # Check k-point correlation (1-point and 2-point)
    assert kernel.k_point_correlation([0]) == flint.fmpq(2, 3)
    # det of top-left 2x2: (2/3)^2 - (1/3)^2 = 4/9 - 1/9 = 3/9 = 1/3
    assert kernel.k_point_correlation([0, 1]) == flint.fmpq(1, 3)

    # Check gap probability (probability of no points in subset [0, 1])
    # det(I - K_I) = det([[1 - 2/3, -1/3], [-1/3, 1 - 2/3]]) = det([[1/3, -1/3], [-1/3, 1/3]]) = 0
    assert gap_probability_discrete(kernel, [0, 1]) == flint.fmpq(0)


def test_orthogonal_polynomial_kernel() -> None:
    from finitefree.utils.conversion import flint_to_float

    # We will use Hermite polynomials He_0, He_1, He_2, He_3 (norms: sqrt(2pi) * k!)
    # But since they are defined algebraically in orthogonal.py:
    # Monic probabilist's Hermite polynomials have norms: h_k = k!
    norms = [1, 1, 2]  # h_0=0!=1, h_1=1!=1, h_2=2!=2
    polys = [
        hermite_polynomial(0, physicist=False),
        hermite_polynomial(1, physicist=False),
        hermite_polynomial(2, physicist=False),
        hermite_polynomial(3, physicist=False),
    ]

    # Leading coefficients are all 1 since they are monic
    kernel = OrthogonalPolynomialKernel(polys, norms)

    # Naive summation kernel evaluation function: sum_{j=0}^2 p_j(x) p_j(y) / h_j
    def naive_sum(x: float, y: float) -> float:
        val = 0.0
        for j in range(3):
            val += (
                flint_to_float(polys[j].evaluate(x))
                * flint_to_float(polys[j].evaluate(y))
                / norms[j]
            )
        return val

    # Test off-diagonal evaluation
    x, y = 0.5, 1.2
    cd_val = flint_to_float(kernel(x, y))
    assert np.isclose(cd_val, naive_sum(x, y))

    # Test diagonal evaluation (x == y)
    cd_diag = flint_to_float(kernel(x, x))
    assert np.isclose(cd_diag, naive_sum(x, x))

    # Test diagonal evaluation with close points
    cd_close = flint_to_float(kernel(x, x + 1e-9))
    assert np.isclose(cd_close, naive_sum(x, x))


def test_gap_probability_continuous() -> None:
    norms = [1, 1]  # h_0=1, h_1=1
    polys = [
        hermite_polynomial(0, physicist=False),
        hermite_polynomial(1, physicist=False),
        hermite_polynomial(2, physicist=False),
    ]
    kernel = OrthogonalPolynomialKernel(polys, norms)

    # Hermite weight function for probabilist's Hermite polynomials
    import math

    def weight_func(x: float) -> float:
        return math.exp(-(x**2) / 2.0) / math.sqrt(2.0 * math.pi)

    # Verify continuous Fredholm determinant evaluation returns a float between 0 and 1
    prob = gap_probability_continuous(
        kernel, -1.0, 1.0, n_points=10, weight_func=weight_func
    )
    assert 0.0 <= prob <= 1.0


def test_sample_discrete() -> None:
    # A 5x5 projection matrix of trace 3
    # Represented as a projector on first 3 coordinates
    K = np.zeros((5, 5))
    K[0, 0] = 1.0
    K[1, 1] = 1.0
    K[2, 2] = 1.0

    kernel = DiscreteFiniteKernel(K)
    state_space = [0, 1, 2, 3, 4]

    # Sample multiple times and check that trace (3) determines the sampled size
    for _ in range(5):
        sample = sample_discrete(kernel, state_space)
        assert len(sample) == 3
        # Eigenvectors are standard basis e_0, e_1, e_2, so we must always sample {0, 1, 2}
        assert set(sample) == {0, 1, 2}
