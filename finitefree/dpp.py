import abc
from typing import Any, List, Sequence, Union

import flint
import numpy as np
import sympy as sp

from .core import RealRootedPolynomial


def _exact_det(A: List[List[Any]]) -> Any:
    """Computes determinant exactly. Uses flint if all entries are rational, otherwise sympy."""
    n = len(A)
    if n == 0:
        return 1

    # Check if all elements can be converted to flint.fmpq
    try:
        flint_mat = flint.fmpq_mat(n, n)
        for i in range(n):
            for j in range(n):
                flint_mat[i, j] = flint.fmpq(A[i][j])
        return flint_mat.det()
    except (TypeError, ValueError):
        # Fallback to sympy
        sym_mat = sp.Matrix(A)
        return sym_mat.det()


class BaseKernel(abc.ABC):
    """Abstract base class for determinantal point process (DPP) correlation kernels."""

    @abc.abstractmethod
    def __call__(self, x: Any, y: Any) -> Any:
        """Evaluate the kernel at x and y."""
        pass

    def matrix(self, xs: Sequence[Any]) -> List[List[Any]]:
        """Compute the exact kernel matrix for points xs."""
        return [[self(x, y) for y in xs] for x in xs]

    def k_point_correlation(self, xs: Sequence[Any]) -> Any:
        """Evaluate the k-point correlation function exactly: det(K(x_i, x_j))."""
        M = self.matrix(xs)
        return _exact_det(M)


class DiscreteFiniteKernel(BaseKernel):
    """Discrete state space kernel defined by a symmetric matrix representation."""

    def __init__(self, K: Any) -> None:
        """K can be a list of lists, a numpy array, or a flint.fmpq_mat."""
        self._K = K

    def __call__(self, i: Any, j: Any) -> Any:
        # Check if i, j are integers
        try:
            return self._K[i][j]
        except TypeError:
            return self._K[int(i)][int(j)]


class OrthogonalPolynomialKernel(BaseKernel):
    """Correlation kernel constructed from orthogonal polynomials using Christoffel-Darboux formula."""

    def __init__(
        self,
        polys: List[RealRootedPolynomial],
        norms: List[Any],
        leading_coeffs: Union[List[Any], None] = None,
    ) -> None:
        """
        polys: List of RealRootedPolynomial, from p_0 to p_n (length n + 1)
        norms: List of norm constants, from h_0 to h_{n-1} (length n)
        leading_coeffs: Optional list of leading coefficients, from k_0 to k_n (length n + 1)
        """
        self.polys = polys
        self.norms = [sympy_to_exact(h) for h in norms]
        self.n = len(norms)

        if len(polys) < self.n + 1:
            raise ValueError(
                f"polys must contain at least {self.n + 1} polynomials (p_0 to p_n)"
            )

        if leading_coeffs is None:
            self.leading_coeffs = []
            for p in polys:
                if p._is_flint:
                    self.leading_coeffs.append(p._fmpq_poly.coeffs()[-1])
                else:
                    self.leading_coeffs.append(p.coeffs[0])
        else:
            self.leading_coeffs = [sympy_to_exact(k) for k in leading_coeffs]

    def __call__(self, x: Any, y: Any) -> Any:
        # Convert inputs to exact types first
        x = sympy_to_exact(x)
        y = sympy_to_exact(y)

        # Christoffel-Darboux evaluation
        # If x == y (or very close numerically), evaluate diagonal via derivatives to avoid division by zero
        is_diag = x == y
        if (
            not is_diag
            and isinstance(x, (float, np.floating, flint.fmpq))
            and isinstance(y, (float, np.floating, flint.fmpq))
        ):
            # Convert to float for close comparison
            from .utils.conversion import flint_to_float

            is_diag = np.isclose(flint_to_float(x), flint_to_float(y))

        pn = self.polys[self.n]
        pn_minus = self.polys[self.n - 1]
        kn = self.leading_coeffs[self.n]
        kn_minus = self.leading_coeffs[self.n - 1]
        hn_minus = self.norms[self.n - 1]

        factor = kn_minus / (kn * hn_minus)

        if is_diag:
            pn_val = pn.evaluate(x)
            pn_minus_val = pn_minus.evaluate(x)
            pn_deriv_val = (
                pn.derivative().evaluate(x) * pn.degree if pn.degree > 0 else 0
            )
            pn_minus_deriv_val = (
                pn_minus.derivative().evaluate(x) * pn_minus.degree
                if pn_minus.degree > 0
                else 0
            )
            return factor * (pn_deriv_val * pn_minus_val - pn_minus_deriv_val * pn_val)
        else:
            pn_x = pn.evaluate(x)
            pn_minus_y = pn_minus.evaluate(y)
            pn_minus_x = pn_minus.evaluate(x)
            pn_y = pn.evaluate(y)
            numerator = pn_x * pn_minus_y - pn_minus_x * pn_y
            return (factor * numerator) / (x - y)


def sympy_to_exact(val: Any) -> Any:
    """Converts a value to flint.fmpq or sympy Rational/float."""
    try:
        from .utils.conversion import sympy_to_fmpq

        return sympy_to_fmpq(sp.sympify(val))
    except Exception:
        return val


def gap_probability_discrete(kernel: BaseKernel, points_in_gap: Sequence[Any]) -> Any:
    """Computes exact gap probability over a discrete state space subset: det(I - K_I)."""
    n = len(points_in_gap)
    if n == 0:
        return 1
    K_mat = kernel.matrix(points_in_gap)

    # We construct exact I - K_I
    I_minus_K = [[-K_mat[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        I_minus_K[i][i] = 1 + I_minus_K[i][i]

    return _exact_det(I_minus_K)


def gap_probability_continuous(
    kernel: BaseKernel, a: float, b: float, n_points: int = 50, weight_func: Any = None
) -> float:
    """Approximates the continuous Fredholm determinant gap probability over [a, b] using Nyström discretization."""
    import scipy.special

    from .utils.conversion import flint_to_float

    pts, w = scipy.special.roots_legendre(n_points)
    # Map points and weights from [-1, 1] to [a, b]
    pts_mapped = 0.5 * (b - a) * pts + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w

    D = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            val = kernel(pts_mapped[i], pts_mapped[j])
            weight = 1.0
            if weight_func is not None:
                weight = np.sqrt(
                    weight_func(pts_mapped[i]) * weight_func(pts_mapped[j])
                )
            D[i, j] = (
                np.sqrt(w_mapped[i])
                * flint_to_float(val)
                * weight
                * np.sqrt(w_mapped[j])
            )

    matrix = np.eye(n_points) - D
    return float(np.linalg.det(matrix))


def sample_discrete(kernel: BaseKernel, state_space: Sequence[Any]) -> List[Any]:
    """HKPV exact projection kernel sampling algorithm for discrete state spaces."""
    K_mat = np.array([[float(kernel(x, y)) for y in state_space] for x in state_space])
    M = len(state_space)

    # Eigen-decomposition for projection component selection
    eigenvalues, eigenvectors = np.linalg.eigh(K_mat)

    selected_indices = []
    for idx, lam in enumerate(eigenvalues):
        p = np.clip(lam, 0.0, 1.0)
        if np.random.rand() < p:
            selected_indices.append(idx)

    V = [eigenvectors[:, idx] for idx in selected_indices]
    k = len(V)

    sampled_indices = []

    for i in range(k, 0, -1):
        probs = np.zeros(M)
        for x_idx in range(M):
            probs[x_idx] = sum(abs(v[x_idx]) ** 2 for v in V) / i

        probs = np.clip(probs, 0, None)
        total_prob = np.sum(probs)
        if total_prob > 1e-12:
            probs /= total_prob
        else:
            probs = np.ones(M) / M

        sampled_idx = np.random.choice(M, p=probs)
        sampled_indices.append(sampled_idx)

        if i > 1:
            best_v_idx = np.argmax([abs(v[sampled_idx]) for v in V])
            v_star = V[best_v_idx]

            V.pop(best_v_idx)

            new_V: List[Any] = []
            for v in V:
                factor = v[sampled_idx] / v_star[sampled_idx]
                updated_v = v - factor * v_star
                # Gram-Schmidt stabilization
                for prev_v in new_V:
                    proj = np.dot(updated_v, prev_v)
                    updated_v = updated_v - proj * prev_v
                norm = np.linalg.norm(updated_v)
                if norm > 1e-12:
                    updated_v /= norm
                    new_V.append(updated_v)
            V = new_V

    return [state_space[idx] for idx in sampled_indices]
