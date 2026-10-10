import abc
import math
import operator
from decimal import Decimal, localcontext
from typing import Any, List, Optional, Sequence, Union

import flint
import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .core import RealRootedPolynomial


def _exact_det(A: List[List[Any]]) -> Any:
    """Computes determinant exactly. Uses flint if all entries are rational, otherwise sympy."""
    n = len(A)
    if n == 0:
        return 1

    # Check if all elements can be converted to flint.fmpq
    try:
        flattened = [flint.fmpq(val) for row in A for val in row]
        flint_mat = flint.fmpq_mat(n, n, flattened)
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
        r"""Evaluate the k-point correlation function exactly: $\det(K(x_i, x_j))$."""
        M = self.matrix(xs)
        return _exact_det(M)


class DiscreteFiniteKernel(BaseKernel):
    """Discrete state space kernel defined by a symmetric matrix representation."""

    def __init__(self, K: Any) -> None:
        """K can be a list of lists, a numpy array, or a flint.fmpq_mat."""
        self._K = K

    def __call__(self, i: Any, j: Any) -> Any:
        i, j = operator.index(i), operator.index(j)
        if isinstance(self._K, flint.fmpq_mat):
            return self._K[i, j]
        return self._K[i][j]


class OrthogonalPolynomialKernel(BaseKernel):
    """Orthogonal-polynomial kernel with exact Christoffel-Darboux evaluation.

    Owns its basis and exact norms. A scaled/shifted Hermite basis is verified
    by exact monic recurrence and norm ratios before numerical evaluation by a
    normalized three-term sum. Basis scaling and measure mass are preserved.
    Other families use Christoffel-Darboux, with a finite basis sum at nearby
    distinct floats. Consistent generic polynomials and norms are assumed.
    """

    def __init__(
        self,
        polys: List[RealRootedPolynomial],
        norms: List[Any],
        leading_coeffs: Union[List[Any], None] = None,
    ) -> None:
        r"""
        polys: List of RealRootedPolynomial, from $p_0$ to $p_n$ (length $n + 1$)
        norms: List of norm constants, from $h_0$ to $h_{n-1}$ (length $n$)
        leading_coeffs: Optional list of leading coefficients, from $k_0$ to $k_n$ (length $n + 1$)
        """
        self._polys = tuple(self._copy_polynomial(p) for p in polys)
        self._norms = tuple(sympy_to_exact(h) for h in norms)
        self.n = len(norms)

        if len(polys) < self.n + 1:
            raise ValueError(
                f"polys must contain at least {self.n + 1} polynomials (p_0 to p_n)"
            )

        if leading_coeffs is None:
            coefficients = []
            for p in self._polys:
                if p._is_flint:
                    coefficients.append(p._fmpq_poly.coeffs()[-1])
                else:
                    coefficients.append(p.coeffs[0])
        else:
            coefficients = [sympy_to_exact(k) for k in leading_coeffs]
        if len(coefficients) < self.n + 1:
            raise ValueError("leading_coeffs must contain k_0 through k_n")
        self._leading_coeffs = tuple(coefficients)
        self._hermite_parameters = self._verify_hermite_basis()
        self._hermite_steps: tuple[tuple[float, float], ...] = ()
        self._hermite_center = 0.0
        self._hermite_initial = 1.0
        self._hermite_range_error = False
        if self._hermite_parameters is not None:
            center, variance, mass = self._hermite_parameters
            from .utils.conversion import flint_to_float

            try:
                self._hermite_center = flint_to_float(center)
                self._hermite_initial = self._positive_sqrt_float64(1 / mass)
                self._hermite_steps = tuple(
                    (
                        self._positive_sqrt_float64(1 / ((j + 1) * variance)),
                        math.sqrt(j / (j + 1)),
                    )
                    for j in range(self.n - 1)
                )
                self._hermite_range_error = not math.isfinite(self._hermite_center)
            except (OverflowError, ValueError):
                self._hermite_range_error = True

        # Precompute derivative objects to avoid dynamic instantiation overhead
        self._pn = self._polys[self.n]
        self._pn_minus = self._polys[self.n - 1]
        self._pn_deriv = (
            self._pn.derivative(monic=False) if self._pn.degree > 0 else None
        )
        self._pn_minus_deriv = (
            self._pn_minus.derivative(monic=False)
            if self._pn_minus.degree > 0
            else None
        )

    @staticmethod
    def _copy_polynomial(p: RealRootedPolynomial) -> RealRootedPolynomial:
        return RealRootedPolynomial(
            p._fmpq_poly if p._is_flint else p.coeffs,
            monic=False,
            assume_real_rooted=p._is_verified,
        )

    @property
    def polys(self) -> List[RealRootedPolynomial]:
        """Return caller-owned polynomial copies of the stored basis."""
        return [self._copy_polynomial(p) for p in self._polys]

    @property
    def norms(self) -> List[Any]:
        """Return a caller-owned list of the exact norms."""
        return list(self._norms)

    @property
    def leading_coeffs(self) -> List[Any]:
        """Return a caller-owned list of the leading coefficients."""
        return list(self._leading_coeffs)

    @staticmethod
    def _positive_sqrt_float64(value: Any) -> float:
        # Convert AFTER square root: tiny/huge rational ratios can have a
        # representable root even when the ratio itself over/underflows.
        with localcontext() as context:
            context.prec = 40
            result = float((Decimal(int(value.p)) / Decimal(int(value.q))).sqrt())
        if not math.isfinite(result) or result <= 0:
            raise ValueError("Hermite normalization exceeds float64 range")
        return result

    def _verify_hermite_basis(self) -> Any:
        """Infer and verify (center, variance, measure mass) from owned data.

        No provenance/root metadata is trusted: it does not encode measure mass
        and caller-supplied polynomials, norms or leading coefficients may differ.
        All normalization and recurrence comparisons precede float conversion.
        """
        if self.n == 0:
            return None
        monic = []
        norms = []
        for j, p in enumerate(self._polys[: self.n + 1]):
            if not p._is_flint or p.degree != j:
                return None
            leading = p._fmpq_poly[j]
            if self._leading_coeffs[j] != leading:
                return None
            monic.append(p._fmpq_poly / leading)
            if j < self.n:
                norm = self._norms[j]
                if not isinstance(norm, flint.fmpq) or norm <= 0:
                    return None
                norms.append(norm / leading**2)
        center = -monic[1][0]
        # A rank-one constant kernel needs no variance parameter.
        variance = norms[1] / norms[0] if self.n > 1 else flint.fmpq(1)
        shifted_x = flint.fmpq_poly([-center, 1])
        previous = flint.fmpq_poly([])
        expected = flint.fmpq_poly([1])
        expected_norm = norms[0]
        for j, polynomial in enumerate(monic):
            if polynomial != expected:
                return None
            if j < self.n:
                if norms[j] != expected_norm:
                    return None
                expected_norm *= (j + 1) * variance
            previous, expected = (
                expected,
                shifted_x * expected - j * variance * previous,
            )
        return center, variance, norms[0]

    def _hermite_sum_float64(self, x: float, y: float) -> float:
        if self._hermite_range_error:
            raise RuntimeError("Hermite kernel exceeds finite float64 range")
        x -= self._hermite_center
        y -= self._hermite_center

        def terms() -> Any:
            previous_x = previous_y = 0.0
            current_x = current_y = self._hermite_initial
            yield current_x * current_y
            for inverse_sqrt, ratio in self._hermite_steps:
                previous_x, current_x = (
                    current_x,
                    x * inverse_sqrt * current_x - ratio * previous_x,
                )
                previous_y, current_y = (
                    current_y,
                    y * inverse_sqrt * current_y - ratio * previous_y,
                )
                yield current_x * current_y

        try:
            result = math.fsum(terms())
        except (OverflowError, ValueError) as error:
            raise RuntimeError("Hermite kernel exceeds finite float64 range") from error
        if not math.isfinite(result):
            raise RuntimeError("Hermite kernel exceeds finite float64 range")
        return result

    def __call__(self, x: Any, y: Any) -> Any:
        if self.n == 0:
            return 0

        if isinstance(x, (float, np.floating)) or isinstance(y, (float, np.floating)):
            from .utils.conversion import flint_to_float

            x_f = float(x)
            y_f = float(y)
            if not math.isfinite(x_f) or not math.isfinite(y_f):
                raise ValueError("Numerical kernel coordinates must be finite")
            if self._hermite_parameters is not None:
                return self._hermite_sum_float64(x_f, y_f)
            separation = abs(x_f - y_f)
            close_scale = max(1.0, abs(x_f), abs(y_f))
            if 0 < separation <= math.sqrt(np.finfo(float).eps) * close_scale:
                # This is K(x,y), not the confluent approximation K(x,x).
                # Summing the basis is slower, but avoids the CD quotient's
                # cancellation for this small subset of floating evaluations.
                return math.fsum(
                    flint_to_float(self._polys[j].evaluate(x_f))
                    * flint_to_float(self._polys[j].evaluate(y_f))
                    / flint_to_float(self._norms[j])
                    for j in range(self.n)
                )
            kn_f = flint_to_float(self._leading_coeffs[self.n])
            kn_minus_f = flint_to_float(self._leading_coeffs[self.n - 1])
            hn_minus_f = flint_to_float(self._norms[self.n - 1])
            factor_f = kn_minus_f / (kn_f * hn_minus_f)
            if x_f == y_f:
                pn_val = self._pn.evaluate(x_f)
                pn_minus_val = self._pn_minus.evaluate(x_f)
                pn_deriv_val = self._pn_deriv.evaluate(x_f) if self._pn_deriv else 0.0
                pn_minus_deriv_val = (
                    self._pn_minus_deriv.evaluate(x_f) if self._pn_minus_deriv else 0.0
                )
                return factor_f * (
                    pn_deriv_val * pn_minus_val - pn_minus_deriv_val * pn_val
                )
            else:
                pn_x = self._pn.evaluate(x_f)
                pn_minus_y = self._pn_minus.evaluate(y_f)
                pn_minus_x = self._pn_minus.evaluate(x_f)
                pn_y = self._pn.evaluate(y_f)
                numerator = pn_x * pn_minus_y - pn_minus_x * pn_y
                return (factor_f * numerator) / (x_f - y_f)

        # Convert inputs to exact types first
        x = sympy_to_exact(x)
        y = sympy_to_exact(y)

        kn = self._leading_coeffs[self.n]
        kn_minus = self._leading_coeffs[self.n - 1]
        hn_minus = self._norms[self.n - 1]

        factor = kn_minus / (kn * hn_minus)

        if x == y:
            pn_val = self._pn.evaluate(x)
            pn_minus_val = self._pn_minus.evaluate(x)
            pn_deriv_val = self._pn_deriv.evaluate(x) if self._pn_deriv else 0
            pn_minus_deriv_val = (
                self._pn_minus_deriv.evaluate(x) if self._pn_minus_deriv else 0
            )
            return factor * (pn_deriv_val * pn_minus_val - pn_minus_deriv_val * pn_val)
        else:
            pn_x = self._pn.evaluate(x)
            pn_minus_y = self._pn_minus.evaluate(y)
            pn_minus_x = self._pn_minus.evaluate(x)
            pn_y = self._pn.evaluate(y)
            numerator = pn_x * pn_minus_y - pn_minus_x * pn_y
            return (factor * numerator) / (x - y)


def sympy_to_exact(val: Any) -> Any:
    """Converts a value to flint.fmpq or sympy Rational/float."""
    if isinstance(val, flint.fmpq):
        return val
    if isinstance(val, int):
        return flint.fmpq(val)
    if isinstance(val, float):
        num, den = val.as_integer_ratio()
        return flint.fmpq(num, den)
    try:
        from .utils.conversion import sympy_to_fmpq

        return sympy_to_fmpq(sp.sympify(val))
    except Exception:
        return val


def gap_probability_discrete(kernel: BaseKernel, points_in_gap: Sequence[Any]) -> Any:
    r"""Computes exact gap probability over a discrete state space subset: $\det(I - K_I)$."""
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
    r"""Approximates the continuous Fredholm determinant gap probability over $[a, b]$ using Nyström discretization."""
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


def _validated_discrete_eigensystem(
    kernel: Union[BaseKernel, NDArray[Any]], state_space: Sequence[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate a numerical correlation kernel before any random draws."""
    M = len(state_space)
    if len(set(state_space)) != M:
        raise ValueError("state_space must contain distinct states.")
    if M == 0:
        if isinstance(kernel, np.ndarray) and kernel.shape != (0, 0):
            raise ValueError("Kernel shape must match state_space.")
        return np.empty((0, 0)), np.empty(0), np.empty((0, 0))

    if isinstance(kernel, np.ndarray):
        K_mat = kernel
    else:
        K_mat = np.array(
            [[float(kernel(x, y)) for y in state_space] for x in state_space]
        )

    if np.iscomplexobj(K_mat):
        raise ValueError("Kernel must be real symmetric.")
    K_mat = np.asarray(K_mat, dtype=np.float64)
    if K_mat.shape != (M, M):
        raise ValueError("Kernel shape must match state_space.")
    if not np.all(np.isfinite(K_mat)):
        raise ValueError("Kernel entries must be finite.")
    tolerance = 1e-10
    if not np.allclose(K_mat, K_mat.T, rtol=0.0, atol=tolerance):
        raise ValueError("Kernel must be real symmetric.")
    K_mat = (K_mat + K_mat.T) / 2

    # Eigen-decomposition for projection component selection
    eigenvalues, eigenvectors = np.linalg.eigh(K_mat)
    if np.any(eigenvalues < -tolerance) or np.any(eigenvalues > 1 + tolerance):
        raise ValueError("Kernel eigenvalues must lie in [0, 1].")
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    return K_mat, eigenvalues, eigenvectors


def _sample_discrete_eigensystem(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    state_space: Sequence[Any],
    rng: Optional[np.random.Generator],
) -> List[Any]:
    """Use the same Bernoulli selection and projection HKPV for both APIs."""
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")
    random_draw = np.random.rand if rng is None else rng.random
    choice = np.random.choice if rng is None else rng.choice
    M = len(state_space)

    selected_indices = []
    for idx, lam in enumerate(eigenvalues):
        if random_draw() < lam:
            selected_indices.append(idx)

    if not selected_indices:
        return []

    V_mat = eigenvectors[:, selected_indices].T  # shape (k, M)
    k = len(selected_indices)
    sampled_indices: list[int] = []

    for i in range(k, 0, -1):
        probs = np.sum(V_mat**2, axis=0) / i
        probs[sampled_indices] = 0
        probs = np.clip(probs, 0, None)
        total_prob = np.sum(probs)
        if total_prob > 1e-12:
            probs /= total_prob
        else:
            raise RuntimeError("Projection sampling lost its orthonormal basis.")

        sampled_idx = int(choice(M, p=probs))
        sampled_indices.append(sampled_idx)

        if i > 1:
            best_v_idx = np.argmax(np.abs(V_mat[:, sampled_idx]))
            v_star = V_mat[best_v_idx]

            # Delete the chosen eigenvector row
            V_remaining = np.delete(V_mat, best_v_idx, axis=0)

            # Vectorized projection:
            factors = V_remaining[:, sampled_idx] / v_star[sampled_idx]
            V_updated = V_remaining - factors[:, None] * v_star

            # Orthonormalize rows using QR decomposition
            Q, _ = np.linalg.qr(V_updated.T)
            V_mat = Q.T

    return [state_space[idx] for idx in sampled_indices]


class PreparedDiscreteDPP:
    """Owned, validated kernel state for repeated real symmetric DPP sampling.

    Preparation evaluates/copies the kernel and computes its eigensystem once.
    Stored arrays have immutable backing; public arrays are read-only copies.
    The state container is copied to a tuple, retaining the supplied hashable
    labels. Mutating the original matrix or state list cannot change sampling.
    There is no cache keyed by mutable array identity. Preparation draws no
    random numbers; sample() uses the ordinary Bernoulli and QR-based HKPV law.
    """

    __slots__ = ("_matrix", "_eigenvalues", "_eigenvectors", "_states")

    def __init__(
        self, kernel: Union[BaseKernel, NDArray[Any]], state_space: Sequence[Any]
    ) -> None:
        states = tuple(state_space)
        matrix, eigenvalues, eigenvectors = _validated_discrete_eigensystem(
            kernel, states
        )

        def freeze(array: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)

        self._matrix = freeze(matrix)
        self._eigenvalues = freeze(eigenvalues)
        self._eigenvectors = freeze(eigenvectors)
        self._states = states

    @property
    def state_space(self) -> tuple[Any, ...]:
        """The owned state order; individual labels retain their original identity."""
        return self._states

    @property
    def kernel_matrix(self) -> NDArray[np.float64]:
        """Return a caller-owned read-only copy of the symmetrized kernel."""
        result = self._matrix.copy()
        result.setflags(write=False)
        return result

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Return a caller-owned read-only copy of the clipped eigenvalues."""
        result = self._eigenvalues.copy()
        result.setflags(write=False)
        return result

    def sample(self, *, rng: Optional[np.random.Generator] = None) -> List[Any]:
        """Draw one sample; None preserves the legacy global NumPy RNG behavior."""
        return _sample_discrete_eigensystem(
            self._eigenvalues, self._eigenvectors, self._states, rng
        )


def sample_discrete(
    kernel: Union[BaseKernel, NDArray[Any], PreparedDiscreteDPP],
    state_space: Optional[Sequence[Any]] = None,
    *,
    rng: Optional[np.random.Generator] = None,
) -> List[Any]:
    """Sample a real symmetric DPP correlation kernel by spectral HKPV.

    Raw kernels are validated and decomposed on every call. PreparedDiscreteDPP
    owns a validated snapshot for repeated sampling; its state_space can be
    omitted or must match its stored order. Nonprojection kernels use independent
    Bernoulli eigenvector selection, followed by the same QR-based projection
    sampler. None uses global NumPy randomness; rng accepts a Generator.
    An ndarray is indexed in state_space order; a BaseKernel is evaluated on
    those states. Kernels must be finite, symmetric, with spectrum in [0, 1],
    up to 1e-10 absolute roundoff. States must be distinct and hashable.
    """
    if isinstance(kernel, PreparedDiscreteDPP):
        if state_space is not None and tuple(state_space) != kernel.state_space:
            raise ValueError("state_space must match the prepared kernel's state order")
        return kernel.sample(rng=rng)
    if state_space is None:
        raise ValueError("state_space is required for an unprepared kernel")
    states = tuple(state_space)
    _, eigenvalues, eigenvectors = _validated_discrete_eigensystem(kernel, states)
    return _sample_discrete_eigensystem(eigenvalues, eigenvectors, states, rng)
