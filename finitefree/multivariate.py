import functools
import math
import operator
from typing import Any, Dict, Mapping, Optional, Sequence, SupportsIndex, Tuple, Union

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .hyperbolic import MultiplicativeMatrixPencil, SymmetricMatrixPencil
from .utils.modular import _as_modular_array, crt, prime_generator
from .utils.parallel import ParallelScheduler, _eval_prime_worker


@functools.lru_cache(maxsize=None)
def get_monomial_exponents(dim: int, deg: int) -> list[tuple[int, ...]]:
    if dim == 1:
        return [(deg,)]
    if deg == 0:
        return [(0,) * dim]
    exps = []
    for power in range(deg + 1):
        for rem in get_monomial_exponents(dim - 1, deg - power):
            exps.append((power,) + rem)
    return exps


@functools.lru_cache(maxsize=None)
def get_monomials(vars_tuple: tuple[sp.Symbol, ...], deg: int) -> list[sp.Expr]:
    if len(vars_tuple) == 1:
        return [vars_tuple[0] ** deg]
    if deg == 0:
        return [sp.Integer(1)]
    monoms = []
    for power in range(deg + 1):
        rem_monoms = get_monomials(vars_tuple[1:], deg - power)
        for rm in rem_monoms:
            monoms.append((vars_tuple[0] ** power) * rm)
    return monoms


@functools.lru_cache(maxsize=None)
def get_grid_points(
    dim: int, grid_vals_tuple: tuple[int, ...]
) -> list[tuple[int, ...]]:
    if dim == 1:
        return [(val,) for val in grid_vals_tuple]
    pts = []
    for val in grid_vals_tuple:
        for rest in get_grid_points(dim - 1, grid_vals_tuple):
            pts.append((val,) + rest)
    return pts


from .core import Polynomial, RealRootedPolynomial


class MultivariatePolynomial(Polynomial):
    r"""
    Store a rational multivariate polynomial in FLINT's sparse fmpq_mpoly.
    Homogeneous geometry operations are available, but construction does not
    require homogeneity; use is_homogeneous() when that assumption is needed.
    Inputs must be convertible to rational coefficients.
    Variable order is explicit. Coefficients and exported native objects are owned
    copies. Algebra does not certify stability, hyperbolicity or real-rootedness.
    """

    _mpoly: Any
    _float_terms_cached: Optional[Tuple[Tuple[Tuple[int, ...], float], ...]]

    def evaluate(self, x: Sequence[Any]) -> Any:
        r"""
        Evaluate at a rational point $x = (x_1, \dots, x_m)$.
        Floating coordinates retain their stored binary ratios. The result is an
        exact FLINT rational; use evaluate_float64 for approximate batch evaluation.
        """
        from .utils.conversion import sympy_to_fmpq

        if len(x) != len(self.variables):
            raise ValueError(f"Expected {len(self.variables)} values, got {len(x)}")
        x_fmpq = [sympy_to_fmpq(xi) for xi in x]
        return self._mpoly(*x_fmpq)

    def evaluate_float64(self, points: Any) -> Union[float, NDArray[np.float64]]:
        """Evaluate real points shaped (..., variable_count) with NumPy.

        A single point returns a float; batches retain their leading dimensions.
        Coefficients are converted once, and terms are evaluated across each batch.
        Nonfinite inputs, coefficients or results raise errors. Finite cancellation,
        underflow and rounding remain possible; outputs have no certified error bound.
        """
        if np.iscomplexobj(points):
            raise ValueError("Numerical coordinates must be real.")
        try:
            coordinates = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Numerical coordinates must be finite real values."
            ) from error
        if coordinates.ndim == 0 or coordinates.shape[-1] != len(self._variables):
            raise ValueError(
                f"Expected final coordinate dimension {len(self._variables)}."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Numerical coordinates must be finite.")
        if self._float_terms_cached is None:
            from .utils.conversion import flint_to_float

            terms = tuple(
                (tuple(int(k) for k in alpha), flint_to_float(c))
                for alpha, c in self._mpoly.to_dict().items()
            )
            if not all(math.isfinite(c) for _, c in terms):
                raise RuntimeError("Polynomial coefficients exceed the float64 range.")
            self._float_terms_cached = terms
        result = np.zeros(coordinates.shape[:-1], dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for alpha, coefficient in self._float_terms_cached:
                    term = np.full(result.shape, coefficient, dtype=np.float64)
                    for i, exponent in enumerate(alpha):
                        if exponent:
                            term *= coordinates[..., i] ** exponent
                    result += term
        except FloatingPointError as error:
            raise RuntimeError(
                "Numerical polynomial evaluation is nonfinite."
            ) from error
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Numerical polynomial evaluation is nonfinite.")
        return float(result) if result.ndim == 0 else result

    @property
    def variables(self) -> list[sp.Symbol]:
        """Return a caller-owned list in the stored coordinate order."""
        return list(self._variables)

    def __init__(self, expr: Any, variables: Sequence[sp.Symbol]) -> None:
        import flint

        from .utils.conversion import sympy_to_fmpq

        self._variables = tuple(variables)
        if not self._variables or not all(
            isinstance(x, sp.Symbol) for x in self._variables
        ):
            raise ValueError("Variables must be a nonempty sequence of SymPy symbols.")
        names = tuple(x.name for x in self._variables)
        if len(set(names)) != len(names):
            raise ValueError("Variable names must be distinct.")
        self._ctx = flint.fmpq_mpoly_ctx.get(names=names)
        self._float_terms_cached = None

        if isinstance(expr, flint.fmpq_mpoly):
            if expr.context().names() != names:
                raise ValueError("Native polynomial context must match variable order.")
            self._mpoly = (
                expr + 0
                if expr.context() == self._ctx
                else self._ctx.from_dict(expr.to_dict())
            )
        else:
            poly_sym = sp.Poly(sp.expand(sp.sympify(expr)), self._variables)
            flint_dict = {}
            for exp, c in poly_sym.as_dict().items():
                flint_dict[exp] = sympy_to_fmpq(c)
            self._mpoly = self._ctx.from_dict(flint_dict)

    @classmethod
    def from_coefficients(
        cls, coefficients: Mapping[Tuple[int, ...], Any], variables: Sequence[sp.Symbol]
    ) -> "MultivariatePolynomial":
        """Construct from sparse nonnegative exponent tuples and rational values."""
        from .utils.conversion import sympy_to_fmpq

        result = cls(0, variables)
        terms = {
            result._nonnegative_indices(alpha, "Exponent"): sympy_to_fmpq(c)
            for alpha, c in coefficients.items()
        }
        result._mpoly = result._ctx.from_dict(terms)
        return result

    def coefficients(self) -> Dict[Tuple[int, ...], sp.Rational]:
        """Return an owned sparse mapping of exponent tuples to exact coefficients."""
        return {
            tuple(int(k) for k in alpha): sp.Rational(int(c.p), int(c.q))
            for alpha, c in self._mpoly.to_dict().items()
        }

    def _nonnegative_indices(
        self, values: Sequence[SupportsIndex], label: str
    ) -> Tuple[int, ...]:
        if len(values) != len(self._variables):
            raise ValueError(f"{label} count must match variable count.")
        indices = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{label} values must be integers, not booleans.")
            try:
                index = operator.index(value)
            except TypeError as error:
                raise TypeError(f"{label} values must be integers.") from error
            if index < 0:
                raise ValueError(f"{label} values must be nonnegative.")
            indices.append(index)
        return tuple(indices)

    def _coerce_operand(self, other: Any) -> Any:
        if isinstance(other, MultivariatePolynomial):
            if other._variables != self._variables:
                raise ValueError("Polynomial variable sequences must match exactly.")
            return other._mpoly
        from .utils.conversion import sympy_to_fmpq

        return sympy_to_fmpq(other)

    def __add__(self, other: Any) -> "MultivariatePolynomial":
        return type(self)(self._mpoly + self._coerce_operand(other), self._variables)

    def __radd__(self, other: Any) -> "MultivariatePolynomial":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "MultivariatePolynomial":
        return type(self)(self._mpoly - self._coerce_operand(other), self._variables)

    def __rsub__(self, other: Any) -> "MultivariatePolynomial":
        return type(self)(self._coerce_operand(other) - self._mpoly, self._variables)

    def __mul__(self, other: Any) -> "MultivariatePolynomial":
        return type(self)(self._mpoly * self._coerce_operand(other), self._variables)

    def __rmul__(self, other: Any) -> "MultivariatePolynomial":
        return self.__mul__(other)

    def __neg__(self) -> "MultivariatePolynomial":
        return type(self)(-self._mpoly, self._variables)

    def __pow__(self, exponent: SupportsIndex) -> "MultivariatePolynomial":
        """Raise to a nonnegative integer power within the rational polynomial ring."""
        if isinstance(exponent, (bool, np.bool_)):
            raise TypeError("Exponent must be an integer, not a boolean.")
        power = operator.index(exponent)
        if power < 0:
            raise ValueError("Exponent must be nonnegative.")
        return type(self)(self._mpoly**power, self._variables)

    @property
    def expr(self) -> sp.Expr:
        """Returns the polynomial as a SymPy expression."""
        flint_dict = self._mpoly.to_dict()
        res_expr = sp.Integer(0)
        for exp, c in flint_dict.items():
            term = sp.Rational(int(c.p), int(c.q))
            for x_i, power in zip(self.variables, exp):
                if power > 0:
                    term *= x_i**power
            res_expr += term
        return res_expr

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.expr})"

    def __repr__(self) -> str:
        return self.__str__()

    def degree(self) -> int:
        """Returns the total degree of the multivariate polynomial."""
        return int(self._mpoly.total_degree())

    def is_homogeneous(self) -> bool:
        r"""
        Verifies if the polynomial is homogeneous.
        $P(t x_1, \dots, t x_m) = t^d P(x_1, \dots, x_m)$
        """
        d = self.degree()
        return all(sum(alpha) == d for alpha in self._mpoly.monoms())

    def directional_derivative(
        self, direction: Sequence[Any]
    ) -> "MultivariatePolynomial":
        r"""
        Computes the directional derivative of the polynomial:
        $D_e P(x) = \langle e, \nabla P(x) \rangle = \sum e_i \frac{\partial P}{\partial x_i}$
        """
        if len(direction) != len(self.variables):
            raise ValueError(
                f"Direction length ({len(direction)}) must match "
                f"variable count ({len(self.variables)})."
            )

        from .utils.conversion import sympy_to_fmpq

        deriv_poly = self._ctx.from_dict({})
        for i, e_i in enumerate(direction):
            if e_i != 0:
                deriv_poly += self._mpoly.derivative(i) * sympy_to_fmpq(e_i)

        return MultivariatePolynomial(deriv_poly, self.variables)

    def mixed_partial_derivative(
        self, orders: Sequence[SupportsIndex]
    ) -> "MultivariatePolynomial":
        """
        Computes mixed partial derivatives exactly and efficiently by
        performing C-level differentiation.
        """
        indices = self._nonnegative_indices(orders, "Derivative order")
        if sum(indices) > self.degree():
            return MultivariatePolynomial(0, self.variables)

        res = self._mpoly
        for i, ord_val in enumerate(indices):
            for _ in range(ord_val):
                res = res.derivative(i)

        return MultivariatePolynomial(res, self.variables)

    def partial_derivative(
        self, variable: sp.Symbol, order: SupportsIndex = 1
    ) -> "MultivariatePolynomial":
        """Differentiate in a stored variable with a nonnegative integer order."""
        if variable not in self._variables:
            raise ValueError("Derivative variable must be in the stored variables.")
        if isinstance(order, (bool, np.bool_)):
            raise TypeError("Derivative order must be an integer, not a boolean.")
        orders = [0] * len(self._variables)
        orders[self._variables.index(variable)] = operator.index(order)
        return self.mixed_partial_derivative(orders)

    def gradient(self) -> list["MultivariatePolynomial"]:
        """Return exact polynomial derivatives in stored variable order."""
        return [self.partial_derivative(x) for x in self._variables]

    def hessian(self) -> list[list["MultivariatePolynomial"]]:
        """Return exact second polynomial derivatives, including at singular points."""
        return [
            [p.partial_derivative(y) for y in self._variables] for p in self.gradient()
        ]

    def restrict_line(
        self, base_point: Sequence[Any], direction: Sequence[Any]
    ) -> "RealRootedPolynomial":
        """Form P(base_point + t*direction) exactly, retaining its leading scalar.

        The returned univariate object certifies real-rootedness lazily. A generic
        restriction can have complex roots; no hyperbolicity assumption is inferred.
        Identically zero restrictions raise ValueError because the univariate class
        cannot represent the zero polynomial.
        """
        import flint

        from .utils.conversion import sympy_to_fmpq

        if len(base_point) != len(self._variables) or len(direction) != len(
            self._variables
        ):
            raise ValueError("Line coordinates must match variable count.")
        factors = [
            flint.fmpq_poly([sympy_to_fmpq(a), sympy_to_fmpq(b)])
            for a, b in zip(base_point, direction)
        ]
        result = flint.fmpq_poly([])
        for alpha, coefficient in self._mpoly.to_dict().items():
            term = flint.fmpq_poly([coefficient])
            for factor, exponent in zip(factors, alpha):
                if exponent:
                    term *= factor**exponent
            result += term
        if result.degree() < 0:
            raise ValueError("Line restriction is identically zero.")
        return RealRootedPolynomial(result, monic=False)

    def normalized_coefficients(self) -> Dict[Tuple[int, ...], Any]:
        r"""
        Extracts the normalized coefficients:
        $\tilde{c}_\alpha = c_\alpha / \binom{d}{\alpha}$
        where $\binom{d}{\alpha}$ is the multinomial coefficient.
        """
        import flint

        if not self.is_homogeneous():
            raise ValueError(
                "Normalized coefficients require a homogeneous polynomial."
            )
        d = self.degree()

        def multinomial_coeff(total: int, alpha: Tuple[int, ...]) -> int:
            num = math.factorial(total)
            den = 1
            for a in alpha:
                den *= math.factorial(a)
            return num // den

        normalized = {}
        for alpha, c in self._mpoly.to_dict().items():
            weight = multinomial_coeff(d, alpha)
            val = c / flint.fmpq(weight, 1)
            normalized[alpha] = sp.Rational(int(val.p), int(val.q))

        return normalized

    def to_fmpq_mpoly(self) -> Any:
        r"""
        Return a caller-owned copy of the FLINT sparse polynomial.
        Evaluation/substitution costs depend on monomials, degrees and coefficient
        sizes; exposing the object does not make those operations constant-time.
        """
        return self._mpoly + 0

    @classmethod
    def from_symmetric_matrix_pencil_interpolated(
        cls,
        pencil: Union[SymmetricMatrixPencil, MultiplicativeMatrixPencil],
        parallel: bool = False,
        backend: str = "threads",
    ) -> "MultivariatePolynomial":
        r"""
        Constructs the multivariate polynomial $\det(x_1 A_1 + \dots + x_m A_m)$
        by evaluating the determinants modulo prime numbers exactly using fast
        C-level modular matrix mathematics and reconstructing exact coefficients
        over $\mathbb{Q}$ using the Chinese Remainder Theorem (CRT) and Rational Reconstruction.
        """
        import math

        import flint

        n = pencil.n
        m = pencil.m
        variables = [sp.Symbol(f"x{i}") for i in range(1, m + 1)]

        if m == 1:
            exact_A = pencil._get_matrices_sympy()[0]
            det_val = sp.Matrix(exact_A).det()
            expr = det_val * (variables[0] ** n)
            return cls(expr, variables)

        exps = get_monomial_exponents(m, n)
        N = len(exps)

        grid_vals = tuple(range(n + 1))
        full_grid_pts = get_grid_points(m - 1, grid_vals)

        # Convert all matrices to exact Rational representation to clear denominators
        exact_matrices = pencil._get_matrices_sympy()

        # Find the global common denominator D
        denominators = []
        for exact_A in exact_matrices:
            for r in range(n):
                for c in range(n):
                    val = exact_A[r][c]
                    if isinstance(val, sp.Rational):
                        denominators.append(val.q)
                    else:
                        denominators.append(1)

        D = 1
        for den in denominators:
            D = (D * den) // math.gcd(D, den)

        # Construct integer matrices A_prime
        integer_matrices = []
        for exact_A in exact_matrices:
            int_A = []
            for r in range(n):
                row = []
                for c in range(n):
                    val = exact_A[r][c] * D
                    row.append(int(val))
                int_A.append(row)
            integer_matrices.append(int_A)

        primes_gen = prime_generator(1000000007)
        reconstructed = None
        primes_used = []
        coeffs_by_prime = []

        effective_backend = backend if parallel else "sequential"
        if pencil.n <= 3:
            effective_backend = "sequential"

        with ParallelScheduler(backend=effective_backend) as scheduler:
            batch_size = max(4, scheduler.max_workers)
            while True:
                batch_primes = [next(primes_gen) for _ in range(batch_size)]
                results = scheduler.evaluate(
                    _eval_prime_worker,
                    batch_primes,
                    integer_matrices,
                    full_grid_pts,
                    n,
                    m,
                    exps,
                )
                for p_res, c_p in results:
                    if c_p is not None:
                        coeffs_by_prime.append(c_p)
                        primes_used.append(p_res)

                if len(primes_used) >= 2:
                    current_reconstruction = []
                    for i in range(N):
                        vals = [coeffs_by_prime[k][i] for k in range(len(primes_used))]
                        current_reconstruction.append(crt(vals, primes_used))

                    if (
                        reconstructed is not None
                        and current_reconstruction == reconstructed
                    ):
                        reconstructed = current_reconstruction
                        break
                    reconstructed = current_reconstruction

        names = tuple(x.name for x in variables)
        ctx = flint.fmpq_mpoly_ctx.get(names=names)
        flint_dict = {}
        denom_scale = D**n
        for exp, val in zip(exps, reconstructed):
            if val % denom_scale == 0:
                flint_dict[exp] = flint.fmpq(val // denom_scale, 1)
            else:
                flint_dict[exp] = flint.fmpq(val, denom_scale)

        poly = ctx.from_dict(flint_dict)
        return cls(poly, variables)

    @classmethod
    def from_symmetric_matrix_pencil_sparse(
        cls, pencil: Union[SymmetricMatrixPencil, MultiplicativeMatrixPencil]
    ) -> "MultivariatePolynomial":
        r"""
        Constructs the multivariate polynomial $\det(x_1 A_1 + \dots + x_m A_m)$
        by evaluating the determinants modulo prime numbers exactly using fast
        C-level modular matrix mathematics and reconstructing exact coefficients
        over $\mathbb{Q}$ using Zippel's sparse interpolation algorithm.
        """
        import math
        import random

        import flint

        n = pencil.n
        m = pencil.m
        variables = [sp.Symbol(f"x{i}") for i in range(1, m + 1)]

        if m == 1:
            exact_A = pencil._get_matrices_sympy()[0]
            det_val = sp.Matrix(exact_A).det()
            expr = det_val * (variables[0] ** n)
            return cls(expr, variables)

        exact_matrices = pencil._get_matrices_sympy()

        denominators = []
        for exact_A in exact_matrices:
            for r in range(n):
                for c in range(n):
                    val = exact_A[r][c]
                    if isinstance(val, sp.Rational):
                        denominators.append(val.q)
                    else:
                        denominators.append(1)

        D = 1
        for den in denominators:
            D = (D * den) // math.gcd(D, den)

        integer_matrices = []
        for exact_A in exact_matrices:
            int_A = []
            for r in range(n):
                row = []
                for c in range(n):
                    val = exact_A[r][c] * D
                    row.append(int(val))
                int_A.append(row)
            integer_matrices.append(int_A)

        def eval_point_mod_p(pt: tuple[int, ...], p: int) -> int:
            M_pt = flint.nmod_mat(n, n, p)
            for r in range(n):
                for c in range(n):
                    val = 0
                    for pt_val, int_A in zip(pt, integer_matrices):
                        val = (val + pt_val * int_A[r][c]) % p
                    M_pt[r, c] = val
            return int(M_pt.det())

        def zippel_mod_p(p: int) -> dict[tuple[int, ...], int]:
            rand_gen = random.Random(42)
            S: dict[tuple[int, ...], int] = {(): 1}

            for i in range(1, m):
                t = [rand_gen.randint(2, p - 2) for _ in range(m - 1 - i)]
                candidates = []
                for beta in S:
                    sum_beta = sum(beta)
                    for j in range(n - sum_beta + 1):
                        candidates.append(beta + (j,))

                K = len(candidates)
                if K == 0:
                    break

                solved = False
                attempts = 0
                while not solved and attempts < 5:
                    attempts += 1
                    test_pts = []
                    for _ in range(K):
                        test_pts.append(
                            tuple(rand_gen.randint(2, p - 2) for _ in range(i))
                        )

                    try:
                        import os

                        if os.environ.get("PYFFP_DISABLE_CYTHON") == "1":
                            raise ImportError(
                                "Cython explicitly disabled via environment variable"
                            )
                        import numpy as np  # noqa: I001
                        from .utils.modular_fast import (
                            construct_zippel_vandermonde_mod_p,
                        )  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: I001

                        test_pts_np = np.array(test_pts, dtype=np.int64)
                        candidates_np = np.array(candidates, dtype=np.int64)
                        V_memview = construct_zippel_vandermonde_mod_p(
                            test_pts_np, candidates_np, p
                        )
                        V_np = np.array(V_memview, dtype=np.int64)
                        V_flat = V_np.flatten().tolist()
                        V = flint.nmod_mat(K, K, V_flat, p)
                    except ImportError:
                        V = flint.nmod_mat(K, K, p)
                        for r_idx in range(K):
                            pt_val = test_pts[r_idx]
                            for c_idx in range(K):
                                exp = candidates[c_idx]
                                term = 1
                                for val, power in zip(pt_val, exp):
                                    term = (term * pow(val, power, p)) % p
                                V[r_idx, c_idx] = term

                    y = flint.nmod_mat(K, 1, p)
                    try:
                        import os

                        if os.environ.get("PYFFP_DISABLE_CYTHON") == "1":
                            raise ImportError(
                                "Cython explicitly disabled via environment variable"
                            )
                        import numpy as np  # noqa: I001

                        from .utils.modular_fast import eval_points_grid_mod_p  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: I001

                        grid_pts = [
                            test_pts[r_idx] + tuple(t) + (1,) for r_idx in range(K)
                        ]
                        grid_pts_np = np.array(grid_pts, dtype=np.int64)
                        matrices_np = _as_modular_array(integer_matrices, p)
                        dets = eval_points_grid_mod_p(matrices_np, grid_pts_np, p)
                        for r_idx in range(K):
                            y[r_idx, 0] = dets[r_idx]
                    except ImportError:
                        for r_idx in range(K):
                            full_pt = test_pts[r_idx] + tuple(t) + (1,)
                            y[r_idx, 0] = eval_point_mod_p(full_pt, p)

                    try:
                        c_flint = V.solve(y)
                        solved = True
                        C = [int(c_flint[r_idx, 0]) for r_idx in range(K)]
                    except Exception:
                        continue

                if not solved:
                    raise ValueError(f"Zippel interpolation failed mod {p}")

                S = {}
                for coeff, exp in zip(C, candidates):
                    if coeff != 0:
                        S[exp] = coeff

            return S

        primes_gen = prime_generator(1000000007)
        reconstructed = None
        primes_used = []
        coeffs_by_prime = []
        exps = []

        while True:
            p = next(primes_gen)
            try:
                S_p = zippel_mod_p(p)
            except ValueError:
                continue

            for exp in S_p:
                if exp not in exps:
                    exps.append(exp)

            coeffs_by_prime.append(S_p)
            primes_used.append(p)

            if len(primes_used) >= 2:
                current_reconstruction = {}
                for exp in exps:
                    vals = []
                    for k in range(len(primes_used)):
                        vals.append(coeffs_by_prime[k].get(exp, 0))
                    current_reconstruction[exp] = crt(vals, primes_used)

                if (
                    reconstructed is not None
                    and current_reconstruction == reconstructed
                ):
                    reconstructed = current_reconstruction
                    break
                reconstructed = current_reconstruction

        names = tuple(x.name for x in variables)
        ctx = flint.fmpq_mpoly_ctx.get(names=names)
        flint_dict = {}
        denom_scale = D**n
        for exp, val in reconstructed.items():
            full_exp = list(exp) + [n - sum(exp)]
            if val % denom_scale == 0:
                flint_dict[tuple(full_exp)] = flint.fmpq(val // denom_scale, 1)
            else:
                flint_dict[tuple(full_exp)] = flint.fmpq(val, denom_scale)

        poly = ctx.from_dict(flint_dict)
        return cls(poly, variables)

    @classmethod
    def from_symmetric_matrix_pencil(
        cls, pencil: Union[SymmetricMatrixPencil, MultiplicativeMatrixPencil]
    ) -> "MultivariatePolynomial":
        r"""
        Constructs the multivariate polynomial $\det(x_1 A_1 + \dots + x_m A_m)$
        using exact arithmetic. Uses exact grid-based rational polynomial
        interpolation for large pencils ($n \ge 4$) to bypass the exponential
        symbolic determinant bottleneck, and direct Berkowitz determinant
        for small pencils.
        """
        if pencil.n >= 4:
            return cls.from_symmetric_matrix_pencil_interpolated(pencil)

        # Create symbols: x0, x1, ..., x(m-1)
        variables = [sp.Symbol(f"x{i}") for i in range(1, pencil.m + 1)]
        n = pencil.n

        # Construct symbolic matrix
        M = sp.zeros(n, n)
        for xi, Ai in zip(variables, pencil._get_matrices_sympy()):
            for r in range(n):
                for c in range(n):
                    val = Ai[r][c]
                    M[r, c] += xi * val

        expr = M.berkowitz_det()
        return cls(expr, variables)
