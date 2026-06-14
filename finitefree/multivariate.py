import functools
import math
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from .hyperbolic import MultiplicativeMatrixPencil, SymmetricMatrixPencil
from .utils.modular import crt, prime_generator
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


from .core import Polynomial


class MultivariatePolynomial(Polynomial):
    """
    Represents a homogeneous multivariate polynomial exactly using Flint's fmpq_mpoly
    as the primary computational backend.
    """

    def evaluate(self, x: Sequence[Any]) -> Any:
        """Evaluates the multivariate polynomial at a point x = (x_1, ..., x_m)."""
        from .utils.conversion import sympy_to_fmpq

        if len(x) != len(self.variables):
            raise ValueError(f"Expected {len(self.variables)} values, got {len(x)}")
        x_fmpq = [sympy_to_fmpq(xi) for xi in x]
        return self._mpoly.evaluate(x_fmpq)  # type: ignore[attr-defined, unused-ignore]

    @property
    def variables(self) -> list[sp.Symbol]:
        return self._variables

    def __init__(self, expr: Any, variables: Sequence[sp.Symbol]) -> None:
        import flint

        from .utils.conversion import sympy_to_fmpq

        self._variables = list(variables)
        names = tuple(x.name for x in self._variables)
        self._ctx = flint.fmpq_mpoly_ctx.get(names=names)

        if isinstance(expr, flint.fmpq_mpoly):
            self._mpoly = expr
        else:
            poly_sym = sp.Poly(sp.expand(sp.sympify(expr)), self._variables)
            flint_dict = {}
            for exp, c in poly_sym.as_dict().items():
                flint_dict[exp] = sympy_to_fmpq(c)
            self._mpoly = self._ctx.from_dict(flint_dict)

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
        """
        Verifies if the polynomial is homogeneous.
        P(t x_1, ..., t x_m) == t^d P(x_1, ..., x_m)
        """
        d = self.degree()
        return all(sum(alpha) == d for alpha in self._mpoly.monoms())

    def directional_derivative(
        self, direction: Sequence[Any]
    ) -> "MultivariatePolynomial":
        """
        Computes the directional derivative of the polynomial:
        D_e P(x) = <e, grad P(x)> = sum e_i * dP/dx_i
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
        self, orders: Sequence[int]
    ) -> "MultivariatePolynomial":
        """
        Computes mixed partial derivatives exactly and efficiently by
        performing C-level differentiation.
        """
        if len(orders) != len(self.variables):
            raise ValueError(
                f"Orders length ({len(orders)}) must match "
                f"variable count ({len(self.variables)})."
            )

        res = self._mpoly
        for i, ord_val in enumerate(orders):
            for _ in range(ord_val):
                res = res.derivative(i)

        return MultivariatePolynomial(res, self.variables)

    def normalized_coefficients(self) -> Dict[Tuple[int, ...], Any]:
        """
        Extracts the normalized coefficients:
        \\tilde{c}_\\alpha = c_\\alpha / \\binom{d}{\\alpha}
        where \\binom{d}{\\alpha} is the multinomial coefficient.
        """
        import flint

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
        """
        Returns the compiled C-level fmpq_mpoly sparse polynomial.
        """
        return self._mpoly

    @classmethod
    def from_symmetric_matrix_pencil_interpolated(
        cls,
        pencil: Union[SymmetricMatrixPencil, MultiplicativeMatrixPencil],
        parallel: bool = False,
        backend: str = "threads",
    ) -> "MultivariatePolynomial":
        """
        Constructs the multivariate polynomial det(x_1 A_1 + ... + x_m A_m)
        by evaluating the determinants modulo prime numbers exactly using fast
        C-level modular matrix mathematics and reconstructing exact coefficients
        over Q using the Chinese Remainder Theorem (CRT) and Rational Reconstruction.
        """
        import math

        import flint

        n = pencil.n
        m = pencil.m
        variables = [sp.Symbol(f"x{i}") for i in range(1, m + 1)]

        if m == 1:
            exact_A = []
            for r in range(n):
                row = []
                for c in range(n):
                    val = sp.sympify(pencil.matrices[0][r, c])
                    if isinstance(val, (int, float, np.number)):
                        val = sp.Rational(val)
                    row.append(val)
                exact_A.append(row)
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
        """
        Constructs the multivariate polynomial det(x_1 A_1 + ... + x_m A_m)
        by evaluating the determinants modulo prime numbers exactly using fast
        C-level modular matrix mathematics and reconstructing exact coefficients
        over Q using Zippel's sparse interpolation algorithm.
        """
        import math
        import random

        import flint

        n = pencil.n
        m = pencil.m
        variables = [sp.Symbol(f"x{i}") for i in range(1, m + 1)]

        if m == 1:
            exact_A = []
            for r in range(n):
                row = []
                for c in range(n):
                    val = sp.sympify(pencil.matrices[0][r, c])
                    if isinstance(val, (int, float, np.number)):
                        val = sp.Rational(val)
                    row.append(val)
                exact_A.append(row)
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
                        matrices_np = np.array(integer_matrices, dtype=np.int64)
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
        """
        Constructs the multivariate polynomial det(x_1 A_1 + ... + x_m A_m)
        using exact arithmetic. Uses exact grid-based rational polynomial
        interpolation for large pencils (n >= 4) to bypass the exponential
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
        for xi, Ai in zip(variables, pencil.matrices):
            for r in range(n):
                for c in range(n):
                    # sympify as Rational to preserve exact integers/rationals
                    val = sp.sympify(Ai[r, c])
                    if isinstance(val, (int, float, np.number)):
                        val = sp.Rational(val)
                    M[r, c] += xi * val

        expr = M.berkowitz_det()
        return cls(expr, variables)
