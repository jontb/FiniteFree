import math
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from .hyperbolic import MultiplicativeMatrixPencil, SymmetricMatrixPencil


class MultivariatePolynomial:
    """
    Represents a multivariate homogeneous polynomial P(x_1, ..., x_m) exactly
    using SymPy as the symbolic backend.
    """

    def __init__(
        self, expr: Union[sp.Expr, Any], variables: Sequence[sp.Symbol]
    ) -> None:
        self.expr = sp.expand(sp.sympify(expr))
        self.variables = list(variables)
        self._degree = int(sp.total_degree(self.expr, *self.variables))

    def degree(self) -> int:
        """Returns the total degree of the multivariate polynomial."""
        return self._degree

    def is_homogeneous(self) -> bool:
        """
        Verifies if the polynomial is homogeneous.
        P(t x_1, ..., t x_m) == t^d P(x_1, ..., x_m)
        """
        poly = sp.Poly(self.expr, self.variables)
        d = self.degree()
        return all(sum(alpha) == d for alpha in poly.monoms())

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

        deriv_expr = 0
        for e_i, x_i in zip(direction, self.variables):
            deriv_expr += e_i * sp.diff(self.expr, x_i)

        return MultivariatePolynomial(deriv_expr, self.variables)

    def mixed_partial_derivative(
        self, orders: Sequence[int]
    ) -> "MultivariatePolynomial":
        """
        Computes mixed partial derivatives exactly and efficiently by
        performing dictionary-based arithmetic over monomial exponents.
        """
        if len(orders) != len(self.variables):
            raise ValueError(
                f"Orders length ({len(orders)}) must match "
                f"variable count ({len(self.variables)})."
            )

        poly = sp.Poly(self.expr, self.variables)
        coeffs_dict = poly.as_dict()

        new_coeffs = {}
        for alpha, c in coeffs_dict.items():
            vanishes = False
            factor = 1
            new_alpha = []
            for a_val, ord_val in zip(alpha, orders):
                if a_val < ord_val:
                    vanishes = True
                    break
                f = 1
                for k in range(ord_val):
                    f *= a_val - k
                factor *= f
                new_alpha.append(a_val - ord_val)

            if not vanishes:
                new_coeffs[tuple(new_alpha)] = c * factor

        if not new_coeffs:
            deriv_expr = sp.Integer(0)
        else:
            terms = []
            for alpha, coeff in new_coeffs.items():
                factors = []
                for x, a in zip(self.variables, alpha):
                    if a > 0:
                        factors.append(x**a)
                if factors:
                    terms.append(coeff * sp.Mul(*factors))
                else:
                    terms.append(sp.sympify(coeff))
            deriv_expr = sp.Add(*terms)

        return MultivariatePolynomial(deriv_expr, self.variables)

    def normalized_coefficients(self) -> Dict[Tuple[int, ...], Any]:
        """
        Extracts the normalized coefficients:
        \\tilde{c}_\\alpha = c_\\alpha / \\binom{d}{\\alpha}
        where \\binom{d}{\\alpha} is the multinomial coefficient.
        """
        poly = sp.Poly(self.expr, self.variables)
        coeffs_dict = poly.as_dict()
        d = self.degree()

        def multinomial_coeff(total: int, alpha: Tuple[int, ...]) -> int:
            num = math.factorial(total)
            den = 1
            for a in alpha:
                den *= math.factorial(a)
            return num // den

        normalized = {}
        for alpha, c in coeffs_dict.items():
            weight = multinomial_coeff(d, alpha)
            if isinstance(c, (int, np.integer)) and isinstance(weight, int):
                if c % weight == 0:
                    normalized[alpha] = c // weight
                else:
                    normalized[alpha] = sp.Rational(c, weight)
            else:
                # If c is already a sympy Expression, divide exactly
                normalized[alpha] = c / sp.Rational(weight)

        return normalized

    def to_fmpq_mpoly(self) -> Any:
        """
        Converts the SymPy homogeneous multivariate polynomial into a compiled
        C-level fmpq_mpoly sparse polynomial.
        """
        import flint

        if not hasattr(flint, "fmpq_mpoly_ctx"):
            raise NotImplementedError(
                "fmpq_mpoly_ctx is not supported in the installed "
                "version of python-flint."
            )

        names = tuple(x.name for x in self.variables)
        ctx = flint.fmpq_mpoly_ctx.get(names=names)

        poly = sp.Poly(self.expr, self.variables)
        flint_dict = {}
        for exp, c in poly.as_dict().items():
            if isinstance(c, sp.Rational):
                flint_dict[exp] = flint.fmpq(int(c.p), int(c.q))
            else:
                flint_dict[exp] = flint.fmpq(int(c), 1)

        return ctx.from_dict(flint_dict)

    @classmethod
    def from_symmetric_matrix_pencil_interpolated(
        cls,
        pencil: Union[SymmetricMatrixPencil, MultiplicativeMatrixPencil],
        parallel: bool = False,
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

        def get_monomials(vars_list: list[sp.Symbol], deg: int) -> list[sp.Expr]:
            if len(vars_list) == 1:
                return [vars_list[0] ** deg]
            if deg == 0:
                return [sp.Integer(1)]
            monoms = []
            for power in range(deg + 1):
                rem_monoms = get_monomials(vars_list[1:], deg - power)
                for rm in rem_monoms:
                    monoms.append((vars_list[0] ** power) * rm)
            return monoms

        def get_grid_points(dim: int, grid_vals: list[int]) -> list[tuple[int, ...]]:
            if dim == 1:
                return [(val,) for val in grid_vals]
            pts = []
            for val in grid_vals:
                for rest in get_grid_points(dim - 1, grid_vals):
                    pts.append((val,) + rest)
            return pts

        exps = get_monomial_exponents(m, n)
        N = len(exps)

        grid_vals = list(range(n + 1))
        full_grid_pts = get_grid_points(m - 1, grid_vals)

        # Convert all matrices to exact Rational representation to clear denominators
        exact_matrices = []
        for A in pencil.matrices:
            exact_A = []
            for r in range(n):
                row = []
                for c in range(n):
                    val = sp.sympify(A[r, c])
                    if isinstance(val, (int, float, np.number)):
                        val = sp.Rational(val)
                    row.append(val)
                exact_A.append(row)
            exact_matrices.append(exact_A)

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

        # modular determinant grid evaluations mod p
        def prime_generator(start: int = 1000000007) -> Any:
            curr = start
            while True:
                if flint.fmpz(curr).is_probable_prime():
                    yield curr
                curr += 1

        def get_inverse_vandermonde_matrices(
            max_k: int, p: int
        ) -> dict[int, list[list[int]]]:
            invs = {}
            for k in range(1, max_k + 1):
                V = flint.nmod_mat(k, k, p)
                for r in range(k):
                    for c in range(k):
                        V[r, c] = pow(r, c, p)
                identity_matrix = flint.nmod_mat(k, k, p)
                for r in range(k):
                    identity_matrix[r, r] = 1
                try:
                    V_inv = V.solve(identity_matrix)
                    invs[k] = [[int(V_inv[r, c]) for c in range(k)] for r in range(k)]
                except Exception as e:
                    raise ValueError(
                        f"Vandermonde matrix of size {k} not invertible mod {p}"
                    ) from e
            return invs

        def interpolate_full_grid(
            values: dict[tuple[int, ...], int],
            v: int,
            d: int,
            p: int,
            inv_vands: dict[int, list[list[int]]],
        ) -> dict[tuple[int, ...], int]:
            current_values = dict(values)
            for step in range(v):
                from collections import defaultdict

                grouped = defaultdict(list)
                for pt, val in current_values.items():
                    vi = pt[step]
                    rem = pt[:step] + pt[step + 1 :]
                    grouped[rem].append((vi, val))

                next_values = {}
                V_inv = inv_vands[d + 1]
                for rem, vi_vals in grouped.items():
                    vi_vals.sort(key=lambda x: x[0])
                    y = [val for _, val in vi_vals]
                    c = [0] * (d + 1)
                    for r in range(d + 1):
                        s = 0
                        for col in range(d + 1):
                            s = (s + V_inv[r][col] * y[col]) % p
                        c[r] = s
                    for j, coeff in enumerate(c):
                        new_pt = rem[:step] + (j,) + rem[step:]
                        next_values[new_pt] = coeff
                current_values = next_values
            return current_values

        def eval_point_mod_p(pt: tuple[int, ...], p: int) -> int:
            M_pt = flint.nmod_mat(n, n, p)
            for r in range(n):
                for c in range(n):
                    val = 0
                    for pt_val, int_A in zip(pt, integer_matrices):
                        val = (val + pt_val * int_A[r][c]) % p
                    M_pt[r, c] = val
            return int(M_pt.det())

        def crt(modulo_values: list[int], primes: list[int]) -> int:
            n_p = len(primes)
            mix = list(modulo_values)
            c = [1] * n_p
            for i in range(1, n_p):
                c_val = 1
                for j in range(i):
                    c_val = (c_val * primes[j]) % primes[i]
                c_inv = pow(c_val, -1, primes[i])
                c[i] = c_inv
            u = [0] * n_p
            u[0] = mix[0] % primes[0]
            for i in range(1, n_p):
                val = u[0]
                p_prod = 1
                for j in range(1, i):
                    p_prod = (p_prod * primes[j - 1]) % primes[i]
                    val = (val + u[j] * p_prod) % primes[i]
                u[i] = ((mix[i] - val) * c[i]) % primes[i]
            x = u[0]
            p_prod = 1
            M = primes[0]
            for i in range(1, n_p):
                p_prod *= primes[i - 1]
                x += u[i] * p_prod
                M *= primes[i]
            if x > M // 2:
                x -= M
            return x

        primes_gen = prime_generator(1000000007)
        reconstructed = None
        primes_used = []
        coeffs_by_prime = []

        while True:
            p = next(primes_gen)
            try:
                inv_vands = get_inverse_vandermonde_matrices(n + 1, p)
                values = {}
                for pt in full_grid_pts:
                    values[pt] = eval_point_mod_p(pt + (1,), p)
                interpolated = interpolate_full_grid(values, m - 1, n, p, inv_vands)
                c_p = [interpolated.get(exp[:-1], 0) for exp in exps]
            except ValueError:
                continue

            coeffs_by_prime.append(c_p)
            primes_used.append(p)

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

        c_coeffs = []
        denom_scale = D**n
        for val in reconstructed:
            if val % denom_scale == 0:
                c_coeffs.append(val // denom_scale)
            else:
                c_coeffs.append(sp.Rational(val, denom_scale))

        expr = sp.Integer(0)
        monoms = get_monomials(variables, n)
        for coeff, monom in zip(c_coeffs, monoms):
            expr += coeff * monom

        return cls(expr, variables)

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

        exact_matrices = []
        for A in pencil.matrices:
            exact_A = []
            for r in range(n):
                row = []
                for c in range(n):
                    val = sp.sympify(A[r, c])
                    if isinstance(val, (int, float, np.number)):
                        val = sp.Rational(val)
                    row.append(val)
                exact_A.append(row)
            exact_matrices.append(exact_A)

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

        def crt(modulo_values: list[int], primes: list[int]) -> int:
            n_p = len(primes)
            mix = list(modulo_values)
            c = [1] * n_p
            for i in range(1, n_p):
                c_val = 1
                for j in range(i):
                    c_val = (c_val * primes[j]) % primes[i]
                c_inv = pow(c_val, -1, primes[i])
                c[i] = c_inv
            u = [0] * n_p
            u[0] = mix[0] % primes[0]
            for i in range(1, n_p):
                val = u[0]
                p_prod = 1
                for j in range(1, i):
                    p_prod = (p_prod * primes[j - 1]) % primes[i]
                    val = (val + u[j] * p_prod) % primes[i]
                u[i] = ((mix[i] - val) * c[i]) % primes[i]
            x = u[0]
            p_prod = 1
            M = primes[0]
            for i in range(1, n_p):
                p_prod *= primes[i - 1]
                x += u[i] * p_prod
                M *= primes[i]
            if x > M // 2:
                x -= M
            return x

        def prime_generator(start: int = 1000000007) -> Any:
            curr = start
            while True:
                if flint.fmpz(curr).is_probable_prime():
                    yield curr
                curr += 1

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

        c_coeffs = {}
        denom_scale = D**n
        for exp, val in reconstructed.items():
            if val % denom_scale == 0:
                c_coeffs[exp] = val // denom_scale
            else:
                c_coeffs[exp] = sp.Rational(val, denom_scale)

        expr = sp.Integer(0)
        for exp, coeff in c_coeffs.items():
            monom_factors = []
            for i in range(m - 1):
                if exp[i] > 0:
                    monom_factors.append(variables[i] ** exp[i])
            power_last = n - sum(exp)
            if power_last > 0:
                monom_factors.append(variables[-1] ** power_last)

            if monom_factors:
                expr += coeff * sp.Mul(*monom_factors)
            else:
                expr += coeff

        return cls(expr, variables)

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
