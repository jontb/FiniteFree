import math
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from .hyperbolic import SymmetricMatrixPencil


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
        Computes mixed partial derivatives.
        orders: list of derivative order for each variable.
        """
        if len(orders) != len(self.variables):
            raise ValueError(
                f"Orders length ({len(orders)}) must match "
                f"variable count ({len(self.variables)})."
            )

        deriv_expr = self.expr
        for x_i, order in zip(self.variables, orders):
            if order > 0:
                deriv_expr = sp.diff(deriv_expr, x_i, order)

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
        cls, pencil: SymmetricMatrixPencil, parallel: bool = False
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
        grid_pts = get_grid_points(m - 1, grid_vals)
        grid_pts.sort(key=sum)

        points = [pt + (1,) for pt in grid_pts[:N]]

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
        PRIMES = [
            1000000007,
            1000000009,
            1000000021,
            1000000033,
            1000000087,
            1000000093,
            1000000097,
            1000000103,
        ]

        def eval_point_mod_p(pt: tuple[int, ...], p: int) -> int:
            M_pt = flint.nmod_mat(n, n, p)
            for r in range(n):
                for c in range(n):
                    val = 0
                    for pt_val, int_A in zip(pt, integer_matrices):
                        val = (val + pt_val * int_A[r][c]) % p
                    M_pt[r, c] = val
            return int(M_pt.det())

        def solve_system_mod_p(p: int) -> list[int]:
            # Solve modular Vandermonde system V_flint * c_p = y_flint modulo p
            V_flint = flint.nmod_mat(N, N, p)
            for r in range(N):
                pt = points[r]
                for c in range(N):
                    exp = exps[c]
                    term = 1
                    for pt_val, power in zip(pt, exp):
                        term = (term * (pt_val**power)) % p
                    V_flint[r, c] = term

            y_flint = flint.nmod_mat(N, 1, p)
            for r in range(N):
                y_flint[r, 0] = eval_point_mod_p(points[r], p)

            try:
                c_flint = V_flint.solve(y_flint)
                return [int(c_flint[r, 0]) for r in range(N)]
            except Exception as e:
                raise ValueError(f"System not invertible modulo {p}") from e

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

        reconstructed = None
        primes_used = []
        coeffs_by_prime = []

        for p in PRIMES:
            try:
                c_p = solve_system_mod_p(p)
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
                    break
                reconstructed = current_reconstruction
        else:
            if reconstructed is None:
                raise ValueError(
                    "Modular determinant interpolation failed to stabilize."
                )

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
    def from_symmetric_matrix_pencil(
        cls, pencil: SymmetricMatrixPencil
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
