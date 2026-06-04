from typing import Any, Sequence, Tuple

import flint
import sympy as sp

from .core import RealRootedPolynomial
from .multivariate import MultivariatePolynomial

# Global cache for Jack polynomials to prevent redundant recomputations
# Keyed by (m, partition_tuple, alpha_sym)
_JACK_CACHE: dict[
    Tuple[int, Tuple[int, ...], sp.Rational],
    dict[Tuple[int, ...], sp.Rational],
] = {}


def sympy_to_fmpq(val: Any) -> flint.fmpq:
    """Converts a SymPy rational value exactly to a Flint fmpq."""
    val = sp.Rational(sp.sympify(val))
    return flint.fmpq(int(val.p), int(val.q))


def fmpq_poly_to_sympy_coeffs(poly: flint.fmpq_poly) -> list[sp.Rational]:
    """Converts flint.fmpq_poly (ascending) to SymPy coefficients (descending)."""
    coeffs_asc = poly.coeffs()
    coeffs = []
    for val in reversed(coeffs_asc):
        coeffs.append(sp.Rational(str(val)))
    return coeffs




def jacobi_polynomial(n: int, alpha: Any, beta: Any) -> RealRootedPolynomial:
    """
    Computes the Jacobi polynomial P_n^(alpha, beta)(x) of degree n
    exactly using the three-term recurrence relation over Q with flint.fmpq_poly.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    alpha_sym = sp.Rational(sp.sympify(alpha))
    beta_sym = sp.Rational(sp.sympify(beta))

    p0 = flint.fmpq_poly([1])
    if n == 0:
        return RealRootedPolynomial([1], assume_real_rooted=True)

    c0 = sympy_to_fmpq((alpha_sym - beta_sym) / 2)
    c1 = sympy_to_fmpq((alpha_sym + beta_sym + 2) / 2)
    p1 = flint.fmpq_poly([c0, c1])
    if n == 1:
        return RealRootedPolynomial(
            fmpq_poly_to_sympy_coeffs(p1), assume_real_rooted=True
        )

    # 3-term recurrence: A_k P_{k+1} = (B_k x + C_k) P_k - D_k P_{k-1}
    p_prev = p0
    p_curr = p1
    for k in range(1, n):
        ak = sympy_to_fmpq(
            2
            * (k + 1)
            * (k + 1 + alpha_sym + beta_sym)
            * (2 * k + alpha_sym + beta_sym)
        )
        bk = sympy_to_fmpq(
            (2 * k + alpha_sym + beta_sym + 1)
            * (2 * k + alpha_sym + beta_sym + 2)
            * (2 * k + alpha_sym + beta_sym)
        )
        ck = sympy_to_fmpq(
            (2 * k + alpha_sym + beta_sym + 1) * (alpha_sym**2 - beta_sym**2)
        )
        dk = sympy_to_fmpq(
            2 * (k + alpha_sym) * (k + beta_sym) * (2 * k + alpha_sym + beta_sym + 2)
        )

        factor = flint.fmpq_poly([ck, bk])
        p_next = (factor * p_curr - dk * p_prev) * (1 / ak)
        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(p_curr), assume_real_rooted=True
    )


def hahn_polynomial(n: int, alpha: Any, beta: Any, N: int) -> RealRootedPolynomial:
    """
    Computes the Hahn polynomial Q_n(x; alpha, beta, N) of degree n
    exactly using sequential O(n) running products for coefficients.
    """
    if n < 0 or n > N:
        raise ValueError("n must satisfy 0 <= n <= N")

    alpha_sym = sp.Rational(sp.sympify(alpha))
    beta_sym = sp.Rational(sp.sympify(beta))
    N_sym = sp.Integer(N)

    # Initial term for k = 0
    coeff = sp.Integer(1)
    total_poly = flint.fmpq_poly([sympy_to_fmpq(coeff)])
    neg_x_k = flint.fmpq_poly([1])

    for k in range(1, n + 1):
        neg_x_k *= flint.fmpq_poly([k - 1, -1])

        # Recurrence relation for Pochhammer coefficients:
        # coeff_k = coeff_{k-1} * (-n + k - 1) * (n + alpha + beta + k) /
        #                         ((alpha + k) * (-N + k - 1) * k)
        num = (-n + k - 1) * (n + alpha_sym + beta_sym + k)
        den = (alpha_sym + k) * (-N_sym + k - 1) * k

        coeff *= sp.Rational(num, den)
        coeff_fmpq = sympy_to_fmpq(coeff)

        total_poly += coeff_fmpq * neg_x_k

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(total_poly), assume_real_rooted=True
    )


def get_conjugate(part: tuple[int, ...]) -> tuple[int, ...]:
    """Computes the conjugate of a partition shape."""
    if not part:
        return ()
    max_val = part[0]
    conj = [0] * max_val
    for val in part:
        for idx in range(val):
            conj[idx] += 1
    return tuple(conj)


def compute_beta(kappa: tuple[int, ...], mu: tuple[int, ...], alpha: Any) -> Any:
    """Computes the beta_kappa_mu coefficient for the Jack recurrence."""
    kappa_conj = get_conjugate(kappa)
    mu_conj = get_conjugate(mu)
    alpha_sym = sp.Rational(sp.sympify(alpha))

    def get_product(nu: tuple[int, ...], nu_conj: tuple[int, ...]) -> Any:
        prod = sp.Integer(1)
        for r_idx, row_len in enumerate(nu):
            r = r_idx + 1
            for c in range(1, row_len + 1):
                k_conj_c = kappa_conj[c - 1] if c - 1 < len(kappa_conj) else 0
                m_conj_c = mu_conj[c - 1] if c - 1 < len(mu_conj) else 0

                nu_conj_c = nu_conj[c - 1] if c - 1 < len(nu_conj) else 0
                nu_r = nu[r - 1] if r - 1 < len(nu) else 0

                if k_conj_c == m_conj_c:
                    val = nu_conj_c - r + alpha_sym * (nu_r - c + 1)
                else:
                    val = nu_conj_c - r + 1 + alpha_sym * (nu_r - c)
                prod *= val
        return prod

    num = get_product(kappa, kappa_conj)
    den = get_product(mu, mu_conj)
    return num / den


def get_horizontal_strips(kappa: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Generates all partitions mu such that kappa/mu is a horizontal strip."""
    m = len(kappa)
    mu_choices = []

    def backtrack(idx: int, current_mu: list[int]) -> None:
        if idx == m:
            normalized_mu = list(current_mu)
            while normalized_mu and normalized_mu[-1] == 0:
                normalized_mu.pop()
            mu_choices.append(tuple(normalized_mu))
            return

        lower_bound = kappa[idx + 1] if idx + 1 < m else 0
        upper_bound = kappa[idx]
        for val in range(lower_bound, upper_bound + 1):
            current_mu.append(val)
            backtrack(idx + 1, current_mu)
            current_mu.pop()

    backtrack(0, [])
    return mu_choices


def _jack_recursive(
    m: int, kappa: tuple[int, ...], alpha_sym: sp.Rational
) -> dict[tuple[int, ...], sp.Rational]:
    """Recursive helper for the Jack polynomial variables recursion."""
    if len([p for p in kappa if p > 0]) > m:
        return {}
    if not kappa or sum(kappa) == 0:
        return {(0,) * m: sp.Rational(1)}

    cache_key = (m, kappa, alpha_sym)
    if cache_key in _JACK_CACHE:
        return _JACK_CACHE[cache_key]

    if m == 1:
        k = sum(kappa)
        prod = sp.Integer(1)
        for j in range(1, k + 1):
            prod *= 1 + (j - 1) * alpha_sym
        res: dict[tuple[int, ...], sp.Rational] = {(k,): sp.Rational(prod)}
        _JACK_CACHE[cache_key] = res
        return res

    kappa_padded = kappa + (0,) * (m - len(kappa))
    strips = get_horizontal_strips(kappa_padded)

    poly_dict: dict[tuple[int, ...], sp.Rational] = {}
    for mu in strips:
        beta = compute_beta(kappa_padded, mu, alpha_sym)
        j_mu_dict = _jack_recursive(m - 1, mu, alpha_sym)
        deg_diff = sum(kappa) - sum(mu)
        for exp, coeff in j_mu_dict.items():
            new_exp = exp + (deg_diff,)
            new_coeff = coeff * beta
            poly_dict[new_exp] = poly_dict.get(new_exp, sp.Rational(0)) + new_coeff

    _JACK_CACHE[cache_key] = poly_dict
    return poly_dict


def jack_polynomial(
    m: int, partition: Sequence[int], alpha: Any
) -> MultivariatePolynomial:
    """
    Computes the symmetric Jack polynomial J_lambda^(alpha)(x_1, ..., x_m)
    recursively in m variables with parameter alpha using transition coefficients.
    """
    kappa = list(partition)
    while kappa and kappa[-1] == 0:
        kappa.pop()
    kappa_tup = tuple(kappa)

    alpha_sym = sp.Rational(sp.sympify(alpha))
    poly_dict = _jack_recursive(m, kappa_tup, alpha_sym)
    variables = [sp.Symbol(f"x{i}") for i in range(1, m + 1)]

    terms = []
    for exp, coeff in poly_dict.items():
        factors = []
        for var, power in zip(variables, exp):
            if power > 0:
                factors.append(var**power)
        if factors:
            terms.append(coeff * sp.Mul(*factors))
        else:
            terms.append(coeff)

    expr = sp.Add(*terms) if terms else sp.Integer(0)
    return MultivariatePolynomial(expr, variables)
