from typing import Any, Sequence, Tuple

import flint
import sympy as sp

from .core import RealRootedPolynomial, UnitaryPolynomial
from .multivariate import MultivariatePolynomial
from .utils.conversion import fmpq_poly_to_sympy_coeffs, sympy_to_fmpq

# Global cache for Jack polynomials to prevent redundant recomputations
# Keyed by (m, partition_tuple, alpha_sym)
_JACK_CACHE: dict[
    Tuple[int, Tuple[int, ...], sp.Rational],
    dict[Tuple[int, ...], sp.Rational],
] = {}


def jacobi_polynomial(n: int, alpha: Any, beta: Any) -> RealRootedPolynomial:
    r"""
    Computes the Jacobi polynomial $P_n^{(\alpha, \beta)}(x)$ of degree $n$
    exactly using the three-term recurrence relation over $\mathbb{Q}$ with flint.fmpq_poly.
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

    alpha_fmpq = sympy_to_fmpq(alpha_sym)
    beta_fmpq = sympy_to_fmpq(beta_sym)

    # 3-term recurrence: A_k P_{k+1} = (B_k x + C_k) P_k - D_k P_{k-1}
    p_prev = p0
    p_curr = p1
    for k in range(1, n):
        ak = (
            2
            * (k + 1)
            * (k + 1 + alpha_fmpq + beta_fmpq)
            * (2 * k + alpha_fmpq + beta_fmpq)
        )
        bk = (
            (2 * k + alpha_fmpq + beta_fmpq + 1)
            * (2 * k + alpha_fmpq + beta_fmpq + 2)
            * (2 * k + alpha_fmpq + beta_fmpq)
        )
        ck = (2 * k + alpha_fmpq + beta_fmpq + 1) * (alpha_fmpq**2 - beta_fmpq**2)
        dk = (
            2
            * (k + alpha_fmpq)
            * (k + beta_fmpq)
            * (2 * k + alpha_fmpq + beta_fmpq + 2)
        )

        factor = flint.fmpq_poly([ck, bk])
        p_next = (factor * p_curr - dk * p_prev) * (1 / ak)
        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(p_curr, assume_real_rooted=True)


def hahn_polynomial(n: int, alpha: Any, beta: Any, N: int) -> RealRootedPolynomial:
    r"""
    Computes the Hahn polynomial $Q_n(x; \alpha, \beta, N)$ of degree $n$
    exactly using sequential $O(n)$ running products for coefficients.
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

    return RealRootedPolynomial(total_poly, assume_real_rooted=True)


_CONJUGATE_CACHE: dict[tuple[int, ...], tuple[int, ...]] = {}


def get_conjugate(part: tuple[int, ...]) -> tuple[int, ...]:
    """Computes the conjugate of a partition shape with caching."""
    if not part:
        return ()
    if part in _CONJUGATE_CACHE:
        return _CONJUGATE_CACHE[part]
    max_val = part[0]
    conj = [0] * max_val
    for val in part:
        for idx in range(val):
            conj[idx] += 1
    res = tuple(conj)
    _CONJUGATE_CACHE[part] = res
    return res


_BETA_CACHE: dict[Tuple[tuple[int, ...], tuple[int, ...], sp.Rational], Any] = {}


def compute_beta(kappa: tuple[int, ...], mu: tuple[int, ...], alpha: Any) -> Any:
    """Computes the beta_kappa_mu coefficient for the Jack recurrence with caching."""
    if isinstance(alpha, sp.Rational):
        alpha_sym = alpha
    else:
        alpha_sym = sp.Rational(sp.sympify(alpha))
    cache_key = (kappa, mu, alpha_sym)
    if cache_key in _BETA_CACHE:
        return _BETA_CACHE[cache_key]

    kappa_conj = get_conjugate(kappa)
    mu_conj = get_conjugate(mu)

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
    res = num / den
    _BETA_CACHE[cache_key] = res
    return res


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
    r"""
    Computes the symmetric Jack polynomial $J_\lambda^{(\alpha)}(x_1, \dots, x_m)$
    recursively in $m$ variables with parameter $\alpha$ using transition coefficients.
    """
    kappa = list(partition)
    while kappa and kappa[-1] == 0:
        kappa.pop()
    kappa_tup = tuple(kappa)

    alpha_sym = sp.Rational(sp.sympify(alpha))
    poly_dict = _jack_recursive(m, kappa_tup, alpha_sym)
    variables = [sp.Symbol(f"x{i}") for i in range(1, m + 1)]

    names = tuple(x.name for x in variables)
    ctx = flint.fmpq_mpoly_ctx.get(names=names)
    flint_dict = {exp: sympy_to_fmpq(coeff) for exp, coeff in poly_dict.items()}
    mpoly = ctx.from_dict(flint_dict)

    return MultivariatePolynomial(mpoly, variables)


def hermite_polynomial(n: int, physicist: bool = True) -> RealRootedPolynomial:
    r"""
    Computes the Hermite polynomial (Physicist's $H_n(x)$ or Probabilist's $He_n(x)$)
    exactly using the three-term recurrence relation over $\mathbb{Q}$ with flint.fmpq_poly.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    p0 = flint.fmpq_poly([1])
    if n == 0:
        return RealRootedPolynomial([1], assume_real_rooted=True)

    if physicist:
        p1 = flint.fmpq_poly([0, 2])
    else:
        p1 = flint.fmpq_poly([0, 1])

    if n == 1:
        return RealRootedPolynomial(
            fmpq_poly_to_sympy_coeffs(p1), assume_real_rooted=True
        )

    p_prev = p0
    p_curr = p1

    for k in range(1, n):
        if physicist:
            factor = flint.fmpq_poly([0, 2])
            term_prev = sympy_to_fmpq(2 * k) * p_prev
            p_next = factor * p_curr - term_prev
        else:
            factor = flint.fmpq_poly([0, 1])
            term_prev = sympy_to_fmpq(k) * p_prev
            p_next = factor * p_curr - term_prev

        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(p_curr), assume_real_rooted=True
    )


def laguerre_polynomial(n: int, alpha: Any) -> RealRootedPolynomial:
    r"""
    Computes the generalized Laguerre polynomial $L_n^{(\alpha)}(x)$
    exactly using the three-term recurrence relation over $\mathbb{Q}$ with flint.fmpq_poly.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    alpha_sym = sp.Rational(sp.sympify(alpha))

    p0 = flint.fmpq_poly([1])
    if n == 0:
        return RealRootedPolynomial([1], assume_real_rooted=True)

    p1 = flint.fmpq_poly([sympy_to_fmpq(1 + alpha_sym), sympy_to_fmpq(-1)])
    if n == 1:
        return RealRootedPolynomial(
            fmpq_poly_to_sympy_coeffs(p1), assume_real_rooted=True
        )

    alpha_fmpq = sympy_to_fmpq(alpha_sym)

    p_prev = p0
    p_curr = p1

    for k in range(1, n):
        ak = flint.fmpq(1, k + 1)
        factor = flint.fmpq_poly([2 * k + 1 + alpha_fmpq, -1])
        term_prev = (k + alpha_fmpq) * p_prev

        p_next = (factor * p_curr - term_prev) * ak
        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(p_curr), assume_real_rooted=True
    )


def krawtchouk_polynomial(n: int, p: Any, N: int) -> RealRootedPolynomial:
    r"""
    Computes the Krawtchouk polynomial $K_n(x; p, N)$
    exactly using the three-term recurrence relation over $\mathbb{Q}$ with flint.fmpq_poly.
    """
    if n < 0 or n > N:
        raise ValueError("n must satisfy 0 <= n <= N")
    if N <= 0:
        raise ValueError("N must be a positive integer")

    p_sym = sp.Rational(sp.sympify(p))
    if p_sym <= 0 or p_sym >= 1:
        raise ValueError("p must be in the open interval (0, 1)")

    p0 = flint.fmpq_poly([1])
    if n == 0:
        return RealRootedPolynomial([1], assume_real_rooted=True)

    p1 = flint.fmpq_poly([sympy_to_fmpq(1), sympy_to_fmpq(sp.Rational(-1, N * p_sym))])
    if n == 1:
        return RealRootedPolynomial(
            fmpq_poly_to_sympy_coeffs(p1), assume_real_rooted=True
        )

    p_fmpq = sympy_to_fmpq(p_sym)

    p_prev = p0
    p_curr = p1

    for k in range(1, n):
        scale = 1 / ((N - k) * p_fmpq)
        factor = flint.fmpq_poly([p_fmpq * (N - 2 * k) + k, -1])
        term_prev = (k * (1 - p_fmpq)) * p_prev

        p_next = (factor * p_curr - term_prev) * scale
        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(p_curr), assume_real_rooted=True
    )


def unitary_hermite_polynomial(d: int, t: Any) -> UnitaryPolynomial:
    r"""
    Computes the Unitary Hermite polynomial $H_d(z; t)$ of degree $d$
    defined on the unit circle. Uses SymPy exp and binomial coefficients exactly.
    """
    if d < 0:
        raise ValueError("d must be non-negative")

    t_sym = sp.sympify(t)

    coeffs = []
    for k in range(d, -1, -1):
        if d > 0:
            num = -t_sym * k * (d - k)
            den = 2 * d
            exp_term = sp.exp(sp.Rational(num, den))
        else:
            exp_term = sp.Integer(1)
        coeff = ((-1) ** k) * sp.binomial(d, k) * exp_term
        coeffs.append(coeff)

    return UnitaryPolynomial(coeffs)


def chebyshev_t_polynomial(n: int) -> RealRootedPolynomial:
    r"""
    Computes the Chebyshev polynomial of the first kind $T_n(x)$ of degree $n$
    exactly using the three-term recurrence relation over $\mathbb{Q}$ with flint.fmpq_poly.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    p0 = flint.fmpq_poly([1])
    if n == 0:
        return RealRootedPolynomial([1], assume_real_rooted=True)

    p1 = flint.fmpq_poly([0, 1])
    if n == 1:
        return RealRootedPolynomial(
            fmpq_poly_to_sympy_coeffs(p1), assume_real_rooted=True
        )

    p_prev = p0
    p_curr = p1

    for _ in range(1, n):
        factor = flint.fmpq_poly([0, 2])
        p_next = factor * p_curr - p_prev
        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(p_curr), assume_real_rooted=True
    )


def chebyshev_u_polynomial(n: int) -> RealRootedPolynomial:
    r"""
    Computes the Chebyshev polynomial of the second kind $U_n(x)$ of degree $n$
    exactly using the three-term recurrence relation over $\mathbb{Q}$ with flint.fmpq_poly.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    p0 = flint.fmpq_poly([1])
    if n == 0:
        return RealRootedPolynomial([1], assume_real_rooted=True)

    p1 = flint.fmpq_poly([0, 2])
    if n == 1:
        return RealRootedPolynomial(
            fmpq_poly_to_sympy_coeffs(p1), assume_real_rooted=True
        )

    p_prev = p0
    p_curr = p1

    for _ in range(1, n):
        factor = flint.fmpq_poly([0, 2])
        p_next = factor * p_curr - p_prev
        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(p_curr), assume_real_rooted=True
    )


def legendre_polynomial(n: int) -> RealRootedPolynomial:
    r"""
    Computes the Legendre polynomial $P_n(x)$ of degree $n$
    exactly using the three-term recurrence relation over $\mathbb{Q}$ with flint.fmpq_poly.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    p0 = flint.fmpq_poly([1])
    if n == 0:
        return RealRootedPolynomial([1], assume_real_rooted=True)

    p1 = flint.fmpq_poly([0, 1])
    if n == 1:
        return RealRootedPolynomial(
            fmpq_poly_to_sympy_coeffs(p1), assume_real_rooted=True
        )

    p_prev = p0
    p_curr = p1

    for k in range(1, n):
        # (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
        ak = flint.fmpq(1, k + 1)
        factor = flint.fmpq_poly([0, flint.fmpq(2 * k + 1, 1)])
        p_next = (factor * p_curr - flint.fmpq(k, 1) * p_prev) * ak
        p_prev = p_curr
        p_curr = p_next

    return RealRootedPolynomial(
        fmpq_poly_to_sympy_coeffs(p_curr), assume_real_rooted=True
    )
