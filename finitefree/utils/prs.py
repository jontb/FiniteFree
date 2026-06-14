from typing import List

import flint


def prem(A: flint.fmpz_poly, B: flint.fmpz_poly) -> flint.fmpz_poly:
    """
    Computes the pseudo-remainder of A divided by B in Z[x].
    This computes the remainder of c^delta * A divided by B, where
    c is the leading coefficient of B and delta = deg(A) - deg(B) + 1.
    """
    m = A.degree()
    n = B.degree()
    if m < n:
        return A
    c = B.leading_coefficient()
    R = A
    steps = m - n + 1
    for i in range(steps):
        if R.is_zero():
            break
        deg_R = R.degree()
        if deg_R < n:
            R = R * (c ** (steps - i))
            break
        lc_R = R.leading_coefficient()
        R = R * c - B.left_shift(deg_R - n) * lc_R
    return R


def sturm_monic_prs(A: flint.fmpq_poly, B: flint.fmpq_poly) -> List[flint.fmpq_poly]:
    """
    Computes a Sturm sequence for A and B using the Monic-Normalized PRS algorithm.
    This scales the remainder at each step by its absolute leading coefficient
    to keep rational coefficients extremely small and avoid exponential coefficient growth.
    """
    prs = [A, B]
    while True:
        P_prev = prs[-2]
        P_curr = prs[-1]

        R = P_prev % P_curr
        if R.is_zero():
            break

        try:
            lc = R.leading_coefficient()
        except AttributeError:
            lc = R.coeffs()[-1] if R.coeffs() else flint.fmpq(0, 1)
        abs_lc = lc if lc > 0 else -lc
        P_next = -R / abs_lc
        prs.append(P_next)

    return prs


def sturm_subresultant_prs(
    A: flint.fmpq_poly, B: flint.fmpq_poly
) -> List[flint.fmpq_poly]:
    """
    Computes a Sturm sequence for A and B using the signed integer Subresultant PRS algorithm.
    All polynomial divisions are performed via exact integer pseudo-division in Z[x] to eliminate
    rational arithmetic overhead while strictly preserving the sign-variation parity.
    """
    # Coerce input rational polynomials into integer polynomials by extracting their numerators.
    # Flint denominators are positive, so this is a positive scalar scaling that does not
    # alter the signs or sign-variations.
    A_num = A.numer()
    B_num = B.numer()

    if B_num.is_zero():
        return [flint.fmpq_poly(A_num)]

    P = [A_num, B_num]

    # Initialize subresultant scalar sequences psi and beta
    psi = [-1, -1]  # index 0 is unused, index 1 is psi_1
    beta = [1, 1]  # index 0 is unused, index 1 is beta_1

    d_1 = P[0].degree() - P[1].degree()
    beta[1] = (-1) ** (d_1 + 1)

    k = 2
    while True:
        R_k = prem(P[k - 2], P[k - 1])
        if R_k.is_zero():
            break

        div_factor = beta[k - 1]
        P_next = -R_k / div_factor
        P.append(P_next)

        d_k_minus_1 = P[k - 2].degree() - P[k - 1].degree()
        c_k_minus_1 = P[k - 1].leading_coefficient()

        psi_val = (-c_k_minus_1) ** d_k_minus_1
        pow_val = 1 - d_k_minus_1
        if pow_val >= 0:
            psi_k = psi_val * (psi[k - 1] ** pow_val)
        else:
            psi_k = psi_val // (psi[k - 1] ** (-pow_val))

        beta_k = -c_k_minus_1 * (psi_k**d_k_minus_1)

        psi.append(psi_k)
        beta.append(beta_k)

        k += 1

    return [flint.fmpq_poly(p) for p in P]
