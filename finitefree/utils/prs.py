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
    Computes an integer Sturm sequence for A and B by sign-corrected primitive
    pseudo-remainders.

    Each returned term is a positive rational multiple of the corresponding
    term of the classical Euclidean sequence A, B, -rem(A, B), ...; sign
    variations, and hence Sturm real-root counts, are therefore the same,
    including for abnormal sequences whose degrees drop by more than one.
    Rational inputs are scaled by their positive denominators. Since
    prem(P, Q) = lc(Q)**delta * rem(P, Q) with delta = deg(P) - deg(Q) + 1,
    the pseudo-remainder is negated and multiplied by sign(lc(Q))**delta;
    its positive integer content is then divided out. The historical name is
    kept for compatibility: Brown-Collins subresultant scalings can introduce
    negative factors and are not used. All returned terms have integer
    coefficients.
    """
    # Flint denominators are positive, so taking numerators is a positive
    # scaling that preserves signs and sign variations.
    A_num = A.numer()
    B_num = B.numer()

    if B_num.is_zero():
        return [flint.fmpq_poly(A_num)]

    P = [A_num, B_num]
    while True:
        previous = P[-2]
        current = P[-1]
        R = prem(previous, current)
        if R.is_zero():
            break
        delta = previous.degree() - current.degree() + 1
        if current.leading_coefficient() < 0 and delta % 2 == 1:
            R = -R  # R is now a positive multiple of rem(previous, current).
        next_term = -R
        P.append(next_term // next_term.content())

    return [flint.fmpq_poly(p) for p in P]
