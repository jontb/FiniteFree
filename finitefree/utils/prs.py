from typing import List

import flint


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


# Alias for compatibility
sturm_subresultant_prs = sturm_monic_prs
