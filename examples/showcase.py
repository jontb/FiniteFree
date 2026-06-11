import numpy as np
import sympy as sp

from finitefree.convolutions import symmetric_additive
from finitefree.core import RealRootedPolynomial
from finitefree.transforms import FiniteSTransform


def build_symmetric_roots_poly(d: int) -> RealRootedPolynomial:
    """
    Builds p(x) = (x^2 - 1)^(d//2) * x^(d % 2)
    This has roots at -1 and 1, symmetrically.
    """
    x = sp.Symbol("x")
    poly_expr = ((x**2 - 1) ** (d // 2)) * (x ** (d % 2))
    poly = sp.Poly(poly_expr, x)
    # coeffs are returned in descending order
    coeffs = [int(c) for c in poly.all_coeffs()]
    return RealRootedPolynomial(coeffs, assume_real_rooted=True)


def build_positive_roots_poly(d: int) -> RealRootedPolynomial:
    """
    Builds p(x) = (x - 2)^d, roots at x=2 > 0.
    """
    x = sp.Symbol("x")
    poly_expr = (x - 2) ** d
    poly = sp.Poly(poly_expr, x)
    coeffs = [int(c) for c in poly.all_coeffs()]
    return RealRootedPolynomial(coeffs, assume_real_rooted=True)


def showcase_asymptotics() -> None:
    print("--- FiniteFree Asymptotic Showcase ---")
    print(r"1. Asymptotic Limit of Finite Free Additive Convolution (p \boxplus_d p)")
    print("   We convolve p_d(x) = (x^2 - 1)^(d/2) with itself.")
    print(
        r"   As d -> infinity, the empirical root distribution of (p_d \boxplus_d p_d)"
    )
    print("   should converge to a known free additive convolution limit.")

    for d in [10, 20, 40]:
        p = build_symmetric_roots_poly(d)
        res = symmetric_additive(p, p, d)
        roots = np.sort(np.roots(np.array(res.coeffs, dtype=float)))
        # Root support will stretch out and the density will approach
        # the free convolution
        print(
            f"   d={d} -> Roots range from {np.min(roots):.4f} to {np.max(roots):.4f}"
        )
        # Variance of roots
        print(f"             Variance of roots: {np.var(roots):.4f}")

    print("\n2. Asymptotic Limit of Finite S-Transform")
    print("   We evaluate S_p^(d)(-t) for p_d(x) = (x-2)^d at t = 0.5")
    print("   For p(x) = (x-a)^d, the roots are all 'a'.")
    print(r"   The classical S-transform of \delta_a is S(z) = 1/a.")
    print("   So for a=2, S(z) = 0.5. Let's see if S_p^(d) converges to 0.5.")

    for d in [10, 20, 40]:
        p = build_positive_roots_poly(d)
        # S_p^(d) has length d, index k-1 corresponds to -k/d
        # We want t = 0.5 -> k/d = 0.5 -> k = d//2
        S = FiniteSTransform(p)
        k = d // 2
        val = S[k - 1]
        print(f"   d={d} -> S_p^({d})(-0.5) = {float(val):.4f}")


if __name__ == "__main__":
    showcase_asymptotics()


def build_hermite_poly(d: int) -> RealRootedPolynomial:
    """
    Builds the monic probabilist's Hermite polynomial He_d(x).
    Roots converge to the Wigner Semicircle law when scaled by 1/sqrt(d).
    Coefficients are integers evaluated exactly in O(d).
    """
    import math

    coeffs = [0] * (d + 1)
    for m in range(d // 2 + 1):
        # Coefficient for x^{d - 2m}
        # c = d! / (m! * (d - 2m)! * (-2)^m) -- wait, He_n(x) has integers?
        # Actually He_n has integers. c = (-1)^m * d! / (m! * (d - 2m)! * 2^m)
        num = math.factorial(d)
        den = math.factorial(m) * math.factorial(d - 2 * m) * (2**m)
        val = ((-1) ** m) * (num // den)
        coeffs[2 * m] = val  # index is 2m, corresponding to x^{d - 2m}
    return RealRootedPolynomial(coeffs, assume_real_rooted=True)


def build_laguerre_poly(d: int, c_ratio: float = 2.0) -> RealRootedPolynomial:
    """
    Builds the monic Laguerre polynomial associated with the Marchenko-Pastur law.
    Evaluated exactly via explicit combinations in O(d).
    """
    import math

    alpha = int(d * c_ratio - d)
    coeffs = [0] * (d + 1)
    for i in range(d + 1):
        # Monic coefficient of x^i is (-1)^{d+i} * d! * binom(d+alpha, d-i) / i!
        # index k for x^{d-k} is d-i.
        k = d - i
        sign = (-1) ** (d + i)
        val = (
            sign * math.factorial(d) * math.comb(d + alpha, d - i) // math.factorial(i)
        )
        coeffs[k] = val
    return RealRootedPolynomial(coeffs, assume_real_rooted=True)


def showcase_semicircle_mp() -> None:
    print("\n3. Wigner Semicircle Law (Hermite Polynomials)")
    print("   We compute the additive convolution of He_d(x) with itself.")
    print("   The variance of He_d is d. The convolution should have variance 2d.")
    for d in [10, 20]:
        p = build_hermite_poly(d)
        res = symmetric_additive(p, p, d)
        roots_p = np.roots(np.array(p.coeffs, dtype=float))
        roots_res = np.roots(np.array(res.coeffs, dtype=float))
        print(f"   d={d} -> Variance of He_d roots: {np.var(roots_p):.4f}")
        print(
            "             Variance of (He_d \\boxplus_d He_d) roots: "
            f"{np.var(roots_res):.4f}"
        )

    print("\n4. Marchenko-Pastur Law (Laguerre Polynomials)")
    print("   We evaluate the Finite S-Transform of the Laguerre polynomial.")
    print("   For c_ratio=2, the classical S-transform is S(z) = 1 / (z + c_ratio).")
    print("   At t = 0.5 (z = -0.5), classical S(-0.5) = 1 / 1.5 = 0.6667.")
    for d in [10, 20, 40]:
        c_ratio = 2
        p = build_laguerre_poly(d, c_ratio=c_ratio)
        S = FiniteSTransform(p)
        # We want S(-t) for t=0.5 -> index k = d/2
        k = d // 2
        val = S[k - 1]
        print(f"   d={d} -> S_p^({d})(-0.5) = {float(val):.4f}")


def showcase_basics() -> None:

    from finitefree.convolutions import multiplicative, symmetric_additive
    from finitefree.transforms import FiniteCauchyTransform, FiniteRTransform

    print("\n--- FiniteFree Basic Operations Showcase ---")

    print("\n1. Polynomial Initialization & Validation")
    # p(x) = (x-1)(x-2) = x^2 - 3x + 2
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=False)
    p.verify_real_rootedness()
    print(
        f"   Polynomial p(x) = x^2 - 3x + 2 is verified real-rooted: {p._is_verified}"
    )
    print(f"   Standard coefficients (descending): {p.coeffs}")
    print(f"   Normalized coefficients (e~_k): {p.normalized_coeffs()}")

    print(r"2. Symmetric Additive Convolution (p \boxplus_2 q)")
    # p(x) = x^2 - 1 (roots 1, -1)
    # q(x) = x^2 - 4 (roots 2, -2)
    p2 = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
    q2 = RealRootedPolynomial([1, 0, -4], assume_real_rooted=True)
    res_add = symmetric_additive(p2, q2, 2)
    print("   p(x) = x^2 - 1")
    print("   q(x) = x^2 - 4")
    print(rf"   (p \boxplus_2 q)(x) coefficients: {res_add.coeffs}")

    print(r"3. Multiplicative Convolution (p \boxtimes_2 q)")
    # p(x) = (x-1)^2 = x^2 - 2x + 1
    # q(x) = (x-2)^2 = x^2 - 4x + 4
    p3 = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)
    q3 = RealRootedPolynomial([1, -4, 4], assume_real_rooted=True)
    res_mul = multiplicative(p3, q3, 2)
    print("   p(x) = (x-1)^2")
    print("   q(x) = (x-2)^2")
    print(rf"   (p \boxtimes_2 q)(x) coefficients: {res_mul.coeffs}")

    print("\n4. Analytical Transforms")
    print("   Finite Cauchy Transform of p(x) = x^2 - 1:")
    cauchy = FiniteCauchyTransform(p2)
    print(f"   G_p^(2)(z) = {cauchy}")

    print("   Finite R-Transform (Cumulants) of p(x) = x^2 - 3x + 2:")
    cumulants = FiniteRTransform(p, order=3)
    print(f"   R_2(y) cumulants: {cumulants}")

    print("\n5. Interlacing Monotonicity of Limiting Polynomial (Phi_d)")
    # Construct two interlacing polynomials p, q of degree 2
    # p roots: {1, 10} -> x^2 - 11x + 10
    # q roots: {2, 11} -> x^2 - 13x + 22
    p_int = RealRootedPolynomial([1, -11, 10], assume_real_rooted=True)
    q_int = RealRootedPolynomial([1, -13, 22], assume_real_rooted=True)
    print("   p(x) roots (1, 10) and q(x) roots (2, 11) strictly interlace: p << q")

    phi_p = p_int.phi_d()
    phi_q = q_int.phi_d()
    roots_phi_p = phi_p.evaluate_roots_float64()
    roots_phi_q = phi_q.evaluate_roots_float64()
    print(f"   Phi_2(p) roots: {list(roots_phi_p)}")
    print(f"   Phi_2(q) roots: {list(roots_phi_q)}")

    # Verify interlacing: r1 <= s1 <= r2 <= s2
    interlaces = roots_phi_p[0] <= roots_phi_q[0] <= roots_phi_p[1] <= roots_phi_q[1]
    print(f"   Phi_2(p) roots and Phi_2(q) roots interlace: {interlaces}")


if __name__ == "__main__":
    showcase_basics()
    showcase_asymptotics()
    showcase_semicircle_mp()
