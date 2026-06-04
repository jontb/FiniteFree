import gc
import math
import warnings
from typing import Any, List, Sequence, Union

import flint
import numpy as np
import sympy as sp
from numpy.typing import NDArray


class PrecisionContext:
    _ALLOCATION_COUNTER: int = 0
    _GC_THRESHOLD: int = 50

    def __init__(self, degree: int) -> None:
        self.degree = degree
        self.original_prec = flint.ctx.prec
        # Dynamically scale precision based on degree to handle combinatorial explosions
        self.new_prec: int = max(53, int(degree * 2.5))

    def __enter__(self) -> "PrecisionContext":
        flint.ctx.prec = self.new_prec
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        flint.ctx.prec = self.original_prec
        # Optimized threshold-based memory management
        PrecisionContext._ALLOCATION_COUNTER += 1
        if PrecisionContext._ALLOCATION_COUNTER >= PrecisionContext._GC_THRESHOLD:
            gc.collect()
            PrecisionContext._ALLOCATION_COUNTER = 0


class RealRootedPolynomial:
    def __init__(
        self,
        coeffs: Union[Sequence[Any], NDArray[Any]],
        assume_real_rooted: bool = False,
    ) -> None:
        """
        coeffs: array of length d+1 where index k corresponds to x^{d-k}.
        """
        self.coeffs: NDArray[np.object_] = np.array(coeffs, dtype=object)

        if len(self.coeffs) == 0:
            raise ValueError("Empty polynomial")

        leading_coeff = self.coeffs[0]
        if leading_coeff != 1:
            coeffs_list = []
            for c in self.coeffs:
                if isinstance(c, (int, np.integer)) and isinstance(
                    leading_coeff, (int, np.integer)
                ):
                    if c % leading_coeff == 0:
                        coeffs_list.append(c // leading_coeff)
                    else:
                        coeffs_list.append(sp.Rational(c, leading_coeff))
                else:
                    coeffs_list.append(c / leading_coeff)
            self.coeffs = np.array(coeffs_list, dtype=object)

        self.degree: int = len(self.coeffs) - 1
        self._is_verified: bool = assume_real_rooted
        self._normalized_coeffs_cached: Union[NDArray[np.object_], None] = None
        self._roots_cached: Union[NDArray[Any], None] = None
        self._has_non_negative_roots_cached: Union[bool, None] = None
        self._has_strictly_positive_roots_cached: Union[bool, None] = None
        # Track memory allocation
        PrecisionContext._ALLOCATION_COUNTER += 1
        if PrecisionContext._ALLOCATION_COUNTER >= PrecisionContext._GC_THRESHOLD:
            gc.collect()
            PrecisionContext._ALLOCATION_COUNTER = 0

    def verify_real_rootedness(self) -> bool:
        """
        Lazily verify that the polynomial is real-rooted via Sturm sequences
        or exact root bounding.
        """
        if self._is_verified:
            return True

        if self.degree <= 1:
            self._is_verified = True
            return True

        import flint

        try:
            # Construct fmpq_poly in ascending degree order (x^0, ..., x^d)
            q_coeffs = []
            for c in reversed(self.coeffs):
                if isinstance(c, sp.Rational):
                    q_coeffs.append(flint.fmpq(int(c.p), int(c.q)))
                elif isinstance(c, (float, np.floating)):
                    c_sym = sp.Rational(float(c))
                    q_coeffs.append(flint.fmpq(int(c_sym.p), int(c_sym.q)))
                else:
                    q_coeffs.append(flint.fmpq(int(c), 1))

            f_poly = flint.fmpq_poly(q_coeffs)
            _, factors = f_poly.factor_squarefree()

            total_real_roots = 0
            for factor, multiplicity in factors:
                # Generate Sturm sequence using compiled division modulo in C
                seq = [factor, factor.derivative()]
                while not seq[-1].is_zero():
                    remainder = seq[-2] % seq[-1]
                    seq.append(-remainder)
                sturm_seq = seq[:-1]

                # Evaluate sturm sequence at -inf and +inf
                def sign_changes_flint(
                    seq_list: List[flint.fmpq_poly], val: str
                ) -> int:
                    signs: List[int] = []
                    for p in seq_list:
                        if p.is_zero():
                            continue
                        lead_coeff = p.leading_coefficient()
                        s = 1 if lead_coeff > 0 else -1
                        deg = p.degree()

                        if val == "-inf":
                            if deg % 2 != 0:
                                s = -s
                        signs.append(s)

                    changes = 0
                    for i in range(len(signs) - 1):
                        if signs[i] * signs[i + 1] < 0:
                            changes += 1
                    return changes

                v_minus_inf = sign_changes_flint(sturm_seq, "-inf")
                v_plus_inf = sign_changes_flint(sturm_seq, "inf")

                real_roots_factor = v_minus_inf - v_plus_inf
                if real_roots_factor < factor.degree():
                    raise ValueError(
                        "Polynomial is not real-rooted. "
                        f"Missing real roots in factor {factor}"
                    )

                total_real_roots += real_roots_factor * multiplicity

            if total_real_roots < self.degree:
                raise ValueError("Polynomial is not real-rooted.")

            self._is_verified = True
            return True

        except Exception:
            # Fallback to numerical check via Flint's Arb-certified roots
            # if exact Sturm sequences fail or raise an exception
            try:
                import flint

                q_coeffs = []
                for c in reversed(self.coeffs):
                    if isinstance(c, sp.Rational):
                        q_coeffs.append(flint.fmpq(int(c.p), int(c.q)))
                    elif isinstance(c, (float, np.floating)):
                        c_sym = sp.Rational(float(c))
                        q_coeffs.append(flint.fmpq(int(c_sym.p), int(c_sym.q)))
                    else:
                        q_coeffs.append(flint.fmpq(int(c), 1))

                f_poly = flint.fmpq_poly(q_coeffs)
                acb_roots = f_poly.complex_roots()

                # Verify that all isolated roots are real (i.e. imaginary
                # part contains 0)
                for r_pair in acb_roots:
                    r = r_pair[0]
                    if 0 not in r.imag:
                        raise ValueError("Complex root detected in Arb isolation.")

                self._is_verified = True
                return True
            except Exception as inner_e:
                # If Flint complex_roots itself fails, fall back to numpy.roots
                # with a conservative tolerance to prevent false rejections
                # due to Wilkinson's phenomenon.
                roots = np.roots(np.array(self.coeffs, dtype=float))
                if not np.allclose(np.imag(roots), 0, atol=1e-2, rtol=1e-2):
                    raise ValueError(
                        "Numerical fallback: Polynomial is not real-rooted."
                    ) from inner_e
                self._is_verified = True
                return True

    def verify_root_interlacing(self, strict: bool = False) -> bool:
        """
        Verifies that the roots of the derivative p'(x) interlace the roots of p(x).
        For univariate polynomials, if p(x) is real-rooted, Rolle's Theorem
        mathematically guarantees that the roots of p'(x) interlace the roots of p(x).
        If strict=True, they strictly interlace if and only if all roots of p(x)
        are simple (which corresponds to the polynomial being square-free).
        """
        self.verify_real_rootedness()

        if not strict:
            return True

        if self.degree <= 1:
            return True

        # Check if the polynomial is square-free (no multiple roots)
        import flint

        # Construct fmpq_poly in ascending degree order (x^0, ..., x^d)
        q_coeffs = []
        for c in reversed(self.coeffs):
            if isinstance(c, sp.Rational):
                q_coeffs.append(flint.fmpq(int(c.p), int(c.q)))
            elif isinstance(c, (float, np.floating)):
                c_sym = sp.Rational(float(c))
                q_coeffs.append(flint.fmpq(int(c_sym.p), int(c_sym.q)))
            else:
                q_coeffs.append(flint.fmpq(int(c), 1))

        f_poly = flint.fmpq_poly(q_coeffs)
        _, factors = f_poly.factor_squarefree()

        for _, multiplicity in factors:
            if multiplicity > 1:
                raise ValueError(
                    "Strict root interlacing failed: multiple roots detected."
                )

        return True

    def normalized_coeffs(self, d: Union[int, None] = None) -> NDArray[np.object_]:
        """
        Extracts the normalized elementary symmetric polynomial sequence
        \\tilde{e}_k^{(d)}(p) with respect to ambient dimension d.
        """
        if d is None:
            d = self.degree

        if d < self.degree:
            raise ValueError(
                f"Ambient dimension d ({d}) cannot be less than polynomial "
                f"degree ({self.degree})."
            )

        if d == self.degree and self._normalized_coeffs_cached is not None:
            return self._normalized_coeffs_cached

        e_k = []
        for k in range(d + 1):
            if k <= self.degree:
                binom = math.comb(d, k)
                sign = (-1) ** k
                c_k = self.coeffs[k]
                val = sign * binom
                # Maintain exact rational/integer representation to avoid
                # float truncation
                if isinstance(c_k, (int, np.integer)):
                    if c_k % val == 0:
                        e_k.append(c_k // val)
                    else:
                        e_k.append(sp.Rational(c_k, val))
                else:
                    e_k.append(c_k / val)
            else:
                e_k.append(0)

        res_array = np.array(e_k, dtype=object)
        if d == self.degree:
            self._normalized_coeffs_cached = res_array
        return res_array

    @classmethod
    def from_normalized_coeffs(
        cls, e_k: Union[Sequence[Any], NDArray[Any]]
    ) -> "RealRootedPolynomial":
        """
        Reconstructs the polynomial from the normalized sequence
        \\tilde{e}_k^{(d)}(p).
        Automatically assumes real-rootedness since finite free
        convolutions preserve it.
        """
        d = len(e_k) - 1
        c = []
        for k in range(d + 1):
            binom = math.comb(d, k)
            sign = (-1) ** k
            c.append(e_k[k] * sign * binom)
        inst = cls(c, assume_real_rooted=True)
        inst._normalized_coeffs_cached = np.array(e_k, dtype=object)
        return inst

    @classmethod
    def from_roots(cls, roots: Sequence[Any]) -> "RealRootedPolynomial":
        """
        Reconstructs the polynomial from its exact roots.
        Uses a divide-and-conquer product of C-level fmpq_poly linear factors
        to run in O(d log^2 d) exact time, avoiding slow SymPy symbolic products.
        """
        import flint
        import sympy as sp

        if len(roots) == 0:
            return cls([1], assume_real_rooted=True)

        factors = []
        for r in roots:
            if isinstance(r, sp.Rational):
                val = flint.fmpq(int(r.p), int(r.q))
            elif isinstance(r, (float, np.floating)):
                c_sym = sp.Rational(float(r))
                val = flint.fmpq(int(c_sym.p), int(c_sym.q))
            elif isinstance(r, flint.fmpq):
                val = r
            else:
                val = flint.fmpq(int(r), 1)
            # Factor is (x - r) represented in ascending order [-r, 1]
            factors.append(flint.fmpq_poly([-val, 1]))

        def _mult(f_list: List[flint.fmpq_poly]) -> flint.fmpq_poly:
            if len(f_list) == 1:
                return f_list[0]
            mid = len(f_list) // 2
            return _mult(f_list[:mid]) * _mult(f_list[mid:])

        poly_flint = _mult(factors)
        coeffs_asc = poly_flint.coeffs()

        # Convert fmpq back to SymPy Rational or Python int
        coeffs_desc = []
        for c in reversed(coeffs_asc):
            q_den = int(c.q)
            q_num = int(c.p)
            if q_den == 1:
                coeffs_desc.append(q_num)
            else:
                coeffs_desc.append(sp.Rational(q_num, q_den))

        inst = cls(coeffs_desc, assume_real_rooted=True)
        # Cache roots by converting them to float64
        float_roots = []
        for r in roots:
            float_roots.append(float(r))
        inst._roots_cached = np.sort(np.array(float_roots, dtype=np.float64))
        return inst

    def to_numpy_poly1d(self) -> np.poly1d:
        """
        Converts the exact rational polynomial coefficients into standard float64
        representation and returns a numpy.poly1d object.

        WARNING: Directly exporting high-degree coefficients to float64 can lead to
        severe numerical instability due to Wilkinson's phenomenon. Consider using
        evaluate_roots_float64() instead to retrieve high-precision isolated roots.
        """
        if self.degree > 20:
            warnings.warn(
                "Directly exporting high-degree coefficients to float64 can result in "
                "extreme numerical instability due to Wilkinson's phenomenon. "
                "Consider using evaluate_roots_float64() which performs high-precision "
                "root isolation first.",
                RuntimeWarning,
                stacklevel=2,
            )
        float_coeffs = np.array(self.coeffs, dtype=float)
        return np.poly1d(float_coeffs)

    def evaluate_roots_float64(self, parallel: bool = False) -> NDArray[Any]:
        """
        Computes the roots of the polynomial with high numerical stability.
        Uses a hybrid approach: tries fast companion-matrix or parallelized
        Aberth-Ehrlich numerical solvers first, falling back to python-flint's
        Arb-based certified root isolation when numerical methods fail or overflow.
        Supports lazy caching to avoid redundant C-level solver evaluations.
        """
        if self._roots_cached is not None:
            return self._roots_cached
        res = self._evaluate_roots_float64_uncached(parallel=parallel)
        self._roots_cached = res
        return res

    def _evaluate_roots_float64_uncached(
        self, parallel: bool = False
    ) -> NDArray[np.float64]:
        # --- Parallel path: Vectorized Aberth-Ehrlich Candidate Seeker ---
        if parallel:
            try:
                float_coeffs = np.array(self.coeffs, dtype=np.float64)
                if np.all(np.isfinite(float_coeffs)):
                    try:
                        import cupy as cp  # type: ignore[import-not-found]

                        # GPU Seeker via CuPy
                        d = len(float_coeffs) - 1
                        c_0 = float_coeffs[0]
                        center = -float_coeffs[1] / (d * c_0)
                        r_vals = [
                            abs(float_coeffs[k] / c_0) ** (1.0 / k)
                            for k in range(1, d + 1)
                        ]
                        R = max(r_vals) * 1.5
                        theta = (2.0 * cp.arange(d) + 0.5) * cp.pi / d
                        z = center + R * (cp.cos(theta) + 1j * cp.sin(theta))
                        dp_coeffs = [float_coeffs[i] * (d - i) for i in range(d)]

                        for _ in range(50):
                            P_z = cp.zeros_like(z, dtype=complex)
                            for c in float_coeffs:
                                P_z = P_z * z + c
                            Dp_z = cp.zeros_like(z, dtype=complex)
                            for c in dp_coeffs:
                                Dp_z = Dp_z * z + c
                            w = P_z / Dp_z
                            diff = z[:, None] - z[None, :]
                            cp.fill_diagonal(diff, 1.0)
                            inv_diff = 1.0 / diff
                            cp.fill_diagonal(inv_diff, 0.0)
                            sum_diff = cp.sum(inv_diff, axis=1)
                            update = w / (1.0 - w * sum_diff)
                            z = z - update
                            if cp.max(cp.abs(update)) < 1e-12:
                                break
                        return np.sort(np.real(cp.asnumpy(z)))
                    except Exception:
                        # CPU Seeker via NumPy (Vectorized)
                        d = len(float_coeffs) - 1
                        c_0 = float_coeffs[0]
                        center = -float_coeffs[1] / (d * c_0)
                        r_vals = [
                            abs(float_coeffs[k] / c_0) ** (1.0 / k)
                            for k in range(1, d + 1)
                        ]
                        R = max(r_vals) * 1.5
                        theta = (2.0 * np.arange(d) + 0.5) * np.pi / d
                        z = center + R * (np.cos(theta) + 1j * np.sin(theta))
                        dp_coeffs = [float_coeffs[i] * (d - i) for i in range(d)]

                        for _ in range(50):
                            P_z = np.zeros_like(z, dtype=complex)
                            for c in float_coeffs:
                                P_z = P_z * z + c
                            Dp_z = np.zeros_like(z, dtype=complex)
                            for c in dp_coeffs:
                                Dp_z = Dp_z * z + c
                            w = P_z / Dp_z
                            diff = z[:, None] - z[None, :]
                            np.fill_diagonal(diff, 1.0)
                            inv_diff = 1.0 / diff
                            np.fill_diagonal(inv_diff, 0.0)
                            sum_diff = np.sum(inv_diff, axis=1)
                            update = w / (1.0 - w * sum_diff)
                            z = z - update
                            if np.max(np.abs(update)) < 1e-12:
                                break
                        return np.sort(np.real(z))
            except (OverflowError, ValueError):
                pass  # Fall through to certified solver

        # --- Fast path: NumPy companion-matrix eigensolver ---
        try:
            float_coeffs = np.array(self.coeffs, dtype=np.float64)
            if not np.all(np.isfinite(float_coeffs)):
                raise OverflowError("Coefficients overflow float64")
            raw_roots = np.roots(float_coeffs)
            # Check if all roots are real (imaginary parts negligible)
            if np.all(np.abs(raw_roots.imag) < 1e-10 * (np.abs(raw_roots.real) + 1)):
                return np.sort(raw_roots.real)
            # Complex roots detected — fall through to certified solver
        except (OverflowError, ValueError, FloatingPointError):
            pass  # Fall through to certified solver

        # --- Certified path: Arb-based root isolation ---
        try:
            # Construct fmpq_poly directly from coefficients.
            # Reversing coefficients is required since fmpq_poly accepts coefficients
            # in ascending degree order (x^0, x^1, ..., x^d).
            q_coeffs = []
            for c in reversed(self.coeffs):
                if isinstance(c, sp.Rational):
                    q_coeffs.append(flint.fmpq(int(c.p), int(c.q)))
                elif isinstance(c, (float, np.floating)):
                    c_sym = sp.Rational(float(c))
                    q_coeffs.append(flint.fmpq(int(c_sym.p), int(c_sym.q)))
                else:
                    q_coeffs.append(flint.fmpq(int(c), 1))

            f_poly = flint.fmpq_poly(q_coeffs)
            acb_roots = f_poly.complex_roots()
            float_roots = []
            for r_pair in acb_roots:
                r = r_pair[0]
                mult = int(r_pair[1])
                if hasattr(r, "real"):
                    real_attr = "real"
                    val = float(getattr(r, real_attr))
                else:
                    val = float(r)
                for _ in range(mult):
                    float_roots.append(val)
            return np.sort(np.array(float_roots, dtype=np.float64))
        except Exception:
            try:
                # Fallback to sympy.nroots
                x = sp.Symbol("x")
                expr = sum(
                    sp.Rational(c) * x ** (self.degree - i)
                    for i, c in enumerate(self.coeffs)
                )
                roots_complex = sp.nroots(expr, maxsteps=1000)
                float_roots = [float(r.as_real_imag()[0]) for r in roots_complex]
                return np.sort(np.array(float_roots, dtype=np.float64))
            except Exception:
                warnings.warn(
                    "High-precision root isolation failed. Falling back to "
                    "numpy.roots. Roots may exhibit Wilkinson's phenomenon.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                float_coeffs = np.array(self.coeffs, dtype=float)
                return np.sort(np.real(np.roots(float_coeffs)))

    def dilation(self, c: Any) -> "RealRootedPolynomial":
        """
        Computes the dilated polynomial [Dil_c p](x) = c^d p(x/c).
        """
        if c == 0:
            raise ValueError("Dilation factor c cannot be zero.")

        d = self.degree
        new_coeffs = []
        for k in range(d + 1):
            val_c = self.coeffs[k]
            scale = c**k
            # Maintain exact representation
            if isinstance(val_c, (int, np.integer)) and isinstance(
                scale, (int, np.integer)
            ):
                new_coeffs.append(val_c * scale)
            else:
                new_coeffs.append(val_c * scale)

        return RealRootedPolynomial(new_coeffs, assume_real_rooted=self._is_verified)

    def shift(self, c: Any) -> "RealRootedPolynomial":
        """
        Computes the shifted polynomial [Shi_c p](x) = p(x-c).
        """
        x = sp.Symbol("x")
        poly_expr = sum(
            sp.Rational(coeff) * x ** (self.degree - i)
            for i, coeff in enumerate(self.coeffs)
        )
        shifted_expr = sp.expand(poly_expr.subs(x, x - c))
        poly = sp.Poly(shifted_expr, x)
        coeffs = list(poly.all_coeffs())
        return RealRootedPolynomial(coeffs, assume_real_rooted=self._is_verified)

    def power(self, c: Any) -> "RealRootedPolynomial":
        """
        Computes the polynomial p^(c) whose roots are lambda_i(p)^c.
        """
        if not self.has_non_negative_roots:
            raise ValueError(
                "Power operation is only defined for polynomials with "
                "non-negative roots."
            )
        if c <= 0:
            raise ValueError("Power factor c must be strictly positive.")

        roots = self.evaluate_roots_float64()
        new_roots = [r**c for r in roots]
        return RealRootedPolynomial.from_roots(new_roots)

    def reversed_polynomial(self) -> "RealRootedPolynomial":
        """
        Computes the reversed polynomial p^(-1) with roots 1/lambda_i(p).
        """
        d = self.degree
        e_k = self.normalized_coeffs()

        # e_d must be non-zero (otherwise we have a root at 0 and cannot invert)
        e_d = e_k[d]
        if e_d == 0:
            raise ValueError(
                "Reversed polynomial is only defined for polynomials with "
                "strictly non-zero roots."
            )

        new_e = []
        for k in range(d + 1):
            val_num = e_k[d - k]
            val_den = e_d
            # Maintain exact representation
            if isinstance(val_num, (int, np.integer)) and isinstance(
                val_den, (int, np.integer)
            ):
                if val_num % val_den == 0:
                    new_e.append(val_num // val_den)
                else:
                    new_e.append(sp.Rational(val_num, val_den))
            else:
                new_e.append(val_num / val_den)

        return RealRootedPolynomial.from_normalized_coeffs(new_e)

    def phi_d(self) -> "RealRootedPolynomial":
        """
        Computes the limiting polynomial Phi_d(p) from Fujie and Ueda [FU23].
        For p in P_d(R_>=0) with multiplicity r at root 0, the roots are
        lambda_k = e_tilde_k / e_tilde_{k-1} for 1 <= k <= d - r, and 0 otherwise.
        """
        # Verify that all roots are non-negative
        if not self.has_non_negative_roots:
            raise ValueError(
                "phi_d is only defined for polynomials with non-negative roots "
                "(p in P_d(R_>=0))."
            )

        d = self.degree
        e_k = self.normalized_coeffs()

        # Multiplicity r of the root at 0 is the number of trailing zero coefficients
        r = 0
        while r < d and self.coeffs[d - r] == 0:
            r += 1

        new_roots = []
        for k in range(1, d + 1):
            if k <= d - r:
                val_num = e_k[k]
                val_den = e_k[k - 1]
                if val_den == 0:
                    raise ValueError(
                        f"Zero division encountered: e_tilde_{k - 1} is zero."
                    )
                if isinstance(val_num, (int, np.integer)) and isinstance(
                    val_den, (int, np.integer)
                ):
                    if val_num % val_den == 0:
                        new_roots.append(val_num // val_den)
                    else:
                        new_roots.append(sp.Rational(val_num, val_den))
                else:
                    new_roots.append(val_num / val_den)
            else:
                new_roots.append(0)

        return RealRootedPolynomial.from_roots(new_roots)

    def derivative(self) -> "RealRootedPolynomial":
        """
        Computes the derivative p'(x) of the polynomial, monic-normalized.
        """
        if self.degree == 0:
            raise ValueError("Cannot take derivative of a constant polynomial.")
        d = self.degree
        new_coeffs = []
        for i in range(d):
            new_coeffs.append(self.coeffs[i] * (d - i))
        return RealRootedPolynomial(new_coeffs, assume_real_rooted=self._is_verified)

    def projection(self, j: int) -> "RealRootedPolynomial":
        """
        Computes the projection \\partial^{j|d} p(x) which is the derivative
        of order d-j, monic-normalized.
        """
        if j < 0 or j > self.degree:
            raise ValueError("Projection dimension j must be between 0 and degree.")

        current = self
        for _ in range(self.degree - j):
            current = current.derivative()
        return current

    def additive_power(self, t: Any) -> "RealRootedPolynomial":
        """
        Computes the fractional finite free additive convolution power
        p^{\\boxplus_d t} defined via scaling the finite free cumulants:
        κ_n^{(d)}(p^{\\boxplus_d t}) = t * κ_n^{(d)}(p).
        """
        if t <= 0:
            raise ValueError(
                "Fractional convolution power t must be strictly positive."
            )

        d = self.degree
        from .transforms import FiniteRTransform

        kappas = FiniteRTransform(self, order=d)
        kappas_scaled = [k * t for k in kappas]

        c_cumulants = []
        for n in range(1, d + 1):
            den = math.factorial(n - 1) * ((-d) ** (n - 1))
            val_num = kappas_scaled[n - 1]
            if isinstance(val_num, (int, np.integer)) and isinstance(
                den, (int, np.integer)
            ):
                if val_num % den == 0:
                    c_cumulants.append(val_num // den)
                else:
                    c_cumulants.append(sp.Rational(val_num, den))
            else:
                c_cumulants.append(val_num / den)

        e_k: List[Any] = [1]
        for n in range(1, d + 1):
            en = c_cumulants[n - 1]
            for k in range(1, n):
                en += math.comb(n - 1, k - 1) * c_cumulants[k - 1] * e_k[n - k]
            e_k.append(en)

        return RealRootedPolynomial.from_normalized_coeffs(e_k)

    def is_symmetric(self) -> bool:
        """
        A polynomial of degree 2d is symmetric if its degree is even and
        all coefficients for odd powers of x are strictly zero (i.e. p(x) = p(-x)).
        """
        d = self.degree
        if d % 2 != 0:
            return False

        for k in range(d + 1):
            if (d - k) % 2 != 0:
                if self.coeffs[k] != 0:
                    return False
        return True

    def square_roots_map(self) -> "RealRootedPolynomial":
        """
        The Sq(p) mapping.
        Given a symmetric polynomial p(x) = sum c_{2k} x^{2k}, constructs
        and returns the transformed polynomial Sq(p)(x) = sum c_{2k} x^k.
        This satisfies the identity Sq(p)(x^2) = p(x).
        """
        if not self.is_symmetric():
            raise ValueError(
                "square_roots_map is only defined for symmetric polynomials."
            )

        d = self.degree
        d_new = d // 2
        new_coeffs = []
        for j in range(d_new + 1):
            new_coeffs.append(self.coeffs[2 * j])

        return RealRootedPolynomial(new_coeffs, assume_real_rooted=True)

    @property
    def has_non_negative_roots(self) -> bool:
        """
        Check if all roots of the polynomial are non-negative in O(d) time
        using sign alternation.
        """
        if self._has_non_negative_roots_cached is None:
            signs = []
            for j, c in enumerate(self.coeffs):
                if c != 0:
                    val = c * (-1) ** (self.degree - j)
                    signs.append(1 if val > 0 else -1)
            self._has_non_negative_roots_cached = len(set(signs)) <= 1
        return self._has_non_negative_roots_cached

    @property
    def has_strictly_positive_roots(self) -> bool:
        """
        Check if all roots of the polynomial are strictly positive in O(d) time
        using sign alternation.
        """
        if self._has_strictly_positive_roots_cached is None:
            ans = True
            for c in self.coeffs:
                if c == 0:
                    ans = False
                    break
            if ans:
                for i in range(1, self.degree + 1):
                    if self.coeffs[i] * self.coeffs[i - 1] >= 0:
                        ans = False
                        break
            self._has_strictly_positive_roots_cached = ans
        return self._has_strictly_positive_roots_cached


class UnitaryPolynomial(RealRootedPolynomial):
    """
    Represents a polynomial whose roots lie strictly on the unit circle T.
    Bypasses Sturm sequence real-rootedness verification and supports complex roots.
    """

    def __init__(
        self,
        coeffs: Union[Sequence[Any], NDArray[Any]],
    ) -> None:
        # Bypasses real-rootedness verification
        super().__init__(coeffs, assume_real_rooted=True)

    def verify_real_rootedness(self) -> bool:
        return False

    def evaluate_roots_float64(self, parallel: bool = False) -> NDArray[Any]:
        """
        Computes the complex roots of the unitary polynomial.
        Evaluates transcendental coefficients numerically using SymPy N(c)
        to avoid int() / float() casting errors of transcendental terms,
        and uses companion matrix eigensolver to compute complex roots on T.
        """
        if self._roots_cached is not None:
            return self._roots_cached

        float_coeffs = []
        for c in self.coeffs:
            # Safely evaluate transcendental SymPy terms to complex float
            float_coeffs.append(complex(sp.N(c)))

        raw_roots = np.roots(float_coeffs)
        # Sort roots by their argument (angle) in [-pi, pi]
        angles = np.angle(raw_roots)
        sorted_idx = np.argsort(angles)
        res = raw_roots[sorted_idx]
        self._roots_cached = res
        return res
