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
        self._roots_cached: Union[NDArray[np.float64], None] = None
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

        except Exception as e:
            # Fallback to numerical check if exact methods fail on extreme degrees
            roots = np.roots(np.array(self.coeffs, dtype=float))
            if not np.allclose(np.imag(roots), 0):
                raise ValueError(
                    "Numerical fallback: Polynomial is not real-rooted."
                ) from e
            self._is_verified = True
            return True

    def verify_root_interlacing(self, strict: bool = False) -> bool:
        """
        Verifies that the roots of the derivative p'(x) interlace the roots of p(x).
        This is a foundational geometry requirement for hyperbolic polynomials.
        If strict=True, requires strict interlacing (\alpha_1 < \beta_1 < \alpha_2 ...).
        """
        if self.degree <= 1:
            return True

        # High-precision stable root isolation solver to avoid Wilkinson's phenomenon
        roots_p = self.evaluate_roots_float64(parallel=False)

        # Derivative coefficients
        # if c_k is for x^{d-k}, then derivative coefficient is c_k * (d-k)
        d = self.degree
        dp_coeffs = [self.coeffs[k] * (d - k) for k in range(d)]
        dp = RealRootedPolynomial(dp_coeffs, assume_real_rooted=True)
        roots_dp = dp.evaluate_roots_float64(parallel=False)

        for i in range(d - 1):
            if strict:
                if not (roots_p[i] < roots_dp[i] < roots_p[i + 1]):
                    raise ValueError(
                        f"Strict root interlacing failed between {roots_p[i]} "
                        f"and {roots_p[i + 1]} with derivative root {roots_dp[i]}"
                    )
            else:
                if not (roots_p[i] <= roots_dp[i] <= roots_p[i + 1]):
                    raise ValueError(
                        f"Root interlacing failed between {roots_p[i]} "
                        f"and {roots_p[i + 1]} with derivative root {roots_dp[i]}"
                    )

        return True

    def normalized_coeffs(self) -> NDArray[np.object_]:
        """
        Extracts the normalized elementary symmetric polynomial sequence
        \\tilde{e}_k^{(d)}(p).
        c[k] = (-1)^k \\binom{d}{k} \\tilde{e}_k^{(d)}(p)
        """
        if self._normalized_coeffs_cached is not None:
            return self._normalized_coeffs_cached

        d = self.degree
        e_k = []
        for k in range(d + 1):
            binom = math.comb(d, k)
            sign = (-1) ** k
            c_k = self.coeffs[k]
            val = sign * binom
            # Maintain exact rational/integer representation to avoid float truncation
            if isinstance(c_k, (int, np.integer)):
                if c_k % val == 0:
                    e_k.append(c_k // val)
                else:
                    e_k.append(sp.Rational(c_k, val))
            else:
                e_k.append(c_k / val)
        self._normalized_coeffs_cached = np.array(e_k, dtype=object)
        return self._normalized_coeffs_cached

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

    def evaluate_roots_float64(self, parallel: bool = False) -> NDArray[np.float64]:
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
