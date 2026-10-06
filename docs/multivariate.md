# Rational multivariate workflows

This guide describes the unreleased development implementation. A `MultivariatePolynomial` represents a sparse polynomial over the rationals in an explicit ordered sequence of SymPy symbols. Construction and algebra do not certify stability, hyperbolicity or real-rootedness.

## Construction and exact evaluation

Native FLINT polynomials must have the same variable names in the same order. Inputs are copied; `variables`, `coefficients()` and `to_fmpq_mpoly()` return caller-owned values. Variable names must be distinct, including when SymPy symbols have different assumptions.

```python
import flint
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(x**2 + sp.Rational(1, 3)*y + 2, [x, y])
assert p.evaluate([sp.Rational(1, 2), 3]) == flint.fmpq(13, 4)
q = MultivariatePolynomial.from_coefficients(p.coefficients(), p.variables)
assert q.expr == p.expr
```

`evaluate` accepts integer/rational coordinates and finite floating coordinates interpreted by their stored binary ratios. It returns an exact FLINT rational. Compare with a FLINT rational, or convert explicitly with `sp.Rational(int(value.p), int(value.q))` when comparing to SymPy; SymPy 1.12's implicit FLINT conversion can lose exactness. Coordinate count must match the stored variables. The python-flint 0.9.0 public evaluation path uses its positional callable interface.

## Rational algebra and derivatives

Addition, subtraction, multiplication, unary negation and nonnegative integer powers are supported. Rational scalar operands work on either side. Polynomial operands must have exactly the same ordered symbols; explicitly reconstruct from a SymPy expression when changing order. Negative/fractional polynomial powers and polynomial division are outside this API.

```python
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(x + y, [x, y])
q = MultivariatePolynomial(x - y, [x, y])
assert (p*q).expr == x**2 - y**2
assert (p**2 - 2*p + 1).expr == sp.expand((x+y-1)**2)
gradient = (p*q).gradient()
hessian = (p*q).hessian()
assert [g.evaluate([0, 0]) for g in gradient] == [0, 0]
assert [[h.evaluate([0, 0]) for h in row] for row in hessian] == [[2, 0], [0, -2]]
```

`partial_derivative(symbol, order=1)`, `mixed_partial_derivative(orders)` and `directional_derivative(direction)` retain exact coefficients. Derivative orders are nonnegative integer indices, including NumPy integer scalars, with booleans rejected. An order above the polynomial's total degree gives zero without an enormous loop. Exact polynomial derivatives can be evaluated at a singular determinant point; determinant/inverse SLP formulas have different singular-point limitations.

## Batched numerical evaluation

`evaluate_float64` accepts real arrays with shape `(..., variable_count)`, returns a float for one point and an array with the leading batch shape otherwise. An empty batch is valid. It converts the stored coefficients once and evaluates sparse monomials across the batch.

```python
import numpy as np
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(x**2 + y, [x, y])
points = np.array([[1, 2], [3, 4]], dtype=float)
np.testing.assert_array_equal(p.evaluate_float64(points), [3, 13])
assert p.evaluate_float64([1, 2]) == 3.0
```

Nonfinite coordinates are rejected with `ValueError`; unrepresentable coefficients or nonfinite arithmetic raise `RuntimeError`. Finite cancellation, underflow and rounding are still possible. This is monomial float64 evaluation, not a certified ball-arithmetic or conditioning-aware solver. Large sparse exponents and coefficient sizes can still be expensive.

## Exact restriction to a line

`restrict_line(base_point, direction)` forms `P(base_point + t*direction)` using exact polynomial arithmetic and preserves its leading scalar. It returns the existing univariate class with lazy real-rootedness certification. The identically zero restriction raises `ValueError` because that class cannot represent zero.

```python
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(3*(x+y)**2, [x, y])
q = p.restrict_line([1, 0], [0, 1])
assert list(q.coeffs) == [3, 6, 3]
assert q.verify_real_rootedness()
value = q.evaluate(sp.Rational(1, 3))
assert sp.Rational(int(value.p), int(value.q)) == sp.Rational(16, 3)
```

A generic rational polynomial need not have real-rooted line restrictions:

```python
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
q = MultivariatePolynomial(x**2+y**2, [x, y]).restrict_line([1, 0], [0, 1])
assert list(q.coeffs) == [1, 0, 1]
try:
    q.verify_real_rootedness()
except ValueError:
    pass
else:
    raise AssertionError("This line restriction has complex roots")
```

## Homogeneous normalization and matrix pencils

`normalized_coefficients()` computes `c_alpha / multinomial(d, alpha)` for a homogeneous degree-`d` polynomial. Nonhomogeneous inputs raise `ValueError`; the zero polynomial returns an empty coefficient mapping. The normalization is an algebraic convention and does not imply a multivariate finite convolution or geometric certificate.

Determinant factories continue to accept the exact rational matrix-pencil snapshots described in [the API](api.md#matrix-pencil-input-contract). Direct, grid-interpolated and sparse constructions can all use the same evaluation/algebra workflows above. Interpolation, coefficient count, degree and bit size can still dominate construction.

After clearing denominators, modular reconstruction continues until the product of accepted primes exceeds twice a determinant coefficient bound. For integer matrices, one such bound is `n! * product_r max_c sum_j abs(A_j[r,c])`: each determinant permutation is a product of linear forms, and the coefficient sum of absolute values is bounded by the product of their coefficient sums. This guarantees a unique centered integer reconstruction of every recovered coefficient. Mere agreement between consecutive CRT reconstructions is insufficient.

The dense grid covers the full homogeneous coefficient support. `from_symmetric_matrix_pencil_sparse` keeps randomized Zippel discovery, then independently verifies all coefficients and the complete support with one exact integer determinant. The coefficient bound alone does not prove support completeness.

For a nonzero bound `B`, dehomogenize the integer determinant by setting the last variable to 1. Encode the other exponents in radix `n+1`, using weights `w_j=(n+1)**j`. Every homogeneous monomial then has a distinct encoded exponent. Evaluate at integer coordinates `x_j=b**w_j`, where `b=2*B+1`. Both the true coefficients and an accepted candidate's coefficients lie in `[-B, B]`. A nonzero difference has coefficients of magnitude at most `b-1`; its highest term at `b` strictly exceeds the sum of every lower term. Equality of the candidate value and the exact determinant therefore proves equality of every coefficient, including monomials omitted by discovery. This is a deterministic identity check, not a random point test.

If the candidate fails its coefficient bounds or this identity check, balanced-base digits of the exact determinant recover the complete polynomial directly. Eight failed prime fields also trigger this exact fallback, avoiding unlimited retries of singular randomized systems. Negative digits, zero coefficients and denominator restoration are handled exactly. A zero row bound proves the determinant identically zero without discovery; an empty matrix has determinant 1.

The encoding can grow exponentially with the variable count. The keyword-only `max_verification_bits` defaults to **1,000,000** and caps the conservative bound `(E+1)*b.bit_length()`, where `E=n*(n+1)**(m-2)` for `m>=2`. This bounds the encoded determinant and evaluated matrix-entry sizes. The check runs before randomized discovery or allocation of the large powers. Exceeding it raises `ValueError`; the method never returns a candidate without verification. Increase the limit explicitly only when the larger integers are feasible, or choose a different determinant constructor. This is an integer-size guard, not a total memory or runtime quota: determinant arithmetic, discovery and fallback decoding can still be expensive. One-variable, empty-matrix and proved-zero-row cases bypass the encoding.

```python
import sympy as sp
from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial

pencil = SymmetricMatrixPencil([
    [[1, 0], [0, 0]],
    [[0, 0], [0, 1]],
    [[0, 0], [0, -2]],
])
p = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
    pencil, max_verification_bits=1_000_000
)
x, y, z = p.variables
assert sp.expand(p.expr - x * (y - 2*z)) == 0
```
