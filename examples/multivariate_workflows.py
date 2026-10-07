"""Run exact context changes and batched derivatives with PYTHONPATH=. python this_file."""

import numpy as np
import sympy as sp

from finitefree.multivariate import MultivariatePolynomial


def main() -> None:
    x, y, u, v = sp.symbols("x y u v")
    p = MultivariatePolynomial(x**2 + 3 * x * y + y**2, [x, y])
    reordered = p.reorder_variables([y, x])
    assert reordered.expr == p.expr
    assert reordered.evaluate([2, 1]) == p.evaluate([1, 2])
    swapped = p.substitute({x: y, y: x + 1})
    assert swapped.expr == sp.expand(p.expr.subs({x: y, y: x + 1}, simultaneous=True))

    composed = p.substitute({x: u + v, y: u - v}, variables=[u, v])
    assert composed.expr == 5 * u**2 - v**2
    points = np.stack(
        np.broadcast_arrays(np.arange(3)[:, None], np.arange(2)[None, :]), axis=-1
    )
    gradient, hessian = (
        composed.gradient_float64(points),
        composed.hessian_float64(points),
    )
    assert gradient.shape == (3, 2, 2) and hessian.shape == (3, 2, 2, 2)
    np.testing.assert_array_equal(
        gradient, np.stack([10 * points[..., 0], -2 * points[..., 1]], axis=-1)
    )
    np.testing.assert_array_equal(
        hessian, np.broadcast_to([[10, 0], [0, -2]], hessian.shape)
    )
    assert composed.hessian_float64(np.empty((0, 2))).shape == (0, 2, 2)
    projected = p.substitute({x: sp.Rational(1, 2)}, variables=[y])
    assert projected.expr == y**2 + sp.Rational(3, 2) * y + sp.Rational(1, 4)
    print(
        f"Multivariate composition passed; gradient {gradient.shape}, Hessian {hessian.shape}."
    )


if __name__ == "__main__":
    main()
