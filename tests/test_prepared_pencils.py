import importlib
from typing import Any

import numpy as np
import pytest

from finitefree.hyperbolic import (
    MultiplicativeMatrixPencil,
    PreparedMatrixPencil,
    SymmetricMatrixPencil,
)
from finitefree.multivariate import MultivariatePolynomial


def coefficients() -> Any:
    return np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.float64)


def test_analytic_batch_and_ownership() -> None:
    a = coefficients()
    p = PreparedMatrixPencil(a, point_block_size=1, coefficient_block_size=1)
    x = np.array([[2.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    expected = np.einsum("bi,ijk->bjk", x, a)
    a[:] = 100
    np.testing.assert_allclose(p.evaluate(x), expected)
    f = p.factor(x)
    sign, log = f.slogdet()
    np.testing.assert_allclose(sign, [1, 1, -1])
    np.testing.assert_allclose(log, np.log([4, 3, 4]))
    det = x[:, 0] ** 2 - x[:, 1] ** 2
    grad = np.stack([2 * x[:, 0], -2 * x[:, 1]], axis=-1) / det[:, None]
    np.testing.assert_allclose(f.logabsdet_gradient(), grad)
    v = np.array([0.3, -0.7])
    hess = (
        np.diag([2.0, -2.0])[None] / det[:, None, None]
        - grad[:, :, None] * grad[:, None, :]
    )
    np.testing.assert_allclose(f.logabsdet_hvp(v), hess @ v)
    x[:] = -99
    sign[:] = 99
    f.logabsdet_gradient()[:] = 99
    np.testing.assert_allclose(f.logabsdet_gradient(), grad)
    np.testing.assert_allclose(f.slogdet()[0], [1, 1, -1])


@pytest.mark.parametrize("symmetric", [False, True])
def test_exact_parity_and_finite_difference(symmetric: bool) -> None:
    rng = np.random.default_rng(32)
    a = rng.integers(-3, 4, (4, 3, 3)).astype(np.float64)
    if symmetric:
        a += a.transpose(0, 2, 1)
    cls = SymmetricMatrixPencil if symmetric else MultiplicativeMatrixPencil
    pencil = cls(list(a))
    p = pencil.prepare_numeric(coefficient_block_size=3)
    slp = pencil.characteristic_polynomial_slp()
    x = np.array([4.0, 1.0, 0.0, -2.0])
    v = np.array([0.2, -0.1, 0.4, 0.3])
    f = p.factor(x)
    value = float(slp.evaluate(x, exact=True))
    gradient = np.asarray(slp.gradient(x, exact=True), dtype=float)
    hessian = np.asarray(slp.hessian(x, exact=True), dtype=float)
    g = gradient / value
    h = hessian / value - np.outer(g, g)
    # HVP first must populate the same gradient cache.
    np.testing.assert_allclose(f.logabsdet_hvp(v), h @ v, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(f.logabsdet_gradient(), g, rtol=2e-12)
    eps = 1e-5
    finite = (
        p.factor(x + eps * v).logabsdet_gradient()
        - p.factor(x - eps * v).logabsdet_gradient()
    ) / (2 * eps)
    np.testing.assert_allclose(f.logabsdet_hvp(v), finite, rtol=1e-7, atol=1e-9)
    expected_matrix: Any = pencil.evaluate(x.tolist())
    np.testing.assert_allclose(p.evaluate(x), expected_matrix)
    np.testing.assert_allclose(f.slogdet(), [np.sign(value), np.log(abs(value))])
    pencil.matrices[0][:] = 999
    np.testing.assert_allclose(p.evaluate(x), np.einsum("i,ijk->jk", x, a))


def test_singular_near_singular_and_polynomial_preservation() -> None:
    pencil = SymmetricMatrixPencil(coefficients())
    p = pencil.prepare_numeric()
    f = p.factor(np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 1.0]]))
    np.testing.assert_array_equal(f.slogdet()[0], [0, 0, 1])
    assert np.isneginf(f.slogdet()[1][:2]).all()
    for query in (f.logabsdet_gradient, lambda: f.logabsdet_hvp(np.ones(2))):
        with pytest.raises(np.linalg.LinAlgError, match="nonsingular"):
            query()
    poly = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    np.testing.assert_allclose(poly.gradient_float64([1.0, 1.0]), [2.0, -2.0])
    np.testing.assert_allclose(poly.hessian_float64([1.0, 1.0]), np.diag([2.0, -2.0]))
    diagonal = PreparedMatrixPencil(
        np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
    )
    f = diagonal.factor(np.array([1.0, 1e-100]))
    np.testing.assert_allclose(f.logabsdet_gradient(), [1.0, 1e100])
    np.testing.assert_allclose(f.logabsdet_hvp(np.ones(2)), [-1.0, -1e200])
    huge = diagonal.factor(np.array([1e200, 1e200]))
    np.testing.assert_allclose(huge.slogdet()[1], 400 * np.log(10))


@pytest.mark.parametrize("shape", [(2,), (3, 2), (2, 3, 2), (0, 2), (2, 0, 2)])
def test_shapes_empty_and_strides(shape: tuple[int, ...]) -> None:
    p = PreparedMatrixPencil(coefficients())
    x = np.empty((*shape[:-1], 4))[..., ::2]
    x[..., 0] = 2.0
    x[..., 1] = 0.0
    f = p.factor(x)
    assert p.evaluate(x).shape == (*shape[:-1], 2, 2)
    assert f.slogdet()[0].shape == shape[:-1]
    assert f.logabsdet_gradient().shape == shape
    assert f.logabsdet_hvp(np.ones(2)).shape == shape
    assert f.logabsdet_hvp(np.ones(shape)).shape == shape


def test_validation() -> None:
    p = PreparedMatrixPencil(coefficients())
    bad: Any
    for bad in (
        [1.0, 2.0],
        np.ones(2, dtype=np.float32),
        np.ones(2, dtype=complex),
        np.ones(2, dtype=object),
    ):
        with pytest.raises(TypeError, match="float64"):
            p.factor(bad)
    for bad in (
        np.ones(3),
        np.array(1.0),
        np.ones((2, 1)),
        np.array([np.inf, 0.0]),
        np.array([np.nan, 0.0]),
    ):
        with pytest.raises(ValueError):
            p.factor(bad)
    for bad in (
        np.empty((0, 2, 2)),
        np.empty((2, 0, 0)),
        np.empty((2, 3, 2)),
        np.ones((2, 2)),
    ):
        with pytest.raises(ValueError):
            PreparedMatrixPencil(bad)
    for bad in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            PreparedMatrixPencil(coefficients(), point_block_size=bad)
    with pytest.raises(ValueError, match="backend"):
        PreparedMatrixPencil(coefficients(), backend="auto")
    f = p.factor(np.array([[2.0, 0.0], [3.0, 0.0]]))
    with pytest.raises(ValueError, match="directions"):
        f.logabsdet_hvp(np.ones((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        f.logabsdet_hvp(np.array([np.nan, 0.0]))
    with pytest.raises(FloatingPointError):
        PreparedMatrixPencil(coefficients() * 1e308).factor(np.array([10.0, 10.0]))


def test_optional_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def without_cupy(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("cupy"):
            raise ImportError("test backend absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_cupy)
    PreparedMatrixPencil(coefficients()).factor(np.array([2.0, 0.0]))
    with pytest.raises(ImportError, match="CuPy"):
        PreparedMatrixPencil(coefficients(), backend="cupy")
    with pytest.raises(ImportError, match="CuPy"):
        SymmetricMatrixPencil(coefficients()).prepare_numeric(backend="cupy")


def test_reuses_factorization_and_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    import scipy.linalg as sla

    p = PreparedMatrixPencil(coefficients())
    f = p.factor(np.array([2.0, 1.0]))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unexpected repeated factorization/solve")

    monkeypatch.setattr(sla, "lu_factor", forbidden)
    f.logabsdet_hvp(np.ones(2))
    f.logabsdet_hvp(np.array([1.0, 0.0]))
    monkeypatch.setattr(sla, "lu_solve", forbidden)
    f.logabsdet_gradient()
    f.slogdet()


def test_real_gpu_parity() -> None:
    try:
        cp = importlib.import_module("cupy")
    except ImportError:
        pytest.skip("Real GPU validation pending: CuPy is not installed")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("Real GPU validation pending: no CUDA device")
        cp.zeros(1).sum().item()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"Real GPU validation pending: {error}")
    a = coefficients()
    cpu = PreparedMatrixPencil(a, point_block_size=2, coefficient_block_size=1)
    gpu = SymmetricMatrixPencil(a).prepare_numeric(
        backend="cupy", point_block_size=2, coefficient_block_size=1
    )
    for x in (
        np.array([2.0, 0.0]),
        np.array([[2.0, 1.0], [0.0, 2.0], [3.0, 0.0]]),
        np.empty((0, 2)),
        np.empty((2, 0, 2)),
    ):
        resident = cp.asarray(x)
        f, g = cpu.factor(x), gpu.factor(resident)
        np.testing.assert_allclose(cp.asnumpy(gpu.evaluate(resident)), cpu.evaluate(x))
        for actual, expected in zip(g.slogdet(), f.slogdet()):
            np.testing.assert_allclose(cp.asnumpy(actual), expected)
        np.testing.assert_allclose(
            cp.asnumpy(g.logabsdet_hvp(cp.ones(2))), f.logabsdet_hvp(np.ones(2))
        )
        np.testing.assert_allclose(
            cp.asnumpy(g.logabsdet_gradient()), f.logabsdet_gradient()
        )
    with pytest.raises(TypeError, match="resident"):
        gpu.factor(np.ones(2))
    singular = gpu.factor(cp.array([1.0, 1.0]))
    assert singular.slogdet()[0].item() == 0
    with pytest.raises(np.linalg.LinAlgError):
        singular.logabsdet_gradient()
