import numpy as np
import pytest

from styne.backend import get_backend
from styne.statistics import DenseCovarianceMatrix


@pytest.fixture(params=['numpy', 'jax', 'pytorch'])
def backend(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    return get_backend(request.param)


def test_factor_constructor_matches_dense_covariance_operations(backend):
    factor = np.array([[1.4, 0.0], [0.2, 0.9]], dtype='float32')
    nativeFactor = backend.asarray(factor)
    covariance = DenseCovarianceMatrix.from_cholesky(nativeFactor, scaling=1.3)
    matrix = 1.3 * factor @ factor.T
    point = backend.asarray([0.3, -0.8], dtype='float32')
    expectedInverse = np.linalg.solve(matrix, np.asarray(point))
    np.testing.assert_allclose(covariance.to_dense(), factor @ factor.T, atol=1e-6)
    np.testing.assert_allclose(covariance.to_cholesky(), np.sqrt(1.3) * factor,
                               atol=1e-6)
    np.testing.assert_allclose(covariance.apply(point), matrix @ np.asarray(point),
                               atol=1e-6)
    np.testing.assert_allclose(covariance.apply_inverse(point), expectedInverse,
                               atol=1e-6)
    np.testing.assert_allclose(covariance.apply_chol_factor(point),
                               np.sqrt(1.3) * factor @ np.asarray(point), atol=1e-6)
    np.testing.assert_allclose(covariance.apply_chol_factor_transpose(point),
                               np.sqrt(1.3) * factor.T @ np.asarray(point), atol=1e-6)
    np.testing.assert_allclose(covariance.log_determinant(), np.linalg.slogdet(matrix)[1],
                               atol=1e-6)
    np.testing.assert_allclose(covariance.quadratic_form(point),
                               np.asarray(point) @ matrix @ np.asarray(point), atol=1e-6)
    np.testing.assert_allclose(covariance.dual_quadratic_form(point),
                               np.asarray(point) @ expectedInverse, atol=1e-6)
    assert backend.metadata(covariance.to_cholesky()) == backend.metadata(nativeFactor)


def test_factor_constructor_and_rescaling_do_not_reconstruct_or_refactor(
        backend, monkeypatch):
    factor = backend.asarray([[1.4, 0.0], [0.2, 0.9]], dtype='float32')

    def unexpected(*args, **kwargs):
        raise AssertionError('A supplied factor must be reused.')

    monkeypatch.setattr(backend, 'cholesky', unexpected)
    monkeypatch.setattr(DenseCovarianceMatrix, 'to_dense', unexpected)
    covariance = DenseCovarianceMatrix.from_cholesky(factor, scaling=2.0)
    scaled = covariance.with_scaling(3.0)
    np.testing.assert_allclose(covariance.to_cholesky(), np.sqrt(2) * np.asarray(factor),
                               atol=1e-6)
    np.testing.assert_allclose(scaled.to_cholesky(), np.sqrt(3) * np.asarray(factor),
                               atol=1e-6)


def test_factor_constructor_preserves_batches(backend):
    factors = np.array([[[1.4, 0.0], [0.2, 0.9]],
                        [[0.8, 0.0], [-0.4, 1.2]]], dtype='float32')
    covariance = DenseCovarianceMatrix.from_cholesky(backend.asarray(factors))
    expected = factors @ np.swapaxes(factors, -1, -2)
    np.testing.assert_allclose(covariance.to_dense(), expected, atol=1e-6)
    assert covariance.to_cholesky().shape == (2, 2, 2)


def test_factor_constructor_does_not_share_numpy_input():
    factor = np.array([[1.4, 0.0], [0.2, 0.9]])
    covariance = DenseCovarianceMatrix.from_cholesky(factor)
    factor[0, 0] = 5.0
    np.testing.assert_array_equal(covariance.to_cholesky(), [[1.4, 0.0], [0.2, 0.9]])


@pytest.mark.parametrize('factor, message', [
    (1.0, 'square'),
    ([1.0, 2.0], 'square'),
    ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 'square'),
    ([[1.0, 0.1], [0.0, 1.0]], 'lower triangular'),
    ([[1.0, 0.0], [0.0, 0.0]], 'positive diagonal'),
    ([[1.0, 0.0], [0.0, -1.0]], 'positive diagonal'),
    ([[np.nan, 0.0], [0.0, 1.0]], 'finite'),
    ([[1.0, 0.0], [np.inf, 1.0]], 'finite'),
])
def test_invalid_numpy_factors_are_rejected(factor, message):
    with pytest.raises(ValueError, match=message):
        DenseCovarianceMatrix.from_cholesky(factor)


def test_factor_constructor_remains_differentiable_in_pytorch():
    torch = pytest.importorskip('torch')
    factor = torch.tensor([[1.4, 0.0], [0.2, 0.9]], dtype=torch.float64,
                          requires_grad=True)
    scaling = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    covariance = DenseCovarianceMatrix.from_cholesky(factor, scaling)
    point = torch.tensor([0.3, -0.8], dtype=torch.float64)
    value = covariance.log_determinant() + covariance.dual_quadratic_form(point)
    factorGradient, scalingGradient = torch.autograd.grad(value, (factor, scaling))
    assert torch.isfinite(factorGradient).all()
    assert torch.isfinite(scalingGradient)


def test_factor_constructor_remains_compilable_and_differentiable_in_jax():
    jax = pytest.importorskip('jax')
    backend = get_backend('jax')
    factor = backend.asarray([[1.4, 0.0], [0.2, 0.9]], dtype='float32')
    point = backend.asarray([0.3, -0.8], dtype='float32')

    def objective(factor):
        covariance = DenseCovarianceMatrix.from_cholesky(factor, scaling=1.3)
        return covariance.log_determinant() + covariance.dual_quadratic_form(point)

    np.testing.assert_allclose(jax.jit(objective)(factor), objective(factor), atol=1e-6)
    gradient = jax.jit(jax.grad(objective))(factor)
    assert np.all(np.isfinite(np.asarray(gradient)))
