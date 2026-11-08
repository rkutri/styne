import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc.method.mrw import MRWProposal
from styne.parameter import Vector
from styne.statistics import (
    DenseCovarianceMatrix, DiagonalCovarianceMatrix, IIDCovarianceMatrix,
)
from styne.utility import FullCovarianceAdaptation


@pytest.fixture(params=['numpy', 'jax', 'pytorch'])
def backend(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    return get_backend(request.param)


@pytest.mark.parametrize('matrixType', [
    DenseCovarianceMatrix, DiagonalCovarianceMatrix, IIDCovarianceMatrix,
])
def test_rescaling_shares_owned_factors(backend, matrixType):
    variance = backend.asarray([1.5, 1.5], dtype='float32')
    if matrixType is DenseCovarianceMatrix:
        covariance = matrixType(backend.eye(2, dtype='float32') * variance)
    elif matrixType is IIDCovarianceMatrix:
        covariance = matrixType(2, variance[0])
    else:
        covariance = matrixType(variance)
    scaling = backend.asarray(2.0, dtype='float32')
    scaled = covariance.with_scaling(scaling)
    assert type(scaled) is matrixType
    assert scaled._tree_flatten()[0][0] is covariance._tree_flatten()[0][0]
    if backend.name != 'jax':
        variance[0] = 20.0
        scaling[...] = 20.0
    np.testing.assert_allclose(covariance.to_cholesky(), np.sqrt(1.5) * np.eye(2))
    np.testing.assert_allclose(scaled.to_cholesky(), np.sqrt(3.0) * np.eye(2),
                               rtol=1e-6)


def test_preparation_reuses_covariance_without_factory_calls(backend, monkeypatch):
    strategy = FullCovarianceAdaptation()
    parameter = Vector(backend.zeros(2, dtype='float32'))
    proposal = MRWProposal(IIDCovarianceMatrix(
        2, backend.asarray(1.0, dtype='float32')
    ))
    state = strategy.initial_state(parameter, proposal)

    def unexpected(*args, **kwargs):
        raise AssertionError('Preparation must retain the validated covariance.')

    monkeypatch.setattr(DenseCovarianceMatrix, 'from_cholesky', unexpected)
    monkeypatch.setattr(backend, 'cholesky', unexpected)
    for _ in range(3):
        assert strategy.prepare_proposal(proposal, state).covariance is state.covariance


def test_jax_covariance_state_survives_compilation_without_revalidation(monkeypatch):
    jax = pytest.importorskip('jax')
    backend = get_backend('jax')
    covariance = DenseCovarianceMatrix(backend.eye(2, dtype='float32'))

    def unexpected(*args, **kwargs):
        raise AssertionError('Restoring owned state must not refactor or validate.')

    monkeypatch.setattr(DenseCovarianceMatrix, '__init__', unexpected)
    monkeypatch.setattr(DenseCovarianceMatrix, 'from_cholesky', unexpected)
    restored = jax.jit(lambda matrix: matrix.with_scaling(2.0))(covariance)
    assert isinstance(restored, DenseCovarianceMatrix)
    np.testing.assert_allclose(restored.to_cholesky(), np.sqrt(2) * np.eye(2))
