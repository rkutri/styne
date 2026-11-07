import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc.method.mrw import MRWProposal
from styne.mcmc.transition import EvaluatedState
from styne.parameter import Vector
from styne.statistics import (
    DenseCovarianceMatrix, DiagonalCovarianceMatrix, IIDCovarianceMatrix,
)
from styne.utility import FullCovarianceAdaptation, NotPositiveDefinite


@pytest.fixture(params=['numpy', 'jax', 'pytorch'])
def backend(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    return get_backend(request.param)


def singular_estimate(backend):
    # Float32 Welford covariance for observations [0, 0] and [1000, 1000].
    return backend.full((2, 2), 499999.96875, dtype='float32')


def test_float32_nugget_failure_reports_precision_and_requested_value(backend):
    matrix = singular_estimate(backend)
    with pytest.raises(NotPositiveDefinite, match=r'float32.*nugget=0.01'):
        DenseCovarianceMatrix(matrix, nugget=0.01)
    covariance = DenseCovarianceMatrix(matrix, nugget=0.125)
    np.testing.assert_allclose(covariance.to_dense(),
                               np.asarray(matrix) + 0.125 * np.eye(2), atol=0.04,
                               rtol=0)


def test_small_nugget_is_allowed_when_factorisation_succeeds(backend):
    matrix = backend.eye(2, dtype='float32') * 500000.0
    covariance = DenseCovarianceMatrix(matrix, nugget=0.01)
    np.testing.assert_allclose(covariance.to_dense(), matrix, rtol=1e-6)


@pytest.mark.parametrize('matrixType', [
    DenseCovarianceMatrix, DiagonalCovarianceMatrix, IIDCovarianceMatrix,
])
def test_nugget_is_exact_and_preserves_structure(backend, matrixType):
    values = backend.asarray([0.0, 0.0], dtype='float32')
    if matrixType is DenseCovarianceMatrix:
        arguments = (backend.eye(2, dtype='float32') * values,)
    elif matrixType is IIDCovarianceMatrix:
        arguments = (2, values[0])
    else:
        arguments = (values,)
    covariance = matrixType(*arguments, nugget=0.25, scaling=2.0)
    assert type(covariance) is matrixType
    factor = np.asarray(covariance.to_cholesky())
    np.testing.assert_allclose(factor @ factor.T, 0.5 * np.eye(2), rtol=1e-6)


@pytest.mark.parametrize('entry', [-1.0, np.nan, np.inf])
def test_invalid_covariances_fail_at_construction(backend, entry):
    matrix = backend.asarray([[entry, 0.0], [0.0, 1.0]], dtype='float32')
    with pytest.raises(NotPositiveDefinite):
        DenseCovarianceMatrix(matrix)
    with pytest.raises(NotPositiveDefinite):
        DiagonalCovarianceMatrix(backend.asarray([entry, 1.0], dtype='float32'))


@pytest.mark.parametrize('nugget', [-1.0, np.inf, np.nan, [0.1, 0.2]])
def test_invalid_nuggets_fail_before_factorisation(backend, nugget, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Invalid regularisation must fail before factorisation.')

    monkeypatch.setattr(backend, 'checked_cholesky', unexpected)
    with pytest.raises(ValueError, match='Nugget'):
        DenseCovarianceMatrix(backend.eye(2, dtype='float32'), nugget=nugget)


@pytest.mark.parametrize('factor, message', [
    ([[1.0, 0.1], [0.0, 1.0]], 'lower triangular'),
    ([[1.0, 0.0], [0.0, 0.0]], 'positive diagonal'),
    ([[np.nan, 0.0], [0.0, 1.0]], 'finite'),
])
def test_external_factors_are_validated_on_all_backends(backend, factor, message):
    with pytest.raises(NotPositiveDefinite, match=message):
        DenseCovarianceMatrix.from_cholesky(backend.asarray(factor, dtype='float32'))


def test_adaptation_does_not_replace_failed_covariance(backend):
    strategy = FullCovarianceAdaptation(interval=2, nugget=0.01)
    initial = Vector(backend.zeros(2, dtype='float32'))
    proposal = MRWProposal(DiagonalCovarianceMatrix(
        backend.ones(2, dtype='float32')
    ))
    state = strategy.initial_state(initial, proposal)
    state = strategy.update_state(state, EvaluatedState(initial), None)
    previous = state.covariance
    nextState = EvaluatedState(Vector(backend.full(2, 1000.0, dtype='float32')))
    with pytest.raises(NotPositiveDefinite, match='nugget=0.01'):
        strategy.update_state(state, nextState, None)
    assert state.covariance is previous
    assert state.moments.nData == 1


@pytest.mark.parametrize('backendName', ['jax', 'pytorch'])
def test_compiled_covariance_reports_failure_and_can_be_called_again(backendName):
    pytest.importorskip('torch' if backendName == 'pytorch' else backendName)
    backend = get_backend(backendName)

    def factor(matrix):
        return DenseCovarianceMatrix(matrix, nugget=0.01).to_cholesky()

    options = {'backend': 'eager', 'fullgraph': True} if backendName == 'pytorch' else {}
    execute = backend.compile(factor, **options)
    with pytest.raises(NotPositiveDefinite, match='nugget=0.01'):
        execute(singular_estimate(backend))
    valid = execute(backend.eye(2, dtype='float32'))
    np.testing.assert_allclose(valid, np.sqrt(1.01) * np.eye(2), rtol=1e-6)


def test_jax_checks_selected_branches_and_vectorised_covariances():
    pytest.importorskip('jax')
    backend = get_backend('jax')

    def factor(matrix):
        return DenseCovarianceMatrix(matrix, nugget=0.01).to_cholesky()

    valid = backend.eye(2, dtype='float32')
    invalid = singular_estimate(backend)
    result = backend.cond(backend.asarray(False), factor, lambda _: valid, invalid)
    np.testing.assert_array_equal(result, valid)
    with pytest.raises(NotPositiveDefinite):
        backend.vectorize(factor)(backend.namespace.stack((valid, invalid)))


def test_unrelated_jax_checks_keep_their_exception_type():
    pytest.importorskip('jax')
    from jax.experimental import checkify
    backend = get_backend('jax')

    def function(value):
        checkify.check(value > 0, 'An unrelated user check.')
        return value

    with pytest.raises(checkify.JaxRuntimeError, match='unrelated user check'):
        backend.compile(function)(backend.asarray(-1))


def test_integer_covariances_accept_fractional_nuggets(backend):
    matrix = backend.zeros((2, 2), dtype='int32')
    covariance = DenseCovarianceMatrix(matrix, nugget=0.25)
    np.testing.assert_allclose(covariance.to_cholesky(), 0.5 * np.eye(2))


def test_compiled_jax_validates_external_factors_and_dynamic_scaling():
    pytest.importorskip('jax')
    backend = get_backend('jax')
    execute = backend.compile(
        lambda factor, scale: DenseCovarianceMatrix.from_cholesky(
            factor, scaling=scale
        ).to_cholesky()
    )
    factor = backend.asarray([[1.0, 0.1], [0.0, 1.0]], dtype='float32')
    with pytest.raises(NotPositiveDefinite, match='lower triangular'):
        execute(factor, backend.asarray(1.0, dtype='float32'))
    with pytest.raises(NotPositiveDefinite, match='Scaling'):
        execute(backend.eye(2, dtype='float32'), backend.asarray(-1.0, dtype='float32'))
