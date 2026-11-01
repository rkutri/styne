import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.parameter import Vector
from styne.statistics import (
    DenseCovarianceMatrix, FullCovarianceAccumulator, Gaussian,
    IIDCovarianceMatrix, WelfordAccumulator,
)
from styne.utility import FullCovarianceAdaptation


@pytest.fixture(params=['numpy', 'jax', 'pytorch'])
def backend(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    return get_backend(request.param)


def correlated_samples():
    rng = np.random.default_rng(27)
    transform = np.array([[1.0, 2.0, -1.0], [0.0, 0.2, 1.0], [0.5, 0.0, 0.3]])
    return (rng.normal(size=(37, 3)) @ transform).astype('float32')


def test_records_match_numpy_at_every_prefix_without_mutation(backend):
    samples = correlated_samples()
    initial = FullCovarianceAccumulator.initial_record(backend.asarray(samples[0]))
    metadata = backend.metadata(initial.mean)
    assert initial.nData == 0
    assert initial.mean.shape == (3,)
    assert initial.m2.shape == (3, 3)
    record = initial
    for index, sample in enumerate(samples, start=1):
        previous = record
        previousMean = np.asarray(previous.mean).copy()
        previousM2 = np.asarray(previous.m2).copy()
        observation = backend.asarray(sample)
        record = FullCovarianceAccumulator.update_record(record, observation)
        assert record.nData == index
        assert previous.nData == index - 1
        np.testing.assert_array_equal(previous.mean, previousMean)
        np.testing.assert_array_equal(previous.m2, previousM2)
        np.testing.assert_array_equal(observation, sample)
        np.testing.assert_allclose(record.mean, samples[:index].mean(axis=0), atol=1e-6)
        np.testing.assert_array_equal(np.asarray(record.m2), np.asarray(record.m2).T)
        for value in (record.mean, record.m2):
            assert backend.metadata(value) == metadata
        if index > 1:
            np.testing.assert_allclose(
                np.asarray(record.m2) / (index - 1),
                np.cov(samples[:index], rowvar=False), rtol=2e-5, atol=2e-6,
            )
    np.testing.assert_array_equal(initial.mean, 0)
    np.testing.assert_array_equal(initial.m2, 0)


def test_eager_full_covariance_and_marginal_variances(backend):
    samples = correlated_samples()
    accumulator = FullCovarianceAccumulator()
    diagonal = WelfordAccumulator()
    for sample in samples:
        observation = backend.asarray(sample)
        accumulator.update(observation)
        diagonal.update(observation)
    assert accumulator.nData == len(samples)
    np.testing.assert_allclose(accumulator.mean(), samples.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        accumulator.covariance(), np.cov(samples, rowvar=False), atol=1e-6,
    )
    np.testing.assert_allclose(
        accumulator.marginal_variance(), diagonal.marginal_variance(), atol=1e-6,
    )
    accumulator.reset()
    assert accumulator.nData == 0
    assert accumulator.mean() is None
    with pytest.raises(RuntimeError, match='Insufficient data'):
        accumulator.covariance()
    accumulator.update(backend.asarray(samples[0]))
    assert accumulator.nData == 1
    np.testing.assert_array_equal(accumulator.mean(), samples[0])


@pytest.mark.parametrize('nSamples', [0, 1])
def test_insufficient_observations(nSamples):
    accumulator = FullCovarianceAccumulator()
    for _ in range(nSamples):
        accumulator.update([1.0, 2.0])
    for estimate in (accumulator.covariance, accumulator.marginal_variance):
        with pytest.raises(RuntimeError, match='Insufficient data'):
            estimate()


@pytest.mark.parametrize('samples', [
    [[1.0], [2.0], [4.0]],
    [[1, 2], [3, 4], [2, 6]],
    [[3.0, -7.0]] * 5,
])
def test_sequence_inputs_single_dimension_and_repeated_observations(samples):
    accumulator = FullCovarianceAccumulator()
    for sample in samples:
        accumulator.update(sample)
    assert accumulator.covariance().shape == (len(samples[0]), len(samples[0]))
    np.testing.assert_allclose(accumulator.mean(), np.mean(samples, axis=0))
    np.testing.assert_allclose(
        accumulator.covariance(), np.atleast_2d(np.cov(samples, rowvar=False)),
    )


def test_integer_records_have_fixed_floating_dtype(backend):
    samples = backend.asarray([[1, 2], [3, 4], [2, 6]], dtype='int32')
    record = FullCovarianceAccumulator.initial_record(samples[0])
    metadata = backend.metadata(record.mean)
    for sample in samples:
        record = FullCovarianceAccumulator.update_record(record, sample)
        assert backend.metadata(record.mean) == metadata
        assert backend.metadata(record.m2) == metadata
    np.testing.assert_allclose(record.mean, [2.0, 4.0])
    np.testing.assert_allclose(np.asarray(record.m2) / 2, [[1.0, 1.0], [1.0, 4.0]])


def test_large_repeated_observations_do_not_overflow(backend):
    observation = backend.asarray([1e20, -1e20], dtype='float32')
    accumulator = FullCovarianceAccumulator()
    for _ in range(5):
        accumulator.update(observation)
    np.testing.assert_array_equal(accumulator.mean(), observation)
    np.testing.assert_array_equal(accumulator.covariance(), np.zeros((2, 2)))


def test_large_offset_preserves_covariance():
    samples = correlated_samples().astype('float64') + 1e9
    accumulator = FullCovarianceAccumulator()
    for sample in samples:
        accumulator.update(sample)
    np.testing.assert_allclose(
        accumulator.covariance(), np.cov(samples, rowvar=False),
        rtol=2e-6, atol=2e-7,
    )


@pytest.mark.parametrize('observation', [1.0, [], [[1.0, 2.0]]])
def test_invalid_initial_shapes_are_rejected(observation):
    with pytest.raises(ValueError, match='non-empty one-dimensional'):
        FullCovarianceAccumulator.initial_record(observation)


@pytest.mark.parametrize('dtype', ['float32', 'int32'])
def test_compiled_records_preserve_covariance_and_continuation(dtype):
    jax = pytest.importorskip('jax')
    backend = get_backend('jax')
    samples = backend.asarray(correlated_samples(), dtype=dtype)
    initial = FullCovarianceAccumulator.initial_record(samples[0])

    @jax.jit
    def accumulate(record, observations):
        return jax.lax.scan(
            lambda current, observation: (
                FullCovarianceAccumulator.update_record(current, observation), None,
            ), record, observations,
        )[0]

    first = accumulate(initial, samples[:12])
    resumed = accumulate(first, samples[12:])
    complete = accumulate(initial, samples)
    assert resumed.nData == len(samples)
    np.testing.assert_allclose(resumed.mean, complete.mean, atol=1e-6)
    np.testing.assert_allclose(resumed.m2, complete.m2, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(resumed.m2) / (len(samples) - 1),
        np.cov(np.asarray(samples), rowvar=False), atol=1e-6,
    )
    assert resumed.mean.dtype == initial.mean.dtype
    assert resumed.m2.dtype == initial.m2.dtype
    assert resumed.nData.dtype == initial.nData.dtype


def test_estimated_covariance_can_configure_an_mrw_proposal(backend):
    accumulator = FullCovarianceAccumulator()
    for sample in correlated_samples():
        accumulator.update(backend.asarray(sample))
    metadata = backend.metadata(accumulator.mean())
    matrix = accumulator.covariance() + 0.01 * backend.eye(
        3, dtype=metadata.dtype, device=metadata.device,
    )
    covariance = DenseCovarianceMatrix(matrix)
    np.testing.assert_allclose(covariance.to_dense(), matrix, rtol=1e-5, atol=1e-6)
    start = Vector(backend.zeros(3, dtype=metadata.dtype, device=metadata.device))
    target = Gaussian(IIDCovarianceMatrix(3, backend.asarray(1.0)), start)
    sampler = MetropolisedRandomWalk(
        target.density, covariance, DummyDiagnostics(), rng=backend.random_state(8),
    )
    sampler.run(5, start)
    assert sampler.chain.length == 6
    assert np.all(np.isfinite(np.asarray(sampler.chain.trajectory)))


def test_full_covariance_records_work_in_mh_adaptation(backend):
    start = Vector(backend.zeros(2, dtype='float32'))
    target = Gaussian(IIDCovarianceMatrix(2, backend.asarray(1.0)), start)
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, backend.asarray(0.1)),
        DummyDiagnostics(), rng=backend.random_state(19),
        adaptation=FullCovarianceAdaptation(interval=4, nugget=0.1),
    )
    sampler.run(4, start)
    sampler.continue_run(4)
    record = sampler.currentState.adaptationState.moments
    retained = np.asarray(sampler.chain.trajectory)[1:]
    assert record.nData == 8
    np.testing.assert_allclose(record.mean, retained.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(record.m2) / 7, np.cov(retained, rowvar=False), atol=1e-6,
    )
    fixed = sampler.freeze_adaptation()
    assert fixed.chain.length == 0
    np.testing.assert_allclose(
        fixed.proposalCovariance.to_dense(),
        np.cov(retained, rowvar=False) + 0.1 * np.eye(2), atol=1e-6,
    )
    if backend.name == 'jax':
        execute = sampler.compiled_trajectory(4)
        first = execute(start, backend.random_state(19))
        resumed = execute(first.finalState, first.rng)
        compiled = resumed.finalState.adaptationState.moments
        assert compiled.nData == 8
        np.testing.assert_allclose(compiled.mean, record.mean, atol=1e-6)
        np.testing.assert_allclose(compiled.m2, record.m2, atol=1e-6)
