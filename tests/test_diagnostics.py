import pytest
import numpy as np

from styne.backend import get_backend
from styne.mcmc.transition import TransitionData
from styne.parameter.vector import Vector
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.statistics.welford import WelfordAccumulator


@pytest.mark.parametrize("testLag", [5, 500, 50000])
def test_acceptance_rate_diagnostics(testLag):
    """
    Test AcceptanceRateDiagnostics for global and rolling acceptance rates.
    """
    diagnostics = AcceptanceRateDiagnostics(window=testLag)

    # Create transition data
    transitions = [
        TransitionData(state=None, proposal=None, outcome=outcome)
        for outcome in (
            [TransitionData.ACCEPTED] * testLag
            + [TransitionData.REJECTED] * testLag
        )
    ]

    for t in transitions:
        diagnostics.process(t)

    expectedRate = 0.5
    assert np.isclose(diagnostics.global_acceptance_rate(), expectedRate)

    diagnostics.clear()
    assert len(diagnostics._recent) == 0


@pytest.mark.parametrize("paramDim",
                         [(10, 1),
                          (10, 3),
                          (1000, 5),
                          (10000, 100)])
def test_moment_diagnostics(paramDim):
    """
    Test WelfordAccumulator against NumPy implementations of mean and variance.
    """
    accumulator = WelfordAccumulator()

    stateVectors = [np.random.randn(paramDim[1]) for _ in range(paramDim[0])]

    for vector in stateVectors:
        transitionData = TransitionData(
            state=None,
            proposal=Vector(vector),
            outcome=TransitionData.ACCEPTED)
        accumulator.update(transitionData.proposal.coordinate)

    computedMean = accumulator.mean()
    computedVar = accumulator.marginal_variance()

    # Compute expected results using NumPy
    expectedMean = np.mean(stateVectors, axis=0)
    expectedVar = np.var(stateVectors, axis=0, ddof=1)

    # Assertions
    assert np.allclose(computedMean, expectedMean), \
        f"mean mismatch: {computedMean} vs. {expectedMean}"
    assert np.allclose(computedVar, expectedVar), \
        f"variance mismatch: {computedVar} vs. {expectedVar}"

    accumulator.reset()
    assert accumulator.nData == 0
    assert accumulator.mean() is None


def test_welford_insufficient_data():
    """
    Test that WelfordAccumulator raises RuntimeError if variance is requested with < 2 samples.
    """
    accumulator = WelfordAccumulator()
    
    with pytest.raises(RuntimeError, match="Insufficient data"):
        accumulator.marginal_variance()
        
    accumulator.update(np.array([1.0, 2.0]))
    
    with pytest.raises(RuntimeError, match="Insufficient data"):
        accumulator.marginal_variance()
        
    accumulator.update(np.array([1.5, 2.5]))
    
    # Should not raise now
    accumulator.marginal_variance()


@pytest.mark.parametrize('backendName', ['numpy', 'jax', 'pytorch'])
def test_welford_records_match_numpy_without_mutation(backendName):
    if backendName != 'numpy':
        pytest.importorskip('torch' if backendName == 'pytorch' else backendName)
    backend = get_backend(backendName)
    samples = np.random.default_rng(2).normal(size=(20, 3)).astype('float32')
    initial = WelfordAccumulator.initial_record(backend.asarray(samples[0]))
    record = initial
    for sample in samples:
        record = WelfordAccumulator.update_record(record, backend.asarray(sample))
    assert record.nData == 20
    assert initial.nData == 0
    np.testing.assert_array_equal(np.asarray(initial.mean), 0)
    np.testing.assert_array_equal(np.asarray(initial.m2), 0)
    np.testing.assert_allclose(np.asarray(record.mean), samples.mean(axis=0), atol=1e-6)
    expected = samples.var(axis=0, ddof=1)
    np.testing.assert_allclose(np.asarray(record.m2) / 19, expected,
                               rtol=1e-6, atol=1e-7)
    if backendName == 'jax':
        import jax
        compiled = jax.jit(lambda data: jax.lax.scan(
            lambda current, sample: (
                WelfordAccumulator.update_record(current, sample), None
            ), initial, data
        )[0])(backend.asarray(samples))
        np.testing.assert_allclose(compiled.mean, record.mean, atol=1e-6)
        np.testing.assert_allclose(compiled.m2, record.m2, atol=1e-6)
        assert compiled.mean.dtype == initial.mean.dtype


@pytest.mark.parametrize('samples', [[1.0, 2.0, 4.0],
                                     [[1.0, 2.0], [3.0, 4.0], [2.0, 6.0]]])
def test_eager_welford_keeps_scalar_and_sequence_inputs(samples):
    accumulator = WelfordAccumulator()
    for sample in samples:
        accumulator.update(sample)
    np.testing.assert_allclose(accumulator.mean(), np.mean(samples, axis=0))
    np.testing.assert_allclose(accumulator.marginal_variance(),
                               np.var(samples, axis=0, ddof=1))
    assert accumulator.nData == len(samples)
    accumulator.reset()
    assert accumulator.nData == 0
    assert accumulator.mean() is None


if __name__ == "__main__":
    pytest.main()
