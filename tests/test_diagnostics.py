import pytest
import numpy as np

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


if __name__ == "__main__":
    pytest.main()
