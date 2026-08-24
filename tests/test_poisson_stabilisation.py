import numpy as np
import pytest
import warnings

from styne.statistics.response import PoissonResponse
from styne.model.forwardmap import ForwardMap
from styne.parameter.vector import Vector
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.interface import DensityInterface
from styne.mcmc.method.pmala import PMALAProposal


class MockDifferentiableForwardMap(ForwardMap):
    """Mock model returning pre-defined evaluation and computing adjoint."""

    def __init__(self, evaluation: np.ndarray):
        super().__init__()
        self._response = evaluation

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self) -> int:
        return len(self._response)

    def _prepare(self, parameter):
        return self._response

    def _evaluate(self, preparedState):
        return preparedState

    def adjoint_directional_derivative(self, vector: np.ndarray) -> np.ndarray:
        return vector


class MockNonFiniteGradientDensity(DensityInterface):
    """Mock density that returns non-finite gradient evaluations."""

    def __init__(self, gradientValue: np.ndarray):
        self._gradientValue = gradientValue

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self) -> int:
        return len(self._gradientValue)

    def evaluate_log(self, parameter) -> float:
        return 0.0

    def evaluate_log_gradient(self, parameter) -> np.ndarray:
        return self._gradientValue


def test_poisson_log_likelihood_limits():
    """Verify log-likelihood does not raise overflow warnings and stays finite."""
    response = PoissonResponse()
    yVal = np.array([1.0, 2.0])

    # Normal inputs
    etaNormal = np.array([0.0, 1.0])
    likelihoodNormal = response.log_likelihood(yVal, etaNormal)
    assert np.isfinite(likelihoodNormal)

    # Inputs at the boundary or exceeding clamp limit, including infinity
    etaExtreme = np.array([500.0, 800.0, np.inf, -np.inf])
    yExtreme = np.array([1.0, 1.0, 1.0, 1.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        likelihoodExtreme = response.log_likelihood(yExtreme, etaExtreme)

    assert np.isfinite(likelihoodExtreme)
    # The linear predictor term is clamped to [_etaFloor, _etaCeil].
    # At eta = inf, it is clamped to 30.0, so it remains extremely negative and finite.
    assert likelihoodExtreme < 0.0



def test_poisson_score_limits():
    """Verify score does not raise overflow warnings and returns finite values."""
    response = PoissonResponse()
    yVal = np.array([1.0, 1.0, 1.0, 1.0])
    etaExtreme = np.array([500.0, 800.0, np.inf, -np.inf])
    modelVal = MockDifferentiableForwardMap(etaExtreme)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scoreVal = response.score(yVal, etaExtreme, modelVal)

    assert np.isfinite(scoreVal).all()
    # At extreme positive eta, score = y - exp(clip(eta)) which should be extremely negative.
    # At eta = -inf, score = y - exp(-500.0) ≈ y - 0 = y > 0 (for y=1.0).
    assert scoreVal[0] < 0.0
    assert scoreVal[1] < 0.0
    assert scoreVal[2] < 0.0
    assert scoreVal[3] > 0.0



def test_pmala_drift_non_finite_fallback():
    """Verify PMALA proposal falls back to pCN drift when gradient is non-finite."""
    refCovariance = IIDCovarianceMatrix(2, 1.0)
    refMean = Vector(np.zeros(2))
    priorVal = Gaussian(refCovariance, refMean)

    # Setup non-finite gradient density (contains nan or inf)
    nonFiniteGrad = np.array([np.inf, np.nan])
    derivVal = MockNonFiniteGradientDensity(nonFiniteGrad)
    targetVal = RadonNikodym(priorVal, derivVal)

    betaVal = 0.5
    proposalVal = PMALAProposal(targetVal, betaVal)
    stateVal = Vector(np.array([2.0, 3.0]))

    # Compute proposal drift
    computedDrift = proposalVal._drift(stateVal)

    # Expected drift is pCN drift since gradient is non-finite: m + sqrt(1 - beta^2) * (x - m)
    expectedDrift = refMean.coordinate + np.sqrt(1.0 - betaVal**2) * (
        stateVal.coordinate - refMean.coordinate
    )

    assert np.allclose(computedDrift, expectedDrift)
