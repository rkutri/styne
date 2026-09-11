import numpy as np

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

    def adjoint_derivative(
            self, parameter, cotangent: np.ndarray) -> np.ndarray:
        return cotangent


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


def test_poisson_log_likelihood_preserves_unclipped_target():
    """Extreme predictors use the Poisson target, not a clipped surrogate."""
    response = PoissonResponse()
    yVal = np.array([1.0, 2.0])

    # Normal inputs
    etaNormal = np.array([0.0, 1.0])
    likelihoodNormal = response.log_likelihood(yVal, etaNormal)
    assert np.isfinite(likelihoodNormal)

    etaLarge = np.array([35.0])
    expected = 35.0 - np.exp(35.0)
    assert response.log_likelihood(np.array([1.0]), etaLarge) == expected

    with np.errstate(over="ignore", invalid="ignore"):
        likelihoodExtreme = response.log_likelihood(
            np.array([1.0, 1.0]), np.array([800.0, -np.inf])
        )
    assert np.isneginf(likelihoodExtreme)


def test_poisson_score_preserves_unclipped_target():
    """The score remains the derivative of the unclipped log likelihood."""
    response = PoissonResponse()
    yVal = np.array([1.0, 1.0, 1.0, 1.0])
    etaExtreme = np.array([500.0, 800.0, np.inf, -np.inf])
    with np.errstate(over="ignore"):
        scoreVal = response.score(yVal, etaExtreme)

    assert scoreVal[0] < 0.0
    assert np.isneginf(scoreVal[1])
    assert np.isneginf(scoreVal[2])
    assert scoreVal[3] == 1.0


def test_pmala_drift_preserves_non_finite_gradient():
    refCovariance = IIDCovarianceMatrix(2, 1.0)
    refMean = Vector(np.zeros(2))
    priorVal = Gaussian(refCovariance, refMean)

    # Setup non-finite gradient density (contains nan or inf)
    nonFiniteGrad = np.array([np.inf, np.nan])
    derivVal = MockNonFiniteGradientDensity(nonFiniteGrad)
    targetVal = RadonNikodym(priorVal, derivVal)

    betaVal = 0.5
    proposalVal = PMALAProposal(
        targetVal, betaVal, derivVal.evaluate_log_gradient
    )
    stateVal = Vector(np.array([2.0, 3.0]))

    # Compute proposal drift
    computedDrift = proposalVal._drift(stateVal)

    assert not np.all(np.isfinite(computedDrift))
