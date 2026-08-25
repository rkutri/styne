import numpy as np
import pytest

from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.mala import MALAProposal
from styne.mcmc.method.mlda import MLDAProposal
from styne.mcmc.method.mrw import MRWProposal, MetropolisedRandomWalk
from styne.mcmc.method.pcn import PCNProposal
from styne.mcmc.surrogate import SurrogateTransitionMeasure
from styne.model.model import Model
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.data import Data
from styne.statistics.dirac import DiracMeasure
from styne.statistics.gaussian import Gaussian
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse


class CountingModel(Model):

    def __init__(self):
        super().__init__()
        self.evaluationCount = 0
        self._parameter = None

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return 1

    def _interpolate(self, parameter):
        self._parameter = parameter

    def _evaluate(self):
        self.evaluationCount += 1
        self._evaluation = self._parameter.coordinate.copy()


class AlwaysAccept(AcceptanceProbability):

    def log_probability(self, logMHRatio):
        return 0.0


def assert_preserves_current_state(proposal):
    currentState = Vector(np.array([1.0, -2.0]))
    initialCoordinate = currentState.coordinate.copy()
    proposal.state = currentState

    transition = proposal.generate_proposal(np.random.default_rng(11))

    assert transition.state is currentState
    assert transition.proposal is not currentState
    np.testing.assert_array_equal(currentState.coordinate, initialCoordinate)


def test_mrw_preserves_current_state():
    assert_preserves_current_state(MRWProposal(IIDCovarianceMatrix(2, 0.5)))


def test_mala_preserves_current_state():
    proposal = MALAProposal(2, 0.5, lambda state: -state.coordinate)
    assert_preserves_current_state(proposal)


def test_pcn_preserves_current_state():
    reference = Gaussian(
        IIDCovarianceMatrix(2, 1.0),
        Vector(np.zeros(2)),
    )
    assert_preserves_current_state(PCNProposal(reference, 0.5))


@pytest.mark.parametrize("subchainLength", [0, 1])
def test_mlda_preserves_current_state(subchainLength):
    target = Gaussian(
        IIDCovarianceMatrix(2, 1.0),
        Vector(np.zeros(2)),
    ).density
    innerSampler = MetropolisedRandomWalk(
        target,
        IIDCovarianceMatrix(2, 0.5),
        DummyDiagnostics(),
        acceptance=AlwaysAccept(),
        rng=np.random.default_rng(5),
    )
    surrogateMeasure = SurrogateTransitionMeasure(
        innerSampler,
        subchainLength,
        DiracMeasure(),
    )

    assert_preserves_current_state(MLDAProposal(surrogateMeasure))


def test_mh_reuses_cached_current_state_evaluation():
    data = Data(1, np.zeros((1, 1)))
    data.measurement = np.zeros((1, 1))
    model = CountingModel()
    likelihood = RegressionLikelihood(
        data,
        model,
        GaussianResponse(IIDCovarianceMatrix(1, 1.0)),
    )
    sampler = MetropolisedRandomWalk(
        likelihood,
        IIDCovarianceMatrix(1, 1.0),
        DummyDiagnostics(),
        acceptance=AlwaysAccept(),
        rng=np.random.default_rng(7),
    )

    sampler.run(2, Vector(np.zeros(1)))

    assert model.evaluationCount == 3
