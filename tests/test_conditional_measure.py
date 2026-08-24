import numpy as np

from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.parameter.block import BlockParameter
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.conditional import MetropolisWithinGibbsConditional
from styne.statistics.gaussian import Gaussian
from styne.statistics.measure import ConditionalMeasure
from styne.statistics.radonnikodym import RadonNikodym


class ConditionalGaussian(ConditionalMeasure):
    def __init__(self):
        self._gaussian = Gaussian(
            IIDCovarianceMatrix(1, 1.0), Vector(np.zeros(1))
        )

    @property
    def blockDimension(self):
        return 1

    @property
    def density(self):
        return self._gaussian.density

    def condition_on(self, state):
        self._gaussian.mean = state.block(0)

    def draw(self, rng):
        return self._gaussian.draw(rng)


def test_condition_returns_independent_state():
    conditional = ConditionalGaussian()
    firstState = BlockParameter([Vector([1.0])])
    secondState = BlockParameter([Vector([2.0])])

    first = conditional.condition(firstState)
    second = conditional.condition(secondState)

    assert first is not conditional
    np.testing.assert_array_equal(first.density.mean.coordinate, [1.0])
    np.testing.assert_array_equal(second.density.mean.coordinate, [2.0])
    assert conditional._gaussian.mean is not first.density.mean


def test_metropolis_within_gibbs_condition_isolates_sampler_target():
    reference = Gaussian(IIDCovarianceMatrix(1, 1.0), Vector([0.0]))
    target = RadonNikodym(reference, reference.density)
    sampler = MetropolisedRandomWalk(
        target,
        IIDCovarianceMatrix(1, 1.0),
        AcceptanceRateDiagnostics(),
    )
    conditional = MetropolisWithinGibbsConditional(sampler, blockIdx=0)
    state = BlockParameter([Vector([1.0])])

    conditioned = conditional.condition(state)

    assert conditioned is not conditional
    assert conditioned._sampler is not sampler
    assert conditioned._sampler.target is not sampler.target
    assert conditioned._sampler._chain is not sampler._chain
    assert conditioned._sampler.diagnostics is not sampler.diagnostics
    np.testing.assert_array_equal(conditioned._currentBlock.coordinate, [1.0])
