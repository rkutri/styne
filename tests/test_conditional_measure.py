import numpy as np

from styne.parameter.block import BlockParameter
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian
from styne.statistics.measure import ConditionalMeasure


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
