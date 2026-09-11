from styne.backend import infer_backend
from styne.parameter.vector import Vector
from styne.statistics.measure import ProbabilityMeasure


class Poisson(ProbabilityMeasure):
    """Poisson measure with backend-native rates and sampling."""

    def __init__(self, rate=None):
        self._rate = rate

    def with_rate(self, rate):
        return type(self)(rate)

    @property
    def rate(self):
        return self._rate

    @property
    def mean(self):
        return self._rate

    def sample(self, randomState):
        if self._rate is None:
            raise RuntimeError("Poisson rate not set.")
        values, nextState = infer_backend(self._rate).poisson(
            randomState, self._rate
        )
        return Vector(values), nextState
