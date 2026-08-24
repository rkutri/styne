from styne.backend import infer_backend
from styne.parameter.vector import Vector
from styne.statistics.measure import ProbabilityMeasure


class Binomial(ProbabilityMeasure):
    """Binomial measure with backend-native probabilities and sampling."""

    def __init__(self, n: int, prob=None):
        self._n = n
        self._prob = prob

    def with_prob(self, prob):
        return type(self)(self._n, prob)

    @property
    def prob(self):
        return self._prob

    def sample(self, randomState):
        if self._prob is None:
            raise RuntimeError("Binomial probability not set.")
        values, nextState = infer_backend(self._prob).binomial(
            randomState, self._n, self._prob
        )
        return Vector(values), nextState
