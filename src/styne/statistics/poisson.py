import numpy as np

from numpy.random import Generator

from styne.parameter.vector import Vector
from styne.statistics.measure import ProbabilityMeasure


class Poisson(ProbabilityMeasure):

    def __init__(self):
        self._rate = None

    @property
    def rate(self) -> np.ndarray:
        return self._rate

    @rate.setter
    def rate(self, rate) -> None:
        self._rate = np.asarray(rate, dtype=float)

    @property
    def mean(self) -> np.ndarray:
        return self._rate

    def draw(self, rng: Generator) -> Vector:
        return Vector(rng.poisson(self._rate).astype(float))
