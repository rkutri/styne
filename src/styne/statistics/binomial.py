import numpy as np

from numpy.random import Generator

from styne.parameter.vector import Vector
from styne.statistics.measure import ProbabilityMeasure


class Binomial(ProbabilityMeasure):
    """
    Binomial probability measure.

    Parameters
    ----------
    n : int
        Number of trials.
    """

    def __init__(self, n: int):
        self._n = n
        self._prob = None

    @property
    def prob(self) -> np.ndarray:
        return self._prob

    @prob.setter
    def prob(self, prob) -> None:
        self._prob = np.asarray(prob, dtype=float)

    def draw(self, rng: Generator) -> Vector:
        """
        Draw a Binomial sample at the current success probability.

        Parameters
        ----------
        rng : Generator

        Returns
        -------
        Vector
        """
        return Vector(rng.binomial(self._n, self._prob).astype(float))
