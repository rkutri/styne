import numpy as np


class WelfordAccumulator():

    def __init__(self):
        self._dataSize = 0
        self._mean = None
        self._welfordM2 = None

    def mean(self):
        return self._mean

    def marginal_variance(self):
        """
        unbiased estimate of the marginal variance
        """
        if self._dataSize < 2:
            raise RuntimeError("Insufficient data for variance estimation.")
        return self._welfordM2 / (self._dataSize - 1)

    def condition_number(self):

        margVar = self.marginal_variance()

        tol = 1e-12
        if np.min(margVar) < tol:
            raise RuntimeError("Singular marginal variance.")

        return np.max(margVar) / np.min(margVar)

    @property
    def nData(self):
        return self._dataSize

    def update(self, realisation):
        """
        Welford's algorithm for stable computation of variance.
        """
        if self._mean is None:
            self._mean = np.zeros_like(realisation)
            self._welfordM2 = np.zeros_like(realisation)

        delta = realisation - self._mean

        # update mean
        self._dataSize += 1
        self._mean += delta / self._dataSize

        delta2 = realisation - self._mean

        # update squared differences
        self._welfordM2 += delta * delta2

    def reset(self):
        self._dataSize = 0
        self._mean = None
        self._welfordM2 = None
