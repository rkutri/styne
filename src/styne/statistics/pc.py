import numpy as np
from numpy import log

from styne.statistics.interface import DensityInterface
from styne.parameter.parameter import Parameter


class MaternRangePCPrior(DensityInterface):
    """
    Penalised Complexity (PC) prior for the range parameter rho.

    Parameters
    ----------
    rho0 : float
        Reference range.
    alphaRho : float
        Tail probability, $P(\rho < \rho_0) = \alpha_\rho$.
    """

    def __init__(self, rho0: float, alphaRho: float):
        if not (0.0 < alphaRho < 1.0):
            raise ValueError("alphaRho must be in (0, 1)")
        self._rateParam = -rho0 * log(alphaRho)

    @property
    def domainType(self):
        return Parameter

    @property
    def domainDimension(self) -> int:
        return 1

    def evaluate_log(self, parameter: Parameter) -> float:
        """
        Log-density of the PC prior at `parameter`. Properly normalised,
        includes the full rate-parameter normalising constant, unlike the
        likelihood-side `evaluate_log` methods in this batch.

        Parameters
        ----------
        parameter : Parameter
            Range value to evaluate at.

        Returns
        -------
        float
        """
        rho = float(np.asarray(parameter.coordinate).ravel()[0])
        if rho <= 0:
            return -np.inf
        return log(self._rateParam) - 2. * log(rho) - self._rateParam / rho


class MaternSigmaPCPrior(DensityInterface):
    """
    Penalised Complexity (PC) prior for the marginal standard deviation
    sigma.

    Parameters
    ----------
    sigma0 : float
        Reference standard deviation.
    alphaSigma : float
        Tail probability, $P(\sigma > \sigma_0) = \alpha_\sigma$.
    """

    def __init__(self, sigma0: float, alphaSigma: float):
        if not (0.0 < alphaSigma < 1.0):
            raise ValueError("alphaSigma must be in (0, 1)")
        self._rateParam = -log(alphaSigma) / sigma0

    @property
    def domainType(self):
        return Parameter

    @property
    def domainDimension(self) -> int:
        return 1

    def evaluate_log(self, parameter: Parameter) -> float:
        """
        Log-density of the PC prior at `parameter`. Properly normalised, same
        as `MaternRangePCPrior.evaluate_log`.

        Parameters
        ----------
        parameter : Parameter
            Standard deviation value to evaluate at.

        Returns
        -------
        float
        """
        sigma = float(np.asarray(parameter.coordinate).ravel()[0])
        if sigma < 0:
            return -np.inf
        return log(self._rateParam) - self._rateParam * sigma


class JointMaternPCPrior(DensityInterface):
    """
    Joint PC prior for the Matern hyperparameters (rho, sigma).

    Parameters
    ----------
    rho0 : float
        Reference range.
    alphaRho : float
        Tail probability for the range component.
    sigma0 : float
        Reference standard deviation.
    alphaSigma : float
        Tail probability for the standard deviation component.
    """

    def __init__(self, rho0: float, alphaRho: float, sigma0: float, alphaSigma: float):
        self._rhoPrior = MaternRangePCPrior(rho0, alphaRho)
        self._sigmaPrior = MaternSigmaPCPrior(sigma0, alphaSigma)

    @property
    def domainType(self):
        from styne.parameter.vector import Vector
        return Vector

    @property
    def domainDimension(self) -> int:
        return 2

    def evaluate_log(self, parameter: Parameter) -> float:
        """
        Joint log-density of the PC prior at `parameter`. Properly normalised,
        sum of the two component priors' normalised log-densities, valid under
        the independence assumption between rho and sigma.

        Parameters
        ----------
        parameter : Parameter
            `(rho, sigma)` values to evaluate at.

        Returns
        -------
        float
        """
        coord = np.asarray(parameter.coordinate).ravel()
        rho, sigma = coord[0], coord[1]

        if rho <= 0 or sigma < 0:
            return -np.inf

        logRho = log(self._rhoPrior._rateParam) - 2. * log(rho) - \
            self._rhoPrior._rateParam / rho
        logSigma = log(self._sigmaPrior._rateParam) - \
            self._sigmaPrior._rateParam * sigma

        return float(logRho + logSigma)

    def evaluate_log_gradient(self, parameter: Parameter) -> np.ndarray:
        """Gradient of log-prior with respect to [rho, sigma]."""
        coord = np.asarray(parameter.coordinate).ravel()
        rho, sigma = coord[0], coord[1]

        if rho <= 0 or sigma < 0:
            return np.zeros(2)

        gradRho = -2.0 / rho + self._rhoPrior._rateParam / rho**2
        gradSigma = -self._sigmaPrior._rateParam

        return np.array([gradRho, gradSigma])
