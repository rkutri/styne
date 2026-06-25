import numpy as np
from numpy import ndarray, log, pi
from numpy.random import Generator

from styne.parameter.parameter import Parameter
from styne.statistics.interface import DensityInterface
from styne.statistics.measure import AbsolutelyContinuousProbabilityMeasure
from styne.statistics.covariance import CovarianceMatrix

_LOG2PI = log(2. * pi)


class GaussianDensity(DensityInterface):
    """
    Log-density of a multivariate Gaussian.

    The mean must be set before evaluation. The unnormalised log-density is

        -0.5 * (x - mu)^T Sigma^{-1} (x - mu)

    Parameters
    ----------
    covariance : CovarianceMatrix
        Covariance matrix of the distribution.

    Notes
    -----
    The mean is required and must be set before calling ``evaluate_log``,
    ``draw``, or accessing ``domainType`` / ``domainDimension``. It can be
    updated at any time via the ``mean`` setter.
    """

    def __init__(
            self, covariance: CovarianceMatrix, mean: Parameter = None) -> None:

        self._validate_covariance(covariance)

        self._mean = mean
        self._cov = covariance

    @staticmethod
    def _validate_covariance(covariance):
        if not isinstance(covariance, CovarianceMatrix):
            raise ValueError(
                "Covariance of Gaussian density must be of type "
                f"CovarianceMatrix. Got {type(covariance)} instead."
            )

    def _require_mean(self):
        if self._mean is None:
            raise RuntimeError("Mean not set.")

    @property
    def mean(self) -> Parameter:
        self._require_mean()
        return self._mean

    @mean.setter
    def mean(self, mean: Parameter):
        self._mean = mean

    @property
    def domainType(self):
        self._require_mean()
        return type(self._mean)

    @property
    def domainDimension(self) -> int:
        self._require_mean()
        return self._mean.dimension

    @property
    def covariance(self) -> CovarianceMatrix:
        return self._cov

    @covariance.setter
    def covariance(self, covariance: CovarianceMatrix):
        self._validate_covariance(covariance)
        self._cov = covariance

    def evaluate_log(
            self, parameter: Parameter, normalised=False
    ) -> float:
        """
        Evaluate the log-density at 'parameter'.

        Parameters
        ----------
        parameter : Parameter
            Point at which to evaluate.
        normalised : bool, optional
            If True, include the normalisation constant.

        Returns
        -------
        float
            Log-density value.
        """
        self._require_mean()

        x = parameter.coordinate - self._mean.coordinate
        logDens = -0.5 * self._cov.dual_quadratic_form(x)

        if normalised:
            logDet = self._cov.log_determinant()
            logDens -= 0.5 * (self.domainDimension * _LOG2PI + logDet)

        return logDens

    # satisfies the DifferentiableDensity protocol
    def evaluate_log_gradient(self, parameter: Parameter) -> ndarray:
        """
        Gradient of the log-density with respect to the parameter.

        Returns
        -------
        ndarray
            -Sigma^{-1} (x - mu).
        """
        self._require_mean()

        v = parameter.coordinate - self._mean.coordinate
        return -self._cov.apply_inverse(v)

    # satisfies the TwiceDifferentiableDensity protocol
    def evaluate_log_hessian(self, parameter: Parameter) -> ndarray:
        """
        Hessian of the log-density with respect to the parameter.

        Returns
        -------
        ndarray
            -Sigma^{-1}.
        """
        return -self._cov.apply_inverse(np.eye(self.domainDimension))


class Gaussian(AbsolutelyContinuousProbabilityMeasure):
    """
    Multivariate Gaussian distribution.

    Parameters
    ----------
    covariance : CovarianceMatrix
        Covariance matrix of the distribution.
    """

    def __init__(self, covariance: CovarianceMatrix, mean: Parameter = None):
        self._density = GaussianDensity(covariance, mean)

    @property
    def mean(self) -> Parameter:
        return self._density.mean

    @mean.setter
    def mean(self, mean: Parameter):
        self._density.mean = mean

    @property
    def covariance(self) -> CovarianceMatrix:
        return self._density.covariance

    @covariance.setter
    def covariance(self, covariance: CovarianceMatrix):
        self._density.covariance = covariance

    @property
    def density(self) -> DensityInterface:
        return self._density

    def draw(self, rng: Generator) -> Parameter:
        """
        Draw a sample using the provided random number generator.

        Parameters
        ----------
        rng : Generator
            NumPy random generator instance.

        Returns
        -------
        Parameter
            A sample from this Gaussian.
        """
        m = self.mean.coordinate
        xi = rng.standard_normal(self.density.domainDimension)
        colouredXi = self.density.covariance.apply_chol_factor(xi)

        realisation = self.mean.clone()
        realisation.coordinate = m + colouredXi

        return realisation
