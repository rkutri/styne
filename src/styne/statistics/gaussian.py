from numpy import ndarray, log, pi
from numpy.random import Generator

from styne.backend import infer_backend
from styne.parameter.parameter import Parameter
from styne.statistics.interface import DensityInterface
from styne.statistics.measure import AbsolutelyContinuousProbabilityMeasure
from styne.statistics.covariance import CovarianceMatrix
from styne.model.representation.expansion import backend_constant

_LOG2PI = log(2. * pi)


class GaussianDensity(DensityInterface):
    r"""
    Log-density of a multivariate Gaussian.

    The mean must be set before evaluation. The unnormalised log-density is

    $-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)$

    Parameters
    ----------
    covariance : CovarianceMatrix
        Covariance matrix of the distribution.

    Notes
    -----
    The mean is required and must be set before calling ``evaluate_log``,
    ``draw``, or accessing ``domainType`` / ``domainDimension``. It can be
    Use :meth:`with_mean` or :meth:`with_covariance` to create a modified
    density; instances are immutable.
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

    def with_mean(self, mean: Parameter):
        return type(self)(self._cov, mean)

    @mean.setter
    def mean(self, mean: Parameter):
        # Compatibility for pre-0.3 callers; new code should use with_mean.
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

    def with_covariance(self, covariance: CovarianceMatrix):
        self._validate_covariance(covariance)
        return type(self)(covariance, self._mean)

    @covariance.setter
    def covariance(self, covariance: CovarianceMatrix):
        # Compatibility for pre-0.3 callers; new code should use with_covariance.
        self._validate_covariance(covariance)
        self._cov = covariance

    def evaluate_log(
            self, parameter: Parameter, normalised=False
    ):
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
        half = backend_constant(0.5, x)
        logDens = -half * self._cov.dual_quadratic_form(x)

        if normalised:
            logDet = self._cov.log_determinant()
            log2pi = backend_constant(_LOG2PI, x)
            logDens -= half * (self.domainDimension * log2pi + logDet)

        return logDens

    # Retained concrete legacy derivative API.
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

    # Retained concrete legacy derivative API.
    def evaluate_log_hessian(self, parameter: Parameter) -> ndarray:
        """
        Hessian of the log-density with respect to the parameter.

        Returns
        -------
        ndarray
            -Sigma^{-1}.
        """
        coordinates = self._mean.coordinate
        backend = infer_backend(coordinates)
        metadata = backend.metadata(coordinates)
        identity = backend.eye(
            self.domainDimension, dtype=metadata.dtype, device=metadata.device
        )
        return -self._cov.apply_inverse(identity)


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

    def with_mean(self, mean: Parameter):
        return type(self)(self.covariance, mean)

    @mean.setter
    def mean(self, mean: Parameter):
        self._density.mean = mean

    @property
    def covariance(self) -> CovarianceMatrix:
        return self._density.covariance

    def with_covariance(self, covariance: CovarianceMatrix):
        return type(self)(covariance, self._density._mean)

    @covariance.setter
    def covariance(self, covariance: CovarianceMatrix):
        self._density.covariance = covariance

    @property
    def density(self) -> DensityInterface:
        return self._density

    def sample(self, randomState) -> tuple[Parameter, object]:
        """Draw a reparameterised sample with explicit random-state flow."""
        m = self.mean.coordinate
        backend = infer_backend(m)
        metadata = backend.metadata(m)
        xi, nextState = backend.normal(
            randomState, m.shape, dtype=metadata.dtype, device=metadata.device
        )
        colouredXi = self.density.covariance.apply_chol_factor(xi)

        return self.mean.with_coordinate(m + colouredXi), nextState

    def draw(self, rng: Generator) -> Parameter:
        return self.sample(rng)[0]
