from math import log

from styne.backend import infer_backend
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

    def evaluate_log(self, parameter: Parameter):
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
        coordinate = parameter.coordinate
        rho = coordinate[..., 0]
        backend = infer_backend(coordinate)
        namespace = backend.namespace
        metadata = backend.metadata(coordinate)
        rate = backend.asarray(
            self._rateParam,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        one = backend.ones(
            rho.shape, dtype=metadata.dtype, device=metadata.device
        )
        safeRho = namespace.where(rho > 0, rho, one)
        value = namespace.log(rate) - 2. * namespace.log(safeRho) \
            - rate / safeRho
        negativeInfinity = backend.asarray(
            float("-inf"), dtype=metadata.dtype, device=metadata.device
        )
        return namespace.where(rho > 0, value, negativeInfinity)


class MaternSigmaPCPrior(DensityInterface):
    r"""
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

    def evaluate_log(self, parameter: Parameter):
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
        coordinate = parameter.coordinate
        sigma = coordinate[..., 0]
        backend = infer_backend(coordinate)
        namespace = backend.namespace
        metadata = backend.metadata(coordinate)
        rate = backend.asarray(
            self._rateParam,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        value = namespace.log(rate) - rate * sigma
        negativeInfinity = backend.asarray(
            float("-inf"), dtype=metadata.dtype, device=metadata.device
        )
        return namespace.where(sigma >= 0, value, negativeInfinity)


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

    def evaluate_log(self, parameter: Parameter):
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
        coordinate = parameter.coordinate
        rho, sigma = coordinate[..., 0], coordinate[..., 1]
        backend = infer_backend(coordinate)
        namespace = backend.namespace
        metadata = backend.metadata(coordinate)
        rhoRate = backend.asarray(
            self._rhoPrior._rateParam,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        sigmaRate = backend.asarray(
            self._sigmaPrior._rateParam,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        one = backend.ones(
            rho.shape, dtype=metadata.dtype, device=metadata.device
        )
        safeRho = namespace.where(rho > 0, rho, one)
        logRho = namespace.log(rhoRate) - 2. * namespace.log(safeRho) \
            - rhoRate / safeRho
        logSigma = namespace.log(sigmaRate) - sigmaRate * sigma
        value = logRho + logSigma
        invalid = (rho <= 0) | (sigma < 0)
        negativeInfinity = backend.asarray(
            float("-inf"), dtype=metadata.dtype, device=metadata.device
        )
        return namespace.where(invalid, negativeInfinity, value)

    def evaluate_log_gradient(self, parameter: Parameter):
        """Gradient of log-prior with respect to [rho, sigma]."""
        coordinate = parameter.coordinate
        rho, sigma = coordinate[..., 0], coordinate[..., 1]
        backend = infer_backend(coordinate)
        namespace = backend.namespace
        metadata = backend.metadata(coordinate)
        rhoRate = backend.asarray(
            self._rhoPrior._rateParam,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        sigmaRate = backend.asarray(
            self._sigmaPrior._rateParam,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        one = backend.ones(
            rho.shape, dtype=metadata.dtype, device=metadata.device
        )
        safeRho = namespace.where(rho > 0, rho, one)
        gradRho = -2.0 / safeRho + rhoRate / safeRho**2
        gradSigma = backend.ones(
            rho.shape, dtype=metadata.dtype, device=metadata.device
        ) * -sigmaRate
        gradient = namespace.stack([gradRho, gradSigma], axis=-1)
        invalid = (rho <= 0) | (sigma < 0)
        invalid = namespace.expand_dims(invalid, axis=-1)
        zeros = backend.zeros(
            gradient.shape, dtype=metadata.dtype, device=metadata.device
        )
        return namespace.where(invalid, zeros, gradient)
