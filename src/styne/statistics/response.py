import numpy as np

from abc import abstractmethod
from scipy.special import expit

from styne.model.forwardmap import DifferentiableModel
from styne.parameter.vector import Vector
from styne.statistics.gaussian import Gaussian
from styne.statistics.measure import ProbabilityMeasure
from styne.statistics.poisson import Poisson
from styne.statistics.binomial import Binomial


class ResponseFamily(ProbabilityMeasure):
    """
    Interface for observation response families, link function, likelihood,
    and simulation, given a linear predictor `eta`.

    Notes
    -----
    Subclasses implement `inverse_link`, `log_likelihood`, `score`, and
    `draw`. `simulate` is concrete here, it sets the family's mean via
    `inverse_link(eta)` then draws through `generate_realisation`.
    """

    @abstractmethod
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Map the linear predictor to the response's natural mean parameter.

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor values.

        Returns
        -------
        np.ndarray
            Mean parameter values on the response's own scale.
        """
        ...

    @abstractmethod
    def _set_mean(self, mu: np.ndarray) -> None:
        ...

    def simulate(self, eta: np.ndarray, rng=None) -> Vector:
        """
        Set the response's mean from a linear predictor, then draw a sample.

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor values.
        rng : Generator, optional

        Returns
        -------
        Vector
        """
        self._set_mean(self.inverse_link(eta))
        return self.generate_realisation(rng=rng)

    @abstractmethod
    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        """
        Log-likelihood of observed data given the linear predictor. Unnormalised,
        drops any additive term that doesn't depend on `eta` (log-factorials,
        binomial coefficients, and similar). Fine for MCMC where only relative
        values across proposals matter, not a properly normalised log-likelihood.

        Parameters
        ----------
        y : np.ndarray
            Observed data.
        eta : np.ndarray
            Linear predictor values.

        Returns
        -------
        float
        """
        ...

    @abstractmethod
    def score(
            self, y: np.ndarray, evaluation: np.ndarray,
            model: DifferentiableModel
    ) -> np.ndarray:
        """
        Adjoint gradient of the log-likelihood with respect to the model's
        parameters, via the model's `adjoint_directional_derivative`.

        Parameters
        ----------
        y : np.ndarray
            Observed data.
        evaluation : np.ndarray
            Linear predictor returned by the forward map.
        model : DifferentiableModel
            Forward model the linear predictor came from.

        Returns
        -------
        np.ndarray
        """
        ...

    @abstractmethod
    def draw(self, rng) -> Vector:
        """
        Draw a sample from the response distribution at its current mean.

        Parameters
        ----------
        rng : Generator

        Returns
        -------
        Vector
        """
        ...


class GaussianResponse(ResponseFamily):
    """
    Gaussian observation response family.

    Parameters
    ----------
    covariance : CovarianceMatrix
        Covariance of the observation noise.
    """

    def __init__(self, covariance):
        self._gaussian = Gaussian(covariance)
        self._gaussian.mean = Vector(np.zeros(covariance.dimension))

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta).ravel()

    def _set_mean(self, mu: np.ndarray) -> None:
        self._gaussian.mean = Vector(mu)

    def draw(self, rng) -> Vector:
        """
        Draw a sample from the current Gaussian mean and covariance.

        Parameters
        ----------
        rng : Generator

        Returns
        -------
        Vector
        """
        return self._gaussian.draw(rng)

    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        """
        Gaussian log-likelihood of `y` given linear predictor `eta`, evaluated
        as a zero-mean density on the residual. Unnormalised, calls the
        underlying `GaussianDensity.evaluate_log` with `normalised=False`,
        drops the `-0.5 * (d * log(2*pi) + log_det)` term.

        Parameters
        ----------
        y : np.ndarray
            Observed data.
        eta : np.ndarray
            Linear predictor values.

        Returns
        -------
        float
        """
        residual = Vector(np.asarray(y).ravel() - np.asarray(eta).ravel())
        previousMean = self._gaussian.mean
        self._gaussian.mean = Vector(np.zeros(self._gaussian.density.domainDimension))
        logLikelihood = self._gaussian.density.evaluate_log(residual)
        self._gaussian.mean = previousMean
        return logLikelihood

    def score(
            self, y: np.ndarray, evaluation: np.ndarray,
            model: DifferentiableModel
    ) -> np.ndarray:
        residual = np.asarray(y).ravel() - np.asarray(evaluation).ravel()
        return model.adjoint_directional_derivative(
            self._gaussian.density.covariance.apply_inverse(residual)
        )

    @property
    def density(self):
        return self._gaussian.density


_etaFloor = -500.0
_etaCeil = 30.0


class PoissonResponse(ResponseFamily):
    """
    Poisson observation response family with a log link, clamped internally
    to `[-500, 30]` before exponentiating to avoid overflow.
    """

    def __init__(self):
        self._poisson = Poisson()

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(np.asarray(eta).ravel(), _etaFloor, _etaCeil))

    def _set_mean(self, mu: np.ndarray) -> None:
        self._poisson.rate = mu

    def draw(self, rng) -> Vector:
        """
        Draw a Poisson sample at the current rate.

        Parameters
        ----------
        rng : Generator

        Returns
        -------
        Vector
        """
        return self._poisson.draw(rng)

    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        """
        Poisson log-likelihood of `y` given linear predictor `eta` on the log
        scale, clamped to `[-500, 30]` before exponentiating. Unnormalised,
        drops the `-log(y!)` term, which doesn't depend on `eta`.

        Parameters
        ----------
        y : np.ndarray
            Observed count data.
        eta : np.ndarray
            Linear predictor values (log rate).

        Returns
        -------
        float
        """
        y = np.asarray(y).ravel()
        eta = np.asarray(eta).ravel()
        etaClamped = np.clip(eta, _etaFloor, _etaCeil)
        return float(np.sum(
            y * etaClamped - np.exp(etaClamped)
        ))

    def score(
            self, y: np.ndarray, evaluation: np.ndarray,
            model: DifferentiableModel
    ) -> np.ndarray:
        eta = np.asarray(evaluation).ravel()
        return model.adjoint_directional_derivative(
            np.asarray(y).ravel() - np.exp(np.clip(eta, _etaFloor, _etaCeil))
        )




class BinomialResponse(ResponseFamily):
    """
    Binomial observation response family with a logit link.

    Parameters
    ----------
    n : int
        Number of trials per observation.
    """

    def __init__(self, n: int):
        self._n = n
        self._binomial = Binomial(n)

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return expit(np.asarray(eta).ravel())

    def _set_mean(self, mu: np.ndarray) -> None:
        self._binomial.prob = mu

    def draw(self, rng) -> Vector:
        """
        Draw a Binomial sample at the current success probability.

        Parameters
        ----------
        rng : Generator

        Returns
        -------
        Vector
        """
        return self._binomial.draw(rng)

    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        """
        Binomial log-likelihood of `y` given linear predictor `eta` on the
        logit scale. Unnormalised, drops the `log(n choose y)` term, which
        doesn't depend on `eta`.

        Parameters
        ----------
        y : np.ndarray
            Observed success counts.
        eta : np.ndarray
            Linear predictor values (log-odds).

        Returns
        -------
        float
        """
        y = np.asarray(y).ravel()
        eta = np.asarray(eta).ravel()
        return float(np.sum(y * eta - self._n * np.logaddexp(0, eta)))

    def score(
            self, y: np.ndarray, evaluation: np.ndarray,
            model: DifferentiableModel
    ) -> np.ndarray:
        p = expit(np.asarray(evaluation).ravel())
        return model.adjoint_directional_derivative(
            np.asarray(y).ravel() - self._n * p
        )
