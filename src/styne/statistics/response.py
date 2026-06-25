import numpy as np

from abc import abstractmethod
from scipy.special import expit

from styne.model.model import DifferentiableModel
from styne.parameter.vector import Vector
from styne.statistics.gaussian import Gaussian
from styne.statistics.measure import ProbabilityMeasure
from styne.statistics.poisson import Poisson
from styne.statistics.binomial import Binomial


class ResponseFamily(ProbabilityMeasure):

    @abstractmethod
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def _set_mean(self, mu: np.ndarray) -> None:
        ...

    def simulate(self, eta: np.ndarray, rng=None) -> Vector:
        self._set_mean(self.inverse_link(eta))
        return self.generate_realisation(rng=rng)

    @abstractmethod
    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        ...

    @abstractmethod
    def score(self, y: np.ndarray, model: DifferentiableModel) -> np.ndarray:
        ...

    @abstractmethod
    def draw(self, rng) -> Vector:
        ...


class GaussianResponse(ResponseFamily):

    def __init__(self, covariance):
        self._gaussian = Gaussian(covariance)
        self._gaussian.mean = Vector(np.zeros(covariance.dimension))

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta).ravel()

    def _set_mean(self, mu: np.ndarray) -> None:
        self._gaussian.mean = Vector(mu)

    def draw(self, rng) -> Vector:
        return self._gaussian.draw(rng)

    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        residual = Vector(np.asarray(y).ravel() - np.asarray(eta).ravel())
        previousMean = self._gaussian.mean
        self._gaussian.mean = Vector(np.zeros(self._gaussian.density.domainDimension))
        logLikelihood = self._gaussian.density.evaluate_log(residual)
        self._gaussian.mean = previousMean
        return logLikelihood

    def score(self, y: np.ndarray, model: DifferentiableModel) -> np.ndarray:
        residual = np.asarray(y).ravel() - np.asarray(model.evaluation).ravel()
        return model.adjoint_directional_derivative(
            self._gaussian.density.covariance.apply_inverse(residual)
        )

    @property
    def density(self):
        return self._gaussian.density


_etaFloor = -500.0
_etaCeil = 30.0


class PoissonResponse(ResponseFamily):

    def __init__(self):
        self._poisson = Poisson()

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(np.asarray(eta).ravel(), _etaFloor, _etaCeil))

    def _set_mean(self, mu: np.ndarray) -> None:
        self._poisson.rate = mu

    def draw(self, rng) -> Vector:
        return self._poisson.draw(rng)

    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        eta = np.asarray(eta).ravel()
        etaClamped = np.clip(eta, _etaFloor, _etaCeil)
        return float(np.sum(
            y * etaClamped - np.exp(etaClamped)
        ))

    def score(self, y: np.ndarray, model: DifferentiableModel) -> np.ndarray:
        eta = np.asarray(model.evaluation).ravel()
        return model.adjoint_directional_derivative(
            np.asarray(y).ravel() - np.exp(np.clip(eta, _etaFloor, _etaCeil))
        )




class BinomialResponse(ResponseFamily):

    def __init__(self, n: int):
        self._n = n
        self._binomial = Binomial(n)

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return expit(np.asarray(eta).ravel())

    def _set_mean(self, mu: np.ndarray) -> None:
        self._binomial.prob = mu

    def draw(self, rng) -> Vector:
        return self._binomial.draw(rng)

    def log_likelihood(self, y: np.ndarray, eta: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        eta = np.asarray(eta).ravel()
        return float(np.sum(y * eta - self._n * np.logaddexp(0, eta)))

    def score(self, y: np.ndarray, model: DifferentiableModel) -> np.ndarray:
        p = expit(np.asarray(model.evaluation).ravel())
        return model.adjoint_directional_derivative(
            np.asarray(y).ravel() - self._n * p
        )
