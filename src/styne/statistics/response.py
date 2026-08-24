from abc import abstractmethod

from styne.backend import infer_backend
from styne.parameter.vector import Vector
from styne.statistics.gaussian import Gaussian
from styne.statistics.measure import ProbabilityMeasure

class ResponseFamily(ProbabilityMeasure):
    """Backend-native, stateless observation response family."""

    @abstractmethod
    def inverse_link(self, eta):
        ...

    @abstractmethod
    def sample(self, eta, randomState):
        """Return ``(sample, nextRandomState)`` at predictor ``eta``."""
        ...

    def simulate(self, eta, rng=None):
        return self.sample(eta, rng)[0]

    @abstractmethod
    def log_likelihood(self, y, eta):
        ...


class GaussianResponse(ResponseFamily):
    def __init__(self, covariance):
        self._covariance = covariance

    def inverse_link(self, eta):
        return eta.reshape((-1,))

    def sample(self, eta, randomState):
        backend = infer_backend(eta)
        metadata = backend.metadata(eta)
        noise, nextState = backend.normal(
            randomState, eta.shape, dtype=metadata.dtype, device=metadata.device
        )
        noise = self._covariance.apply_chol_factor(noise)
        return Vector(eta.reshape((-1,)) + noise.reshape((-1,))), nextState

    def log_likelihood(self, y, eta):
        residual = Vector(y.reshape((-1,)) - eta.reshape((-1,)))
        backend = infer_backend(residual.coordinate)
        metadata = backend.metadata(residual.coordinate)
        zero = backend.zeros(
            residual.coordinate.shape, dtype=metadata.dtype, device=metadata.device
        )
        return Gaussian(self._covariance, Vector(zero)).density.evaluate_log(residual)

    # Compatibility helper; likelihood code should use automatic differentiation.
    def score(self, y, evaluation):
        residual = y.reshape((-1,)) - evaluation.reshape((-1,))
        return self._covariance.apply_inverse(residual)

    @property
    def density(self):
        """Zero-mean Gaussian density retained for compatibility."""
        return Gaussian(self._covariance).density


class PoissonResponse(ResponseFamily):
    def inverse_link(self, eta):
        ns = infer_backend(eta).namespace
        return ns.exp(eta.reshape((-1,)))

    def sample(self, eta, randomState):
        backend = infer_backend(eta)
        values, nextState = backend.poisson(randomState, self.inverse_link(eta))
        return Vector(values), nextState

    def log_likelihood(self, y, eta):
        ns = infer_backend(eta).namespace
        eta = eta.reshape((-1,))
        return ns.sum(y.reshape((-1,)) * eta - ns.exp(eta))

    def score(self, y, evaluation):
        return y.reshape((-1,)) - self.inverse_link(evaluation)


class BinomialResponse(ResponseFamily):
    def __init__(self, n: int):
        self._n = n

    def inverse_link(self, eta):
        return infer_backend(eta).namespace.sigmoid(eta.reshape((-1,)))

    def sample(self, eta, randomState):
        backend = infer_backend(eta)
        values, nextState = backend.binomial(
            randomState, self._n, self.inverse_link(eta)
        )
        return Vector(values), nextState

    def log_likelihood(self, y, eta):
        ns = infer_backend(eta).namespace
        eta = eta.reshape((-1,))
        return ns.sum(y.reshape((-1,)) * eta - self._n * ns.logaddexp(0, eta))

    def score(self, y, evaluation):
        return y.reshape((-1,)) - self._n * self.inverse_link(evaluation)
