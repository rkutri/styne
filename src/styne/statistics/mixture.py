import numpy as np

from styne.backend import infer_backend
from styne.statistics.interface import DensityInterface


class FixedWeights:
    """Normalised, state-independent component probabilities."""

    def __init__(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or not weights.size:
            raise ValueError("Mixture weights must form a non-empty vector.")
        total = np.sum(weights)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) \
                or not np.isfinite(total) or total <= 0.0:
            raise ValueError("Mixture weights must be finite and non-negative.")
        self._weights = weights / total

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def logWeights(self):
        with np.errstate(divide="ignore"):
            return np.log(self._weights)

    def draw(self, rng, like):
        if self._weights.size == 1:
            return 0, rng
        backend = infer_backend(like)
        metadata = backend.metadata(like)
        uniform, rng = backend.uniform(
            rng, (), dtype=metadata.dtype, device=metadata.device
        )
        thresholds = backend.asarray(
            np.cumsum(self._weights)[:-1],
            dtype=metadata.dtype, device=metadata.device,
        )
        return backend.namespace.sum(uniform >= thresholds), rng


class Mixture:
    """Components and the fixed weights selecting among them."""

    def __init__(self, components, weights=None):
        self._components = tuple(components)
        if not self._components:
            raise ValueError("A mixture needs at least one component.")
        self._rule = FixedWeights(
            np.ones(len(self._components)) if weights is None else weights
        )
        if self._rule.weights.size != len(self._components):
            raise ValueError("Expected one weight per component.")

    @property
    def components(self):
        return self._components

    @property
    def rule(self):
        return self._rule


class GaussianMixtureDensity(Mixture, DensityInterface):
    """
    Log-density of a Gaussian mixture model.

    Evaluates the log-density of a sum of Gaussian components, each with an
    associated weight. Weights default to uniform if not specified.

    Parameters
    ----------
    components : list of GaussianDensity
        The component densities.
    weights : list of float, optional
        Weights for the components. Must sum to 1. If None, components are
        weighted equally.
    """

    def __init__(self, components, weights=None):
        if weights is not None and not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0.")
        super().__init__(components, weights)
        components = self._components

        domainType = components[0].domainType
        domainDimension = components[0].domainDimension

        for g in components:
            if g.domainType != domainType:
                raise ValueError(
                    "All components must have the same domainType."
                )
            if g.domainDimension != domainDimension:
                raise ValueError(
                    "All components must have the same domainDimension."
                )

        self._weights = self._rule.weights
        self._logWeights = self._rule.logWeights

    @property
    def domainType(self):
        return self._components[0].domainType

    @property
    def domainDimension(self):
        return self._components[0].domainDimension

    def evaluate_log(self, parameter):
        backend = infer_backend(parameter.coordinate)
        namespace = backend.namespace
        metadata = backend.metadata(parameter.coordinate)
        logValues = namespace.stack([
            component.evaluate_log(parameter)
            for component in self._components
        ])
        logWeights = backend.asarray(
            self._logWeights,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        weightIndex = (slice(None),) + (None,) * (logValues.ndim - 1)
        return namespace.logsumexp(
            logWeights[weightIndex] + logValues, axis=0
        )

    def evaluate_log_gradient(self, parameter):
        backend = infer_backend(parameter.coordinate)
        namespace = backend.namespace
        metadata = backend.metadata(parameter.coordinate)
        logValues = namespace.stack([
            component.evaluate_log(parameter)
            for component in self._components
        ])
        gradients = namespace.stack([
            component.evaluate_log_gradient(parameter)
            for component in self._components
        ])
        logWeights = backend.asarray(
            self._logWeights,
            dtype=metadata.dtype,
            device=metadata.device,
        )
        weightIndex = (slice(None),) + (None,) * (logValues.ndim - 1)
        logJoint = logWeights[weightIndex] + logValues
        responsibilities = namespace.exp(
            logJoint - namespace.logsumexp(logJoint, axis=0)
        )
        return namespace.sum(
            namespace.expand_dims(responsibilities, axis=-1) * gradients,
            axis=0,
        )
