import numpy as np
from scipy.special import logsumexp

from styne.statistics.interface import DensityInterface


class GaussianMixtureDensity(DensityInterface):
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
        if not components:
            raise ValueError("components list must not be empty.")

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

        self._components = list(components)
        n_components = len(self._components)

        if weights is None:
            self._weights = np.ones(n_components) / n_components
        else:
            if len(weights) != n_components:
                raise ValueError(
                    f"Expected {n_components} weights, got {len(weights)}."
                )
            weights = np.array(weights, dtype=float)
            if not np.all(weights >= 0):
                raise ValueError("Weights must be non-negative.")
            w_sum = np.sum(weights)
            if not np.isclose(w_sum, 1.0):
                raise ValueError(f"Weights must sum to 1.0, got {w_sum}.")
            self._weights = weights
            
        # Precompute log-weights for logsumexp
        self._logWeights = np.log(self._weights)

    @property
    def domainType(self):
        return self._components[0].domainType

    @property
    def domainDimension(self):
        return self._components[0].domainDimension

    def evaluate_log(self, parameter):
        log_vals = [g.evaluate_log(parameter) for g in self._components]
        return logsumexp(self._logWeights + log_vals)

    def evaluate_log_gradient(self, parameter):
        log_vals = np.array([g.evaluate_log(parameter) for g in self._components])
        gradients = np.array([g.evaluate_log_gradient(parameter) for g in self._components])
        
        # log P(C=k | x) = log(w_k) + log N(x|k) - log(\sum w_j N(x|j))
        log_joint = self._logWeights + log_vals
        log_marginal = logsumexp(log_joint)
        resp = np.exp(log_joint - log_marginal)
        
        return np.sum(resp[:, None] * gradients, axis=0)
