"""
Bayesian logistic regression: posterior density, gradient, data generation,
and SVD surrogate utilities.

Labels are binary y_i in {0, 1}. The gradient formula

    nabla log pi(beta) = -lambda * beta + X^T (y - sigma(X beta))

follows Dwivedi et al. (2019), 'Log-concave sampling: Metropolis-Hastings
algorithms are fast!', JMLR, and is exact for this label convention.
"""

import numpy as np
from scipy.special import expit

from styne.backend import infer_backend
from styne.model.representation.expansion import backend_constant
from styne.parameter.vector import Vector
from styne.statistics.interface import DensityInterface
from styne.statistics.stationary import MaternCovariance1D


class LogisticPosterior(DensityInterface):
    """
    Unnormalised Bayesian logistic regression posterior.

    Target density (unnormalised):

        pi(beta) = exp(-lambda/2 * ||beta||^2)
                   * prod_i sigma(x_i^T beta)^y_i
                            * (1 - sigma(x_i^T beta))^(1 - y_i)

    with labels y_i in {0, 1}. Log-density:

        log pi(beta) = -lambda/2 * ||beta||^2
                       + sum_i [y_i * x_i^T beta
                                - log(1 + exp(x_i^T beta))]

    Gradient:

        nabla log pi(beta) = -lambda * beta + X^T (y - sigma(X beta))

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Design matrix; rows are i.i.d. samples from N(0, Sigma).
    y : ndarray, shape (n,)
        Binary response labels in {0, 1}.
    lambdaPrior : float
        Prior precision (reciprocal variance of the isotropic Gaussian prior).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, lambdaPrior: float):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n, d).")
        if y.ndim != 1 or y.size != X.shape[0]:
            raise ValueError("y must be 1D with length n = X.shape[0].")
        if lambdaPrior <= 0.:
            raise ValueError("lambdaPrior must be positive.")

        self._X = X
        self._y = y
        self._lambda = lambdaPrior
        self._n, self._d = X.shape

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self) -> int:
        return self._d

    def evaluate_log(self, beta: Vector):
        b = beta.coordinate
        backend = infer_backend(b)
        namespace = backend.namespace
        features = backend_constant(self._X, b)
        labels = backend_constant(self._y, b)
        linPred = b @ features.T

        logLik = namespace.sum(
            labels * linPred - namespace.logaddexp(0., linPred),
            axis=-1,
        )
        logPrior = -0.5 * self._lambda * namespace.sum(
            b * b, axis=-1
        )

        return logLik + logPrior

    def evaluate_log_gradient(self, beta: Vector):
        b = beta.coordinate
        backend = infer_backend(b)
        features = backend_constant(self._X, b)
        labels = backend_constant(self._y, b)
        linPred = b @ features.T

        gradLogLik = (labels - backend.namespace.sigmoid(linPred)) @ features
        gradLogPrior = -self._lambda * b

        return gradLogLik + gradLogPrior


def generate_logistic_data(d, n, betaStar, ell, rng=None):
    """
    Synthetic logistic regression data with Matern-correlated design points.

    Rows of X are drawn i.i.d. from N(0, Sigma), where Sigma is the
    Matern(nu=1) covariance on a uniform grid of d points in [0, 1].
    Labels y_i ~ Bernoulli(sigma(x_i^T betaStar)).

    Parameters
    ----------
    d : int
        Parameter dimension.
    n : int
        Number of observations.
    betaStar : ndarray, shape (d,)
        True regression coefficient.
    ell : float
        Matern correlation length controlling spectral decay of X.
    rng : np.random.Generator, optional

    Returns
    -------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,), labels in {0, 1}
    designCov : DenseCovarianceMatrix
    """

    if rng is None:
        rng = np.random.default_rng()

    grid = np.linspace(0., 1., d)
    designCov = MaternCovariance1D(
        ell, nu=1.0, variance=1.0
    ).evaluate_covariance(grid, grid)

    X = np.array([
        designCov.apply_chol_factor(rng.standard_normal(d)) for _ in range(n)
    ])

    linPred = X @ np.asarray(betaStar, dtype=float)
    probs = expit(linPred)
    y = (rng.uniform(size=n) < probs).astype(float)

    return X, y, designCov


__all__ = [
    "LogisticPosterior",
    "generate_logistic_data",
]
