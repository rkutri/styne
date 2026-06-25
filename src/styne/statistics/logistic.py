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

from styne.parameter.vector import Vector
from styne.statistics.interface import DensityInterface
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.linalg import svd_truncate_frob, svd_truncate_rank


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

    def evaluate_log(self, beta: Vector) -> float:
        b = beta.coordinate
        linPred = self._X @ b

        logLik = float(np.sum(self._y * linPred - np.logaddexp(0., linPred)))
        logPrior = -0.5 * self._lambda * float(np.dot(b, b))

        return logLik + logPrior

    def evaluate_log_gradient(self, beta: Vector) -> np.ndarray:
        b = beta.coordinate
        linPred = self._X @ b

        gradLogLik = self._X.T @ (self._y - expit(linPred))
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


# svd_truncate_frob and svd_truncate_rank are re-exported from utility.linalg
__all__ = [
    "LogisticPosterior",
    "generate_logistic_data",
    "svd_truncate_frob",
    "svd_truncate_rank",
]
