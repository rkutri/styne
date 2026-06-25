import numpy as np

from scipy.special import expit

from styne.statistics.stationary import MaternCovariance1D
from styne.statistics.covariance import DenseCovarianceMatrix


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
    designCovArray = MaternCovariance1D(
        lengthScale=ell, smoothness=1.0, marginalVariance=1.0
    ).evaluate_covariance(grid, grid)
    designCov = DenseCovarianceMatrix(designCovArray)

    X = np.array([
        designCov.apply_chol_factor(rng.standard_normal(d)) for _ in range(n)
    ])

    linPred = X @ np.asarray(betaStar, dtype=float)
    probs = expit(linPred)
    y = (rng.uniform(size=n) < probs).astype(float)

    return X, y, designCov
