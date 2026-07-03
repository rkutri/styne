import numpy as np

from math import isclose
from scipy.linalg import toeplitz
from scipy.special import gamma, kv, gammaln
from scipy.spatial.distance import cdist

from styne.statistics.interface import CovarianceFunctionInterface
from styne.statistics.covariance import (
    CovarianceMatrix, DenseCovarianceMatrix
)
from styne.utility.grid import Grid, UniformGrid

_MATERN_MAX_SCALED_DISTANCE = 800.0
_MATERN_MIN_SCALED_DISTANCE = 1e-8


class StationaryCovariance1D(CovarianceFunctionInterface):
    """
    Base class wrapping a 1D stationary covariance function and its Fourier
    transform as plain callables.

    Parameters
    ----------
    statCovCallable : callable
        Takes an array of distances, returns covariance values.
    fourierCallable : callable
        Takes an array of frequencies, returns spectral density values.
    """

    def __init__(self, statCovCallable, fourierCallable):

        self._fcn = statCovCallable
        self._fourier = fourierCallable

    def evaluate_covariance(self, x: np.ndarray,
                            y: np.ndarray) -> np.ndarray:
        """
        Dispatching implementation for 1D covariance evaluation.
        """
        if isinstance(x, (Grid, UniformGrid)):
            xArr = np.asarray(x.to_array()).ravel()
        else:
            xArr = np.asarray(x).ravel()

        if isinstance(y, (Grid, UniformGrid)):
            yArr = np.asarray(y.to_array()).ravel()
        else:
            yArr = np.asarray(y).ravel()

        distances = np.abs(xArr[:, None] - yArr[None, :])
        covFlat = self._fcn(distances.ravel())
        return np.asarray(covFlat).reshape(distances.shape)

    def evaluate_fourier(self, freq: np.ndarray) -> np.ndarray:
        return self._fourier(freq)


def exponential_covariance(delta, alpha, variance):
    delta = np.asarray(delta)
    norms = (np.linalg.norm(delta, axis=-1) if delta.ndim > 1
             else np.abs(delta))
    return variance * np.exp(-alpha * norms)


def matern_kappa(lengthScale, smoothness):
    return np.sqrt(2. * smoothness) / lengthScale


def matern_beta(smoothness, dim):
    return 0.5 * (smoothness + 0.5 * dim)


def matern_covariance(x, lengthScale, smoothness, variance):

    if smoothness < 0.5:
        raise ValueError(f"Invalid smoothness parameter: {smoothness}")

    if isclose(smoothness, 0.5):
        return variance * np.exp(-np.abs(x) / lengthScale)

    kappa = matern_kappa(lengthScale, smoothness)
    scaledDistance = np.abs(x)
    if np.isfinite(kappa):
        scaledDistance *= kappa
    else:
        scaledDistance = np.where(scaledDistance == 0, 0.0, np.inf)

    # Avoid inf * 0 = nan when distance overflows
    safeDist = np.minimum(scaledDistance, _MATERN_MAX_SCALED_DISTANCE)

    if isclose(smoothness, 1.5):
        return variance * (1. + safeDist) * np.exp(-safeDist)

    if isclose(smoothness, 2.5):
        return variance * (1. + safeDist + safeDist**2 / 3.) * \
            np.exp(-safeDist)

    covariance = np.zeros_like(scaledDistance)

    validMask = (
        (scaledDistance > _MATERN_MIN_SCALED_DISTANCE)
        & (scaledDistance < _MATERN_MAX_SCALED_DISTANCE)
    )
    covariance[~validMask & (scaledDistance < _MATERN_MIN_SCALED_DISTANCE)] = variance
    covariance[validMask] = (variance * (2. ** (1. - smoothness)) / gamma(smoothness)) * \
        (scaledDistance[validMask] ** smoothness) * kv(smoothness, scaledDistance[validMask])

    return covariance


def matern_log_rho_gradient(x, lengthScale, smoothness, variance):
    if smoothness < 0.5:
        raise ValueError(f"Invalid smoothness parameter: {smoothness}")

    if isclose(smoothness, 0.5):
        distance = np.abs(x)
        return variance * (distance / lengthScale) * np.exp(-distance / lengthScale)

    kappa = matern_kappa(lengthScale, smoothness)
    scaledDistance = np.abs(x)
    if np.isfinite(kappa):
        scaledDistance *= kappa
    else:
        scaledDistance = np.where(scaledDistance == 0, 0.0, np.inf)

    # Avoid inf * 0 = nan when distance overflows
    safeDist = np.minimum(scaledDistance, _MATERN_MAX_SCALED_DISTANCE)

    if isclose(smoothness, 1.5):
        return variance * (safeDist**2) * np.exp(-safeDist)

    if isclose(smoothness, 2.5):
        return (variance / 3.) * (safeDist**2 + safeDist**3) * \
            np.exp(-safeDist)

    gradient = np.zeros_like(scaledDistance)
    validMask = (
        (scaledDistance > _MATERN_MIN_SCALED_DISTANCE)
        & (scaledDistance < _MATERN_MAX_SCALED_DISTANCE)
    )
    gradient[validMask] = (variance * (2. ** (1. - smoothness)) / gamma(smoothness)) * \
        (scaledDistance[validMask] ** (smoothness + 1.)) * \
        kv(smoothness - 1., scaledDistance[validMask])

    return gradient


def matern_fourier(f, lengthScale, smoothness, variance, d=1):
    f = np.asarray(f, dtype=float)
    if d == 1 and f.ndim == 1:
        f = f[:, np.newaxis]
    kappa = matern_kappa(lengthScale, smoothness)
    kappa = np.clip(kappa, 1e-10, None)
    
    # Vectorized sum over frequency components if multiple dimensions provided
    sSq = np.sum((2. * np.pi * f)**2, axis=-1)

    # Closed-form fast paths for common smoothness values (API contract: vectorized)
    if kappa < 1e30:
        if isclose(smoothness, 0.5):
            if d == 1:
                return (variance * 2. * kappa) / (kappa**2 + sSq)
            elif d == 2:
                return (variance * 2. * np.pi * kappa) / (kappa**2 + sSq)**1.5
        
        if isclose(smoothness, 1.5):
            if d == 1:
                return (variance * 4. * kappa**3) / (kappa**2 + sSq)**2
            elif d == 2:
                return (variance * 6. * np.pi * kappa**3) / (kappa**2 + sSq)**2.5

        if isclose(smoothness, 2.5):
            if d == 1:
                return (variance * (16. / 3.) * kappa**5) / (kappa**2 + sSq)**3
            elif d == 2:
                return (variance * 10. * np.pi * kappa**5) / (kappa**2 + sSq)**3.5

    # Generic Gamma-based evaluation
    beta = matern_beta(smoothness, d)
    logNormConst = (d / 2.0) * np.log(4. * np.pi) + \
        gammaln(2. * beta) - gammaln(smoothness) + \
        (2. * smoothness) * np.log(kappa)

    logSpectral = np.log(variance) + logNormConst - (2. * beta) * np.log(kappa**2 + sSq)
    return np.exp(logSpectral)


class ExponentialCovariance1D(StationaryCovariance1D):
    """
    Exponential covariance function, $C(r) = \sigma^2 \exp(-\alpha |r|)$.

    Parameters
    ----------
    alpha : float
        Decay rate.
    marginalVariance : float
        Marginal variance $\sigma^2$.
    """

    def __init__(self, alpha, marginalVariance):
        super().__init__(
            lambda r: exponential_covariance(r, alpha, marginalVariance),
            lambda f: matern_fourier(f, 1. / alpha, 0.5, marginalVariance, d=1)
        )


class MaternCovariance1D(StationaryCovariance1D):
    """
    Matern covariance in 1D, arbitrary smoothness $\nu$. Closed-form fast
    paths for $\nu \in \{0.5, 1.5, 2.5\}$, general Gamma-based evaluation
    otherwise.

    Parameters
    ----------
    lengthScale : float
        Length scale.
    smoothness : float
        Smoothness parameter $\nu$, must be at least 0.5.
    marginalVariance : float
        Marginal variance $\sigma^2$.
    """

    def __init__(self, lengthScale, smoothness, marginalVariance):
        super().__init__(
            lambda r: matern_covariance(r, lengthScale, smoothness, marginalVariance),
            lambda f: matern_fourier(f, lengthScale, smoothness, marginalVariance, d=1)
        )
        self._lengthScale = lengthScale
        self._smoothness = smoothness
        self._marginalVariance = marginalVariance

    def evaluate_covariance_gradient(self, x: np.ndarray, y: np.ndarray) -> dict:
        if isinstance(x, (Grid, UniformGrid)):
            xArr = np.asarray(x.to_array()).ravel()
        else:
            xArr = np.asarray(x).ravel()

        if isinstance(y, (Grid, UniformGrid)):
            yArr = np.asarray(y.to_array()).ravel()
        else:
            yArr = np.asarray(y).ravel()

        if np.array_equal(xArr, yArr):
            if xArr.size == 1:
                delta = xArr - xArr[0]
                return {
                    'log_rho': toeplitz(matern_log_rho_gradient(delta, self._lengthScale, self._smoothness, self._marginalVariance)),
                    'log_sigma': 2. * toeplitz(matern_covariance(delta, self._lengthScale, self._smoothness, self._marginalVariance))
                }
            if xArr.size > 1 and np.allclose(np.diff(xArr), xArr[1] - xArr[0]):
                delta = xArr - xArr[0]
                return {
                    'log_rho': toeplitz(matern_log_rho_gradient(delta, self._lengthScale, self._smoothness, self._marginalVariance)),
                    'log_sigma': 2. * toeplitz(matern_covariance(delta, self._lengthScale, self._smoothness, self._marginalVariance))
                }

        distances = np.abs(xArr[:, None] - yArr[None, :])
        rhoGrad = matern_log_rho_gradient(distances.ravel(), self._lengthScale, self._smoothness, self._marginalVariance)
        sigmaGrad = 2. * matern_covariance(distances.ravel(), self._lengthScale, self._smoothness, self._marginalVariance)
        
        return {
            'log_rho': rhoGrad.reshape(distances.shape),
            'log_sigma': sigmaGrad.reshape(distances.shape)
        }


class Matern32Covariance1D(MaternCovariance1D):

    def __init__(self, lengthScale, marginalVariance):
        super().__init__(lengthScale, 1.5, marginalVariance)

    def spde_parameters(self) -> tuple:
        kappa = np.sqrt(3.) / self._lengthScale
        tau = np.sqrt(
            gamma(1.5) / (gamma(2.) * np.sqrt(4. * np.pi)
                          * kappa**3 * self._marginalVariance)
        )
        return kappa, tau


class MaternCovariance2D:
    """
    Matern covariance in 2D, arbitrary smoothness $\nu$. Standalone
    implementation, not a `StationaryCovariance1D` subclass. Does not
    formally implement `CovarianceFunctionInterface`, despite matching
    its contract; `isinstance` checks against that interface will fail.

    Parameters
    ----------
    lengthScale : float
        Length scale.
    smoothness : float
        Smoothness parameter $\nu$, must be at least 0.5.
    marginalVariance : float
        Marginal variance $\sigma^2$.
    """

    def __init__(self, lengthScale, smoothness, marginalVariance):
        self._lengthScale = lengthScale
        self._smoothness = smoothness
        self._marginalVariance = marginalVariance

    def evaluate_fourier(self, freq: np.ndarray) -> np.ndarray:
        return matern_fourier(
            freq, self._lengthScale, self._smoothness, self._marginalVariance, d=2
        )

    def evaluate_covariance(self, pts1: np.ndarray,
                            pts2: np.ndarray) -> np.ndarray:
        """
        Evaluate the Matern covariance between two point sets.

        Parameters
        ----------
        pts1 : np.ndarray
            First set of points, or a `Grid`.
        pts2 : np.ndarray
            Second set of points, or a `Grid`.

        Returns
-------
        np.ndarray
            Pairwise covariance matrix.
        """
        if isinstance(pts1, (Grid, UniformGrid)):
            pts1 = pts1.to_array()
        else:
            pts1 = np.atleast_2d(pts1)

        if isinstance(pts2, (Grid, UniformGrid)):
            pts2 = pts2.to_array()
        else:
            pts2 = np.atleast_2d(pts2)

        R = cdist(pts1, pts2)
        return matern_covariance(R, self._lengthScale, self._smoothness,
                                 self._marginalVariance)

    def evaluate_covariance_gradient(self, pts1: np.ndarray, pts2: np.ndarray) -> dict:
        if isinstance(pts1, (Grid, UniformGrid)):
            pts1 = pts1.to_array()
        else:
            pts1 = np.atleast_2d(pts1)

        if isinstance(pts2, (Grid, UniformGrid)):
            pts2 = pts2.to_array()
        else:
            pts2 = np.atleast_2d(pts2)

        R = cdist(pts1, pts2)
        rhoGrad = matern_log_rho_gradient(R.ravel(), self._lengthScale, self._smoothness, self._marginalVariance)
        sigmaGrad = 2. * matern_covariance(R.ravel(), self._lengthScale, self._smoothness, self._marginalVariance)
        
        return {
            'log_rho': rhoGrad.reshape(R.shape),
            'log_sigma': sigmaGrad.reshape(R.shape)
        }


class Matern1Covariance2D(MaternCovariance2D):

    def __init__(self, lengthScale, marginalVariance):
        super().__init__(lengthScale, 1.0, marginalVariance)

    def spde_parameters(self) -> tuple:
        kappa = np.sqrt(2.) / self._lengthScale
        tau = np.sqrt(
            gamma(1.) / (gamma(2.) * 4. * np.pi * kappa**2 * self._marginalVariance)
        )
        return kappa, tau
