import numpy as np

from math import isclose
from scipy.linalg import toeplitz
from scipy.special import gamma, kv, gammaln
from scipy.spatial.distance import cdist

from styne.backend import infer_backend

from styne.statistics.interface import CovarianceFunctionInterface
from styne.statistics.covariance import (
    CovarianceMatrix, DenseCovarianceMatrix
)
from styne.utility.grid import Grid, UniformGrid

_MATERN_MAX_SCALED_DISTANCE = 800.0
_MATERN_MIN_SCALED_DISTANCE = 1e-8


def as_point_array(points, backend=None, metadata=None):
    """Coerce a Grid or raw array to a dense `(nPoints, dimension)` array.

    'Grid.to_array' already returns this shape, `(n, 1)` in 1D and `(n, 2)`
    in 2D, so cdist consumes it directly. A raw 1D array of coordinates is
    promoted to a column.
    """
    if isinstance(points, (Grid, UniformGrid)):
        points = points.to_array()

    if backend is None:
        array = np.asarray(points)
    else:
        array = backend.asarray(
            points, dtype=metadata.dtype, device=metadata.device
        )
    if array.ndim == 1:
        return array.reshape((-1, 1))
    return array


class StationaryCovariance(CovarianceFunctionInterface):
    """
    Stationary covariance function in any spatial dimension, wrapping a
    radial covariance and its spectral density as plain callables.

    Pairwise distances are Euclidean via cdist, so the same implementation
    serves 1D and 2D. 'spatialDimension' is carried only to set the 'd'
    argument of the Fourier callable, which differs between dimensions.

    Parameters
    ----------
    statCovCallable : callable
        Takes an array of Euclidean distances, returns covariance values.
    fourierCallable : callable
        Takes an array of frequencies, returns spectral density values.
    spatialDimension : int
        Domain dimension, 1 or 2.
    """

    def __init__(
            self, statCovCallable, fourierCallable, spatialDimension,
            backendReference=None):
        self._fcn = statCovCallable
        self._fourier = fourierCallable
        self._spatialDimension = spatialDimension
        self._backendReference = backendReference

    @property
    def spatialDimension(self) -> int:
        return self._spatialDimension

    def evaluate_covariance(self, x: np.ndarray,
                            y: np.ndarray) -> np.ndarray:
        reference = (
            self._backendReference
            if hasattr(self._backendReference, "shape") else x
        )
        if isinstance(reference, (Grid, UniformGrid)):
            reference = reference.to_array()
        backend = infer_backend(reference)
        metadata = backend.metadata(reference)
        xPoints = as_point_array(x, backend, metadata)
        yPoints = as_point_array(y, backend, metadata)
        delta = xPoints[:, None, :] - yPoints[None, :, :]
        distances = backend.namespace.sqrt(
            backend.namespace.sum(delta * delta, axis=-1)
        )
        return self._fcn(distances)

    def evaluate_fourier(self, freq: np.ndarray) -> np.ndarray:
        return self._fourier(freq)


def exponential_covariance(delta, alpha, variance):
    reference = alpha if hasattr(alpha, "shape") else delta
    backend = infer_backend(reference)
    metadata = backend.metadata(reference)
    delta = backend.asarray(delta, dtype=metadata.dtype, device=metadata.device)
    return variance * backend.namespace.exp(-alpha * backend.namespace.abs(delta))


def matern_kappa(lengthScale, smoothness):
    return np.sqrt(2. * smoothness) / lengthScale


def matern_beta(smoothness, dim):
    return 0.5 * (smoothness + 0.5 * dim)


def matern_covariance(x, lengthScale, smoothness, variance):

    if smoothness < 0.5:
        raise ValueError(f"Invalid smoothness parameter: {smoothness}")

    reference = lengthScale if hasattr(lengthScale, "shape") else x
    backend = infer_backend(reference)
    metadata = backend.metadata(reference)
    ns = backend.namespace
    x = backend.asarray(x, dtype=metadata.dtype, device=metadata.device)

    if isclose(smoothness, 0.5):
        return variance * ns.exp(-ns.abs(x) / lengthScale)

    kappa = matern_kappa(lengthScale, smoothness)
    scaledDistance = ns.abs(x) * kappa

    # Avoid inf * 0 = nan when distance overflows
    safeDist = ns.minimum(scaledDistance, _MATERN_MAX_SCALED_DISTANCE)

    if isclose(smoothness, 1.5):
        return variance * (1. + safeDist) * ns.exp(-safeDist)

    if isclose(smoothness, 2.5):
        return variance * (1. + safeDist + safeDist**2 / 3.) * \
            ns.exp(-safeDist)

    if backend.name != "numpy":
        raise NotImplementedError(
            "Matern smoothness must be 0.5, 1.5, or 2.5 on this backend."
        )
    covariance = (variance * (2. ** (1. - smoothness)) / gamma(smoothness)) * \
        (scaledDistance ** smoothness) * kv(smoothness, scaledDistance)
    return ns.where(scaledDistance <= _MATERN_MIN_SCALED_DISTANCE, variance, covariance)


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
    reference = lengthScale if hasattr(lengthScale, "shape") else f
    backend = infer_backend(reference)
    metadata = backend.metadata(reference)
    ns = backend.namespace
    f = backend.asarray(f, dtype=metadata.dtype, device=metadata.device)
    if d == 1 and f.ndim == 1:
        f = f.reshape((-1, 1))
    kappa = matern_kappa(lengthScale, smoothness)
    kappa = ns.maximum(kappa, 1e-10)
    
    # Vectorized sum over frequency components if multiple dimensions provided
    sSq = ns.sum((2. * np.pi * f)**2, axis=-1)

    # Closed-form fast paths for common smoothness values (API contract: vectorized)
    if isclose(smoothness, 0.5):
        if d == 1:
            return (variance * 2. * kappa) / (kappa**2 + sSq)
        if d == 2:
            return (variance * 2. * np.pi * kappa) / (kappa**2 + sSq)**1.5

    if isclose(smoothness, 1.5):
        if d == 1:
            return (variance * 4. * kappa**3) / (kappa**2 + sSq)**2
        if d == 2:
            return (variance * 6. * np.pi * kappa**3) / (kappa**2 + sSq)**2.5

    if isclose(smoothness, 2.5):
        if d == 1:
            return (variance * (16. / 3.) * kappa**5) / (kappa**2 + sSq)**3
        if d == 2:
            return (variance * 10. * np.pi * kappa**5) / (kappa**2 + sSq)**3.5

    if backend.name != "numpy":
        raise NotImplementedError(
            "Matern smoothness must be 0.5, 1.5, or 2.5 on this backend."
        )

    # Generic Gamma-based evaluation (NumPy only).
    beta = matern_beta(smoothness, d)
    logNormConst = (d / 2.0) * np.log(4. * np.pi) + \
        gammaln(2. * beta) - gammaln(smoothness) + \
        (2. * smoothness) * ns.log(kappa)

    logSpectral = ns.log(variance) + logNormConst - (2. * beta) * ns.log(kappa**2 + sSq)
    return ns.exp(logSpectral)


class ExponentialCovariance1D(StationaryCovariance):
    r"""
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
            lambda f: matern_fourier(f, 1. / alpha, 0.5, marginalVariance, d=1),
            spatialDimension=1, backendReference=marginalVariance
        )


class MaternCovariance1D(StationaryCovariance):
    r"""
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
            lambda f: matern_fourier(f, lengthScale, smoothness, marginalVariance, d=1),
            spatialDimension=1, backendReference=marginalVariance
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


class MaternCovariance2D(StationaryCovariance):
    r"""
    Matern covariance in 2D, arbitrary smoothness $\nu$. Closed-form fast
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
            lambda f: matern_fourier(f, lengthScale, smoothness, marginalVariance, d=2),
            spatialDimension=2, backendReference=marginalVariance
        )
        self._lengthScale = lengthScale
        self._smoothness = smoothness
        self._marginalVariance = marginalVariance

    def evaluate_covariance_gradient(self, x: np.ndarray, y: np.ndarray) -> dict:
        distances = cdist(as_point_array(x), as_point_array(y))
        rhoGrad = matern_log_rho_gradient(
            distances.ravel(), self._lengthScale, self._smoothness,
            self._marginalVariance
        )
        sigmaGrad = 2. * matern_covariance(
            distances.ravel(), self._lengthScale, self._smoothness,
            self._marginalVariance
        )
        return {
            'log_rho': rhoGrad.reshape(distances.shape),
            'log_sigma': sigmaGrad.reshape(distances.shape)
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
