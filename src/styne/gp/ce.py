import numpy as np
import logging

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from numpy import ndarray
from numpy.linalg import norm
from numpy.random import Generator
from scipy.fft import fft2, ifft2, fftshift

from styne.statistics.measure import ProbabilityMeasure


class CirculantEmbeddingEngine(ProbabilityMeasure, ABC):
    """
    Abstract base for circulant embedding GP samplers.

    Subclasses specialise the eigenvalue computation and sampling step for
    1D (FFT) and 2D (FFT2). The base class handles padding autotuning and
    the `draw` skeleton.

    Parameters
    ----------
    cov_callable : callable
        Stationary covariance function; takes a scalar distance and returns
        the covariance value.
    vertPerDim : int
        Number of grid cells per dimension (output has vertPerDim points).
    domExt : float
        Domain extent (grid spacing = domExt / vertPerDim).
    autotunePadding : bool
        If True, bisect to find minimal valid padding after finding a valid
        embedding.
    padding : int
        Initial zero-padding size.
    maxPadding : int
        Maximum padding before raising an error.
    tol : float
        Tolerance for detecting negative eigenvalues.
    """

    def __init__(self, cov_callable, vertPerDim: int, domExt: float = 1.,
                 autotunePadding: bool = True, padding: int = 0,
                 maxPadding: int = 512, tol: float = 1e-12):

        self._vertPerDim = vertPerDim
        self._nBase = vertPerDim + 1
        self._domExt = domExt
        self._h = domExt / vertPerDim
        self._padding = padding
        self._tol = tol
        self._maxPadding = maxPadding
        self._performAutotune = autotunePadding

        if self._padding > self._maxPadding:
            raise RuntimeError(
                "initial CE padding larger than maximal allowed padding")

        self._eigenvalues = self._compute_eigenvalues(cov_callable)

    def _n_ext(self) -> int:
        return self._nBase + self._padding

    def _is_valid(self, eigenvalues: ndarray) -> bool:
        return eigenvalues.min() >= -self._tol

    def _compute_eigenvalues(self, cov_callable) -> ndarray:
        lam = self._build_eigenvalues(cov_callable, self._n_ext())
        if not self._is_valid(lam):
            logger.warning("initial embedding is not positive definite.")
            lam = self._find_valid_padding(cov_callable)
            if self._performAutotune:
                lam = self._bisect_padding(cov_callable)
        logger.debug(f"Padding used: {self._padding}")
        return lam

    def _find_valid_padding(self, cov_callable) -> ndarray:
        next_pad = max(1, 1 << self._padding.bit_length())
        while next_pad <= self._maxPadding:
            logger.debug(f"check definiteness for padding={next_pad}")
            self._padding = next_pad
            lam = self._build_eigenvalues(cov_callable, self._n_ext())
            if self._is_valid(lam):
                return lam
            logger.warning("...still indefinite")
            next_pad *= 2
        from styne.utility.exceptions import NotPositiveDefinite
        raise NotPositiveDefinite(
            f"Exceeded maximal padding of {self._maxPadding}")

    def _bisect_padding(self, cov_callable, maxBisections: int = 4) -> ndarray:
        logger.debug("determining minimal padding")
        init_pad = self._padding
        upper, lower = self._padding, 0
        lam = None
        is_valid = False

        for _ in range(maxBisections):
            if upper <= lower:
                break
            mid = int(np.rint(0.5 * (lower + upper)))
            self._padding = mid
            lam = self._build_eigenvalues(cov_callable, self._n_ext())
            is_valid = self._is_valid(lam)
            if is_valid:
                upper = mid
            else:
                lower = mid
                self._padding = upper

        if not is_valid:
            logger.warning("no smaller padding possible")
            self._padding = init_pad
            lam = self._build_eigenvalues(cov_callable, self._n_ext())

        return lam

    @abstractmethod
    def _build_eigenvalues(self, cov_callable, n_ext: int) -> ndarray:
        """Real eigenvalues of the extended circulant covariance matrix."""

    @abstractmethod
    def _sample_field(self, eigenvalues: ndarray, rng: Generator) -> ndarray:
        """Draw a raw sample from the full extended field."""

    def draw(self, rng: Generator) -> ndarray:
        return self._sample_field(self._eigenvalues, rng)[:self._vertPerDim]


class CirculantEmbeddingEngine1D(CirculantEmbeddingEngine):
    """1D circulant embedding using np.fft.fft."""

    def _build_eigenvalues(self, cov_callable, n_ext: int) -> ndarray:
        nRed = 2 * n_ext - 1
        offsets = np.arange(-(n_ext - 1), n_ext)
        c = np.array([cov_callable(abs(k) * self._h) for k in offsets])

        c_tilde = np.zeros(2 * n_ext)
        c_tilde[1:2 * n_ext] = c
        c_tilde = np.fft.fftshift(c_tilde)

        return (2 * n_ext * np.fft.ifft(c_tilde)).real

    def _sample_field(self, eigenvalues: ndarray, rng: Generator) -> ndarray:
        N = eigenvalues.shape[0]
        coeff = np.sqrt(np.maximum(eigenvalues, 0.))
        xi = rng.standard_normal(N) + 1.j * rng.standard_normal(N)
        z = np.fft.fft(coeff * xi) / np.sqrt(N)
        return np.real(z)


class CirculantEmbeddingEngine2D(CirculantEmbeddingEngine):
    """2D circulant embedding using scipy.fft.fft2. Returns a single field."""

    def _build_eigenvalues(self, cov_callable, n_ext: int) -> ndarray:
        nRed = 2 * n_ext - 1
        redCov = np.zeros((nRed, nRed))
        for i in range(nRed):
            for j in range(nRed):
                x = (i - (n_ext - 1)) * self._h
                y = (j - (n_ext - 1)) * self._h
                redCov[i, j] = cov_callable(norm([x, y]))

        redCovTilde = np.zeros((2 * n_ext, 2 * n_ext))
        redCovTilde[1:2 * n_ext, 1:2 * n_ext] = redCov
        redCovTilde = fftshift(redCovTilde)

        N_sq = (2 * n_ext)**2
        return (N_sq * ifft2(redCovTilde)).real

    def _sample_field(self, eigenvalues: ndarray, rng: Generator) -> ndarray:
        nExt = eigenvalues.shape[0]
        coeff = np.sqrt(np.maximum(eigenvalues, 0.))
        xi = (rng.standard_normal((nExt, nExt))
              + 1.j * rng.standard_normal((nExt, nExt)))
        z = fft2(coeff * xi) / np.sqrt(nExt**2)
        return np.real(z)

    def draw(self, rng: Generator) -> ndarray:
        n = self._vertPerDim
        return self._sample_field(self._eigenvalues, rng)[:n, :n]


class ApproximateCirculantEmbeddingEngine1D(CirculantEmbeddingEngine1D):
    """1D circulant embedding that clamps negative eigenvalues to zero.

    The resulting covariance is approximate; use when the exact embedding
    is indefinite and no padding can fix it.
    """

    def __init__(self, cov_callable, vertPerDim: int, domExt: float = 1.,
                 padding: int = 0):
        super().__init__(cov_callable, vertPerDim, domExt,
                         autotunePadding=False, padding=padding)

    def _compute_eigenvalues(self, cov_callable) -> ndarray:
        logger.debug(f"Padding used: {self._padding}")
        return np.maximum(
            self._build_eigenvalues(cov_callable, self._n_ext()), 0.)


class ApproximateCirculantEmbeddingEngine2D(CirculantEmbeddingEngine2D):
    """2D circulant embedding that clamps negative eigenvalues to zero."""

    def __init__(self, cov_callable, vertPerDim: int, domExt: float = 1.,
                 padding: int = 0):
        super().__init__(cov_callable, vertPerDim, domExt,
                         autotunePadding=False, padding=padding)

    def _compute_eigenvalues(self, cov_callable) -> ndarray:
        logger.debug(f"Padding used: {self._padding}")
        return np.maximum(
            self._build_eigenvalues(cov_callable, self._n_ext()), 0.)
