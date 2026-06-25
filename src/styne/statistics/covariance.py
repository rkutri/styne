import numpy as np
from abc import abstractmethod

from numpy import sqrt
from scipy.linalg import cholesky, solve_triangular

from styne.statistics.interface import CovarianceOperatorInterface


class CovarianceMatrix(CovarianceOperatorInterface):
    """
    Abstract base class for covariance matrices.

    Subclasses provide dimension, log-determinant, Cholesky application,
    inverse application, and direct application. All operations respect
    the `scaling` property (default 1.0), which multiplies the represented
    covariance by a positive scalar without altering stored data.
    """

    @property
    @abstractmethod
    def dimension(self):
        pass

    @abstractmethod
    def log_determinant(self):
        pass

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def quadratic_form(self, x: np.ndarray) -> float:
        """Evaluate the quadratic form x^T C x."""
        pass

    @abstractmethod
    def dual_quadratic_form(self, x: np.ndarray) -> float:
        """Evaluate the dual quadratic form x^T C^{-1} x."""
        pass

    @property
    def scaling(self) -> float:
        return getattr(self, '_scaling', 1.0)

    @scaling.setter
    def scaling(self, value: float):
        if value <= 0:
            raise ValueError(f"Scaling must be positive. Got {value}.")
        self._scaling = float(value)

    @abstractmethod
    def clone(self) -> 'CovarianceMatrix':
        """
        Return an independent copy with identical behavior. 
        
        Subclasses must ensure that underlying coordinate arrays are copied 
        to prevent aliasing between instances.
        """
        ... 


class DiagonalCovarianceMatrix(CovarianceMatrix):
    """
    Covariance matrix of independent random variables.

    Stored internally via the square root of the marginal variances
    (for Cholesky application) and the precision (reciprocal variances).

    Parameters
    ----------
    marginalVariance : ndarray
        One-dimensional array of positive marginal variances.
    """

    def __init__(self, marginalVariance):

        marginalVariance = np.asarray(marginalVariance, dtype=np.float64)
        self._validate_variance(marginalVariance)

        self._sqrtMV = np.sqrt(marginalVariance)
        self._precision = np.reciprocal(marginalVariance)

    def clone(self) -> 'DiagonalCovarianceMatrix':
        """
        Create an independent copy with cloned variance arrays. 
        
        Copying the underlying arrays avoids aliasing if the marginal variances 
        are later modified via the setter on either instance.
        """
        # Create a new instance with the same marginal variance (handles array copy)
        newObj = DiagonalCovarianceMatrix(np.square(self._sqrtMV))
        if hasattr(self, '_scaling'):
            newObj.scaling = self.scaling
        return newObj

    @staticmethod
    def _validate_variance(variance):
        if np.any(variance <= 0.):
            raise ValueError(
                "All marginal variances must be strictly positive.")

    @property
    def marginalVariance(self):
        return np.square(self._sqrtMV)

    @marginalVariance.setter
    def marginalVariance(self, mVar):
        mVar = np.asarray(mVar, dtype=np.float64)
        self._validate_variance(mVar)
        self._sqrtMV = np.sqrt(mVar)
        self._precision = np.reciprocal(mVar)

    @property
    def dimension(self):
        return self._precision.size

    def apply_chol_factor(self, x: np.ndarray) -> np.ndarray:
        return sqrt(self.scaling) * self._sqrtMV * x

    def apply_chol_factor_transpose(self, x: np.ndarray) -> np.ndarray:
        return sqrt(self.scaling) * self._sqrtMV * x

    def log_determinant(self) -> float:
        base = float(np.sum(np.log(np.square(self._sqrtMV))))
        return base + self.dimension * np.log(self.scaling)

    def apply_inverse(self, x: np.ndarray) -> np.ndarray:
        return self._precision * x / self.scaling

    def quadratic_form(self, x: np.ndarray) -> float:
        return float(np.sum(np.square(x) * self.marginalVariance)) * self.scaling

    def dual_quadratic_form(self, x: np.ndarray) -> float:
        return float(np.sum(np.square(x) * self._precision)) / self.scaling

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.scaling * self.marginalVariance * x


class IIDCovarianceMatrix(DiagonalCovarianceMatrix):
    """
    Covariance matrix of i.i.d. random variables: variance * I.

    Parameters
    ----------
    dimension : int
        Size of the state space.
    variance : float
        Common marginal variance (must be positive).
    """

    def __init__(self, dimension, variance):
        margVar = np.full(dimension, variance)
        super().__init__(margVar)


class DenseCovarianceMatrix(CovarianceMatrix):
    """
    General (dense) symmetric positive-definite covariance matrix.

    Stored internally via its lower-triangular Cholesky factor.

    Parameters
    ----------
    denseCovMat : ndarray
        Two-dimensional symmetric positive-definite matrix.
    """

    def __init__(self, denseCovMat):

        denseCovMat = np.asarray(denseCovMat, dtype=np.float64)

        s = denseCovMat.shape
        if denseCovMat.ndim != 2 or s[0] != s[1]:
            raise ValueError(
                f"Covariance matrix must be square. Got shape {s} instead."
            )

        if not np.allclose(denseCovMat, denseCovMat.T):
            raise ValueError("Covariance matrix must be symmetric.")

        self._dim = s[0]
        from scipy.linalg import LinAlgError
        try:
            self._cholFactor = cholesky(denseCovMat, lower=True)
        except LinAlgError as e:
            from styne.utility.exceptions import NotPositiveDefinite
            raise NotPositiveDefinite("Covariance matrix is not positive definite.") from e

    def clone(self) -> 'DenseCovarianceMatrix':
        """
        Create an independent copy with a cloned Cholesky factor.
        """
        # We can't easily call __init__ without the original dense matrix,
        # so we use __class__.__new__ and manually set the state.
        newObj = self.__class__.__new__(self.__class__)
        newObj._dim = self._dim
        newObj._cholFactor = self._cholFactor.copy()
        if hasattr(self, '_scaling'):
            newObj.scaling = self.scaling
        return newObj

    @property
    def dimension(self):
        return self._dim

    def apply_chol_factor(self, x: np.ndarray) -> np.ndarray:
        return sqrt(self.scaling) * (self._cholFactor @ x)

    def apply_chol_factor_transpose(self, x: np.ndarray) -> np.ndarray:
        return sqrt(self.scaling) * (self._cholFactor.T @ x)

    def log_determinant(self) -> float:
        base = float(2. * np.sum(np.log(np.diag(self._cholFactor))))
        return base + self._dim * np.log(self.scaling)

    def apply_inverse(self, x: np.ndarray) -> np.ndarray:
        y = solve_triangular(self._cholFactor, x, lower=True)
        return solve_triangular(self._cholFactor.T, y, lower=False) / self.scaling

    def quadratic_form(self, x: np.ndarray) -> float:
        y = self._cholFactor.T @ x
        return float(np.sum(np.square(y))) * self.scaling

    def dual_quadratic_form(self, x: np.ndarray) -> float:
        y = solve_triangular(self._cholFactor, x, lower=True)
        return float(np.sum(np.square(y))) / self.scaling

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.scaling * (self._cholFactor @ (self._cholFactor.T @ x))

    def to_dense(self):
        """Reconstruct the full covariance matrix from the Cholesky factor."""
        return np.matmul(self._cholFactor, self._cholFactor.T)
