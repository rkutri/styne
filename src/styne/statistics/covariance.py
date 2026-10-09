import numpy as np
from abc import abstractmethod

from styne.backend import BackendInferenceError, get_backend, infer_backend
from styne.statistics.interface import CovarianceOperatorInterface


def backend_array(value):
    try:
        return infer_backend(value), value
    except BackendInferenceError:
        backend = get_backend("numpy")
        return backend, backend.asarray(value)


def scaling_array(backend, reference, scaling):
    try:
        scalingBackend = infer_backend(scaling)
    except BackendInferenceError:
        metadata = backend.metadata(reference)
        return backend.asarray(
            scaling, dtype=metadata.dtype, device=metadata.device
        )
    if scalingBackend.name != backend.name:
        infer_backend(reference, scaling)
    return scaling


def matrix_apply(matrix, vector):
    return (matrix @ vector[..., None])[..., 0]


class CovarianceMatrix(CovarianceOperatorInterface):
    """Abstract base class for immutable backend-native covariance matrices."""

    def __init__(self, backend, reference, scaling=1.0):
        self._backend = backend
        self._scaling = scaling_array(backend, reference, scaling)
        if backend.name == "numpy" and np.any(self._scaling <= 0.0):
            raise ValueError(f"Scaling must be positive. Got {scaling}.")

    @property
    @abstractmethod
    def dimension(self):
        pass

    @abstractmethod
    def log_determinant(self):
        pass

    @abstractmethod
    def apply(self, x):
        pass

    @abstractmethod
    def to_cholesky(self):
        """Return a dense lower factor of the scaled covariance."""
        pass

    @abstractmethod
    def quadratic_form(self, x):
        """Evaluate ``x.T @ C @ x`` over the final coordinate axis."""
        pass

    @abstractmethod
    def dual_quadratic_form(self, x):
        """Evaluate ``x.T @ C^{-1} @ x`` over the final coordinate axis."""
        pass

    @property
    def scaling(self):
        return self._scaling

    @abstractmethod
    def with_scaling(self, scaling):
        """Return an equivalent covariance with replacement scaling."""
        ...


class DiagonalCovarianceMatrix(CovarianceMatrix):
    """Covariance matrix of independent random variables."""

    def __init__(self, marginalVariance, scaling=1.0):
        backend, marginalVariance = backend_array(marginalVariance)
        if marginalVariance.ndim != 1:
            raise ValueError("Marginal variances must be one-dimensional.")
        if backend.name == "numpy" and np.any(marginalVariance <= 0.0):
            raise ValueError(
                "All marginal variances must be strictly positive."
            )

        super().__init__(backend, marginalVariance, scaling)
        self._sqrtMV = backend.namespace.sqrt(marginalVariance)

    def with_scaling(self, scaling):
        return DiagonalCovarianceMatrix(self.marginalVariance, scaling)

    @property
    def marginalVariance(self):
        return self._backend.namespace.square(self._sqrtMV)

    @property
    def dimension(self):
        return self._sqrtMV.shape[-1]

    def apply_chol_factor(self, x):
        infer_backend(self._sqrtMV, x)
        return self._backend.namespace.sqrt(self.scaling) * self._sqrtMV * x

    def apply_chol_factor_transpose(self, x):
        return self.apply_chol_factor(x)

    def to_cholesky(self):
        metadata = self._backend.metadata(self._sqrtMV)
        diagonal = self._backend.namespace.sqrt(self.scaling) * self._sqrtMV
        return self._backend.eye(
            self.dimension, dtype=metadata.dtype, device=metadata.device
        ) * diagonal

    def log_determinant(self):
        namespace = self._backend.namespace
        return namespace.sum(namespace.log(self.marginalVariance), axis=-1) + \
            self.dimension * namespace.log(self.scaling)

    def apply_inverse(self, x):
        infer_backend(self._sqrtMV, x)
        return x / (self.scaling * self.marginalVariance)

    def quadratic_form(self, x):
        namespace = self._backend.namespace
        return namespace.sum(x * self.apply(x), axis=-1)

    def dual_quadratic_form(self, x):
        namespace = self._backend.namespace
        return namespace.sum(x * self.apply_inverse(x), axis=-1)

    def apply(self, x):
        infer_backend(self._sqrtMV, x)
        return self.scaling * self.marginalVariance * x


class IIDCovarianceMatrix(DiagonalCovarianceMatrix):
    """Covariance matrix with one marginal variance repeated by dimension."""

    def __init__(self, dimension, variance, scaling=1.0):
        backend, variance = backend_array(variance)
        metadata = backend.metadata(variance)
        marginalVariance = backend.ones(
            dimension, dtype=metadata.dtype, device=metadata.device
        ) * variance
        super().__init__(marginalVariance, scaling)

    def with_scaling(self, scaling):
        return IIDCovarianceMatrix(
            self.dimension, self.marginalVariance[0], scaling
        )


class DenseCovarianceMatrix(CovarianceMatrix):
    """Dense symmetric positive-definite covariance matrix."""

    def __init__(self, denseCovMat, scaling=1.0):
        backend, denseCovMat = backend_array(denseCovMat)
        if (
                denseCovMat.ndim < 2
                or denseCovMat.shape[-2] != denseCovMat.shape[-1]):
            raise ValueError(
                "Covariance matrix must be square over its final two axes."
            )
        if backend.name == "numpy" and not np.allclose(
                denseCovMat,
                np.swapaxes(denseCovMat, -1, -2)):
            raise ValueError("Covariance matrix must be symmetric.")

        super().__init__(backend, denseCovMat, scaling)
        try:
            self._cholFactor = backend.cholesky(denseCovMat)
        except np.linalg.LinAlgError as error:
            from styne.utility.exceptions import NotPositiveDefinite
            raise NotPositiveDefinite(
                "Covariance matrix is not positive definite."
            ) from error

    def with_scaling(self, scaling):
        return DenseCovarianceMatrix(self.to_dense(), scaling)

    @property
    def dimension(self):
        return self._cholFactor.shape[-1]

    def apply_chol_factor(self, x):
        infer_backend(self._cholFactor, x)
        return self._backend.namespace.sqrt(self.scaling) * matrix_apply(
            self._cholFactor, x
        )

    def apply_chol_factor_transpose(self, x):
        infer_backend(self._cholFactor, x)
        transpose = self._backend.namespace.swapaxes(self._cholFactor, -1, -2)
        return self._backend.namespace.sqrt(self.scaling) * matrix_apply(
            transpose, x
        )

    def to_cholesky(self):
        return self._backend.namespace.sqrt(self.scaling) * self._cholFactor

    def log_determinant(self):
        namespace = self._backend.namespace
        diagonal = namespace.diagonal(self._cholFactor)
        return 2.0 * namespace.sum(namespace.log(diagonal), axis=-1) + \
            self.dimension * namespace.log(self.scaling)

    def apply_inverse(self, x):
        infer_backend(self._cholFactor, x)
        solution = self._backend.solve(self.to_dense(), x[..., None])
        return solution[..., 0] / self.scaling

    def quadratic_form(self, x):
        namespace = self._backend.namespace
        return namespace.sum(x * self.apply(x), axis=-1)

    def dual_quadratic_form(self, x):
        namespace = self._backend.namespace
        return namespace.sum(x * self.apply_inverse(x), axis=-1)

    def apply(self, x):
        infer_backend(self._cholFactor, x)
        return self.scaling * matrix_apply(self.to_dense(), x)

    def to_dense(self):
        transpose = self._backend.namespace.swapaxes(self._cholFactor, -1, -2)
        return self._cholFactor @ transpose
