import copy
import numpy as np
from abc import abstractmethod

from styne.backend import BackendInferenceError, get_backend, infer_backend
from styne.statistics.interface import CovarianceOperatorInterface
from styne.utility.exceptions import NotPositiveDefinite


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


def square_matrix(matrix):
    """Backend and array of a square matrix, checked for symmetry on NumPy."""
    backend, matrix = backend_array(matrix)
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(
            "Covariance matrix must be square over its final two axes."
        )
    if backend.name == "numpy" and not np.allclose(
            matrix, np.swapaxes(matrix, -1, -2)):
        raise ValueError("Covariance matrix must be symmetric.")
    return backend, matrix


def cholesky_factor(backend, matrix):
    try:
        return backend.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise NotPositiveDefinite(
            "Covariance matrix is not positive definite."
        ) from error


def triangular_solve(backend, factor, x, *, transpose=False):
    """Solve with a lower triangular factor, or its transpose, over the final axis."""
    namespace = backend.namespace
    if transpose:
        factor = namespace.swapaxes(factor, -1, -2)
    if factor.ndim == 2:
        # One factor can solve all samples as multiple right-hand sides.
        columns = namespace.swapaxes(x.reshape((-1, x.shape[-1])), 0, 1)
        solution = backend.solve_triangular(factor, columns, lower=not transpose)
        return namespace.swapaxes(solution, 0, 1).reshape(x.shape)
    batchShape = np.broadcast_shapes(factor.shape[:-2], x.shape[:-1])
    factor = namespace.broadcast_to(factor, batchShape + factor.shape[-2:])
    columns = namespace.broadcast_to(x, batchShape + x.shape[-1:])[..., None]
    return backend.solve_triangular(factor, columns, lower=not transpose)[..., 0]


class CovarianceMatrix(CovarianceOperatorInterface):
    """Abstract base class for immutable backend-native covariance matrices."""

    def __init__(self, backend, reference, scaling=1.0):
        self._backend = backend
        self._scaling = scaling_array(backend, reference, scaling) * 1.0
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

    def with_scaling(self, scaling):
        """Return an equivalent covariance with replacement scaling.

        The stored factorisation does not involve the scaling, so the result
        shares it instead of factorising again.
        """
        result = copy.copy(self)
        CovarianceMatrix.__init__(result, self._backend, self._scaling, scaling)
        return result


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


class DenseCovarianceMatrix(CovarianceMatrix):
    """Dense symmetric positive-definite covariance matrix."""

    def __init__(self, denseCovMat, scaling=1.0):
        backend, denseCovMat = square_matrix(denseCovMat)
        super().__init__(backend, denseCovMat, scaling)
        self._cholFactor = cholesky_factor(backend, denseCovMat)

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
        whitened = triangular_solve(self._backend, self._cholFactor, x)
        return triangular_solve(
            self._backend, self._cholFactor, whitened, transpose=True
        ) / self.scaling

    def quadratic_form(self, x):
        namespace = self._backend.namespace
        return namespace.sum(
            namespace.square(self.apply_chol_factor_transpose(x)), axis=-1
        )

    def dual_quadratic_form(self, x):
        infer_backend(self._cholFactor, x)
        namespace = self._backend.namespace
        whitened = triangular_solve(self._backend, self._cholFactor, x)
        return namespace.sum(namespace.square(whitened), axis=-1) / self.scaling

    def apply(self, x):
        infer_backend(self._cholFactor, x)
        transpose = self._backend.namespace.swapaxes(self._cholFactor, -1, -2)
        return self.scaling * matrix_apply(
            self._cholFactor, matrix_apply(transpose, x)
        )

    def to_dense(self):
        transpose = self._backend.namespace.swapaxes(self._cholFactor, -1, -2)
        return self._cholFactor @ transpose


class SemidefiniteCovarianceMatrix(CovarianceMatrix):
    """Dense symmetric positive-semidefinite covariance matrix.

    Its factor comes from the eigendecomposition, with negative round-off of
    the eigenvalues set to zero. Inverse-type operations require a
    nonsingular matrix and raise ``NotPositiveDefinite`` otherwise. The
    spectrum is validated on NumPy only.
    """

    def __init__(self, denseCovMat, scaling=1.0):
        backend, denseCovMat = square_matrix(denseCovMat)
        super().__init__(backend, denseCovMat, scaling)
        eigenvalues, self._eigenvectors = backend.eigh(denseCovMat)
        self._singular = False
        if backend.name == "numpy":
            # The default rank tolerance of numpy.linalg.matrix_rank separates
            # round-off of exact zeros from genuine eigenvalues.
            tolerance = denseCovMat.shape[-1] * np.finfo(
                eigenvalues.dtype
            ).eps * np.max(np.abs(eigenvalues), axis=-1, keepdims=True)
            if np.any(eigenvalues < -tolerance):
                raise ValueError(
                    "Covariance matrix must be positive semidefinite."
                )
            self._singular = bool(np.any(eigenvalues <= tolerance))
        self._matrix = denseCovMat * 1.0
        self._eigenvalues = backend.namespace.maximum(eigenvalues, 0.0)

    @property
    def dimension(self):
        return self._matrix.shape[-1]

    def _require_nonsingular(self):
        if self._singular:
            raise NotPositiveDefinite("Covariance matrix is singular.")

    def apply_chol_factor(self, x):
        infer_backend(self._matrix, x)
        namespace = self._backend.namespace
        return namespace.sqrt(self.scaling) * matrix_apply(
            self._eigenvectors, namespace.sqrt(self._eigenvalues) * x
        )

    def apply_chol_factor_transpose(self, x):
        infer_backend(self._matrix, x)
        namespace = self._backend.namespace
        transpose = namespace.swapaxes(self._eigenvectors, -1, -2)
        return namespace.sqrt(self.scaling * self._eigenvalues) * matrix_apply(
            transpose, x
        )

    def to_cholesky(self):
        """Return a lower Cholesky factor; requires a positive-definite matrix."""
        self._require_nonsingular()
        return cholesky_factor(self._backend, self.scaling * self._matrix)

    def to_factor(self):
        """Return the spectral square root, which need not be triangular."""
        namespace = self._backend.namespace
        diagonal = namespace.sqrt(
            self.scaling * self._eigenvalues
        )
        return self._eigenvectors * diagonal[..., None, :]

    def log_determinant(self):
        self._require_nonsingular()
        namespace = self._backend.namespace
        return (
            namespace.sum(namespace.log(self._eigenvalues), axis=-1)
            + self.dimension * namespace.log(self.scaling)
        )

    def apply_inverse(self, x):
        self._require_nonsingular()
        infer_backend(self._matrix, x)
        transpose = self._backend.namespace.swapaxes(self._eigenvectors, -1, -2)
        return matrix_apply(
            self._eigenvectors,
            matrix_apply(transpose, x) / self._eigenvalues,
        ) / self.scaling

    def quadratic_form(self, x):
        namespace = self._backend.namespace
        return namespace.sum(x * self.apply(x), axis=-1)

    def dual_quadratic_form(self, x):
        self._require_nonsingular()
        infer_backend(self._matrix, x)
        namespace = self._backend.namespace
        transpose = namespace.swapaxes(self._eigenvectors, -1, -2)
        rotated = matrix_apply(transpose, x)
        return namespace.sum(
            rotated * rotated / self._eigenvalues, axis=-1
        ) / self.scaling

    def apply(self, x):
        infer_backend(self._matrix, x)
        return self.scaling * matrix_apply(self._matrix, x)


class PrecisionCovarianceMatrix(CovarianceMatrix):
    """Covariance matrix given by its symmetric positive-definite precision.

    The covariance is the scaled inverse of the precision. Storing the
    precision's Cholesky factor R makes the dual quadratic form one
    multiplication and applying the covariance two triangular solves. An
    existing CovarianceMatrix may supply the precision; dense operators share
    their validated factor instead of reconstructing and factorising it.
    """

    def __init__(self, precision, scaling=1.0):
        precisionScaling = 1.0
        if isinstance(precision, DenseCovarianceMatrix):
            backend, factor = precision._backend, precision._cholFactor
            precisionScaling = precision.scaling
        elif isinstance(precision, CovarianceMatrix):
            backend, factor = precision._backend, precision.to_cholesky()
        else:
            backend, precision = square_matrix(precision)
            factor = cholesky_factor(backend, precision)
        super().__init__(backend, factor, scaling)
        self._precisionFactor = factor
        self._precisionScaling = scaling_array(backend, factor, precisionScaling) * 1.0

    @property
    def _factorScaling(self):
        return self.scaling / self._precisionScaling

    @property
    def precision(self):
        """The unscaled precision matrix."""
        transpose = self._backend.namespace.swapaxes(
            self._precisionFactor, -1, -2
        )
        return self._precisionScaling * (self._precisionFactor @ transpose)

    @property
    def precisionFactor(self):
        """Lower Cholesky factor of the unscaled precision."""
        return self._backend.namespace.sqrt(
            self._precisionScaling
        ) * self._precisionFactor

    @property
    def dimension(self):
        return self._precisionFactor.shape[-1]

    def apply_chol_factor(self, x):
        # R^-T is a square root of the inverse of the precision R R^T.
        infer_backend(self._precisionFactor, x)
        return self._backend.namespace.sqrt(self._factorScaling) * triangular_solve(
            self._backend, self._precisionFactor, x, transpose=True
        )

    def apply_chol_factor_transpose(self, x):
        infer_backend(self._precisionFactor, x)
        return self._backend.namespace.sqrt(self._factorScaling) * triangular_solve(
            self._backend, self._precisionFactor, x
        )

    def to_cholesky(self):
        metadata = self._backend.metadata(self._precisionFactor)
        identity = self._backend.eye(
            self.dimension, dtype=metadata.dtype, device=metadata.device
        )
        namespace = self._backend.namespace
        identity = namespace.broadcast_to(identity, self._precisionFactor.shape)
        inverseFactor = self._backend.solve_triangular(
            self._precisionFactor, identity, lower=True
        )
        dense = self._factorScaling * (
            namespace.swapaxes(inverseFactor, -1, -2) @ inverseFactor
        )
        return self._backend.cholesky(
            0.5 * (dense + self._backend.namespace.swapaxes(dense, -1, -2))
        )

    def log_determinant(self):
        namespace = self._backend.namespace
        diagonal = namespace.diagonal(self._precisionFactor)
        return (
            self.dimension * namespace.log(self._factorScaling)
            - 2.0 * namespace.sum(namespace.log(diagonal), axis=-1)
        )

    def apply_inverse(self, x):
        infer_backend(self._precisionFactor, x)
        transpose = self._backend.namespace.swapaxes(
            self._precisionFactor, -1, -2
        )
        return matrix_apply(
            self._precisionFactor, matrix_apply(transpose, x)
        ) / self._factorScaling

    def quadratic_form(self, x):
        namespace = self._backend.namespace
        return namespace.sum(
            namespace.square(self.apply_chol_factor_transpose(x)), axis=-1
        )

    def dual_quadratic_form(self, x):
        infer_backend(self._precisionFactor, x)
        namespace = self._backend.namespace
        transpose = namespace.swapaxes(self._precisionFactor, -1, -2)
        return namespace.sum(
            namespace.square(matrix_apply(transpose, x)), axis=-1
        ) / self._factorScaling

    def apply(self, x):
        infer_backend(self._precisionFactor, x)
        whitened = triangular_solve(self._backend, self._precisionFactor, x)
        return self._factorScaling * triangular_solve(
            self._backend, self._precisionFactor, whitened, transpose=True
        )
