import copy
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
    if isinstance(scaling, (int, float)):
        metadata = backend.metadata(reference)
        return backend.asarray(scaling, dtype=metadata.dtype, device=metadata.device)
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


def nugget_array(backend, reference, nugget):
    metadata = backend.metadata(reference)
    # Integer covariance inputs still need floating-point regularisation.
    unit = backend.asarray(1, dtype=metadata.dtype, device=metadata.device) * 1.0
    value = scaling_array(backend, unit, nugget)
    value = backend.asarray(value, dtype=backend.metadata(unit).dtype,
                            device=metadata.device)
    if value.ndim != 0:
        raise ValueError('Nugget must be a scalar.')
    backend.check_positive_definite(
        backend.namespace.isfinite(value) & (value >= 0),
        'Nugget must be finite and non-negative.',
    )
    return value


def factorisation_message(backend, reference, nugget):
    dtype = backend.metadata(reference).dtype
    value = nugget if isinstance(nugget, (int, float)) else 'array-valued'
    return (
        f'Covariance factorisation failed (dtype={dtype}, nugget={value}). '
        'The regularised covariance must admit a finite Cholesky factor '
        'with positive diagonal. '
        'Check the covariance, increase the nugget explicitly, or use higher precision.'
    )


class CovarianceMatrix(CovarianceOperatorInterface):
    """Abstract base class for immutable backend-native covariance matrices."""

    def __init__(self, backend, reference, scaling=1.0):
        self._backend = backend
        self._scaling = scaling_array(backend, reference, scaling) * 1.0
        namespace = backend.namespace
        backend.check_positive_definite(
            namespace.all(namespace.isfinite(self._scaling) & (self._scaling > 0)),
            'Scaling must be finite and positive.',
        )

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
        """Replace scaling while sharing the owned factorisation."""
        result = copy.copy(self)
        CovarianceMatrix.__init__(result, self._backend, self._scaling, scaling)
        return result

    def _tree_flatten(self):
        return (getattr(self, self._factorName), self._scaling), self._backend

    @classmethod
    def _restore(cls, backend, factor, scaling):
        """Restore owned numerical state without validating or copying it again."""
        result = cls.__new__(cls)
        result._backend = backend
        result._scaling = scaling
        setattr(result, cls._factorName, factor)
        return result


class DiagonalCovarianceMatrix(CovarianceMatrix):
    """Independent variances; 'nugget' is added before applying 'scaling'."""
    _factorName = "_sqrtMV"

    def __init__(self, marginalVariance, scaling=1.0, *, nugget=0.0):
        backend, marginalVariance = backend_array(marginalVariance)
        if marginalVariance.ndim != 1:
            raise ValueError("Marginal variances must be one-dimensional.")
        regularisation = nugget_array(backend, marginalVariance, nugget)
        marginalVariance = marginalVariance + regularisation
        namespace = backend.namespace
        message = factorisation_message(backend, marginalVariance, nugget)
        backend.check_positive_definite(
            namespace.all(namespace.isfinite(marginalVariance) & (marginalVariance > 0)),
            message,
        )
        super().__init__(backend, marginalVariance, scaling)
        self._sqrtMV = namespace.sqrt(marginalVariance)

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

    def __init__(self, dimension, variance, scaling=1.0, *, nugget=0.0):
        backend, variance = backend_array(variance)
        metadata = backend.metadata(variance)
        marginalVariance = backend.ones(
            dimension, dtype=metadata.dtype, device=metadata.device
        ) * variance
        super().__init__(marginalVariance, scaling, nugget=nugget)

class DenseCovarianceMatrix(CovarianceMatrix):
    """Dense covariance; 'nugget * I' is added before applying 'scaling'."""
    _factorName = "_cholFactor"

    def __init__(self, denseCovMat, scaling=1.0, *, nugget=0.0):
        backend, denseCovMat = backend_array(denseCovMat)
        if (
                denseCovMat.ndim < 2
                or denseCovMat.shape[-2] != denseCovMat.shape[-1]):
            raise ValueError(
                "Covariance matrix must be square over its final two axes."
            )
        regularisation = nugget_array(backend, denseCovMat, nugget)
        message = factorisation_message(backend, denseCovMat, nugget)
        namespace = backend.namespace
        backend.check_positive_definite(
            namespace.all(namespace.isfinite(denseCovMat)), message,
        )
        if backend.name == "numpy" and not np.allclose(
                denseCovMat,
                np.swapaxes(denseCovMat, -1, -2)):
            raise ValueError("Covariance matrix must be symmetric.")

        if not isinstance(nugget, (int, float)) or nugget != 0:
            metadata = backend.metadata(denseCovMat)
            denseCovMat = denseCovMat + regularisation * backend.eye(
                denseCovMat.shape[-1], dtype=metadata.dtype, device=metadata.device,
            )
        super().__init__(backend, denseCovMat, scaling)
        self._cholFactor = backend.checked_cholesky(denseCovMat, message)
        backend.check_positive_definite(
            namespace.all(namespace.isfinite(self._cholFactor)) &
            namespace.all(namespace.diagonal(self._cholFactor) > 0),
            message,
        )

    @classmethod
    def from_cholesky(cls, cholFactor, scaling=1.0):
        """Build from a finite lower factor with positive diagonal, without factoring.

        The covariance is 'scaling * L @ L.T'. External factors are validated
        and copied once; compiled checks use the backend's checked boundary.
        """
        backend, cholFactor = backend_array(cholFactor)
        if cholFactor.ndim < 2 or cholFactor.shape[-2] != cholFactor.shape[-1]:
            raise ValueError(
                "Cholesky factor must be square over its final two axes."
            )
        namespace = backend.namespace
        backend.check_positive_definite(
            namespace.all(namespace.isfinite(cholFactor)),
            'Cholesky factor must be finite.',
        )
        indices = backend.arange(cholFactor.shape[-1],
                                  device=backend.metadata(cholFactor).device)
        upper = namespace.where(indices[:, None] < indices[None, :], cholFactor, 0)
        backend.check_positive_definite(
            namespace.all(upper == 0), 'Cholesky factor must be lower triangular.',
        )
        backend.check_positive_definite(
            namespace.all(namespace.diagonal(cholFactor) > 0),
            'Cholesky factor must have a positive diagonal.',
        )
        # Isolate mutable inputs without detaching their autodiff graph.
        factor = cholFactor * 1.0
        result = cls.__new__(cls)
        CovarianceMatrix.__init__(result, backend, factor, scaling)
        result._cholFactor = factor
        return result

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
