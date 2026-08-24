from __future__ import annotations
import numpy as np

from styne.gp.dnautility import (
    BC, BoundaryCondition,
)
from styne.backend import infer_backend
from styne.model.representation.expansion import (
    BoundLinearExpansion,
    LinearExpansion,
    backend_constant,
)
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.utility.grid import Grid, UniformGrid
from styne.utility.interpolation import (
    linear_interpolation_matrix, bilinear_interpolation_matrix
)


def _validate_coefficient(coefficient):
    if coefficient.ndim < 1:
        raise ValueError(
            "coefficient must have shape (..., dimension)"
        )
    return coefficient


def _backend_zeros(reference, shape):
    backend = infer_backend(reference)
    metadata = backend.metadata(reference)
    return backend.zeros(
        shape, dtype=metadata.dtype, device=metadata.device
    )


def _cos_series_backend(coefficient):
    backend = infer_backend(coefficient)
    namespace = backend.namespace
    rootTwo = backend_constant(np.sqrt(2.0), coefficient)
    first = coefficient[..., :1]
    rest = coefficient[..., 1:] / rootTwo
    zero = _backend_zeros(
        coefficient, coefficient.shape[:-1] + (1,)
    )
    padded = namespace.concatenate(
        (first, rest, zero),
        axis=-1,
    )
    return backend.dct1(padded, axis=-1)


def _sin_series_backend(coefficient):
    backend = infer_backend(coefficient)
    namespace = backend.namespace
    rootTwo = backend_constant(np.sqrt(2.0), coefficient)
    interior = backend.dst1(
        coefficient[..., 1:] / rootTwo, axis=-1
    )
    zero = _backend_zeros(coefficient, coefficient.shape[:-1] + (1,))
    return namespace.concatenate((zero, interior, zero), axis=-1)


def _series_backend(coefficient, boundaryCondition, axis):
    moved = coefficient if axis == -1 else coefficient.swapaxes(axis, -1)
    result = _sin_series_backend(moved) \
        if boundaryCondition == BC.DIRICHLET \
        else _cos_series_backend(moved)
    return result if axis == -1 else result.swapaxes(axis, -1)


def _as_axis_tuple(value, d):
    if np.isscalar(value):
        return (value,) * d
    t = tuple(value)
    if len(t) != d:
        raise ValueError(
            f"expected scalar or length-{d} sequence, got {t!r}"
        )
    return t


# ---- DNA Fourier component ----


class DNAFourierComponentExpansion(LinearExpansion):
    """
    Expansion for one DNA Fourier component with a fixed boundary condition.

    Coefficient is the flat row-major array of spectral weights ξ_μ for all
    modes μ in the BC block: indices 0..q (NEUMANN) or 1..q (DIRICHLET) per
    axis, with total length bc.block_size(q).
    """

    def __init__(
            self, bc: BoundaryCondition, q, alpha=1.0,
            spectralWeights=None):
        self._bc = bc
        self._q = _as_axis_tuple(q, bc.d)
        self._alpha = _as_axis_tuple(alpha, bc.d)
        if spectralWeights is None:
            spectralWeights = np.ones(self.dimension)
        elif not hasattr(spectralWeights, "shape"):
            spectralWeights = np.asarray(spectralWeights)
        if spectralWeights.shape != (self.dimension,):
            raise ValueError(
                f"Expected weights of dimension {self.dimension}, "
                f"got {spectralWeights.shape}."
            )
        self._weights = spectralWeights

    @property
    def spectralWeights(self):
        return self._weights

    @property
    def dimension(self) -> int:
        return self._bc.block_size(self._q)

    @property
    def spatialDimension(self) -> int:
        return self._bc.d

    @property
    def resolution(self) -> tuple:
        return self._q

    @property
    def alpha(self):
        return self._alpha

    @property
    def nativeGrid(self):
        if self._bc.d == 1:
            return UniformGrid(0., self._alpha[0], self._q[0] + 2)
        return UniformGrid(
            (0., self._alpha[0], self._q[0] + 2),
            (0., self._alpha[1], self._q[1] + 2),
        )

    def _validate(self, coefficient):
        coefficient = _validate_coefficient(coefficient)
        if coefficient.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected coefficient of dimension {self.dimension}, "
                f"got {coefficient.shape[-1]}."
            )
        return coefficient

    def evaluate_interior(self, coefficient) -> np.ndarray:
        coefficient = self._validate(coefficient)
        if self._bc.d == 1:
            return self._evaluate_interior_1d(coefficient)
        return self._evaluate_interior_2d(coefficient)

    def _evaluate_interior_1d(self, coefficient) -> np.ndarray:
        c = coefficient * backend_constant(self._weights, coefficient)

        if self._bc[0] == BC.NEUMANN:
            return _cos_series_backend(c)

        # Dirichlet (sine) coefficients have size q, but sin_series expects
        # size q+1 with a[0]=0, so prepend the zero mode before synthesis.
        padding = _backend_zeros(coefficient, coefficient.shape[:-1] + (1,))
        paddedCoefficient = infer_backend(coefficient).namespace.concatenate(
            (padding, c), axis=-1
        )
        return _sin_series_backend(paddedCoefficient)

    def _evaluate_interior_2d(self, coefficient) -> np.ndarray:
        nX = (self._q[0] + 1) if self._bc[0] == BC.NEUMANN else self._q[0]
        nY = (self._q[1] + 1) if self._bc[1] == BC.NEUMANN else self._q[1]

        backend = infer_backend(coefficient)
        namespace = backend.namespace
        c = coefficient * backend_constant(self._weights, coefficient)
        spectral = c.reshape(coefficient.shape[:-1] + (nX, nY))

        if self._bc[1] == BC.DIRICHLET:
            zeroColumn = _backend_zeros(
                coefficient, spectral.shape[:-1] + (1,)
            )
            spectral = namespace.concatenate(
                (zeroColumn, spectral), axis=-1
            )
        if self._bc[0] == BC.DIRICHLET:
            zeroRow = _backend_zeros(
                coefficient, spectral.shape[:-2] + (1, spectral.shape[-1])
            )
            spectral = namespace.concatenate((zeroRow, spectral), axis=-2)

        result = _series_backend(spectral, self._bc[1], axis=-1)
        return _series_backend(result, self._bc[0], axis=-2)

    def evaluate_native(self, coefficient) -> np.ndarray:
        """Evaluate explicit component coefficients on the native grid."""
        coefficient = self._validate(coefficient)
        interior = self.evaluate_interior(coefficient)
        if self._bc.d == 1:
            return interior
        if coefficient.ndim == 1:
            return interior.ravel()
        return interior.reshape(coefficient.shape[:-1] + (-1,))

    def _bind(self, grid: Grid) -> BoundLinearExpansion:
        return _DNAEvaluation(self, grid)

# ---- Full DNA field ----


class DNAFourierExpansion(LinearExpansion):
    """
    Expansion for the full DNA GRF, averaging 2^d independent component
    expansions. Coefficients are a flat concatenation of component blocks in
    ``BoundaryCondition.all_combinations`` order.
    """

    def __init__(self, q, d: int, alpha=1.0, spectralWeights=None):
        self._q = _as_axis_tuple(q, d)
        self._d = d
        self._alpha = _as_axis_tuple(alpha, d)
        bcs = BoundaryCondition.all_combinations(d)
        dimensions = [bc.block_size(self._q) for bc in bcs]
        totalDimension = sum(dimensions)
        if spectralWeights is None:
            spectralWeights = np.ones(totalDimension)
        elif not hasattr(spectralWeights, "shape"):
            spectralWeights = np.asarray(spectralWeights)
        if spectralWeights.shape != (totalDimension,):
            raise ValueError(
                f"Expected weights of dimension {totalDimension}, "
                f"got {spectralWeights.shape}."
            )

        self._spectralWeights = spectralWeights

        self._slices = []
        self._components = []
        offset = 0
        for bc, dimension in zip(bcs, dimensions):
            coefficientSlice = slice(offset, offset + dimension)
            self._slices.append(coefficientSlice)
            self._components.append(DNAFourierComponentExpansion(
                bc, self._q, self._alpha,
                spectralWeights[coefficientSlice]
            ))
            offset += dimension

    @property
    def dimension(self) -> int:
        return sum(component.dimension for component in self._components)

    @property
    def spatialDimension(self) -> int:
        return self._d

    @property
    def resolution(self) -> tuple:
        return self._q

    @property
    def alpha(self):
        return self._alpha

    @property
    def spectralWeights(self):
        return self._spectralWeights

    def with_spectral_weights(self, spectralWeights):
        """Return an equivalent expansion with replacement static weights."""
        return type(self)(
            self._q, self._d, self._alpha, spectralWeights
        )

    def _validate(self, coefficient):
        coefficient = _validate_coefficient(coefficient)
        if coefficient.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected coefficient of dimension {self.dimension}, "
                f"got {coefficient.shape[-1]}."
            )
        return coefficient

    def evaluate_native(self, coefficient) -> np.ndarray:
        """Evaluate explicit coefficients on the native DNA grid."""
        coefficient = self._validate(coefficient)
        scale = backend_constant(2. ** (-self._d / 2.), coefficient)
        fields = [
            component.evaluate_native(coefficient[..., coefficientSlice])
            for component, coefficientSlice in zip(
                self._components, self._slices
            )
        ]
        namespace = infer_backend(coefficient).namespace
        return scale * namespace.sum(namespace.stack(fields, axis=0), axis=0)

    @property
    def nativeGrid(self):
        if self._d == 1:
            return UniformGrid(0., self._alpha[0], self._q[0] + 2)
        return UniformGrid(
            (0., self._alpha[0], self._q[0] + 2),
            (0., self._alpha[1], self._q[1] + 2),
        )

    def _bind(self, grid: Grid) -> BoundLinearExpansion:
        return _DNAEvaluation(self, grid)


class _DNAEvaluation(BoundLinearExpansion):

    def __init__(self, expansion, grid):
        self._expansion = expansion
        q = expansion.resolution
        alpha = expansion.alpha
        points = grid.to_array() if isinstance(grid, Grid) \
            else np.asarray(grid)
        if expansion.spatialDimension == 1:
            axis = np.linspace(0., alpha[0], q[0] + 2)
            self._interpolation = linear_interpolation_matrix(
                points.ravel(), axis
            ).toarray()
        else:
            axis0 = np.linspace(0., alpha[0], q[0] + 2)
            axis1 = np.linspace(0., alpha[1], q[1] + 2)
            self._interpolation = bilinear_interpolation_matrix(
                points, axis0, axis1
            ).toarray()

    @property
    def dimension(self):
        return self._expansion.dimension

    def evaluate(self, coefficient):
        native = self._expansion.evaluate_native(coefficient)
        interpolation = backend_constant(self._interpolation, coefficient)
        return native @ interpolation.T


class _DNAGPSpecification:
    """Construction and prediction rules for a DNA GP parametrisation."""

    def __init__(self, q, d: int, alpha=1.0):
        self._q = _as_axis_tuple(q, d)
        self._d = d
        self._alpha = _as_axis_tuple(alpha, d)

    @property
    def spatialDimension(self) -> int:
        return self._d

    @property
    def resolution(self) -> tuple:
        return self._q

    @property
    def nativeGrid(self) -> UniformGrid:
        if self._d == 1:
            return UniformGrid(0., self._alpha[0], self._q[0] + 2)
        return UniformGrid(
            (0., self._alpha[0], self._q[0] + 2),
            (0., self._alpha[1], self._q[1] + 2),
        )

    def build(self, covFcn: CovarianceFunctionInterface):
        """
        Build the whitened covariance for the DNA GRF.

        Under the whitening contract, the latent coefficients are standard
        white noise, so the returned covariance operator is the identity.
        The spectral square root is evaluated here and stored in 'self._weights'
        instead, one weight per mode across all 2^d boundary-condition blocks in
        'BoundaryCondition.all_combinations' order. Expansion evaluation applies
        these weights before synthesis. The weights carry the spectral density's
        square root and a domain-extent factor 'prod(alpha) ** -0.5'.

        Returns
        -------
        DiagonalCovarianceMatrix
            The identity operator (an 'IIDCovarianceMatrix' with unit variance),
            matching the whitened parametrisation.
        """
        from styne.statistics.covariance import IIDCovarianceMatrix
        q, alpha = self._q, self._alpha
        spectralDensities = []

        for bc in BoundaryCondition.all_combinations(self._d):
            if self._d == 1:
                mRange = range(q[0] + 1) if bc[0] == BC.NEUMANN else range(1, q[0] + 1)
                freqs = np.array(list(mRange)) / (2. * alpha[0])
                spectralDensities.append(covFcn.evaluate_fourier(freqs[:, np.newaxis]))
            else:
                xRange = range(q[0] + 1) if bc[0] == BC.NEUMANN else range(1, q[0] + 1)
                yRange = range(q[1] + 1) if bc[1] == BC.NEUMANN else range(1, q[1] + 1)
                X, Y = np.meshgrid(xRange, yRange, indexing='ij')
                freqs = np.column_stack((
                    X.ravel() / (2. * alpha[0]),
                    Y.ravel() / (2. * alpha[1]),
                ))
                spectralDensities.append(covFcn.evaluate_fourier(freqs))

        backend = infer_backend(spectralDensities[0])
        weights = backend.namespace.sqrt(
            backend.namespace.concatenate(spectralDensities, axis=0)
        ) * np.prod(self._alpha) ** (-0.5)
        expansion = DNAFourierExpansion(
            self._q, self._d, self._alpha, weights
        )
        unitVariance = weights[0] * 0.0 + 1.0
        return IIDCovarianceMatrix(weights.shape[0], unitVariance), expansion

    def evaluate_exact_conditional(
            self, expansion, queryGrid, state, covFcn, sites):
        """
        Evaluate the exact conditional predictive mean using the dense exact covariance.
        This provides a benchmark fallback avoiding interpolation, though it is O(N^3).
        """
        from scipy.linalg import cholesky, cho_solve, LinAlgError
        from styne.statistics.covariance import DenseCovarianceMatrix

        coefficient = state.coordinate if hasattr(state, 'coordinate') \
            else state
        uObs = expansion.evaluate(coefficient, sites)

        kObs = covFcn.evaluate_covariance(sites, sites)
        kObsArr = kObs.to_dense() if isinstance(kObs, DenseCovarianceMatrix) else np.asarray(kObs)
        
        # Add a small nugget for numerical stability in the dense exact covariance block
        kObsArr.flat[::kObsArr.shape[0] + 1] += 1e-8

        try:
            L = cholesky(kObsArr, lower=True)
        except LinAlgError:
            # Fallback if numerically unstable
            kObsArr.flat[::kObsArr.shape[0] + 1] += 1e-5
            L = cholesky(kObsArr, lower=True)

        kStar = covFcn.evaluate_covariance(queryGrid, sites)
        kStarArr = kStar.to_dense() if isinstance(kStar, DenseCovarianceMatrix) else np.asarray(kStar)

        return kStarArr @ cho_solve((L, True), uObs)

    def create_predictor(
            self, gpState, queryGrid: Grid,
            coefficient: np.ndarray, observationGrid=None) -> Predictor:
        frozenCoefficient = np.array(coefficient, dtype=float, copy=True)
        mean = gpState.expansion.evaluate(frozenCoefficient, queryGrid)
        return DNAGPPredictor(mean)


class DNAGPPredictor(Predictor):
    """Immutable out-of-sample mean snapshot for the DNA parametrisation."""

    def __init__(self, mean: np.ndarray):
        self._mean = np.array(mean, dtype=float, copy=True)

    def mean(self) -> np.ndarray:
        """Return an independent copy of the snapshotted mean."""
        return self._mean.copy()
