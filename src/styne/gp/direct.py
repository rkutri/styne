import numpy as np
from scipy.linalg import solve_triangular
from styne.gp.engine import GPEngine, GPState
from styne.model.representation.expansion import Expansion
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import CovarianceMatrix, DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.utility.grid import Grid
from styne.utility.interpolation import Interpolation1D


class DirectExpansion(Expansion):
    """
    A Gaussian process parametrisation where the coordinates directly
    correspond to function values at the representation grid points.

    Parameters
    ----------
    grid : Grid
        The grid on which the direct coefficients live.
    dimension : int
        Number of coefficients (grid points).
    """

    def __init__(
            self, grid: Grid, dimension: int,
            shapeCovariance: CovarianceMatrix = None):

        self._dim = dimension
        self._grid = grid
        self._shapeCov = shapeCovariance

    @property
    def dimension(self) -> int:
        return self._dim

    def _validate(self, coeff: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coeff, dtype=float)
        if coeff.ndim < 1 or coeff.shape[-1] != self._dim:
            raise ValueError(
                f"Expected coefficient shape (..., {self._dim}); "
                f"got {coeff.shape}."
            )
        return coeff

    def evaluate(self, coefficient, grid: Grid) -> np.ndarray:
        """
        Linearly interpolate explicit coefficients onto a query grid.

        Parameters
        ----------
        grid : Grid
            Sites at which to evaluate the coefficients.

        Returns
        -------
        np.ndarray
            Interpolated field values at `grid`.

        Raises
        ------
        NotImplementedError
            If the representation grid is 2D. Off-grid evaluation is
            currently only implemented in 1D.
        """

        if self._grid.dimension != 1:
            raise NotImplementedError(
                "DirectExpansion does not support evaluate() in 2D."
            )

        gridArr = self._grid.to_array().ravel()
        coefficient = self._validate(coefficient)

        def evaluate_one(values):
            if self._shapeCov is not None:
                values = self._shapeCov.apply_chol_factor(values)
            return Interpolation1D(
                gridArr, values, degree=1
            ).evaluate(grid)

        if coefficient.ndim == 1:
            return evaluate_one(coefficient)
        batchShape = coefficient.shape[:-1]
        values = np.stack([
            evaluate_one(row)
            for row in coefficient.reshape(-1, self._dim)
        ])
        return values.reshape(batchShape + (values.shape[-1],))


class DirectGPEngine(GPEngine):
    """
    GPEngine for the direct (whitened) GP parametrisation.

    The parameter $\theta = \tilde{z}$ lives in $N(0, I)$. The field at
    the sites is $u = L\tilde{z}$, where $L$ is the Cholesky factor of
    K(\text{sites}, \text{sites})$.
    Changing the sites requires rebuilding the covariance.

    Parameters
    ----------
    grid : Grid
        Sites the GP is initially defined on.
    nugget : float, default 0.0
        Diagonal regularisation added to the covariance before
        factorisation.
    """

    def __init__(self, grid: Grid, nugget: float = 0.0):
        self._grid = grid
        self._nugget = nugget
        self._shapeCovariance = None

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def spatialDimension(self) -> int:
        return self._grid.dimension

    def requires_covariance_rebuild(self) -> bool:
        return True

    def set_sites(self, sites: Grid) -> None:

        if sites is not None:
            self._grid = sites

    def build_expansion(self) -> DirectExpansion:
        """
        Construct a new `DirectExpansion` on the engine's current grid.

        Returns
        -------
        DirectExpansion
            Static direct representation matching the engine's grid.
        """
        return DirectExpansion(
            self._grid, len(self._grid), self._shapeCovariance
        )

    def build_covariance(
            self, covarianceFunction: CovarianceFunctionInterface
    ) -> CovarianceMatrix:

        covarianceMatrix = covarianceFunction.evaluate_covariance(self._grid, self._grid)

        if not isinstance(covarianceMatrix, CovarianceMatrix):
            covarianceMatrix = 0.5 * (covarianceMatrix + covarianceMatrix.T)
            if self._nugget > 0.:
                covarianceMatrix = covarianceMatrix + \
                    self._nugget * np.eye(len(self._grid))

        self._shapeCovariance = covarianceMatrix if \
            isinstance(covarianceMatrix, CovarianceMatrix) \
            else DenseCovarianceMatrix(covarianceMatrix)
        
        return IIDCovarianceMatrix(len(self._grid), 1.0)


    def apply_jacobian(
            self, vector: np.ndarray,
            covariance: CovarianceMatrix) -> np.ndarray:
        if vector.ndim == 1:
            return self._shapeCovariance.apply_chol_factor(vector)
        batchShape = vector.shape[:-1]
        result = np.stack([
            self._shapeCovariance.apply_chol_factor(row)
            for row in vector.reshape(-1, vector.shape[-1])
        ])
        return result.reshape(batchShape + (result.shape[-1],))

    def apply_adjoint_jacobian(
            self, cotangent: np.ndarray,
            covariance: CovarianceMatrix) -> np.ndarray:
        if cotangent.ndim == 1:
            return self._shapeCovariance.apply_chol_factor_transpose(cotangent)
        batchShape = cotangent.shape[:-1]
        result = np.stack([
            self._shapeCovariance.apply_chol_factor_transpose(row)
            for row in cotangent.reshape(-1, cotangent.shape[-1])
        ])
        return result.reshape(batchShape + (result.shape[-1],))

    def evaluate_exact_conditional(self, queryGrid: Grid, state, covFcn: CovarianceFunctionInterface) -> np.ndarray:
        """ Exact conditional mean projection for Direct engine. """
        # Determine coordinate (handle both Expansion and Function)
        z = state.coordinate if hasattr(state, 'coordinate') \
            else np.asarray(state)
        
        kStar = covFcn.evaluate_covariance(queryGrid, self._grid)
        kStarArr = kStar.to_dense() if isinstance(kStar, DenseCovarianceMatrix) else np.asarray(kStar)
        
        if isinstance(self._shapeCovariance, DenseCovarianceMatrix):
            L = self._shapeCovariance._cholFactor
            LT_inv_z = solve_triangular(L.T, z, lower=False)
            return kStarArr @ LT_inv_z
        else:
            raise NotImplementedError("Exact conditional only supported for DenseCovarianceMatrix shape covariance.")

    def evaluate_hyper_gradient(self, state: Expansion, zTilde: np.ndarray, covFcn: CovarianceFunctionInterface) -> dict:
        if not hasattr(covFcn, 'evaluate_covariance_gradient'):
            raise NotImplementedError("Exact gradients not implemented for this covariance.")
        
        gradK = covFcn.evaluate_covariance_gradient(self._grid, self._grid)
        
        if not isinstance(self._shapeCovariance, DenseCovarianceMatrix):
             raise NotImplementedError("Hyper gradient requires dense covariance matrix.")
             
        L = self._shapeCovariance._cholFactor
        z = state.coordinate if hasattr(state, 'coordinate') \
            else np.asarray(state)
        result = {}
        for paramName, dK in gradK.items():
            dKArr = dK.to_dense() if isinstance(dK, DenseCovarianceMatrix) else np.asarray(dK)
            
            temp = solve_triangular(L, dKArr, lower=True)
            M = solve_triangular(L, temp.T, lower=True).T
            
            X = np.tril(M)
            np.fill_diagonal(X, 0.5 * np.diag(M))
            
            du = L @ (X @ z)
            result[paramName] = float(np.dot(zTilde, du))
            
        return result

    def create_predictor(
            self, gpState: GPState, queryGrid: Grid,
            coefficient: np.ndarray) -> Predictor:
        shapeCovariance = self._shapeCovariance
        if not isinstance(shapeCovariance, DenseCovarianceMatrix):
            raise NotImplementedError(
                "Only DenseCovarianceMatrix is supported."
            )

        kStar = gpState.covarianceFunction.evaluate_covariance(
            queryGrid, self._grid)
        kStarArray = kStar.to_dense() if isinstance(
            kStar, DenseCovarianceMatrix) else np.asarray(kStar)
        frozenCoefficient = np.array(coefficient, dtype=float, copy=True)
        frozenFactor = np.array(
            shapeCovariance._cholFactor, dtype=float, copy=True)
        mean = kStarArray @ solve_triangular(
            frozenFactor.T, frozenCoefficient, lower=False)
        return DirectGPPredictor(mean)


class DirectGPPredictor(Predictor):
    """
    Immutable out-of-sample mean snapshot for the direct GP engine.

    Parameters
    ----------
    mean : np.ndarray
        Predictive mean computed from the coefficient and covariance state
        at construction time.
    """

    def __init__(self, mean: np.ndarray):
        self._mean = np.array(mean, dtype=float, copy=True)

    def mean(self) -> np.ndarray:
        """Return an independent copy of the snapshotted mean."""
        return self._mean.copy()
