import numpy as np
from scipy.linalg import solve_triangular
from styne.gp.engine import GPEngine, GPState
from styne.model.representation.expansion import Expansion
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import CovarianceMatrix, DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.utility.grid import Grid
from styne.utility.interpolation import Interpolation1D


class DirectRealisation(Expansion):
    """
    A Gaussian process parametrisation where the coordinates directly
    correspond to the values of the realisation at the grid points.

    Parameters
    ----------
    grid : Grid
        The grid the realisation's coefficients live on.
    dimension : int
        Number of coefficients (grid points).
    """

    def __init__(self, grid: Grid, dimension: int):

        self._dim = dimension
        self._grid = grid
        self._coeff = None
        self._shapeCov = None

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def coefficient(self) -> np.ndarray:

        if self._coeff is None:
            raise ValueError("DirectRealisation has no coefficient set.")

        return self._coeff.copy()

    @coefficient.setter
    def coefficient(self, coeff: np.ndarray) -> None:
        self.project(self.validate(coeff))

    def validate(self, coeff: np.ndarray) -> np.ndarray:

        coeff = np.asarray(coeff, dtype=float)

        if coeff.size != self._dim:
            raise ValueError(
                f"Expected {self._dim} values, got {coeff.size}.")

        return coeff

    def project(self, coeff: np.ndarray) -> None:

        if coeff.size != self._dim:
            raise ValueError(
                f"Expected {self._dim} values, got {coeff.size}.")

        self._coeff = coeff

    def evaluate(self, queryGrid: Grid) -> np.ndarray:
        """
        Linearly interpolate the realisation onto a query grid.

        Parameters
        ----------
        queryGrid : Grid
            Sites to evaluate the realisation at.

        Returns
        -------
        np.ndarray
            Interpolated field values at `queryGrid`.

        Raises
        ------
        NotImplementedError
            If the realisation's own grid is 2D. Off-grid evaluation is
            currently only implemented in 1D.
        """

        if self._grid.dimension != 1:
            raise NotImplementedError(
                "DirectRealisation does not support evaluate() in 2D."
            )

        gridArr = self._grid.to_array().ravel()
        fieldValues = self._coeff
        if self._shapeCov is not None:
            fieldValues = self._shapeCov.apply_chol_factor(fieldValues)

        return Interpolation1D(gridArr, fieldValues, degree=1).evaluate(queryGrid)

    def clone(self) -> 'DirectRealisation':
        result = DirectRealisation(self._grid, self._dim)
        if self._coeff is not None:
            result._coeff = self._coeff.copy()
        result._shapeCov = self._shapeCov
        return result


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

    def build_realisation(self) -> DirectRealisation:
        """
        Construct a new `DirectRealisation` on the engine's current grid.

        Returns
        -------
        DirectRealisation
            A fresh, uninitialised realisation matching the engine's grid.
        """
        result = DirectRealisation(self._grid, len(self._grid))
        result._shapeCov = self._shapeCovariance
        return result

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
            self, v: np.ndarray, covariance: CovarianceMatrix) -> np.ndarray:
        return self._shapeCovariance.apply_chol_factor(v)

    def apply_adjoint_jacobian(
            self, w: np.ndarray, covariance: CovarianceMatrix) -> np.ndarray:
        return self._shapeCovariance.apply_chol_factor_transpose(w)

    def evaluate_exact_conditional(self, queryGrid: Grid, state, covFcn: CovarianceFunctionInterface) -> np.ndarray:
        """ Exact conditional mean projection for Direct engine. """
        from scipy.linalg import cholesky, cho_solve
        
        # Determine coordinate (handle both Expansion and Function)
        z = state.coordinate if hasattr(state, 'coordinate') else state.coefficient
        
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
        z = state.coefficient
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
