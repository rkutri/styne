from __future__ import annotations
import numpy as np

from styne.gp.dnautility import (
    BC, BoundaryCondition,
    cos_series, sin_series,
    sin_series_rows, sin_series_cols,
    cos_series_rows, cos_series_cols,
    adj_cos_series_1d, adj_sin_series_1d,
    adj_cos_series_rows, adj_cos_series_cols,
    adj_sin_series_rows, adj_sin_series_cols,
)
from styne.gp.engine import GPEngine, GPState
from styne.model.representation.expansion import Expansion
from styne.parameter.block import BlockParameter
from styne.parameter.function import Function
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import CovarianceMatrix, DiagonalCovarianceMatrix
from styne.utility.grid import Grid, UniformGrid
from styne.utility.interpolation import (
    Interpolation1D, GridInterpolation2D,
    linear_interpolation_matrix, bilinear_interpolation_matrix
)


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


class DNAFourierComponentRealisation(Expansion):
    """
    Expansion for one DNA Fourier component with a fixed boundary condition.

    Coefficient is the flat row-major array of spectral weights ξ_μ for all
    modes μ in the BC block: indices 0..q (NEUMANN) or 1..q (DIRICHLET) per
    axis, with total length bc.block_size(q).
    """

    def __init__(
            self, bc: BoundaryCondition, q, alpha=1.0):
        self._bc = bc
        self._q = _as_axis_tuple(q, bc.d)
        self._alpha = _as_axis_tuple(alpha, bc.d)
        self._coeff = np.zeros(self.dimension)
        self._weights = np.ones(self.dimension)
        self._scratch = None

    @property
    def spectralWeights(self) -> np.ndarray:
        return self._weights

    @spectralWeights.setter
    def spectralWeights(self, value: np.ndarray) -> None:
        if value is not None and value.size != self.dimension:
            raise ValueError(
                f"Expected weights of dimension {self.dimension}, got {value.size}."
            )
        self._weights = value

    @property
    def dimension(self) -> int:
        return self._bc.block_size(self._q)

    @property
    def coefficient(self) -> np.ndarray:
        if self._coeff is None:
            raise ValueError(
                "DNAFourierComponentRealisation coefficient not set.")
        return self._coeff

    @coefficient.setter
    def coefficient(self, value: np.ndarray) -> None:
        self.project(self.validate(value))

    def project(self, coeff: np.ndarray) -> None:
        currentDim = coeff.shape[1] if coeff.ndim == 2 else coeff.size
        if currentDim != self.dimension:
            raise ValueError(
                f"Expected coefficient of dimension {self.dimension}, got {currentDim}."
            )
        self._coeff = np.asarray(coeff, dtype=float)
        if self._scratch is not None and self._scratch.shape[0] != (self._q[0] + 2):
            self._scratch = None

    def evaluate_interior(self) -> np.ndarray:
        if self._coeff is None:
            raise ValueError(
                "DNAFourierComponentRealisation coefficient not set.")
        if self._bc.d == 1:
            return self._evaluate_interior_1d()
        return self._evaluate_interior_2d()

    def _evaluate_interior_1d(self) -> np.ndarray:
        isBatch = self._coeff.ndim == 2
        nG = self._q[0] + 2
        expectedShape = (self._coeff.shape[0], nG) if isBatch else (nG,)
        
        if self._scratch is None or self._scratch.shape != expectedShape:
            self._scratch = np.zeros(expectedShape)

        c = self._coeff * self._weights if self._weights is not None else self._coeff

        if self._bc[0] == BC.NEUMANN:
             return cos_series(c, axis=-1, out=self._scratch)

        # Dirichlet (sine) path. _coeff has size q, but sin_series expects
        # size q+1 with a[0]=0, so prepend the zero mode before synthesis.
        if self._coeff.ndim == 1:
            padded_coeff = np.concatenate([[0.], c])
        else:
            nBatch = self._coeff.shape[0]
            padded_coeff = np.zeros((nBatch, self.dimension + 1))
            padded_coeff[:, 1:] = c

        return sin_series(padded_coeff, axis=-1, out=self._scratch)

    def _evaluate_interior_2d(self) -> np.ndarray:
        nX = (self._q[0] + 1) if self._bc[0] == BC.NEUMANN else self._q[0]
        nY = (self._q[1] + 1) if self._bc[1] == BC.NEUMANN else self._q[1]

        c = self._coeff * self._weights if self._weights is not None else self._coeff
        C = c.reshape(nX, nY)

        A = np.zeros((self._q[0] + 1, self._q[1] + 1))
        r0 = 1 if self._bc[0] == BC.DIRICHLET else 0
        c0 = 1 if self._bc[1] == BC.DIRICHLET else 0
        A[r0:, c0:] = C

        rowFn = sin_series_rows if self._bc[1] == BC.DIRICHLET else cos_series_rows
        colFn = sin_series_cols if self._bc[0] == BC.DIRICHLET else cos_series_cols

        return colFn(rowFn(A))

    def evaluate_native(self) -> np.ndarray:
        """
        Evaluate component native field. Accumulates into '_scratch' buffer via 
        internal series functions to minimise allocations.
        """
        interior = self.evaluate_interior()
        if interior.ndim == 1:
            return interior
        if self._bc.d == 1:
            return interior # (nBatch, nG)
        return interior.reshape(interior.shape[0], -1)

    def clone(self) -> DNAFourierComponentRealisation:
        """Return a copy with cloned coefficients and shared weight reference."""
        import copy
        newObj = copy.copy(self)
        if self._coeff is not None:
            newObj._coeff = self._coeff.copy()
        newObj._scratch = None
        return newObj

    def evaluate(self, queryGrid: Grid) -> np.ndarray:
        raise NotImplementedError(
            "DNAFourierComponentRealisation does not support arbitrary-site "
            "evaluation. Use DNAFourierEngine to evaluate at sites."
        )

    def adjoint_synthesis(self, r: np.ndarray) -> np.ndarray:
        """Adjoint of evaluate_native (scaled): native space -> R^{block_size}."""
        res = self._adjoint_synthesis_1d(r) if self._bc.d == 1 \
            else self._adjoint_synthesis_2d(r)
        
        if self._weights is not None:
            return res * self._weights
        return res

    def _adjoint_synthesis_1d(self, r: np.ndarray) -> np.ndarray:
        q = self._q[0]
        if self._bc[0] == BC.NEUMANN:
            return adj_cos_series_1d(r, q)
        return adj_sin_series_1d(r, q)

    def _adjoint_synthesis_2d(self, r: np.ndarray) -> np.ndarray:
        r_mat = r.reshape(self._q[0] + 2, self._q[1] + 2)

        if self._bc[0] == BC.NEUMANN:
            col_adj = adj_cos_series_cols(r_mat, self._q[0])
        else:
            col_adj = adj_sin_series_cols(r_mat, self._q[0])

        if self._bc[1] == BC.NEUMANN:
            a_adj = adj_cos_series_rows(col_adj, self._q[1])
        else:
            a_adj = adj_sin_series_rows(col_adj, self._q[1])

        r0 = 1 if self._bc[0] == BC.DIRICHLET else 0
        c0 = 1 if self._bc[1] == BC.DIRICHLET else 0
        nX = (self._q[0] + 1) if self._bc[0] == BC.NEUMANN else self._q[0]
        nY = (self._q[1] + 1) if self._bc[1] == BC.NEUMANN else self._q[1]

        return a_adj[r0:r0 + nX, c0:c0 + nY].ravel()


# ---- Full DNA field ----


class DNAFourierRealisation(Expansion):
    """
    Expansion for the full DNA GRF, averaging 2^d independent component
    realisations. The coefficient is the flat concatenation of all component
    spectral weights, managed via a BlockParameter of Function objects
    (one Function per component, in BoundaryCondition.all_combinations order).
    """

    def __init__(self, q, d: int, alpha=1.0):
        self._q = _as_axis_tuple(q, d)
        self._d = d
        self._alpha = _as_axis_tuple(alpha, d)
        bcs = BoundaryCondition.all_combinations(d)
        self._param = BlockParameter(
            [Function(DNAFourierComponentRealisation(bc, self._q, self._alpha))
             for bc in bcs])
        self._nativeBuffer = None

    @property
    def dimension(self) -> int:
        return self._param.dimension

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
    def coefficient(self) -> np.ndarray:
        return self._param.coordinate

    @coefficient.setter
    def coefficient(self, value: np.ndarray) -> None:
        self.project(self.validate(value))

    @property
    def spectralWeights(self) -> np.ndarray:
        """Concatenated spectral weights across all BC blocks."""
        return np.concatenate([
            self._param.block(i).function.spectralWeights
            for i in range(self._param.nBlocks)
        ])

    @spectralWeights.setter
    def spectralWeights(self, value: np.ndarray) -> None:
        if value is None:
            return
        offset = 0
        for i in range(self._param.nBlocks):
            comp = self._param.block(i).function
            n = comp.dimension
            comp.spectralWeights = value[offset:offset + n]
            offset += n

    def project(self, coeff: np.ndarray) -> None:
        self._param.coordinate = coeff

    def evaluate_native(self) -> np.ndarray:
        """
        Evaluate full DNA field on its native grid. Returns a view of the
        internal buffer; copy if cross-step persistence is needed.
        """
        isBatch = self._param.coordinate.ndim == 2
        nBatch = self._param.coordinate.shape[0] if isBatch else 1

        nG0 = self._q[0] + 2
        nG1 = self._q[1] + 2 if self._d == 2 else 1
        nTotal = nG0 * nG1 if self._d == 2 else nG0
        scale = 2. ** (-self._d / 2.)

        expectedShape = (nBatch, nTotal) if isBatch else (nTotal,)
        if self._nativeBuffer is None or self._nativeBuffer.shape != expectedShape:
            self._nativeBuffer = np.zeros(expectedShape)
        else:
            self._nativeBuffer.fill(0.0)

        if self._d == 1:
            for i in range(self._param.nBlocks):
                self._nativeBuffer += self._param.block(i).function.evaluate_native()
        else:
            for i in range(self._param.nBlocks):
                self._nativeBuffer += (
                    self._param.block(i).function.evaluate_native()
                    .reshape(self._nativeBuffer.shape)
                )

        self._nativeBuffer *= scale
        return self._nativeBuffer

    def clone(self) -> DNAFourierRealisation:
        """Return a copy with cloned components and shared weight references."""
        import copy
        newObj = copy.copy(self)
        newObj._param = self._param.clone()
        newObj._nativeBuffer = None
        return newObj

    def evaluate(self, queryGrid: Grid) -> np.ndarray:
        native = self.evaluate_native()

        if self._d == 1:
            axis = np.linspace(0., self._alpha[0], self._q[0] + 2)
            return Interpolation1D(axis, native, degree=1).evaluate(queryGrid)

        nG0 = self._q[0] + 2
        nG1 = self._q[1] + 2
        axis0 = np.linspace(0., self._alpha[0], nG0)
        axis1 = np.linspace(0., self._alpha[1], nG1)
        return GridInterpolation2D(
            axis0, axis1, native.reshape(nG0, nG1)).evaluate(queryGrid)

    def adjoint_synthesis(self, r: np.ndarray) -> np.ndarray:
        """
        Adjoint of evaluate_native: native space -> R^{total_spectral_dim}.

        For each BC block b, computes scale * W_b^T @ r and concatenates
        the results. The scale factor 2^{-d/2} matches evaluate_native.
        """
        scale = 2. ** (-self._d / 2.)
        return np.concatenate([
            scale * self._param.block(i).function.adjoint_synthesis(r)
            for i in range(self._param.nBlocks)
        ])


class DNAFourierEngine(GPEngine):
    """
    GPEngine for the full DNA GRF.

    Evaluates the spectral density once across all 2^d BC blocks in
    BoundaryCondition.all_combinations order. The resulting
    DiagonalCovarianceMatrix aligns with DNAFourierRealisation's BlockParameter.
    """

    def __init__(self, q, d: int, alpha=1.0):
        self._q = _as_axis_tuple(q, d)
        self._d = d
        self._alpha = _as_axis_tuple(alpha, d)
        self._interpMat = None
        self._nativeSites = False
        self._weights = None

    @property
    def spatialDimension(self) -> int:
        return self._d

    @property
    def resolution(self) -> tuple:
        return self._q

    @property
    def spectralWeights(self) -> np.ndarray:
        return self._weights

    @property
    def nativeGrid(self) -> UniformGrid:
        if self._d == 1:
            return UniformGrid(0., self._alpha[0], self._q[0] + 2)
        return UniformGrid(
            (0., self._alpha[0], self._q[0] + 2),
            (0., self._alpha[1], self._q[1] + 2),
        )

    def build_realisation(self) -> DNAFourierRealisation:
        return DNAFourierRealisation(self._q, self._d, self._alpha)

    def build_covariance(
        self, covFcn: CovarianceFunctionInterface
    ) -> DiagonalCovarianceMatrix:
        """
        Build the whitened covariance for the DNA GRF.

        Under the whitening contract, the latent coefficients are standard
        white noise, so the returned covariance operator is the identity.
        The spectral square root is evaluated here and stored in 'self._weights'
        instead, one weight per mode across all 2^d boundary-condition blocks in
        'BoundaryCondition.all_combinations' order. 'apply_jacobian' multiplies by
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

        self._weights = np.sqrt(np.concatenate(spectralDensities))
        self._weights *= np.prod(self._alpha) ** (-0.5)
        return IIDCovarianceMatrix(self._weights.size, 1.0)

    def set_sites(self, sites: Grid) -> None:

        self._sites = sites

        if sites is None:
            self._interpMat = None
            self._nativeSites = False
            return

        nG0 = self._q[0] + 2
        axis0 = np.linspace(0., self._alpha[0], nG0)

        if isinstance(sites, UniformGrid):
            if self._d == 1:
                nativeMatch = (
                    len(sites) == nG0
                    and np.isclose(sites.axis[0], 0.)
                    and np.isclose(sites.axis[-1], self._alpha[0])
                )
            else:
                nG1 = self._q[1] + 2
                nativeMatch = (
                    len(sites) == nG0 * nG1
                    and np.isclose(sites.xAxis[0], 0.)
                    and np.isclose(sites.xAxis[-1], self._alpha[0])
                    and np.isclose(sites.yAxis[0], 0.)
                    and np.isclose(sites.yAxis[-1], self._alpha[1])
                )
            if nativeMatch:
                self._interpMat = None
                self._nativeSites = True
                return

        self._nativeSites = False

        if self._d == 1:
            self._interpMat = linear_interpolation_matrix(
                sites.to_array().ravel(), axis0)
        else:
            nG1 = self._q[1] + 2
            axis1 = np.linspace(0., self._alpha[1], nG1)
            self._interpMat = bilinear_interpolation_matrix(
                sites.to_array(), axis0, axis1)

    def at_sites(self, realisation, sites: Grid) -> np.ndarray:
        # Handle both Expansion and Function objects
        expansion = realisation.function if hasattr(realisation, 'function') else realisation
        native = expansion.evaluate_native()
        if self._nativeSites:
            return native
        
        if native.ndim == 1:
            return self._interpMat @ native
        
        # Batch: (nBatch, nGrid) @ (nGrid, nSites)
        return native @ self._interpMat.T

    def apply_jacobian(
            self, v: np.ndarray, _covariance) -> np.ndarray:
        """
        Apply the spectral synthesis Jacobian W and interpolate to sites.

        Computes I_obs @ (scale * sum_b W_b @ v_b) without building spline
        interpolants, since evaluate_native uses only the spectral coefficients.
        """
        scale = 2. ** (-self._d / 2.)
        nG0 = self._q[0] + 2
        q = self._q
        isBatch = v.ndim > 1
        nBatch = v.shape[1] if isBatch else 1

        if self._d == 1:
            interior = np.zeros((nG0, nBatch)) if isBatch else np.zeros(nG0)
            offset = 0
            for bc in BoundaryCondition.all_combinations(1):
                n = bc.block_size(q)
                block = v[offset:offset + n]
                if self._weights is not None:
                    w_block = self._weights[offset:offset + n]
                    if isBatch:
                        block = block * w_block[:, np.newaxis]
                    else:
                        block = block * w_block

                if bc[0] == BC.NEUMANN:
                    interior += cos_series(block, axis=0)
                else:
                    if isBatch:
                        padding = np.zeros((1, nBatch))
                        padded = np.concatenate([padding, block], axis=0)
                        interior += sin_series(padded, axis=0)
                    else:
                        interior += sin_series(np.concatenate([[0.], block]))
                offset += n

            native = scale * interior
            if self._interpMat is None:
                return native if isBatch else native.ravel()

            res = self._interpMat @ native
            return res if isBatch else res.ravel()

        # 2D
        nG1 = self._q[1] + 2
        interior = np.zeros((nG0, nG1, nBatch)) if isBatch else np.zeros((nG0, nG1))
        offset = 0
        for bc in BoundaryCondition.all_combinations(2):
            nX = q[0] + 1 if bc[0] == BC.NEUMANN else q[0]
            nY = q[1] + 1 if bc[1] == BC.NEUMANN else q[1]

            block = v[offset:offset + nX * nY]
            if self._weights is not None:
                w_block = self._weights[offset:offset + nX * nY]
                if isBatch:
                    block = block * w_block[:, np.newaxis]
                else:
                    block = block * w_block

            C = block.reshape(nX, nY, -1) if isBatch else block.reshape(nX, nY)

            A = np.zeros((q[0] + 1, q[1] + 1, nBatch)) if isBatch else np.zeros((q[0] + 1, q[1] + 1))
            r0 = 1 if bc[0] == BC.DIRICHLET else 0
            c0 = 1 if bc[1] == BC.DIRICHLET else 0

            if isBatch:
                A[r0:, c0:, :] = C
            else:
                A[r0:, c0:] = C

            rowFn = sin_series_rows if bc[1] == BC.DIRICHLET else cos_series_rows
            colFn = sin_series_cols if bc[0] == BC.DIRICHLET else cos_series_cols

            interior += colFn(rowFn(A))
            offset += nX * nY

        res = scale * interior.reshape(-1, nBatch) if isBatch else scale * interior.ravel()
        if self._interpMat is None:
            return res

        return self._interpMat @ res

    def apply_adjoint_jacobian(
            self, w: np.ndarray, _covariance) -> np.ndarray:
        """
        Apply the adjoint of the spectral synthesis Jacobian.

        Computes scale * W^T @ I_obs^T @ w for each BC block without
        building spline interpolants, since adjoint_synthesis is spline-free.
        """
        if self._interpMat is None:
            r = w
        else:
            r = self._interpMat.T @ w
        scale = 2. ** (-self._d / 2.)
        q = self._q
        isBatch = w.ndim > 1
        parts = []

        if self._d == 1:
            for bc in BoundaryCondition.all_combinations(1):
                if bc[0] == BC.NEUMANN:
                    parts.append(adj_cos_series_1d(r, q[0], axis=0))
                else:
                    parts.append(adj_sin_series_1d(r, q[0], axis=0))

            res = scale * np.concatenate(parts, axis=0)
            if self._weights is not None:
                if isBatch:
                    return res * self._weights[:, np.newaxis]
                return res * self._weights
            return res

        # 2D
        r_mat = r.reshape(q[0] + 2, q[1] + 2, -1) if isBatch else r.reshape(q[0] + 2, q[1] + 2)
        for bc in BoundaryCondition.all_combinations(2):
            col_adj = adj_cos_series_cols(r_mat, q[0]) if bc[0] == BC.NEUMANN \
                else adj_sin_series_cols(r_mat, q[0])
            a_adj = adj_cos_series_rows(col_adj, q[1]) if bc[1] == BC.NEUMANN \
                else adj_sin_series_rows(col_adj, q[1])
            r0 = 1 if bc[0] == BC.DIRICHLET else 0
            c0 = 1 if bc[1] == BC.DIRICHLET else 0
            nX = q[0] + 1 if bc[0] == BC.NEUMANN else q[0]
            nY = q[1] + 1 if bc[1] == BC.NEUMANN else q[1]

            block = a_adj[r0:r0 + nX, c0:c0 + nY]
            parts.append(block.reshape(-1, block.shape[-1]) if isBatch else block.ravel())

        res = scale * np.concatenate(parts, axis=0)
        if self._weights is not None:
            if isBatch:
                return res * self._weights[:, np.newaxis]
            return res * self._weights
        return res

    def compute_log_length_multiplier(
        self, nu: float, lengthScale: float
    ) -> np.ndarray:
        """ Analytically compute spectral multipliers for log-lengthscale grad. """
        multipliers = []
        alpha, q, d = self._alpha, self._q, self._d
        l2 = lengthScale**2
        factor = 2 * nu + d

        for bc in BoundaryCondition.all_combinations(d):
            if d == 1:
                modes = np.array(range(q[0] + 1) if bc[0] == BC.NEUMANN else range(1, q[0] + 1))
                w2 = (np.pi / alpha[0])**2 * (modes**2)
                mVal = 0.5 * (d - factor * (l2 * w2) / (2 * nu + l2 * w2))
                multipliers.append(mVal)
            else:
                xRange = range(q[0] + 1) if bc[0] == BC.NEUMANN else range(1, q[0] + 1)
                yRange = range(q[1] + 1) if bc[1] == BC.NEUMANN else range(1, q[1] + 1)
                X, Y = np.meshgrid(xRange, yRange, indexing='ij')
                w2 = ((np.pi / alpha[0])**2 * X.ravel()**2
                      + (np.pi / alpha[1])**2 * Y.ravel()**2)
                mVal = 0.5 * (d - factor * (l2 * w2) / (2 * nu + l2 * w2))
                multipliers.append(mVal)

        return np.concatenate(multipliers)

    def evaluate_exact_conditional(self, queryGrid: Grid, state: Expansion, covFcn: CovarianceFunctionInterface) -> np.ndarray:
        """
        Evaluate the exact conditional predictive mean using the dense exact covariance.
        This provides a benchmark fallback avoiding interpolation, though it is O(N^3).
        """
        if not hasattr(self, '_sites') or self._sites is None:
            raise RuntimeError("Cannot compute exact conditional: no observation sites set.")
            
        from scipy.linalg import cholesky, cho_solve, LinAlgError
        from styne.statistics.covariance import DenseCovarianceMatrix

        uObs = self.at_sites(state, self._sites)

        kObs = covFcn.evaluate_covariance(self._sites, self._sites)
        kObsArr = kObs.to_dense() if isinstance(kObs, DenseCovarianceMatrix) else np.asarray(kObs)
        
        # Add a small nugget for numerical stability in the dense exact covariance block
        kObsArr.flat[::kObsArr.shape[0] + 1] += 1e-8

        try:
            L = cholesky(kObsArr, lower=True)
        except LinAlgError:
            # Fallback if numerically unstable
            kObsArr.flat[::kObsArr.shape[0] + 1] += 1e-5
            L = cholesky(kObsArr, lower=True)

        kStar = covFcn.evaluate_covariance(queryGrid, self._sites)
        kStarArr = kStar.to_dense() if isinstance(kStar, DenseCovarianceMatrix) else np.asarray(kStar)

        return kStarArr @ cho_solve((L, True), uObs)

    def create_predictor(
            self, gpState: GPState, queryGrid: Grid,
            coefficient: np.ndarray) -> Predictor:
        if self._d == 1:
            interpMat = linear_interpolation_matrix(
                queryGrid.to_array().ravel(), self.nativeGrid.xAxis
            )
        else:
            interpMat = bilinear_interpolation_matrix(
                queryGrid.to_array(), self.nativeGrid.xAxis, self.nativeGrid.yAxis
            )
        realisation = gpState.parameter.function.clone()
        realisation.coefficient = np.array(
            coefficient, dtype=float, copy=True)
        native = realisation.evaluate_native()
        if native.ndim == 1:
            mean = interpMat @ native
        else:
            mean = native @ interpMat.T
        return DNAGPPredictor(mean)


class DNAGPPredictor(Predictor):
    """Immutable out-of-sample mean snapshot for the DNA GP engine."""

    def __init__(self, mean: np.ndarray):
        self._mean = np.array(mean, dtype=float, copy=True)

    def mean(self) -> np.ndarray:
        """Return an independent copy of the snapshotted mean."""
        return self._mean.copy()
