import numpy as np

from typing import Optional

from styne.model.model import Model
from styne.model.trend import Trend
from styne.parameter.parameter import Parameter
from styne.parameter.vector import Vector
from styne.parameter.function import Function
from styne.parameter.block import BlockParameter
from styne.gp.gaussianprocess import GaussianProcess
from styne.utility.grid import Grid
from styne.statistics.interface import Predictor


class SGLMM(Model):
    """
    Template class for (Spatial) Generalised Linear Mixed Models.

    The linear predictor is
        eta_i = u_{zeta}(s_i) [+ X_i @ beta] [+ trend(s_i)],
    where u is the latent GP, zeta its parameters, X an optional design matrix
    for parametric fixed effects, and trend an optional deterministic
    additive component (not part of the MCMC parameter).

    Parameter convention
    --------------------
    features=None
        Parameter is Vector — the parameters of the latent field.
    features=X
        Parameter is BlockParameter([zeta, beta]): block 0 is the latent
        field parameters, block 1 the fixed-effect coefficients.

    The trend argument (if given) is evaluated once at construction and
    adds a fixed offset that contributes nothing to any gradient.

    Note
    ----
    A GLMM with uncorrelated (i.i.d.) random effects and no spatial structure
    is currently not supported.
    """


    def __init__(
        self,
        gp: GaussianProcess,
        obsSites: Grid,
        features=None,
        trend: Optional[Trend] = None,
    ):
        super().__init__()

        if not isinstance(obsSites, Grid):
            raise TypeError("obsSites must be a Grid")

        if trend is not None and not isinstance(trend, Trend):
            raise TypeError("trend must implement the Trend protocol")

        self._fixedEffect = None
        self._trendValues = None

        self._obsSites = obsSites

        self._gp = gp
        # If the GP has no sites set, we default to the observation sites.
        # Otherwise, we respect the existing parametrisation (e.g. for Cholesky baselines).
        if self._gp.sites is None:
            self._gp.sites = self._obsSites

        if features is None:
            self._features = None
        else:
            featureArray = np.asarray(features)
            if featureArray.ndim == 1:
                featureArray = featureArray[:, None]

            if featureArray.shape[0] != len(self._obsSites):
                raise ValueError("features must have shape (N, p)")

            self._features = featureArray

        self._trend = trend

        if self._trend is not None:
            self._trendValues = self._trend.evaluate(self._obsSites)



    @property
    def pType(self):
        return BlockParameter if self._features is not None else (Vector, Function)

    @property
    def pDim(self) -> int:

        latentDim = self._gp.parameter.dimension

        if self._features is not None:
            return self._features.shape[1] + latentDim

        return latentDim


    def _interpolate(self, parameter) -> None:

        if self._features is not None:

            self._gp.parameter.coordinate = parameter.block(0).coordinate
            self._fixedEffect = parameter.block(1).coordinate

        else:
            self._gp.parameter.coordinate = parameter.coordinate

    def _evaluate(self) -> None:
        if self._gp.sites is not self._obsSites:
            self._gp.sites = self._obsSites
            
        self._evaluation = self._gp.at_sites()

        if self._features is not None:
            self._evaluation = self._evaluation \
                + self._features @ self._fixedEffect

        if self._trendValues is not None:
            self._evaluation = self._evaluation + self._trendValues

    def directional_derivative(self, parameter: Parameter) -> np.ndarray:
        if self._gp.sites is not self._obsSites:
            self._gp.sites = self._obsSites

        coord = parameter.block(0).coordinate if self._features is not None else parameter.coordinate
        deriv = self._gp.directional_derivative(coord)

        if self._features is not None:
            deriv = deriv + self._features @ parameter.block(1).coordinate
        return deriv

    def adjoint_directional_derivative(self, w: np.ndarray) -> np.ndarray:
        if self._gp.sites is not self._obsSites:
            self._gp.sites = self._obsSites

        w = np.asarray(w).ravel()
        gpAdj = self._gp.adjoint_directional_derivative(w)

        if self._features is None:
            return gpAdj

        return np.concatenate([
            gpAdj,
            self._features.T @ w
        ])

    def create_predictor(self, queryGrid: Grid, features=None) -> 'SGLMMPredictor':
        gpPredictor = self._gp.engine.create_predictor(self._gp, queryGrid)
        return SGLMMPredictor(self, gpPredictor, queryGrid, features)


class SGLMMPredictor(Predictor):
    """
    Out-of-sample predictor for the SGLMM model.
    """
    def __init__(
        self, model: SGLMM, gpPredictor: Predictor, 
        queryGrid: Grid, features=None
    ):
        self._model = model
        self._gpPredictor = gpPredictor
        self._trendValues = model._trend.evaluate(queryGrid) if model._trend else None
        
        if model._features is not None:
            if features is None:
                raise ValueError("Out-of-sample features required for prediction.")
            self._features = np.asarray(features)
            if self._features.ndim == 1:
                self._features = self._features[:, None]
            if len(self._features) != len(queryGrid):
                raise ValueError("features must have shape (len(queryGrid), p)")
        else:
            self._features = None

    def mean(self) -> np.ndarray:
        val = self._gpPredictor.mean()
        if self._trendValues is not None:
            val = val + self._trendValues
        if self._features is not None:
            val = val + self._features @ self._model._fixedEffect
        return val
