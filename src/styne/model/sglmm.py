import numpy as np

from typing import Optional

from styne.model.forwardmap import ForwardMap
from styne.model.trend import Trend
from styne.parameter.parameter import Parameter
from styne.parameter.vector import Vector
from styne.parameter.function import Function
from styne.parameter.block import BlockParameter
from styne.model.representation.expansion import backend_constant
from styne.gp.gaussianprocess import GaussianProcess
from styne.utility.grid import Grid
from styne.statistics.interface import Predictor


class SGLMM(ForwardMap):
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

        self._trendValues = None

        self._obsSites = obsSites

        self._gp = gp

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


    def _prepare(self, parameter):
        if self._features is not None:
            return parameter.block(0).coordinate, parameter.block(1).coordinate

        return parameter.coordinate, None

    def _evaluate(self, preparedState):
        latentCoordinate, fixedEffect = preparedState
        evaluation = self._gp.evaluate(
            latentCoordinate, self._obsSites
        )

        if self._features is not None:
            features = backend_constant(self._features, fixedEffect)
            evaluation = evaluation + fixedEffect @ features.T

        if self._trendValues is not None:
            evaluation = evaluation + self._trendValues

        return evaluation

    def directional_derivative(
            self, parameter: Parameter,
            direction: Parameter) -> np.ndarray:
        """
        Apply the model's Jacobian to a parameter direction.

        Routes the latent-field block through the GP's own
        `directional_derivative` and, when `features` is set, adds the
        fixed-effect block's contribution via the design matrix directly.

        Parameters
        ----------
        parameter : Parameter
            Point in parameter space.
        direction : Parameter
            Direction in parameter space, `Vector` or `BlockParameter`
            depending on whether `features` was set at construction.

        Returns
        -------
        np.ndarray
        """
        latentCoordinate, _ = self._prepare(parameter)
        directionCoordinate, _ = self._prepare(direction)
        deriv = self._gp.directional_derivative(
            latentCoordinate, directionCoordinate, self._obsSites
        )

        if self._features is not None:
            fixedDirection = direction.block(1).coordinate
            features = backend_constant(self._features, fixedDirection)
            deriv = deriv + fixedDirection @ features.T
        return deriv

    def adjoint_derivative(
            self, parameter: Parameter,
            cotangent: np.ndarray) -> np.ndarray:
        """
        Apply the adjoint of the model's Jacobian to a cotangent.

        Splits the same way as `directional_derivative`, adjoint GP action for
        the latent block, concatenated with `features.T @ w` for the
        fixed-effect block when present.

        Parameters
        ----------
        parameter : Parameter
            Point in parameter space.
        cotangent : np.ndarray
            Cotangent in observation space.

        Returns
        -------
        np.ndarray
            `Vector`-shaped if no fixed effects, otherwise concatenated with
            the fixed-effect block's adjoint contribution.
        """
        latentCoordinate, _ = self._prepare(parameter)
        gpAdj = self._gp.adjoint_derivative(
            latentCoordinate, cotangent, self._obsSites
        )

        if self._features is None:
            return gpAdj

        features = backend_constant(self._features, cotangent)
        fixedAdjoint = cotangent @ features
        backend = parameter.backend
        return backend.namespace.concatenate(
            (gpAdj, fixedAdjoint), axis=-1
        )

    def create_predictor(
            self, preparedState, queryGrid: Grid, features=None
    ) -> 'SGLMMPredictor':
        """
        Build an out-of-sample predictor at new sites.

        Parameters
        ----------
        preparedState
            State returned by 'prepare' for the parameter to predict.
        queryGrid : Grid
            Sites to predict at.
        features : np.ndarray, optional
            Design matrix at the query sites, required if the model was
            constructed with fixed effects.

        Returns
        -------
        SGLMMPredictor
        """
        latentCoordinate, fixedEffect = preparedState
        gpPredictor = self._gp.create_predictor(
            latentCoordinate, queryGrid, self._obsSites
        )
        return SGLMMPredictor(gpPredictor, queryGrid, features, fixedEffect, self._trend)


class SGLMMPredictor(Predictor):
    """
    Immutable out-of-sample mean snapshot for the SGLMM model.
    """
    def __init__(
        self, gpPredictor: Predictor, queryGrid: Grid, features, fixedEffect, trend
    ):
        mean = np.array(gpPredictor.mean(), dtype=float, copy=True)

        if trend is not None:
            mean = mean + np.array(
                trend.evaluate(queryGrid), dtype=float, copy=True)

        if fixedEffect is not None:
            if features is None:
                raise ValueError("Out-of-sample features required for prediction.")
            frozenFeatures = np.array(features, dtype=float, copy=True)
            if frozenFeatures.ndim == 1:
                frozenFeatures = frozenFeatures[:, None]
            if len(frozenFeatures) != len(queryGrid):
                raise ValueError("features must have shape (len(queryGrid), p)")
            frozenFixedEffect = np.array(
                fixedEffect, dtype=float, copy=True)
            mean = mean + frozenFeatures @ frozenFixedEffect

        self._mean = np.array(mean, dtype=float, copy=True)

    def mean(self) -> np.ndarray:
        """
        Predictive mean at the query sites.

        Returns
        -------
        np.ndarray
        """
        return self._mean.copy()
