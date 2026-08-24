from typing import Optional

from styne.model.forwardmap import ForwardMap
from styne.model.trend import Trend
from styne.parameter.parameter import as_coordinate
from styne.parameter.vector import Vector
from styne.parameter.block import BlockParameter
from styne.model.representation.expansion import backend_constant
from styne.gp.gaussianprocess import GaussianProcess
from styne.utility.grid import Grid


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
            featureArray = as_coordinate(features)
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
        return BlockParameter if self._features is not None else Vector

    @property
    def pDim(self) -> int:
        latentDim = self._gp.parameterDimension

        if self._features is not None:
            return self._features.shape[1] + latentDim

        return latentDim

    def _prepare(self, parameter):
        if self._features is not None:
            if parameter.nBlocks != 2:
                raise ValueError(
                    "SGLMM fixed effects require latent and fixed blocks."
                )
            latent, fixedEffect = parameter.block(0), parameter.block(1)
            if not isinstance(latent, Vector) or not isinstance(
                    fixedEffect, Vector):
                raise TypeError("SGLMM blocks must be Vector parameters.")
            if latent.dimension != self._gp.parameterDimension:
                raise ValueError("Latent block has the wrong dimension.")
            if fixedEffect.dimension != self._features.shape[1]:
                raise ValueError("Fixed-effect block has the wrong dimension.")
            return latent.coordinate, fixedEffect.coordinate

        if parameter.dimension != self._gp.parameterDimension:
            raise ValueError("Latent parameter has the wrong dimension.")
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
            evaluation = evaluation + backend_constant(
                self._trendValues, evaluation
            )

        return evaluation

    def directional_derivative(self, parameter, direction):
        latentCoordinate, _ = self._prepare(parameter)
        directionCoordinate, _ = self._prepare(direction)
        derivative = self._gp.directional_derivative(
            latentCoordinate, directionCoordinate, self._obsSites
        )

        if self._features is not None:
            fixedDirection = direction.block(1).coordinate
            features = backend_constant(self._features, fixedDirection)
            derivative = derivative + fixedDirection @ features.T
        return derivative

    def adjoint_derivative(self, parameter, cotangent):
        latentCoordinate, _ = self._prepare(parameter)
        latentAdjoint = self._gp.adjoint_derivative(
            latentCoordinate, cotangent, self._obsSites
        )

        if self._features is None:
            return latentAdjoint

        features = backend_constant(self._features, cotangent)
        fixedAdjoint = cotangent @ features
        return parameter.backend.namespace.concatenate(
            (latentAdjoint, fixedAdjoint), axis=-1
        )

    def predict(self, preparedState, queryGrid: Grid, features=None):
        """
        Evaluate the linear predictor at new sites.

        Parameters
        ----------
        preparedState
            State returned by 'prepare' for the parameter to predict.
        queryGrid : Grid
            Sites to predict at.
        features : array-like, optional
            Design matrix at the query sites, required if the model was
            constructed with fixed effects.

        Returns
        -------
        array-like
            Backend-native linear-predictor values at ``queryGrid``.
        """
        latentCoordinate, fixedEffect = preparedState
        mean = self._gp.evaluate(latentCoordinate, queryGrid)

        if self._trend is not None:
            mean = mean + backend_constant(
                self._trend.evaluate(queryGrid), mean
            )

        if fixedEffect is not None:
            if features is None:
                raise ValueError(
                    "Out-of-sample features required for prediction."
                )
            featureArray = as_coordinate(features)
            if featureArray.ndim == 1:
                featureArray = featureArray[:, None]
            if (
                    len(featureArray) != len(queryGrid)
                    or featureArray.shape[1] != self._features.shape[1]):
                raise ValueError(
                    "features must have shape (len(queryGrid), p)."
                )
            featureArray = backend_constant(featureArray, fixedEffect)
            mean = mean + fixedEffect @ featureArray.T

        return mean
