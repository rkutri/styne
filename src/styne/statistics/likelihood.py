import numpy as np

from styne.model.forwardmap import ForwardMap, DifferentiableModel
from styne.model.sglmm import SGLMM
from styne.parameter.parameter import Parameter
from styne.utility.memoisation import EvaluationCache
from styne.statistics.interface import LikelihoodInterface
from styne.statistics.data import Data
from styne.statistics.response import ResponseFamily


class RegressionLikelihood(LikelihoodInterface):
    r"""
    Likelihood L given by

        L(\theta) = \rho(y \mid G(\theta)),

    where \theta is the parameter, y represents a
    generic measurement, G is the forward map, and
    \rho is the response family's density.

    Parameters
    ----------
    data : Data
        Observed data.
    forwardMap : ForwardMap
        Forward model mapping the parameter to the model
        response G(\theta) at the observation sites.
    noise : ResponseFamily
        Observation response family.
    cacheSize : int, default 5
        Number of recent evaluations cached for `evaluate_log` and
        `evaluate_log_gradient`.

    Notes
    -----
    evaluate_log_gradient is available when the model satisfies the
    DifferentiableModel protocol.

    All distributional assumptions about the observations live in the
    response family. For `GaussianResponse`, the noise structure
    (i.i.d., heteroscedastic, or correlated) is carried entirely by
    the covariance matrix.
    """

    def __init__(
        self,
        data: Data,
        forwardMap: ForwardMap,
        noise: ResponseFamily,
        cacheSize: int = 5,
    ):
        self._data = data
        self._forwardMap = forwardMap
        self._response = noise
        self._logLikelihoodCache = EvaluationCache(cacheSize)
        self._gradientCache = EvaluationCache(cacheSize)

    @property
    def domainType(self):
        return self._forwardMap.pType

    @property
    def domainDimension(self):
        return self._forwardMap.pDim

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._forwardMap

    @property
    def response(self) -> ResponseFamily:
        return self._response

    def evaluate_log(self, parameter: Parameter) -> float:
        """
        Log-likelihood at `parameter`, cached by evaluation. Unnormalised,
        inherits that from `ResponseFamily.log_likelihood`, see its docstring.

        Parameters
        ----------
        parameter : Parameter
            Point to evaluate at.

        Returns
        -------
        float
        """

        if self._logLikelihoodCache.contains(parameter):
            return self._logLikelihoodCache.retrieve(parameter)

        evaluation = self._forwardMap(parameter)

        result = self._response.log_likelihood(
            self._data.measurement, evaluation)

        self._logLikelihoodCache.add(parameter, result)

        return result

    def evaluate_log_gradient(self, parameter: Parameter) -> np.ndarray:

        if not isinstance(self._forwardMap, DifferentiableModel):
            raise RuntimeError(
                f"{type(self._forwardMap).__name__} does not satisfy "
                "DifferentiableModel protocol."
            )

        if self._gradientCache.contains(parameter):
            return self._gradientCache.retrieve(parameter)

        evaluation = self._forwardMap(parameter)

        cotangent = self._response.score(
            self._data.measurement, evaluation
        )
        result = self._forwardMap.adjoint_derivative(
            parameter, cotangent
        )
        self._gradientCache.add(parameter, result)
        return result

    def condition_on(self, state: Parameter) -> None:
        """State is ignored; signature maintained for downward cache
        invalidation propagation from RadonNikodym."""
        self._logLikelihoodCache.clear()
        self._gradientCache.clear()


class SGLMMLikelihood(LikelihoodInterface):
    r"""
    Likelihood for an SGLMM, given by

        L(\theta) = \rho(y \mid \eta(\theta)),

    where \theta is the parameter, y represents a
    generic measurement, \eta is the SGLMM linear
    predictor, and \rho is the response family's
    density conditioned on it.

    Notes
    -----
    Functionally equivalent to RegressionLikelihood. Will be deprecated in 0.3.0

    Parameters
    ----------
    data : Data
        Observed data.
    predictor : SGLMM
        SGLMM forward model mapping the parameter to the
        linear predictor \eta(\theta) at the observation sites.
    response : ResponseFamily
        Observation response family.
    cacheSize : int, default 5
        Number of recent evaluations cached for `evaluate_log` and
        `evaluate_log_gradient`.

    Notes
    -----
    evaluate_log_gradient is available when the model satisfies the
    DifferentiableModel protocol.

    Conditional independence of the observations given \eta is built
    into the non-Gaussian response families (`PoissonResponse`,
    `BinomialResponse`) as a sum over observations. For
    `GaussianResponse`, the noise structure is carried by the
    covariance matrix it wraps.
    """

    def __init__(
            self, data: Data, predictor: SGLMM,
            response: ResponseFamily, cacheSize: int = 5):

        self._data = data
        self._predictor = predictor
        self._response = response
        self._logLikelihoodCache = EvaluationCache(cacheSize)
        self._gradientCache = EvaluationCache(cacheSize)

    @property
    def domainType(self):
        return self._predictor.pType

    @property
    def domainDimension(self):
        return self._predictor.pDim

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._predictor

    @property
    def response(self) -> ResponseFamily:
        return self._response

    def evaluate_log(self, parameter: Parameter) -> float:
        """
        Log-likelihood at `parameter`, cached by evaluation. Unnormalised,
        inherits that from `ResponseFamily.log_likelihood`, see its docstring.

        Parameters
        ----------
        parameter : Parameter
            Point to evaluate at.

        Returns
        -------
        float
        """

        if self._logLikelihoodCache.contains(parameter):
            return self._logLikelihoodCache.retrieve(parameter)

        evaluation = self._predictor(parameter)

        result = self._response.log_likelihood(
            self._data.measurement, evaluation
        )

        self._logLikelihoodCache.add(parameter, result)
        return result

    def evaluate_log_gradient(self, parameter: Parameter) -> np.ndarray:

        if not isinstance(self._predictor, DifferentiableModel):
            raise RuntimeError(
                f"{type(self._predictor).__name__} does not satisfy "
                "DifferentiableModel protocol."
            )

        if self._gradientCache.contains(parameter):
            return self._gradientCache.retrieve(parameter)

        evaluation = self._predictor(parameter)

        cotangent = self._response.score(
            self._data.measurement, evaluation
        )
        result = self._predictor.adjoint_derivative(
            parameter, cotangent
        )
        self._gradientCache.add(parameter, result)
        return result

    def condition_on(self, state: Parameter) -> None:
        """State is ignored; signature maintained for downward cache
        invalidation propagation from RadonNikodym."""
        self._logLikelihoodCache.clear()
        self._gradientCache.clear()
