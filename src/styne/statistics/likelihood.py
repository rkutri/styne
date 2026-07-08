import numpy as np

from styne.model.model import Model, DifferentiableModel
from styne.model.sglmm import SGLMM
from styne.parameter.parameter import Parameter
from styne.utility.memoisation import EvaluationCache
from styne.statistics.interface import LikelihoodInterface
from styne.statistics.data import Data
from styne.statistics.response import ResponseFamily


class RegressionLikelihood(LikelihoodInterface):
    """
    evaluate_log_gradient is available when the model satisfies the
    DifferentiableModel protocol.

    Parameters
    ----------
    data : Data
        Observed data.
    forwardMap : Model
        Forward model mapping the parameter to the linear predictor.
    noise : ResponseFamily
        Observation response family.
    cacheSize : int, default 5
        Number of recent evaluations cached for `evaluate_log` and
        `evaluate_log_gradient`.
    """

    def __init__(
        self,
        data: Data,
        forwardMap: Model,
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

        self._forwardMap.interpolate(parameter)
        self._forwardMap.evaluate()

        result = self._response.log_likelihood(
            self._data.measurement, self._forwardMap.evaluation)

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

        self._forwardMap.interpolate(parameter)
        self._forwardMap.evaluate()

        result = self._response.score(self._data.measurement, self._forwardMap)
        self._gradientCache.add(parameter, result)
        return result

    def condition_on(self, state: Parameter) -> None:
        """State is ignored; signature maintained for downward cache
        invalidation propagation from RadonNikodym."""
        self._logLikelihoodCache.clear()
        self._gradientCache.clear()
        self._forwardMap.reset()


class SGLMMLikelihood(LikelihoodInterface):
    """
    Likelihood for an SGLMM.

    Parameters
    ----------
    data : Data
        Observed data.
    predictor : SGLMM
        SGLMM forward model.
    response : ResponseFamily
        Observation response family.
    cacheSize : int, default 5
        Number of recent evaluations cached for `evaluate_log` and
        `evaluate_log_gradient`.
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

        self._predictor.interpolate(parameter)
        self._predictor.evaluate()

        result = self._response.log_likelihood(
            self._data.measurement, self._predictor.evaluation
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

        self._predictor.interpolate(parameter)
        self._predictor.evaluate()

        result = self._response.score(self._data.measurement, self._predictor)
        self._gradientCache.add(parameter, result)
        return result

    def condition_on(self, state: Parameter) -> None:
        """State is ignored; signature maintained for downward cache
        invalidation propagation from RadonNikodym."""
        self._logLikelihoodCache.clear()
        self._gradientCache.clear()
        self._predictor.reset()