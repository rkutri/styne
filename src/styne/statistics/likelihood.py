from numpy import ndarray

from styne.model.forwardmap import ForwardMap, DifferentiableModel
from styne.parameter.parameter import Parameter
from styne.statistics.interface import LikelihoodInterface
from styne.statistics.data import Data
from styne.statistics.response import ResponseFamily


class RegressionLikelihood(LikelihoodInterface):
    """Pure likelihood composition of data, a forward map, and a response."""

    def __init__(self, data: Data, forwardMap: ForwardMap, noise: ResponseFamily):
        self._data = data
        self._forwardMap = forwardMap
        self._response = noise

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

    def evaluate_log(self, parameter: Parameter):
        evaluation = self._forwardMap(parameter)
        return self._response.log_likelihood(self._data.measurement, evaluation)

    def evaluate_log_gradient(self, parameter: Parameter) -> ndarray:
        """Legacy explicit gradient; differentiate ``evaluate_log`` instead."""
        if not isinstance(self._forwardMap, DifferentiableModel):
            raise RuntimeError(
                f"{type(self._forwardMap).__name__} does not provide an adjoint."
            )
        evaluation = self._forwardMap(parameter)
        return self._forwardMap.adjoint_derivative(
            parameter, self._response.score(self._data.measurement, evaluation)
        )


class SGLMMLikelihood(RegressionLikelihood):
    """Compatibility name for the unified :class:`RegressionLikelihood`."""

    def __init__(self, data: Data, predictor: ForwardMap, response: ResponseFamily):
        super().__init__(data, predictor, response)
