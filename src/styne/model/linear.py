from styne.model.representation.expansion import backend_constant
from styne.model.forwardmap import ForwardMap
from styne.parameter.parameter import as_coordinate
from styne.parameter.vector import Vector


class LinearForwardMap(ForwardMap):
    """
    Base class for linear models. The model response is the linear predictor,
        eta = X @ beta
    where X is a design (feature) matrix and beta is the parameter vector.

    Parameters
    ----------
    features : array-like, shape (N, p)
        Backend-native design matrix for fixed effects.
    """

    def __init__(self, features):
        super().__init__()

        self._features = as_coordinate(features)

        if not self._features.ndim == 2:
            raise ValueError("features must be a 2D array")

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return self._features.shape[1]

    def _prepare(self, parameter: Vector):
        return parameter.coordinate

    def _evaluate(self, preparedState):
        features = backend_constant(self._features, preparedState)
        return preparedState @ features.T

    def directional_derivative(self, parameter, direction):
        features = backend_constant(
            self._features, direction.coordinate
        )
        return direction.coordinate @ features.T

    def adjoint_derivative(self, parameter, cotangent):
        features = backend_constant(self._features, cotangent)
        return cotangent @ features
