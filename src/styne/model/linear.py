import numpy as np

from styne.model.representation.expansion import backend_constant
from styne.model.forwardmap import ForwardMap
from styne.parameter.vector import Vector


class LinearForwardMap(ForwardMap):
    """
    Base class for linear models. The model response is the linear predictor,
        eta = X @ beta
    where X is a design (feature) matrix and beta is the parameter vector.

    Parameters
    ----------
    features : ndarray, shape (N, p)
        Design matrix for fixed effects.
    """

    def __init__(self, features: np.ndarray):
        
        super().__init__()

        self._features = np.asarray(features)

        if not self._features.ndim == 2:
            raise ValueError("features must be a 2D array")

    @property
    def pType(self):
        return Vector
    
    @property
    def pDim(self):
        return self._features.shape[1]
    
    def _prepare(self, parameter: Vector) -> np.ndarray:
        return parameter.coordinate

    def _evaluate(self, preparedState: np.ndarray) -> np.ndarray:
        features = backend_constant(self._features, preparedState)
        return preparedState @ features.T

    def directional_derivative(
            self, parameter: Vector, direction: Vector) -> Vector:
        """
        Apply the model's Jacobian to a parameter direction.

        For a linear model the Jacobian is the design matrix itself, constant
        in the parameter, so this is `features @ direction.coordinate`.

        Parameters
        ----------
        parameter : Vector
            Point in parameter space. It does not affect this linear map.
        direction : Vector
            Direction in parameter space.

        Returns
        -------
        Vector
        """

        features = backend_constant(
            self._features, direction.coordinate
        )
        return direction.with_coordinate(
            direction.coordinate @ features.T
        )

    def adjoint_derivative(
            self, parameter: Vector, cotangent: np.ndarray) -> np.ndarray:
        """
        Apply the adjoint of the model's Jacobian.

        Parameters
        ----------
        parameter : Vector
            Point in parameter space. It does not affect this linear map.
        cotangent : np.ndarray
            Cotangent in observation space.

        Returns
        -------
        np.ndarray
        """
        features = backend_constant(self._features, cotangent)
        return cotangent @ features
