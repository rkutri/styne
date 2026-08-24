import numpy as np

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
        return self._features @ preparedState

    def directional_derivative(self, parameter: Vector) -> Vector:
        """
        Apply the model's Jacobian to a parameter direction.

        For a linear model the Jacobian is the design matrix itself, constant
        in the parameter, so this is `features @ parameter.coordinate`.

        Parameters
        ----------
        parameter : Vector
            Direction in parameter space.

        Returns
        -------
        Vector
        """

        return parameter.with_coordinate(
            self._features @ parameter.coordinate
        )

    def adjoint_directional_derivative(self, w: np.ndarray) -> np.ndarray:
        """
        Apply the adjoint of the model's Jacobian, `features.T @ w`.

        Parameters
        ----------
        w : np.ndarray
            Vector in observation space.

        Returns
        -------
        np.ndarray
        """
        return self._features.T @ np.asarray(w).ravel()
