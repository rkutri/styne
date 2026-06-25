import numpy as np

from styne.model.model import Model
from styne.parameter.vector import Vector


class LinearModel(Model):
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

        self._beta = None


    @property
    def pType(self):
        return Vector
    
    @property
    def pDim(self):
        return self._features.shape[1]
    
    def _interpolate(self, parameter: Vector) -> None:
        self._beta = parameter.coordinate

    def _evaluate(self) -> None:
        self._evaluation = self._features @ self._beta

    def directional_derivative(self, parameter: Vector) -> Vector:

        v = parameter.clone()
        v.coordinate = self._features @ parameter.coordinate
        return v

    def adjoint_directional_derivative(self, w: np.ndarray) -> np.ndarray:
        return self._features.T @ np.asarray(w).ravel()

        