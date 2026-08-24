from abc import ABC, abstractmethod

from numpy import ndarray

from styne.model.representation.expansion import Expansion
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import CovarianceMatrix
from styne.utility.grid import Grid
from styne.parameter.function import Function
from typing import Protocol

class GPState(Protocol):
    """
    Gaussian-process configuration available while building a predictor.

    Engines may inspect the basis and covariance at construction time, but
    the returned predictor must not retain this live state.
    """
    @property
    def parameter(self) -> Function: ...
    @property
    def covarianceFunction(self) -> CovarianceFunctionInterface: ...


class GPEngine(ABC):
    """
    Base class for Gaussian process parametrisation engines.

    The engine defines the mapping between latent coefficients (parameters) and 
    the GP function. Implementations must follow the **whitening contract**:
    the latent parameters are assumed to be independent white noise vectors 
    sampled from a standard normal distribution. The engine is responsible for 
    applying the square root of the covariance operator (e.g. Cholesky factor 
    or spectral square root) during forward maps, adjoint maps, and evaluations.
    """

    @abstractmethod
    def set_sites(self, sites: Grid) -> None:
        ...

    def requires_covariance_rebuild(self) -> bool:
        return False

    def evaluate(
            self, coefficient: ndarray,
            covariance: CovarianceMatrix) -> ndarray:
        """Synthesise values from an explicit whitened coefficient.

        All engine operations use a trailing feature axis. Coefficients have
        shape ``(..., parameter_dimension)`` and the result has shape
        ``(..., site_count)``.
        """
        return self.apply_jacobian(coefficient, covariance)

    @property
    @abstractmethod
    def spatialDimension(self) -> int:
        """Number of spatial dimensions of the GP domain."""
        ...

    @abstractmethod
    def build_expansion(self) -> Expansion:
        ...

    @abstractmethod
    def build_covariance(
            self, covFcn: CovarianceFunctionInterface) -> CovarianceMatrix:
        ...

    @abstractmethod
    def apply_jacobian(
            self, vector: ndarray,
            covariance: CovarianceMatrix) -> ndarray:
        """Map ``(..., parameter_dimension)`` to ``(..., site_count)``."""
        ...

    @abstractmethod
    def apply_adjoint_jacobian(
            self, cotangent: ndarray,
            covariance: CovarianceMatrix) -> ndarray:
        """Map ``(..., site_count)`` to ``(..., parameter_dimension)``."""
        ...

    @abstractmethod
    def create_predictor(
            self, gpState: GPState, queryGrid: Grid,
            coefficient: ndarray) -> Predictor:
        """Build an immutable predictor snapshot.

        The returned predictor owns all inputs needed for prediction. Later
        mutation of the GP, coefficient, covariance, features, or arrays
        returned by the predictor cannot change its predictions.
        """
        ...
