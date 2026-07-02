from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from styne.model.representation.expansion import Expansion
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import CovarianceMatrix
from styne.utility.grid import Grid
from styne.parameter.function import Function
from typing import Protocol

class GPState(Protocol):
    """
    Protocol exposing the dynamically changing state of a Gaussian process.

    Used by predictors to read the current parameter and covariance
    function without depending on the concrete class that provides them.

    Attributes
    ----------
    parameter : Function
        The GP's current realisation.
    covarianceFunction : CovarianceFunctionInterface
        The GP's current covariance function.
    """
    @property
    def parameter(self) -> Function: ...
    @property
    def covarianceFunction(self) -> CovarianceFunctionInterface: ...


class GPEngine(ABC):
    """
    Base class for Gaussian process parametrisation engines.

    The engine defines the mapping between latent coefficients (parameters) and 
    the GP realisation. Implementations must follow the **whitening contract**: 
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

    def at_sites(self, realisation: Expansion, sites: Grid) -> ndarray:
        return realisation.evaluate(sites)

    @property
    @abstractmethod
    def spatialDimension(self) -> int:
        """Number of spatial dimensions of the GP domain."""
        ...

    @abstractmethod
    def build_realisation(self) -> Expansion:
        ...

    @abstractmethod
    def build_covariance(
            self, covFcn: CovarianceFunctionInterface) -> CovarianceMatrix:
        ...

    @abstractmethod
    def apply_jacobian(
            self, v: ndarray, covariance: CovarianceMatrix) -> ndarray:
        ...

    @abstractmethod
    def apply_adjoint_jacobian(
            self, w: ndarray, covariance: CovarianceMatrix) -> ndarray:
        ...

    @abstractmethod
    def create_predictor(self, gpState: GPState, queryGrid: Grid) -> Predictor:
        ...
