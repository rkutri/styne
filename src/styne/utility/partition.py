import numpy as np

from typing import List
from numpy import ndarray

from styne.parameter.parameter import Parameter
from styne.statistics.interface import DensityInterface
from styne.utility.densityarithmetic import ProductWrapper


from abc import ABC, abstractmethod

class PartitionRule(ABC):
    """
    Template pattern for parameter partitioning rules.
    """

    @property
    @abstractmethod
    def numComponents(self) -> int:
        ... 

    @abstractmethod
    def indices(self, idx: int) -> ndarray:
        """
        Return the coordinate indices belonging to partition component idx.
        """
        ... 

    def component_dimension(self, idx: int) -> int:
        return len(self.indices(idx))

    def total_dimension(self) -> int:
        return sum(self.component_dimension(i) for i in range(self.numComponents))

    def extract(self, idx: int, coordinate: ndarray) -> ndarray:
        return coordinate[self.indices(idx)]

    def merge(self, components: List[ndarray]) -> ndarray:
        if len(components) != self.numComponents:
            raise ValueError(
                f"Expected {self.numComponents} components, got {len(components)}."
            )
        result = np.empty(self.total_dimension())
        for i, comp in enumerate(components):
            result[self.indices(i)] = comp
        return result


class Partition:
    """
    Class representing the partitioning of a global parameter,
    into a direct sum of component parameters.

    Internally stores a reference to a single global parameter and manages read-only
    access to the components of this parameter according to the partitionRule.
    """

    def __init__(self, partitionRule: PartitionRule, parameter: Parameter):

        if partitionRule.total_dimension() != parameter.dimension:
            raise ValueError("Partition rule total dimension must match parameter dimension.")

        self._globalParameter = parameter
        self._rule = partitionRule

    @property
    def rule(self) -> PartitionRule:
        return self._rule

    @property
    def parameter(self) -> Parameter:
        return self._globalParameter
    
    @parameter.setter
    def parameter(self, newParameter: Parameter) -> None:

        if newParameter.dimension != self._rule.total_dimension():
            raise ValueError("New parameter dimension must match partition rule total dimension.")

        self._globalParameter = newParameter

    def component(self, idx: int) -> Parameter:
        from styne.parameter.vector import Vector
        return Vector(self._rule.extract(idx, self._globalParameter.coordinate))

    def global_domain_type(self) -> Parameter:
        return type(self._globalParameter)
    
    def global_dimension(self) -> int:
        return self._globalParameter.dimension


class IndependentPartitionDensity(DensityInterface):
    """
    Density based on a partition of the underlying parameter
    into two or more statistically independent components.
    """

    def __init__(self, partition: Partition, componentDensities: List[DensityInterface]):

        if partition.rule.numComponents != len(componentDensities):
            raise ValueError("Number of components in partition must match number of densities.")

        for i, dens in enumerate(componentDensities):
            if dens.domainDimension != partition.rule.component_dimension(i):
                raise ValueError(f"Density dimension {dens.domainDimension} at index {i} "
                                 f"does not match partition component dimension "
                                 f"{partition.rule.component_dimension(i)}.")

        self._partition = partition
        self._densities = componentDensities

    @property
    def domainType(self) -> Parameter:
        return self._partition.global_domain_type()

    @property
    def parameter(self) -> Parameter:
        """Parameter template retaining the partition's static metadata."""
        return self._partition.parameter

    @property
    def domainDimension(self) -> int:
        return self._partition.global_dimension()

    def evaluate_log(self, state: Parameter) -> float:

        self._partition.parameter = state

        stateEval = 0.
        for i, dens in enumerate(self._densities):

            componentParameter = self._partition.component(i)
            stateEval += dens.evaluate_log(componentParameter)

        return stateEval
    
    def evaluate_log_gradient(self, state: Parameter):

        self._partition.parameter = state

        gradients = []
        for i, dens in enumerate(self._densities):
            if not callable(getattr(dens, "evaluate_log_gradient", None)):
                raise RuntimeError(
                    f"Component density at index {i} must expose "
                    "evaluate_log_gradient."
                )
            
            componentParameter = self._partition.component(i)
            gradients.append(dens.evaluate_log_gradient(componentParameter))

        return self._partition.rule.merge(gradients)
