from __future__ import annotations

from abc import ABC, abstractmethod

import copy

from numpy import ndarray, asarray

from styne.parameter.parameter import Parameter
from styne.utility.grid import Grid


class GridFunctionInterface(ABC):

    @abstractmethod
    def evaluate(self, grid: Grid) -> Grid:
        pass


class Expansion(GridFunctionInterface):

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def coefficient(self) -> ndarray:
        pass

    @coefficient.setter
    @abstractmethod
    def coefficient(self, coefficient: ndarray) -> None:
        pass

    @abstractmethod
    def project(self, coefficient: ndarray) -> None:
        pass

    @staticmethod
    def validate(coefficient: ndarray) -> ndarray:
        coefficient = asarray(coefficient, dtype=float)
        if coefficient.ndim not in [1, 2]:
            raise ValueError("coefficient must be vector or (nBatch, nParam) matrix")
        return coefficient

    def clone(self) -> Expansion:
        return copy.deepcopy(self)
