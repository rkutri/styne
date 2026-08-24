import numpy as np
import pytest

import styne.model.representation as representation
from styne.model.representation.expansion import (
    Expansion,
    GridFunction,
)


class MatrixExpansion(Expansion):

    def __init__(self, designMatrix):
        self._designMatrix = designMatrix

    @property
    def dimension(self):
        return self._designMatrix.shape[1]

    def evaluate(self, coefficient, grid):
        return coefficient @ self._designMatrix.T


def test_expansion_evaluates_multiple_coefficients_without_mutation():
    designMatrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    expansion = MatrixExpansion(designMatrix)
    firstCoefficient = np.array([1.0, -1.0])
    secondCoefficient = np.array([2.0, 3.0])

    firstResult = expansion.evaluate(firstCoefficient, grid=None)
    secondResult = expansion.evaluate(secondCoefficient, grid=None)

    np.testing.assert_array_equal(firstResult, [-1.0, -1.0])
    np.testing.assert_array_equal(secondResult, [8.0, 18.0])
    assert expansion._designMatrix is designMatrix
    assert set(vars(expansion)) == {"_designMatrix"}


def test_expansion_contract_contains_only_static_representation_operations():
    assert set(Expansion.__abstractmethods__) == {"dimension", "evaluate"}
    for removedName in ("coefficient", "project", "validate", "clone"):
        assert removedName not in Expansion.__dict__


def test_grid_function_is_a_structural_protocol():
    assert getattr(GridFunction, "_is_protocol", False)


def test_representation_package_exports_the_function_contracts():
    assert representation.Expansion is Expansion
    assert representation.GridFunction is GridFunction


def test_expansion_requires_explicit_evaluation():

    class IncompleteExpansion(Expansion):

        @property
        def dimension(self):
            return 1

    with pytest.raises(TypeError, match="evaluate"):
        IncompleteExpansion()
