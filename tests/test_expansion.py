import numpy as np
import pytest

import styne.model.representation as representation
from styne.model.representation.expansion import (
    BoundExpansion,
    BoundLinearExpansion,
    Expansion,
    GridFunction,
    LinearExpansion,
    backend_constant,
)
from styne.backend import BackendCapabilityError
from styne.parameter.function import Function


class MatrixEvaluation(BoundLinearExpansion):

    def __init__(self, designMatrix):
        self._designMatrix = designMatrix

    @property
    def dimension(self):
        return self._designMatrix.shape[1]

    def evaluate(self, coefficient):
        matrix = backend_constant(self._designMatrix, coefficient)
        return coefficient @ matrix.T

    def _adjoint_derivative(self, coefficient, cotangent):
        return cotangent @ self._designMatrix


class MatrixExpansion(LinearExpansion):

    def __init__(self, designMatrix):
        self._designMatrix = designMatrix

    @property
    def dimension(self):
        return self._designMatrix.shape[1]

    def _bind(self, grid):
        return MatrixEvaluation(self._designMatrix)


class SquareEvaluation(BoundExpansion):

    def __init__(self, dimension):
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension

    def evaluate(self, coefficient):
        return coefficient**2


class SquareExpansion(Expansion):

    def __init__(self, dimension):
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension

    def _bind(self, grid):
        return SquareEvaluation(self._dimension)


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
    assert set(vars(expansion)) == {
        "_designMatrix", "_boundEvaluations"
    }


def test_expansion_reuses_grid_binding():
    expansion = MatrixExpansion(np.eye(2))

    assert expansion.bind(None) is expansion.bind(None)


def test_expansion_contract_contains_only_static_representation_operations():
    assert set(Expansion.__abstractmethods__) == {"dimension", "_bind"}
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

    with pytest.raises(TypeError, match="_bind"):
        IncompleteExpansion()


def test_nonlinear_expansion_does_not_claim_numpy_derivatives():
    function = Function(np.array([2.0, -3.0]), SquareExpansion(2))

    np.testing.assert_array_equal(function.evaluate(None), [4.0, 9.0])
    with pytest.raises(BackendCapabilityError):
        function.directional_derivative(np.ones(2), None)
    with pytest.raises(BackendCapabilityError):
        function.adjoint_derivative(np.ones(2), None)


def test_jax_automatically_differentiates_nonlinear_expansion():
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    expansion = SquareExpansion(2)

    directional = jax.jit(lambda coefficient, direction: Function(
        coefficient, expansion
    ).directional_derivative(direction, None))
    adjoint = jax.jit(lambda coefficient, cotangent: Function(
        coefficient, expansion
    ).adjoint_derivative(cotangent, None))

    coefficient = jnp.array([2.0, -3.0])
    direction = jnp.array([0.5, 2.0])
    cotangent = jnp.array([4.0, -1.0])
    np.testing.assert_allclose(
        directional(coefficient, direction), 2 * coefficient * direction
    )
    np.testing.assert_allclose(
        adjoint(coefficient, cotangent), 2 * coefficient * cotangent
    )


def test_pytorch_automatically_differentiates_nonlinear_expansion():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    expansion = SquareExpansion(2)
    coefficient = torch.tensor([2.0, -3.0], requires_grad=True)
    direction = torch.tensor([0.5, 2.0])
    cotangent = torch.tensor([4.0, -1.0])
    function = Function(coefficient, expansion)

    torch.testing.assert_close(
        function.directional_derivative(direction, None),
        2 * coefficient * direction,
    )
    torch.testing.assert_close(
        function.adjoint_derivative(cotangent, None),
        2 * coefficient * cotangent,
    )
