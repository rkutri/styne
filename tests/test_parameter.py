import importlib

import numpy as np
import pytest

import styne.parameter as parameterModule
from styne.model.representation.expansion import Expansion
from styne.parameter import Function, Parameter, Scalar, Vector
from styne.statistics import Data


class ExampleParameter(Parameter):

    def __init__(self, coordinate):
        self._coordinate = coordinate

    @property
    def dimension(self):
        return self._coordinate.size

    @property
    def coordinate(self):
        return self._coordinate

    def with_coordinate(self, coordinate):
        return self.__class__(coordinate)


class MatrixExpansion(Expansion):

    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def dimension(self):
        return self.matrix.shape[1]

    def evaluate(self, coefficient, grid):
        return coefficient @ self.matrix.T


class IdentityExpansion(Expansion):

    def __init__(self, dimension):
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension

    def evaluate(self, coefficient, grid):
        return coefficient


def test_parameter_contract_is_immutable_without_clone():
    parameter = ExampleParameter(np.array([1.0, 2.0]))

    assert Parameter.coordinate.fset is None
    with pytest.raises(AttributeError):
        parameter.coordinate = np.array([3.0, 4.0])

    replacement = parameter.with_coordinate(np.array([3.0, 4.0]))

    assert not hasattr(parameter, "clone")
    np.testing.assert_array_equal(parameter.coordinate, [1.0, 2.0])
    np.testing.assert_array_equal(replacement.coordinate, [3.0, 4.0])


def test_parameter_does_not_define_numerical_equality():
    assert "__eq__" not in Parameter.__dict__
    assert "__hash__" not in Parameter.__dict__


def test_parameter_backend_and_metadata_follow_coordinate():
    parameter = ExampleParameter(np.array([1.0], dtype=np.float32))

    assert parameter.backend.name == "numpy"
    assert parameter.backendMetadata == parameter.backend.metadata(
        parameter.coordinate
    )


def test_with_coordinate_is_required_by_abstract_contract():

    class IncompleteParameter(Parameter):

        @property
        def dimension(self):
            return 1

        @property
        def coordinate(self):
            return np.array([1.0])

    with pytest.raises(TypeError, match="with_coordinate"):
        IncompleteParameter()


@pytest.mark.parametrize(
    ("parameterType", "shape", "dimension"),
    (
        (Vector, (3,), 3),
        (Vector, (2, 3), 3),
        (Scalar, (1,), 1),
        (Scalar, (2, 1), 1),
    ),
)
def test_numeric_parameters_preserve_numpy_arrays(
        parameterType, shape, dimension):
    coordinate = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    parameter = parameterType(coordinate)

    assert parameter.coordinate is coordinate
    assert parameter.coordinate.dtype == np.dtype("float32")
    assert parameter.dimension == dimension
    assert parameter.backend.name == "numpy"


@pytest.mark.parametrize("parameterType", (Vector, Scalar))
def test_numeric_parameters_normalise_scalar_arrays(parameterType):
    coordinate = np.array(2, dtype=np.int16)

    parameter = parameterType(coordinate)

    assert parameter.coordinate.shape == (1,)
    assert parameter.coordinate.dtype == np.dtype("int16")
    assert coordinate.shape == ()


def test_vector_defaults_non_array_input_to_numpy_without_dtype_coercion():
    parameter = Vector([1, 2])

    assert isinstance(parameter.coordinate, np.ndarray)
    assert np.issubdtype(parameter.coordinate.dtype, np.integer)


@pytest.mark.parametrize(
    ("parameterType", "shape"),
    (
        (Vector, (2, 3, 4)),
        (Scalar, (2,)),
        (Scalar, (2, 2)),
        (Scalar, (2, 1, 1)),
    ),
)
def test_numeric_parameters_reject_invalid_shapes(parameterType, shape):
    with pytest.raises(ValueError, match="shape"):
        parameterType(np.zeros(shape))


@pytest.mark.parametrize("parameterType", (Vector, Scalar))
def test_numeric_parameter_reconstruction_is_immutable(parameterType):
    coordinate = np.array([1.0])
    replacementCoordinate = np.array([2.0])
    parameter = parameterType(coordinate)

    replacement = parameter.with_coordinate(replacementCoordinate)

    assert parameterType.coordinate.fset is None
    assert parameter.coordinate is coordinate
    assert replacement.coordinate is replacementCoordinate


@pytest.mark.parametrize("parameterType", (Vector, Scalar))
def test_numpy_numeric_parameter_clone_has_independent_storage(parameterType):
    parameter = parameterType(np.array([1.0], dtype=np.float32))

    clone = parameter.clone()

    assert clone.coordinate.dtype == parameter.coordinate.dtype
    assert not np.shares_memory(clone.coordinate, parameter.coordinate)


def test_jax_numeric_parameters_preserve_graph_and_metadata():
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    coordinate = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    parameter = Vector(coordinate)
    scalar = Scalar(jnp.array(2, dtype=jnp.int16))

    assert parameter.coordinate is coordinate
    assert parameter.dimension == 3
    assert parameter.backend.name == "jax"
    assert parameter.backendMetadata.dtype == jnp.dtype("float32")
    assert scalar.coordinate.shape == (1,)
    assert scalar.coordinate.dtype == jnp.dtype("int16")

    rebuilt = jax.jit(
        lambda value: Vector(value).with_coordinate(value + 1).coordinate
    )(coordinate)
    np.testing.assert_array_equal(rebuilt, np.asarray(coordinate) + 1)


def test_pytorch_numeric_parameters_preserve_graph_and_metadata():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    coordinate = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    coordinate.requires_grad_()

    parameter = Vector(coordinate)
    replacement = parameter.with_coordinate(coordinate * 2)
    clone = parameter.clone()
    scalar = Scalar(torch.tensor(2, dtype=torch.int16))

    assert parameter.coordinate is coordinate
    assert parameter.dimension == 3
    assert parameter.backend.name == "pytorch"
    assert parameter.backendMetadata.dtype == torch.float64
    assert parameter.backendMetadata.device == coordinate.device
    assert clone.coordinate.data_ptr() != coordinate.data_ptr()
    assert scalar.coordinate.shape == (1,)
    assert scalar.coordinate.dtype == torch.int16

    replacement.coordinate.sum().backward()
    torch.testing.assert_close(coordinate.grad, torch.full_like(coordinate, 2))


def test_numeric_wrapper_and_data_type_marker_are_removed():
    assert "Numeric" not in parameterModule.__all__
    assert not hasattr(parameterModule, "Numeric")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("styne.parameter.numeric")

    data = Data(1, np.zeros((1, 1)))
    assert not hasattr(data, "dType")


def test_function_reconstruction_shares_expansion_without_mutation():
    matrix = np.eye(2)
    expansion = MatrixExpansion(matrix)
    coordinate = np.array([1.0, 2.0])
    replacementCoordinate = np.array([3.0, 4.0])
    batchCoordinate = np.array([[5.0, 6.0], [7.0, 8.0]])
    parameter = Function(coordinate, expansion)

    replacement = parameter.with_coordinate(replacementCoordinate)
    batch = parameter.with_coordinate(batchCoordinate)

    assert Function.coordinate.fset is None
    assert parameter.coordinate is coordinate
    assert replacement.coordinate is replacementCoordinate
    assert batch.coordinate is batchCoordinate
    assert parameter.expansion is expansion
    assert replacement.expansion is expansion
    assert batch.expansion is expansion
    assert set(vars(expansion)) == {"matrix"}
    assert expansion.matrix is matrix
    np.testing.assert_array_equal(
        parameter.evaluate(None), coordinate
    )
    np.testing.assert_array_equal(
        replacement.evaluate(None),
        replacementCoordinate,
    )


@pytest.mark.parametrize("shape", ((), (3,), (2, 3)))
def test_function_rejects_invalid_coordinate_shape(shape):
    expansion = IdentityExpansion(2)

    with pytest.raises(ValueError, match="shape"):
        Function(np.zeros(shape), expansion)


def test_function_preserves_arbitrary_leading_batch_dimensions():
    coordinate = np.arange(24.).reshape(2, 3, 4)
    expansion = MatrixExpansion(np.eye(4))

    parameter = Function(coordinate, expansion)

    assert parameter.dimension == 4
    np.testing.assert_array_equal(parameter.evaluate(None), coordinate)


def test_jax_function_preserves_graph_and_shares_expansion():
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    expansion = MatrixExpansion(np.eye(2))
    coordinate = jnp.arange(8., dtype=jnp.float32).reshape(2, 2, 2)
    parameter = Function(coordinate, expansion)

    rebuilt = jax.jit(
        lambda value: parameter.with_coordinate(value + 1).evaluate(None)
    )(coordinate)

    assert parameter.coordinate is coordinate
    assert parameter.expansion is expansion
    assert parameter.backend.name == "jax"
    np.testing.assert_array_equal(parameter.evaluate(None), coordinate)
    np.testing.assert_array_equal(rebuilt, np.asarray(coordinate) + 1)


def test_pytorch_function_preserves_graph_and_shares_expansion():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    expansion = IdentityExpansion(2)
    coordinate = torch.arange(8., requires_grad=True).reshape(2, 2, 2)
    coordinate.retain_grad()
    parameter = Function(coordinate, expansion)
    replacement = parameter.with_coordinate(coordinate * 2)

    assert parameter.coordinate is coordinate
    assert replacement.expansion is expansion
    assert parameter.backend.name == "pytorch"

    evaluated = replacement.evaluate(None)
    torch.testing.assert_close(evaluated, coordinate * 2)

    evaluated.sum().backward()
    torch.testing.assert_close(coordinate.grad, torch.full_like(coordinate, 2))
