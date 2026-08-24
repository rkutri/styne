import styne
import numpy as np

from styne.model import ForwardMap
from styne.model.forwardmap import ForwardMap as DeepForwardMap
from styne.parameter.vector import Vector


class ExplicitForwardMap(ForwardMap):
    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return 1

    def _prepare(self, parameter):
        return parameter.coordinate + 1.0

    def _evaluate(self, preparedState):
        return 2.0 * preparedState


def test_forward_map_public_exports():
    assert styne.ForwardMap is ForwardMap
    assert ForwardMap is DeepForwardMap
    assert styne.__all__ == [
        "enable_logging",
        "__version__",
        "ForwardMap",
        "DifferentiableForwardMap",
        "Parameter",
        "MetropolisHastings",
    ]


def test_prepare_replaces_interpolate():
    assert hasattr(ForwardMap, "prepare")
    assert not hasattr(ForwardMap, "interpolate")


def test_forward_map_uses_explicit_prepared_state():
    model = ExplicitForwardMap()
    firstState = model.prepare(Vector(np.array([1.0])))
    secondState = model.prepare(Vector(np.array([3.0])))

    np.testing.assert_array_equal(model.evaluate(firstState), [4.0])
    np.testing.assert_array_equal(model.evaluate(secondState), [8.0])
    np.testing.assert_array_equal(model(Vector(np.array([2.0]))), [6.0])
    assert not hasattr(model, "evaluation")
    assert not hasattr(model, "reset")
