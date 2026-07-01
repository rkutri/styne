import numpy as np
import pytest
from styne.parameter.vector import Vector
from styne.utility.memoisation import EvaluationCache


def test_hits_and_misses_counters():
    cache = EvaluationCache(3)
    parameter1 = Vector(np.array([1.0]))
    parameter2 = Vector(np.array([2.0]))

    cache.add(parameter1, 2e-4)

    with pytest.raises(RuntimeError):
        cache.retrieve(parameter2)

    assert cache.misses == 1
    assert cache.hits == 0

    value = cache.retrieve(parameter1)
    assert value == 2e-4
    assert cache.hits == 1

