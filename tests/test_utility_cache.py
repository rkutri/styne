import pytest

from numpy import array
from styne.utility.memoisation import EvaluationCache


class MockParameterInterface:

    def __init__(self, coeff):
        self.coeff = coeff

    def __eq__(self, other):
        return self.coeff == other.coeff


@pytest.fixture
def cache():
    return EvaluationCache(3)


@pytest.mark.skip()
def test_cache_add_and_contains(cache):

    param1 = MockParameterInterface(1)
    param2 = MockParameterInterface(2)

    cache.add(param1, 1.5)
    cache.add(param2, -3.226)

    assert len(cache.keys) == 2
    assert len(cache._cache) == 2

    assert cache.contains(param1) is True
    assert cache.contains(param2) is True
    assert cache.contains(MockParameterInterface(3)) is False


@pytest.mark.skip()
def test_cache_misses(cache):

    param1 = MockParameterInterface(1)
    cache.add(param1, 2e-4)

    assert cache.contains(MockParameterInterface(2)) is False
    assert cache.misses == 1


@pytest.mark.skip()
def test_cache_eviction(cache):

    param1 = MockParameterInterface(1)
    param2 = MockParameterInterface(2)
    param3 = MockParameterInterface(3)
    param4 = MockParameterInterface(4)

    cache.add(param1, 0.1)
    cache.add(param2, 0.2)
    cache.add(param3, 0.3)
    cache.add(param4, 0.4)

    assert cache.contains(param1) is False
    assert cache.contains(param2) is True
    assert cache.contains(param3) is True
    assert cache.contains(param4) is True


@pytest.mark.skip()
def test_cache_call(cache):

    param1 = MockParameterInterface(1)
    cache.add(param1, "mockValue")

    assert cache(param1) == "mockValue"


if __name__ == '__main__':
    pytest.main()
