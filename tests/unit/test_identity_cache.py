import numpy as np
import pytest
from styne.utility.memoisation import EvaluationCache
from styne.parameter.vector import Vector


class TestIdentityCache:
    def test_same_object_is_hit(self):
        cache = EvaluationCache(3)
        p = Vector(np.array([1., 2., 3.]))
        cache.add(p, 42.0)
        assert cache.contains(p) is True
        assert cache.retrieve(p) == 42.0

    def test_clone_is_miss(self):
        """A cloned parameter with identical coordinates must miss."""
        cache = EvaluationCache(3)
        p = Vector(np.array([1., 2., 3.]))
        cache.add(p, 42.0)
        q = p.clone()
        assert cache.contains(q) is False

    def test_eviction_removes_oldest(self):
        cache = EvaluationCache(2)
        p1 = Vector(np.array([1.]))
        p2 = Vector(np.array([2.]))
        p3 = Vector(np.array([3.]))
        cache.add(p1, 10.)
        cache.add(p2, 20.)
        cache.add(p3, 30.)
        assert cache.contains(p1) is False
        assert cache.contains(p2) is True
        assert cache.contains(p3) is True

    def test_clear_empties_cache(self):
        cache = EvaluationCache(3)
        p = Vector(np.array([1.]))
        cache.add(p, 10.)
        cache.clear()
        assert cache.contains(p) is False

    def test_cached_object_not_garbage_collected(self):
        """Strong references prevent GC from reusing the id."""
        import gc
        cache = EvaluationCache(3)
        p = Vector(np.array([1., 2.]))
        cache.add(p, 99.)
        pid = id(p)
        assert cache.contains(p)
        gc.collect()
        assert cache.contains(p)

    def test_duplicate_add_is_idempotent(self):
        cache = EvaluationCache(3)
        p = Vector(np.array([1.]))
        cache.add(p, 10.)
        cache.add(p, 20.)
        assert cache.retrieve(p) == 10.


class TestModelInterpolateGuard:
    """Verify Model.interpolate uses identity, not equality."""

    def test_same_object_skips_reinterpolation(self):
        from styne.gp.gaussianprocess import GaussianProcess
        from styne.statistics.stationary import MaternCovariance1D
        from styne.model.sglmm import SGLMM
        from styne.utility.grid import UniformGrid

        grid = UniformGrid(0., 1., 20)
        covFcn = MaternCovariance1D(0.3, 1.5, 1.0)
        gp = GaussianProcess.dna(covFcn, q=5, d=1)
        model = SGLMM(gp, grid)

        param = gp.parameter.clone()
        param.coordinate = np.random.randn(
            param.dimension
        )
        model.interpolate(param)
        model.evaluate()
        eval1 = model.evaluation.copy()

        # Same object, no reset: should skip _interpolate
        model.interpolate(param)
        model.evaluate()
        eval2 = model.evaluation

        np.testing.assert_array_equal(eval1, eval2)

    def test_clone_triggers_reinterpolation(self):
        from styne.gp.gaussianprocess import GaussianProcess
        from styne.statistics.stationary import MaternCovariance1D
        from styne.model.sglmm import SGLMM
        from styne.utility.grid import UniformGrid

        grid = UniformGrid(0., 1., 20)
        covFcn = MaternCovariance1D(0.3, 1.5, 1.0)
        gp = GaussianProcess.dna(covFcn, q=5, d=1)
        model = SGLMM(gp, grid)

        param1 = gp.parameter.clone()
        param1.coordinate = np.random.randn(
            param1.dimension
        )
        model.interpolate(param1)
        model.evaluate()

        param2 = gp.parameter.clone()
        param2.coordinate = np.random.randn(
            param2.dimension
        )
        model.reset()
        model.interpolate(param2)
        model.evaluate()

        assert not np.allclose(
            model.evaluation, 0.
        )
