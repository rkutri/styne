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


class TestForwardMapExplicitState:
    def test_distinct_prepared_states_do_not_interfere(self):
        from styne.gp.gaussianprocess import GaussianProcess
        from styne.statistics.stationary import MaternCovariance1D
        from styne.model.sglmm import SGLMM
        from styne.utility.grid import UniformGrid

        grid = UniformGrid(0., 1., 20)
        covFcn = MaternCovariance1D(0.3, 1.5, 1.0)
        gp = GaussianProcess.dna(covFcn, q=5, d=1)
        model = SGLMM(gp, grid)

        rng = np.random.default_rng(20240720)
        firstParameter = gp.parameter.with_coordinate(
            rng.standard_normal(gp.parameter.dimension)
        )
        secondParameter = gp.parameter.with_coordinate(
            rng.standard_normal(gp.parameter.dimension)
        )

        firstState = model.prepare(firstParameter)
        secondState = model.prepare(secondParameter)

        firstEvaluation = model.evaluate(firstState)
        secondEvaluation = model.evaluate(secondState)

        assert not np.allclose(firstEvaluation, secondEvaluation)
        np.testing.assert_allclose(firstEvaluation, model.evaluate(firstState))

    def test_gradient_is_stable_across_interleaved_evaluation(self):
        from styne.gp.gaussianprocess import GaussianProcess
        from styne.statistics.stationary import MaternCovariance2D
        from styne.model.sglmm import SGLMM
        from styne.statistics.response import PoissonResponse
        from styne.statistics.likelihood import SGLMMLikelihood
        from styne.statistics import Data
        from styne.utility.grid import Grid

        covariance = MaternCovariance2D(0.3, 1.5, 1.0)
        gp = GaussianProcess.dna(covariance, q=2, d=2)
        sites = Grid(np.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]]))
        model = SGLMM(gp, sites)

        data = Data(1, sites.to_array())
        data.measurement = np.array([[1.0], [0.0], [3.0]])
        likelihood = SGLMMLikelihood(data, model, PoissonResponse())

        target = Vector(np.linspace(-0.3, 0.5, gp.parameterDimension))
        interleaved = Vector(np.linspace(0.4, -0.6, gp.parameterDimension))
        targetAgain = Vector(np.linspace(-0.3, 0.5, gp.parameterDimension))

        isolatedGradient = likelihood.evaluate_log_gradient(target)
        likelihood.evaluate_log(interleaved)
        recomputedGradient = likelihood.evaluate_log_gradient(targetAgain)

        np.testing.assert_allclose(
            isolatedGradient, recomputedGradient, rtol=0.0, atol=1e-12
        )
