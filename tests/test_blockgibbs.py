"""
Tests for GibbsSampler / BlockGibbs.

Covers:
  1. Structural test: chain length and per-block entry count.
  2. Invariant measure test: 2D correlated Gaussian with exact conditional
     draws. Uses rho = 0.8 so a factored test would pass marginals but fail
     the covariance check.
"""

import numpy as np
import pytest

from styne.backend import infer_backend
from styne.mcmc.method.gibbs import BlockGibbs, GibbsBuilder
from styne.statistics.bayes import HierarchicalBayes, HierarchicalBayesModelBuilder
from styne.statistics.measure import ConditionalMeasure
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.parameter.vector import Vector
from styne.parameter.block import BlockParameter


# ---------------------------------------------------------------------------
# Test-local ConditionalMeasure for a 2D correlated Gaussian
#
# Target: p(x0, x1) = N(0, [[1, rho], [rho, 1]])
# Conditionals:
#   p(x0 | x1) = N(rho * x1,  1 - rho^2)
#   p(x1 | x0) = N(rho * x0,  1 - rho^2)
# ---------------------------------------------------------------------------

class CorrelatedGaussianConditional(ConditionalMeasure):
    """Exact Gaussian conditional for one block of a 2D correlated Gaussian."""

    def __init__(self, blockIdx: int, rho: float):
        self._blockIdx = blockIdx
        self._otherIdx = 1 - blockIdx
        self._rho = rho
        condVar = 1.0 - rho ** 2
        self._gaussian = Gaussian(
            IIDCovarianceMatrix(1, condVar),
            Vector(np.zeros(1))
        )

    @property
    def blockDimension(self) -> int:
        return 1

    @property
    def density(self):
        return self._gaussian.density

    def condition_on(self, state) -> None:
        otherVal = state.block(self._otherIdx).coordinate[0]
        self._gaussian.mean = Vector(np.array([self._rho * otherVal]))

    def draw(self, rng):
        return self._gaussian.draw(rng)


class BackendConditional(ConditionalMeasure):
    """Test conditional that samples on the state array's backend."""

    def __init__(self, blockIdx):
        self._blockIdx = blockIdx
        self._state = None

    @property
    def blockDimension(self):
        return 1

    @property
    def density(self):
        raise NotImplementedError

    def condition_on(self, state):
        self._state = state

    def sample(self, randomState):
        coordinate = self._state.block(self._blockIdx).coordinate
        backend = infer_backend(coordinate)
        metadata = backend.metadata(coordinate)
        noise, nextRng = backend.normal(
            randomState, coordinate.shape,
            dtype=metadata.dtype, device=metadata.device,
        )
        return Vector(coordinate + noise), nextRng


def make_model(rho: float) -> HierarchicalBayes:
    """Build a HierarchicalBayes for a 2D correlated Gaussian with correlation rho."""
    root = Gaussian(IIDCovarianceMatrix(1, 1.0), Vector(np.zeros(1)))
    return (
        HierarchicalBayesModelBuilder()
        .set_root(root)
        .add_conditional(CorrelatedGaussianConditional(0, rho))
        .add_conditional(CorrelatedGaussianConditional(1, rho))
        .build()
    )


def make_init() -> BlockParameter:
    return BlockParameter([Vector(np.zeros(1)), Vector(np.zeros(1))])


# ---------------------------------------------------------------------------
# 1. Structural test
# ---------------------------------------------------------------------------

class TestBlockGibbsStructure:

    def setup_method(self):
        np.random.seed(0)
        self.model = make_model(rho=0.5)
        self.sampler = BlockGibbs(self.model)

    def test_chain_length(self):
        N = 50
        self.sampler.run(N, make_init())
        assert self.sampler.chain.length == N + 1

    def test_block_entry_count(self):
        N = 50
        self.sampler.run(N, make_init())
        assert len(self.sampler.chain.block(0).trajectory) == N + 1
        assert len(self.sampler.chain.block(1).trajectory) == N + 1

    def test_last_state_is_block_parameter(self):
        self.sampler.run(10, make_init())
        assert isinstance(self.sampler.lastState, BlockParameter)
        assert self.sampler.lastState.nBlocks == 2

    def test_builder(self):
        builder = GibbsBuilder()
        builder.model = self.model
        sampler = builder.build()
        sampler.run(10, make_init())
        assert sampler.chain.length == 11

    def test_step_does_not_condition_model_templates(self):
        templates = [BackendConditional(0), BackendConditional(1)]

        class Model:
            nBlocks = 2

            @staticmethod
            def conditional(index, state):
                return templates[index].condition(state)

        sampler = BlockGibbs(Model())
        nextState, _, _ = sampler.step(make_init(), np.random.default_rng(3))

        assert all(template._state is None for template in templates)
        assert isinstance(nextState, BlockParameter)

    def test_step_preserves_jax_blocks_and_random_state(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        templates = [BackendConditional(0), BackendConditional(1)]

        class Model:
            nBlocks = 2

            @staticmethod
            def conditional(index, state):
                return templates[index].condition(state)

        sampler = BlockGibbs(Model())
        initialState = BlockParameter([
            Vector(jnp.zeros(1)), Vector(jnp.zeros(1)),
        ])

        nextState, _, nextRng = sampler.step(initialState, jax.random.key(2))

        assert isinstance(nextState.block(0).coordinate, jax.Array)
        assert isinstance(nextState.block(1).coordinate, jax.Array)
        assert isinstance(nextRng, jax.Array)

    def test_step_preserves_pytorch_blocks(self):
        torch = pytest.importorskip("torch")
        templates = [BackendConditional(0), BackendConditional(1)]

        class Model:
            nBlocks = 2

            @staticmethod
            def conditional(index, state):
                return templates[index].condition(state)

        sampler = BlockGibbs(Model())
        initialState = BlockParameter([
            Vector(torch.zeros(1)), Vector(torch.zeros(1)),
        ])

        nextState, _, _ = sampler.step(
            initialState, torch.Generator().manual_seed(2)
        )

        assert isinstance(nextState.block(0).coordinate, torch.Tensor)
        assert isinstance(nextState.block(1).coordinate, torch.Tensor)


# ---------------------------------------------------------------------------
# 2. Invariant measure test — 2D correlated Gaussian, rho = 0.8
# ---------------------------------------------------------------------------

class TestBlockGibbsInvariantMeasure:

    RHO = 0.8
    N_STEPS = 12000
    BURNIN = 2000
    SEED = 42

    def setup_method(self):
        np.random.seed(self.SEED)
        model = make_model(rho=self.RHO)
        sampler = BlockGibbs(model)
        sampler.run(self.N_STEPS, make_init())

        traj0 = np.array(sampler.chain.block(0).trajectory)[self.BURNIN:, 0]
        traj1 = np.array(sampler.chain.block(1).trajectory)[self.BURNIN:, 0]
        self.samples = np.column_stack([traj0, traj1])

    def test_mean_near_zero(self):
        mean = self.samples.mean(axis=0)
        assert np.abs(mean[0]) < 0.05
        assert np.abs(mean[1]) < 0.05

    def test_marginal_variance(self):
        var = self.samples.var(axis=0)
        assert abs(var[0] - 1.0) < 0.1
        assert abs(var[1] - 1.0) < 0.1

    def test_cross_covariance(self):
        cov = np.cov(self.samples, rowvar=False)
        assert abs(cov[0, 1] - self.RHO) < 0.1
