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


def _make_model(rho: float) -> HierarchicalBayes:
    """Build a HierarchicalBayes for a 2D correlated Gaussian with correlation rho."""
    root = Gaussian(IIDCovarianceMatrix(1, 1.0), Vector(np.zeros(1)))
    return (
        HierarchicalBayesModelBuilder()
        .set_root(root)
        .add_conditional(CorrelatedGaussianConditional(0, rho))
        .add_conditional(CorrelatedGaussianConditional(1, rho))
        .build()
    )


def _make_init() -> BlockParameter:
    return BlockParameter([Vector(np.zeros(1)), Vector(np.zeros(1))])


# ---------------------------------------------------------------------------
# 1. Structural test
# ---------------------------------------------------------------------------

class TestBlockGibbsStructure:

    def setup_method(self):
        np.random.seed(0)
        self.model = _make_model(rho=0.5)
        self.sampler = BlockGibbs(self.model)

    def test_chain_length(self):
        N = 50
        self.sampler.run(N, _make_init())
        assert self.sampler.chain.length == N + 1

    def test_block_entry_count(self):
        N = 50
        self.sampler.run(N, _make_init())
        assert len(self.sampler.chain.block(0).trajectory) == N + 1
        assert len(self.sampler.chain.block(1).trajectory) == N + 1

    def test_last_state_is_block_parameter(self):
        self.sampler.run(10, _make_init())
        assert isinstance(self.sampler.lastState, BlockParameter)
        assert self.sampler.lastState.nBlocks == 2

    def test_builder(self):
        builder = GibbsBuilder()
        builder.model = self.model
        sampler = builder.build()
        sampler.run(10, _make_init())
        assert sampler.chain.length == 11


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
        model = _make_model(rho=self.RHO)
        sampler = BlockGibbs(model)
        sampler.run(self.N_STEPS, _make_init())

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
