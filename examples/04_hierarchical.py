"""
Hierarchical Gaussian Sampling

Sample a correlated two-block Gaussian model with exact Gibbs updates while
keeping its joint factorisation separate from its full conditionals.
"""

import numpy as np

from runtime import as_numpy, configure_backend, parse_arguments
from styne.backend import infer_backend
from styne.mcmc import GibbsBuilder
from styne.parameter import BlockParameter, Vector
from styne.statistics import (
    ConditionalMeasure,
    Gaussian,
    HierarchicalBayesModelBuilder,
    IIDCovarianceMatrix,
)


class CorrelatedGaussianConditional(ConditionalMeasure):
    """Exact full conditional for one block of a bivariate Gaussian."""

    def __init__(self, blockIdx, correlation, backend):
        self._blockIdx = blockIdx
        self._otherIdx = 1 - blockIdx
        self._correlation = correlation
        self._backend = backend
        self._gaussian = None

    @property
    def blockDimension(self):
        return 1

    @property
    def density(self):
        if self._gaussian is None:
            raise RuntimeError("Condition the measure before using its density.")
        return self._gaussian.density

    def condition_on(self, state):
        other = state.block(self._otherIdx).coordinate
        variance = self._backend.asarray(1.0 - self._correlation**2)
        self._gaussian = Gaussian(
            IIDCovarianceMatrix(1, variance),
            Vector(self._correlation * other),
        )

    def sample(self, randomState):
        return self._gaussian.sample(randomState)


arguments = parse_arguments(__doc__)
backend, rng = configure_backend(arguments.backend)
correlation = 0.8
root = Gaussian(
    IIDCovarianceMatrix(1, backend.asarray(1.0)),
    Vector(backend.zeros(1)),
)
latentConditional = CorrelatedGaussianConditional(0, correlation, backend)
rootConditional = CorrelatedGaussianConditional(1, correlation, backend)

hierarchy = (
    HierarchicalBayesModelBuilder()
    .set_root(root, name="root")
    .add_conditional(latentConditional, name="latent")
    .add_update(latentConditional)
    .add_update(rootConditional)
    .build()
)
builder = GibbsBuilder()
builder.model = hierarchy
builder.rng = rng
sampler = builder.build()
initial = BlockParameter(
    [Vector(backend.zeros(1)), Vector(backend.zeros(1))],
    names={"latent": 0, "root": 1},
)
steps = 20 if arguments.smoke else 5_000
sampler.run(steps, initial, progress=not arguments.smoke)

trajectory = np.column_stack([
    as_numpy(sampler.chain.block(index).trajectory)
    for index in range(hierarchy.nBlocks)
])
print(f"blocks={hierarchy.blockNames}")
print(f"sample covariance={np.cov(trajectory.T)}")
assert infer_backend(sampler.lastState.coordinate) is backend
