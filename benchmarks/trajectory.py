"""Compare retraced JAX chunks with one reusable compiled trajectory.

Run: python benchmarks/trajectory.py
Includes compilation in each total and synchronises device execution.
This measures dispatch and compilation for a small Gaussian random walk,
not the cost of a model evaluation or general MCMC throughput.
"""
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from styne.mcmc import MetropolisedRandomWalk
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix


def run_chunks(reuse, nChunks=10, nSteps=100):
    start = Vector(jnp.zeros(16))
    target = Gaussian(IIDCovarianceMatrix(16, jnp.asarray(1.0)), start)
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(16, jnp.asarray(0.1)), DummyDiagnostics()
    )
    run = sampler.compiled_trajectory(nSteps) if reuse else None
    state, rng = start, jax.random.key(3)
    started = perf_counter()
    for _ in range(nChunks):
        result = run(state, rng) if reuse else sampler.transformed_trajectory(
            nSteps, state, rng
        )
        state, rng = result.final, result.rng
        state.coordinate.block_until_ready()
    return perf_counter() - started, np.asarray(state.coordinate)


if __name__ == '__main__':
    repeated, repeatedFinal = run_chunks(False)
    reused, reusedFinal = run_chunks(True)
    np.testing.assert_allclose(repeatedFinal, reusedFinal, atol=1e-6)
    print(f'10 chunks of 100 transitions, 16 dimensions: {repeated:.3f}s retraced; '
          f'{reused:.3f}s reused')
