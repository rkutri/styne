"""Warm up a random-walk scale, then sample with its scale fixed.

Run: python examples/adaptation.py --backend numpy --smoke
Requires styne; JAX and PyTorch additionally need their optional dependencies.
See docs/mcmc.md for the transition and adaptation contracts.
"""
import argparse

from styne.backend import get_backend
from styne.mcmc import MetropolisedRandomWalk
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix
from styne.utility import RobbinsMonroAdaptation


def run(backendName, smoke=False):
    backend = get_backend(backendName)
    start = Vector(backend.zeros(2, dtype='float32'))
    target = Gaussian(IIDCovarianceMatrix(2, backend.asarray(1.0)), start)
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, backend.asarray(0.01)),
        AcceptanceRateDiagnostics(), rng=backend.random_state(7),
    )
    warmup = RobbinsMonroAdaptation(sampler)
    state, rng = warmup.run(
        32 if smoke else 2000, warmup.initial_state(start), backend.random_state(11)
    )
    frozen = warmup.freeze(state, rng)
    frozen.run(32 if smoke else 2000, state.parameter)
    assert frozen.chain.length == (33 if smoke else 2001)
    assert sampler.chain.length == 0
    return frozen.diagnostics.global_acceptance_rate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'jax', 'pytorch'], default='numpy')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    acceptance = run(args.backend, args.smoke)
    print(f'{args.backend}: production acceptance {acceptance:.3f}')
