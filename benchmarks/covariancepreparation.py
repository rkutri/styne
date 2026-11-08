"""Compare raw-factor reconstruction with retained covariance preparation."""
from timeit import repeat

import numpy as np

from styne.mcmc.method.mrw import MRWProposal
from styne.parameter import Vector
from styne.statistics import DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.utility import FullCovarianceAdaptation


def main():
    iterations = 1000
    print('dimension  reconstruct (us)  retain (us)  speedup')
    for dimension in (16, 128, 512):
        strategy = FullCovarianceAdaptation()
        proposal = MRWProposal(IIDCovarianceMatrix(dimension, 1.0))
        state = strategy.initial_state(Vector(np.zeros(dimension)), proposal)
        factor = state.cholFactor

        def reconstruct():
            return MRWProposal(DenseCovarianceMatrix.from_cholesky(factor))

        def retain():
            return strategy.prepare_proposal(proposal, state)

        previous, current = (
            min(repeat(function, number=iterations, repeat=3)) / iterations
            for function in (reconstruct, retain)
        )
        print(f'{dimension:9d}  {previous * 1e6:16.2f}  {current * 1e6:11.2f}'
              f'  {previous / current:7.1f}')


if __name__ == '__main__':
    main()
