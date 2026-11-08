"""Learn a full MRW covariance while retaining its Cholesky factor between updates.

Run: python examples/06_fullcovariance.py --backend numpy --smoke
"""
from styne.mcmc import MetropolisedRandomWalk
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.parameter import Vector
from styne.statistics import DenseCovarianceMatrix, Gaussian, IIDCovarianceMatrix
from styne.utility import FullCovarianceAdaptation


if __name__ == '__main__':
    from runtime import configure_backend, parse_arguments

    arguments = parse_arguments(__doc__)
    backend, rng = configure_backend(arguments.backend)
    start = Vector(backend.zeros(2, dtype='float32'))
    target = Gaussian(DenseCovarianceMatrix(
        backend.asarray([[1.0, 0.8], [0.8, 2.0]], dtype='float32'),
    ), start)
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, backend.asarray(0.1)),
        AcceptanceRateDiagnostics(), rng=rng,
        adaptation=FullCovarianceAdaptation(interval=10, nugget=0.01),
    )
    nSteps = 40 if arguments.smoke else 1000
    sampler.run(nSteps, start)
    sampler.continue_run(nSteps)

    fixed = sampler.freeze_adaptation()
    fixed.run(nSteps, sampler.lastState)
    print(f'Learned proposal covariance:\n{sampler.proposalCovariance.to_dense()}')
