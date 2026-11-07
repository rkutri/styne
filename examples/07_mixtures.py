"""Mix step scales or different MCMC methods for one target.

Run: python examples/07_mixtures.py --backend jax --smoke
"""
from styne.mcmc import MALAFactory, MRWFactory, Mixture
from styne.parameter import Vector
from styne.statistics import DiagonalCovarianceMatrix, Gaussian, IIDCovarianceMatrix


if __name__ == '__main__':
    from runtime import as_numpy, configure_backend, parse_arguments

    arguments = parse_arguments(__doc__)
    backend, rng = configure_backend(arguments.backend)
    start = Vector(backend.zeros(2))
    target = Gaussian(
        DiagonalCovarianceMatrix(backend.asarray([1.0, 4.0])), start,
    ).density

    # Small moves explore locally; occasional larger moves use the same target.
    walks = []
    for variance in (0.1, 2.0):
        factory = MRWFactory()
        factory.target = target
        factory.proposalCovariance = IIDCovarianceMatrix(
            2, backend.asarray(variance),
        )
        walks.append(factory.create())
    scales = Mixture(walks, weights=[0.8, 0.2], rng=rng)

    # Components can also use different algorithms. Each performs its own MH step.
    factory = MALAFactory()
    factory.target = target
    factory.stepSize = 0.8
    factory.gradient = target.evaluate_log_gradient
    methods = Mixture(
        [walks[0], factory.create()], weights=[0.3, 0.7],
        rng=backend.random_state(2027),
    )

    nSteps = 40 if arguments.smoke else 2000
    for name, sampler in [('step scales', scales), ('different methods', methods)]:
        sampler.run(nSteps, start)
        # This is one mixture trajectory, including rejected-state repetitions.
        coordinates = backend.namespace.stack(sampler.chain.trajectory)
        mean = as_numpy(coordinates)[nSteps // 4:].mean(axis=0)
        print(f'{name}: weights={sampler.weights}, sample mean={mean}')

    if backend.name == 'jax':
        # Compile once; finalState retains every component's numerical history.
        execute = methods.compiled_trajectory(nSteps)
        first = execute(start, backend.random_state(2028))
        second = execute(first.finalState, first.rng)
        print(f'JAX: two chunks of {first.coordinates.shape[0]} transitions')
