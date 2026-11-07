"""Set up mixtures of sampler settings or sampling methods.

Configure each component as an ordinary sampler, then choose its weight.
Both examples sample the same two-dimensional Gaussian distribution.

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

    # Use a mixture of two MRW configurations with different proposal variances.
    walks = []
    for variance in (0.1, 2.0):
        factory = MRWFactory()
        factory.target = target
        factory.proposalCovariance = IIDCovarianceMatrix(
            2, backend.asarray(variance),
        )
        walks.append(factory.create())
    # Weights follow component order: use the small-step walk 80% of the time.
    # Every component receives the same target object.
    scales = Mixture(walks, weights=[0.8, 0.2], rng=rng)

    # To combine methods, keep the small-step walk and add a MALA sampler.
    # MALA needs the target gradient as well as a step size.
    factory = MALAFactory()
    factory.target = target
    factory.stepSize = 0.8
    factory.gradient = target.evaluate_log_gradient
    # This configuration uses MRW with probability 0.3 and MALA with 0.7.
    methods = Mixture(
        [walks[0], factory.create()], weights=[0.3, 0.7],
        rng=backend.random_state(2027),
    )

    # Both mixtures use the usual sampler API
    nSteps = 40 if arguments.smoke else 2000
    for name, sampler in [('step scales', scales), ('different methods', methods)]:
        componentNames = [component.name for component in sampler.components]
        print(f'{name}: components={componentNames}, weights={sampler.weights}')
        sampler.run(nSteps, start)
        coordinates = backend.namespace.stack(sampler.chain.trajectory)
        mean = as_numpy(coordinates)[nSteps // 4:].mean(axis=0)
        print(f'{name}: sample mean={mean}')

    if backend.name == 'jax':
        # With JAX, compile once and pass finalState and rng to continue sampling.
        execute = methods.compiled_trajectory(nSteps)
        first = execute(start, backend.random_state(2028))
        second = execute(first.finalState, first.rng)
        print(f'JAX: two chunks of {first.coordinates.shape[0]} transitions')
