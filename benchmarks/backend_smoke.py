"""Small backend smoke benchmark for Gaussian sampling and density evaluation."""
from time import perf_counter

from styne.backend import BackendUnavailableError, get_backend
from styne.parameter import Vector
from styne.statistics import Gaussian, GaussianDensity, IIDCovarianceMatrix


def run_backend(backendName, nSteps=100):
    backend = get_backend(backendName)
    mean = Vector(backend.zeros(64, dtype='float32'))
    covariance = IIDCovarianceMatrix(
        64, backend.asarray(1., dtype='float32')
    )
    measure = Gaussian(covariance, mean)
    density = GaussianDensity(covariance, mean)
    randomState = backend.random_state(7)
    started = perf_counter()
    for _ in range(nSteps):
        sample, randomState = measure.sample(randomState)
        density.evaluate_log(sample)
    return perf_counter() - started


if __name__ == '__main__':
    for backendName in ('numpy', 'pytorch', 'jax'):
        try:
            elapsed = run_backend(backendName)
        except BackendUnavailableError:
            continue
        print(f'{backendName}: {elapsed:.3f}s')
