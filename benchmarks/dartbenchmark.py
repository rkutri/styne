"""Shared machinery for the DART benchmark entry points."""

import os
import csv
import gc
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


THREAD_VARIABLES = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)
for threadVariable in THREAD_VARIABLES:
    os.environ.setdefault(threadVariable, '1')


import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402

from styne.backend import (  # noqa: E402
    get_backend as backend_factory,
    infer_backend,
)
from styne.mcmc.diagnostics import DummyDiagnostics  # noqa: E402
from styne.mcmc.method.dartdirect import DirectDART  # noqa: E402
from styne.mcmc.method.mala import MALAFactory  # noqa: E402
from styne.mcmc.method.mlda import MLDAFactory  # noqa: E402
from styne.mcmc.metropolishastings import MetropolisHastings  # noqa: E402
from styne.mcmc.proposal import ProposalMethod  # noqa: E402
from styne.mcmc.transition import TransitionData  # noqa: E402
from styne.parameter import Vector  # noqa: E402
from styne.statistics import DenseCovarianceMatrix, Gaussian  # noqa: E402
from styne.statistics.logistic import LogisticPosterior  # noqa: E402
from styne.utility.postprocessing import multichain_ess_per_iter  # noqa: E402


LOGISTIC_TUNING = 'logistic_tuning'
LOGISTIC_COMPARISON = 'logistic_comparison'
LOCALISATION_PILOT = 'localisation_pilot'
LOCALISATION_ABLATION = 'localisation_ablation'
DART = 'dart_laplace'
MALA = 'mala'
MLDA = 'mlda_untempered'
LOCALISED = 'localised_dart'
UNLOCALISED = 'unlocalised_surrogate'

SURROGATE_CONDITIONS = (
    'laplace',
    'mean_shift',
    'precision_distortion',
    'combined_misspecification',
)
GAMMA_CANDIDATES = (0.01, 0.03, 0.1, 0.3, 1.0)

PRIMARY_ABLATION_METRIC = (
    'minimum_laplace_hessian_eigendirection_ess_'
    'per_target_density_evaluation'
)

RAW_FIELDS = (
    'analysis', 'config', 'run_scope', 'backend', 'method',
    'run_id', 'replicate_id', 'configuration_id', 'evaluation_seed',
    'dataset_seed', 'perturbation_seed', 'start_seed', 'chain_seed',
    'dimension', 'surrogate_condition', 'chain_count',
    'transition_count', 'burnin', 'retained_iterations', 'tempering',
    'gamma_over_l', 'mala_step_size', 'mlda_root_step_times_sqrt_d',
    'mlda_subchain_length', 'tuning_parameter',
    'tuning_value', 'selected_for_evaluation', 'boundary_selection',
    'status', 'error', 'acceptance_rate',
    'max_rank_normalized_split_rhat', 'setup_seconds', 'sampling_seconds',
    'ess_observable_count',
    'eigendirection_ess_per_iteration',
    'eigendirection_rank_normalized_split_rhat',
    'minimum_eigendirection_index',
    'slow_direction_ess_per_iteration',
    'minimum_eigendirection_total_ess',
    PRIMARY_ABLATION_METRIC,
    'hessian_esjd_per_target_density_evaluation',
    'target_density_evaluations',
    'target_gradient_evaluations', 'surrogate_density_evaluations',
    'surrogate_gradient_evaluations',
    'surrogate_quadratic_form_evaluations',
)

N_OBSERVATIONS = 240
ALPHA_PRIOR = 3.0
TEMPERING = 0.5
ABLATION_CONDITION_NUMBER = 16.0
MEAN_SHIFT_MAHALANOBIS = 1.5
PRECISION_DISTORTION_FACTOR = 2.0
CHECKPOINT_SCHEMA_VERSION = 1
NONFINITE_FLOAT_KEY = '__styne_benchmark_nonfinite_float__'


def get_backend(backendName):
    return backend_factory(backendName)


def validate_production_runtime(configuration, backend):
    if configuration.name != 'production':
        return
    if backend.name != 'jax':
        raise ValueError('Production DART benchmarks require JAX.')

    import jax
    import styne

    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError(
            'Production DART benchmarks require JAX float64.'
        )
    expectedPackage = (Path(__file__).parents[1] / 'src/styne').resolve()
    importedPackage = Path(styne.__file__).resolve().parent
    if importedPackage != expectedPackage:
        raise RuntimeError(
            'Production must import styne from the current repository: '
            f'expected {expectedPackage}, found {importedPackage}.'
        )


@dataclass(frozen=True)
class LogisticProblem:
    features: np.ndarray
    responses: np.ndarray
    mapCoordinate: np.ndarray
    hessian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    smoothness: float


@dataclass(frozen=True)
class GaussianProblem:
    precision: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class SamplingMetrics:
    essPerIteration: np.ndarray
    splitRhat: np.ndarray
    acceptanceRate: float
    maximumRhat: float
    samplingSeconds: float
    hessianEsjdPerEvaluation: float
    fineEvaluations: int


class IndependentGaussianProposal(ProposalMethod):
    """Independent draws from one fixed, tempered Gaussian surrogate."""

    def __init__(self, proposalMeasure):
        self._proposalMeasure = proposalMeasure

    def propose(self, state, rng):
        backend = infer_backend(state.coordinate)
        mean = backend.namespace.broadcast_to(
            self._proposalMeasure.mean.coordinate,
            state.coordinate.shape,
        )
        proposalMeasure = self._proposalMeasure.with_mean(
            state.with_coordinate(mean)
        )
        proposal, nextRng = proposalMeasure.sample(rng)
        return TransitionData(state, proposal), nextRng


class UnlocalisedGaussianSurrogate(MetropolisHastings):
    """Exact independence-MH limit of Gaussian DART at zero localisation."""

    name = 'unlocalised surrogate'

    def __init__(self, target, tempering, surrogate):
        covariance = surrogate.covariance.with_scaling(
            surrogate.covariance.scaling / tempering
        )
        proposalMeasure = Gaussian(covariance, surrogate.mean)
        super().__init__(
            target,
            IndependentGaussianProposal(proposalMeasure),
            DummyDiagnostics(),
        )
        self._tempering = tempering
        self._surrogate = surrogate

    def _log_mh_ratio(self, transition):
        stateDifference = (
            transition.state.coordinate - self._surrogate.mean.coordinate
        )
        proposalDifference = (
            transition.proposal.coordinate - self._surrogate.mean.coordinate
        )
        surrogateDifference = 0.5 * self._tempering * (
            self._surrogate.covariance.dual_quadratic_form(
                proposalDifference
            )
            - self._surrogate.covariance.dual_quadratic_form(
                stateDifference
            )
        )
        targetDifference = (
            transition.proposed.logDensity - transition.current.logDensity
        )
        return targetDifference + surrogateDifference


@lru_cache(maxsize=None)
def generate_logistic_problem(dataSeed, dimension):
    randomGenerator = np.random.default_rng(dataSeed)
    covariates = randomGenerator.choice(
        (-1.0, 1.0), size=(N_OBSERVATIONS, dimension)
    )
    covariates /= np.linalg.norm(covariates, axis=1, keepdims=True)
    probabilities = expit(covariates @ np.ones(dimension))
    responses = randomGenerator.binomial(1, probabilities).astype(float)

    sampleCovariance = covariates.T @ covariates / N_OBSERVATIONS
    covarianceEigenvalues, covarianceEigenvectors = np.linalg.eigh(
        sampleCovariance
    )
    inverseSquareRoot = (
        covarianceEigenvectors / np.sqrt(covarianceEigenvalues)
    ) @ covarianceEigenvectors.T
    features = covariates @ inverseSquareRoot

    def objective(coordinate):
        predictor = features @ coordinate
        return (
            np.sum(np.logaddexp(0.0, predictor) - responses * predictor)
            + 0.5 * ALPHA_PRIOR * coordinate @ coordinate
        )

    def gradient(coordinate):
        predictor = features @ coordinate
        return features.T @ (expit(predictor) - responses) \
            + ALPHA_PRIOR * coordinate

    optimisation = minimize(
        objective,
        np.zeros(dimension),
        jac=gradient,
        method='L-BFGS-B',
        options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 1_000},
    )
    if not optimisation.success:
        raise RuntimeError(f'MAP optimisation failed: {optimisation.message}')

    mapCoordinate = optimisation.x
    mapProbabilities = expit(features @ mapCoordinate)
    weights = mapProbabilities * (1.0 - mapProbabilities)
    hessian = (features.T * weights) @ features \
        + ALPHA_PRIOR * np.eye(dimension)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    smoothness = N_OBSERVATIONS / 4.0 + ALPHA_PRIOR
    return LogisticProblem(
        features,
        responses,
        mapCoordinate,
        hessian,
        eigenvalues,
        eigenvectors,
        smoothness,
    )


def generate_gaussian_problem(dataSeed, dimension):
    randomGenerator = np.random.default_rng(dataSeed)
    orthogonal, triangular = np.linalg.qr(
        randomGenerator.normal(size=(dimension, dimension))
    )
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    eigenvectors = orthogonal * signs
    eigenvalues = np.geomspace(1.0, ABLATION_CONDITION_NUMBER, dimension)
    precision = (eigenvectors * eigenvalues) @ eigenvectors.T
    covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
    return GaussianProblem(
        precision,
        covariance,
        eigenvalues,
        eigenvectors,
    )


def derived_seed(baseSeed, purpose, dimension):
    """Derive an order-independent uint32 seed for one benchmark purpose."""
    return int(np.random.SeedSequence(
        (int(baseSeed), int(purpose), int(dimension))
    ).generate_state(1)[0])


def ablation_seeds(evaluationSeed, dimension):
    return {
        'dataset': derived_seed(evaluationSeed, 1, dimension),
        'perturbation': derived_seed(evaluationSeed, 2, dimension),
        'start': derived_seed(evaluationSeed, 3, dimension),
        'chain': derived_seed(evaluationSeed, 4, dimension),
    }


def orthogonal_matrix(randomGenerator, dimension):
    orthogonal, triangular = np.linalg.qr(
        randomGenerator.normal(size=(dimension, dimension))
    )
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    return orthogonal * signs


def surrogate_parameters(
        referenceMean, referencePrecision, eigenvalues, eigenvectors,
        perturbationSeed, condition):
    """Return one predeclared perturbation of a Laplace approximation."""
    dimension = referencePrecision.shape[0]
    randomGenerator = np.random.default_rng(perturbationSeed)
    direction = randomGenerator.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    inverseSquareRoot = (
        eigenvectors / np.sqrt(eigenvalues)
    ) @ eigenvectors.T
    shiftedMean = referenceMean + MEAN_SHIFT_MAHALANOBIS * (
        inverseSquareRoot @ direction
    )

    rotation = orthogonal_matrix(randomGenerator, dimension)
    multipliers = np.geomspace(
        1.0 / PRECISION_DISTORTION_FACTOR,
        PRECISION_DISTORTION_FACTOR,
        dimension,
    )
    relativePrecision = (rotation * multipliers) @ rotation.T
    squareRoot = (
        eigenvectors * np.sqrt(eigenvalues)
    ) @ eigenvectors.T
    distortedPrecision = squareRoot @ relativePrecision @ squareRoot

    if condition == 'laplace':
        return referenceMean.copy(), referencePrecision.copy()
    if condition == 'mean_shift':
        return shiftedMean, referencePrecision.copy()
    if condition == 'precision_distortion':
        return referenceMean.copy(), distortedPrecision
    if condition == 'combined_misspecification':
        return shiftedMean, distortedPrecision
    raise ValueError(f'Unknown surrogate condition: {condition}')


def variance_rhat(chains):
    nChains, nSamples = chains.shape
    if nChains < 2 or nSamples < 2:
        return np.nan
    withinVariance = np.mean(np.var(chains, axis=1, ddof=1))
    if not np.isfinite(withinVariance) or withinVariance <= 0.0:
        return np.inf
    betweenVariance = nSamples * np.var(
        np.mean(chains, axis=1), ddof=1
    )
    varianceEstimate = (
        (nSamples - 1.0) / nSamples * withinVariance
        + betweenVariance / nSamples
    )
    return float(np.sqrt(varianceEstimate / withinVariance))


def rank_normalize(values):
    ranks = rankdata(values.reshape(-1), method='average')
    probabilities = (ranks - 3.0 / 8.0) / (ranks.size + 1.0 / 4.0)
    return norm.ppf(probabilities).reshape(values.shape)


def split_rhat(chains):
    """Return the maximum of rank-normalised and folded split R-hat."""
    chains = np.asarray(chains, dtype=float)
    nChains, nSamples = chains.shape
    half = nSamples // 2
    if nChains < 2 or half < 2:
        return np.nan
    splitChains = np.concatenate(
        (chains[:, :half], chains[:, -half:]), axis=0
    )
    bulkRhat = variance_rhat(rank_normalize(splitChains))
    folded = np.abs(splitChains - np.median(splitChains))
    foldedRhat = variance_rhat(rank_normalize(folded))
    return float(np.maximum(bulkRhat, foldedRhat))


def as_numpy(array):
    if hasattr(array, 'detach'):
        array = array.detach()
    if hasattr(array, 'cpu'):
        array = array.cpu()
    return np.asarray(array)


def projected_metrics(
        coordinates, initialCoordinates, eigenvectors, precision,
        burnin, retained, samplingSeconds, acceptanceOutcomes=None):
    backend = infer_backend(coordinates)
    namespace = backend.namespace
    initial = backend.asarray(
        initialCoordinates,
        dtype=backend.metadata(coordinates).dtype,
        device=backend.metadata(coordinates).device,
    )
    retainedCoordinates = coordinates[burnin:burnin + retained]
    if burnin == 0:
        retainedPrevious = namespace.concatenate(
            (
                namespace.expand_dims(initial, axis=0),
                coordinates[:retained - 1],
            ),
            axis=0,
        )
    else:
        retainedPrevious = coordinates[
            burnin - 1:burnin + retained - 1
        ]
    eigenvectorsBackend = backend.asarray(
        eigenvectors,
        dtype=backend.metadata(coordinates).dtype,
        device=backend.metadata(coordinates).device,
    )
    projections = retainedCoordinates @ eigenvectorsBackend
    projectionArray = np.swapaxes(as_numpy(projections), 0, 1)

    essPerIteration = np.asarray([
        multichain_ess_per_iter(projectionArray[:, :, directionIndex])
        for directionIndex in range(projectionArray.shape[-1])
    ])
    rhatValues = np.asarray([
        split_rhat(projectionArray[:, :, directionIndex])
        for directionIndex in range(projectionArray.shape[-1])
    ])

    retainedChanges = retainedCoordinates - retainedPrevious
    if acceptanceOutcomes is None:
        accepted = namespace.any(retainedChanges != 0.0, axis=-1)
    else:
        accepted = acceptanceOutcomes[burnin:burnin + retained]
    acceptanceCount = float(as_numpy(namespace.sum(accepted)))
    acceptanceRate = acceptanceCount / (
        initialCoordinates.shape[0] * retained
    )
    precisionBackend = backend.asarray(
        precision,
        dtype=backend.metadata(coordinates).dtype,
        device=backend.metadata(coordinates).device,
    )
    allPrevious = namespace.concatenate(
        (
            namespace.expand_dims(initial, axis=0),
            coordinates[:-1],
        ),
        axis=0,
    )
    allChanges = coordinates - allPrevious
    squaredJump = namespace.sum(
        allChanges * (allChanges @ precisionBackend.T), axis=-1
    )
    jumpSum = float(as_numpy(namespace.sum(squaredJump)))
    chainCount = initialCoordinates.shape[0]
    totalSteps = coordinates.shape[0]
    fineEvaluations = chainCount * (totalSteps + 1)
    hessianEsjd = jumpSum / fineEvaluations
    maximumRhat = float(np.max(rhatValues))
    return SamplingMetrics(
        essPerIteration,
        rhatValues,
        acceptanceRate,
        maximumRhat,
        samplingSeconds,
        hessianEsjd,
        fineEvaluations,
    )


def run_transformed_sampler(
        sampler, backend, initialCoordinates, eigenvectors, precision,
        burnin, retained, chainSeed):
    initial = Vector(backend.asarray(initialCoordinates, dtype='float64'))
    randomState = backend.random_state(chainSeed)
    started = time.perf_counter()
    transitionCount = burnin + retained

    def advance(carry, _):
        state, currentRng = carry
        nextState, transition, nextRng = sampler.step(state, currentRng)
        output = (
            nextState.parameter.coordinate,
            transition.outcome,
        )
        return (nextState, nextRng), output

    def execute(parameter, currentRng):
        state = sampler.initial_state(parameter)
        return backend.scan(
            advance,
            (state, currentRng),
            None,
            length=transitionCount,
        )

    compiled = backend.compile(execute)
    _, (coordinates, acceptanceOutcomes) = compiled(
        initial, randomState
    )
    if hasattr(coordinates, 'block_until_ready'):
        coordinates.block_until_ready()
    samplingSeconds = time.perf_counter() - started
    metrics = projected_metrics(
        coordinates,
        initialCoordinates,
        eigenvectors,
        precision,
        burnin,
        retained,
        samplingSeconds,
        acceptanceOutcomes=acceptanceOutcomes,
    )
    del coordinates, acceptanceOutcomes
    gc.collect()
    return metrics


def run_eager_sampler(
        sampler, backend, initialCoordinates, eigenvectors, precision,
        burnin, retained, chainSeed):
    initial = Vector(backend.asarray(initialCoordinates, dtype='float64'))
    randomState = backend.random_state(chainSeed)
    transitionCount = burnin + retained
    started = time.perf_counter()
    currentState = sampler.initial_state(initial)
    coordinateRows = []
    acceptanceRows = []
    for stepIndex in range(transitionCount):
        currentState, transition, randomState = sampler.step(
            currentState, randomState
        )
        coordinateRows.append(currentState.parameter.coordinate)
        acceptanceRows.append(transition.outcome)
    samplingSeconds = time.perf_counter() - started
    backendCoordinates = backend.namespace.stack(coordinateRows, axis=0)
    backendAcceptance = backend.namespace.stack(acceptanceRows, axis=0)
    return projected_metrics(
        backendCoordinates,
        initialCoordinates,
        eigenvectors,
        precision,
        burnin,
        retained,
        samplingSeconds,
        acceptanceOutcomes=backendAcceptance,
    )


def run_sampler(
        sampler, backend, initialCoordinates, eigenvectors, precision,
        burnin, retained, chainSeed):
    if backend.name == 'jax':
        return run_transformed_sampler(
            sampler,
            backend,
            initialCoordinates,
            eigenvectors,
            precision,
            burnin,
            retained,
            chainSeed,
        )
    return run_eager_sampler(
        sampler,
        backend,
        initialCoordinates,
        eigenvectors,
        precision,
        burnin,
        retained,
        chainSeed,
    )


def backend_gaussian(backend, mean, covariance):
    covarianceBackend = backend.asarray(covariance, dtype='float64')
    meanBackend = backend.asarray(mean, dtype='float64')
    return Gaussian(
        DenseCovarianceMatrix(covarianceBackend),
        Vector(meanBackend),
    )


def make_dart_sampler(target, backend, surrogate, precision, gamma):
    dimension = precision.shape[0]
    proposalPrecision = TEMPERING * precision + gamma * np.eye(dimension)
    proposalCovariance = DenseCovarianceMatrix(
        backend.asarray(np.linalg.inv(proposalPrecision), dtype='float64')
    )
    return DirectDART(
        target,
        TEMPERING,
        gamma,
        surrogate,
        proposalCovariance,
        DummyDiagnostics(),
    )


def make_mala_sampler(target, stepSize):
    factory = MALAFactory()
    factory.target = target
    factory.stepSize = stepSize
    factory.gradient = target.evaluate_log_gradient
    factory.diagnostics = DummyDiagnostics()
    return factory.create()


def make_mlda_sampler(
        target, backend, surrogate, rootCovariance, subchainLength):
    factory = MLDAFactory(root='mrw')
    factory.target = target
    factory.surrogate = [surrogate.density]
    factory.nChain = [subchainLength]
    factory.tempering = [1.0]
    factory.root.proposalCovariance = DenseCovarianceMatrix(
        backend.asarray(rootCovariance, dtype='float64')
    )
    factory.diagnostics = DummyDiagnostics()
    return factory.create()


def evaluation_status(metrics):
    finiteValues = np.concatenate((
        metrics.essPerIteration,
        metrics.splitRhat,
        np.asarray([
            metrics.acceptanceRate,
            metrics.maximumRhat,
            metrics.samplingSeconds,
            metrics.hessianEsjdPerEvaluation,
        ]),
    ))
    if not np.all(np.isfinite(finiteValues)):
        return 'nonfinite'
    return 'ok'


def operation_counts(
        method, chainCount, totalSteps, mldaSubchainLength=None):
    counts = {
        'target_density_evaluations': chainCount * (totalSteps + 1),
        'target_gradient_evaluations': 0,
        'surrogate_density_evaluations': 0,
        'surrogate_gradient_evaluations': 0,
        'surrogate_quadratic_form_evaluations': 0,
    }
    if method == MALA:
        counts['target_gradient_evaluations'] = 2 * chainCount * totalSteps
    elif method == MLDA:
        if not isinstance(mldaSubchainLength, int) \
                or mldaSubchainLength <= 0:
            raise ValueError(
                'MLDA operation counts require a positive subchain length.'
            )
        counts['surrogate_density_evaluations'] = (
            (mldaSubchainLength + 3) * chainCount * totalSteps
        )
    elif method in (DART, LOCALISED, UNLOCALISED):
        counts['surrogate_quadratic_form_evaluations'] = (
            2 * chainCount * totalSteps
        )
    return counts


def empty_raw_row(configuration, backendName, analysis, method):
    row = {field: '' for field in RAW_FIELDS}
    row.update({
        'analysis': analysis,
        'config': configuration.name,
        'run_scope': configuration.name,
        'backend': backendName,
        'method': method,
    })
    return row


def metrics_row(
        configuration, backendName, analysis, method, runId, evaluationSeed,
        datasetSeed, startSeed, chainSeed, dimension, condition,
        chainCount, burnin, retained, gammaOverL, malaStepSize, metrics,
        perturbationSeed='', setupSeconds='', mldaSubchainLength=None):
    row = empty_raw_row(configuration, backendName, analysis, method)
    minimumEssRate = float(np.min(metrics.essPerIteration))
    minimumEssIndex = int(np.argmin(metrics.essPerIteration))
    minimumTotalEss = minimumEssRate * retained * chainCount
    transitionCount = metrics.fineEvaluations // chainCount - 1
    counts = operation_counts(
        method,
        chainCount,
        transitionCount,
        mldaSubchainLength=mldaSubchainLength,
    )
    row.update({
        'run_id': runId,
        'replicate_id': evaluationSeed,
        'configuration_id': f'd{dimension}:{condition}',
        'evaluation_seed': evaluationSeed,
        'dataset_seed': datasetSeed,
        'perturbation_seed': perturbationSeed,
        'start_seed': startSeed,
        'chain_seed': chainSeed,
        'dimension': dimension,
        'surrogate_condition': condition,
        'chain_count': chainCount,
        'transition_count': transitionCount,
        'burnin': burnin,
        'retained_iterations': retained,
        'tempering': TEMPERING,
        'gamma_over_l': gammaOverL,
        'mala_step_size': malaStepSize,
        'status': evaluation_status(metrics),
        'error': '',
        'acceptance_rate': metrics.acceptanceRate,
        'max_rank_normalized_split_rhat': metrics.maximumRhat,
        'setup_seconds': setupSeconds,
        'sampling_seconds': metrics.samplingSeconds,
        'ess_observable_count': metrics.essPerIteration.size,
        'eigendirection_ess_per_iteration': json.dumps(
            metrics.essPerIteration.tolist(), separators=(',', ':')
        ),
        'eigendirection_rank_normalized_split_rhat': json.dumps(
            metrics.splitRhat.tolist(), separators=(',', ':')
        ),
        'minimum_eigendirection_index': minimumEssIndex,
        'slow_direction_ess_per_iteration': metrics.essPerIteration[0],
        'minimum_eigendirection_total_ess': minimumTotalEss,
        PRIMARY_ABLATION_METRIC: (
            minimumTotalEss
            / metrics.fineEvaluations
        ),
        'hessian_esjd_per_target_density_evaluation': (
            metrics.hessianEsjdPerEvaluation
        ),
        **counts,
    })
    return row


def failure_row(
        configuration, backendName, analysis, method, runId, evaluationSeed,
        datasetSeed, startSeed, chainSeed, dimension, condition,
        chainCount, burnin, retained, error, perturbationSeed=''):
    row = empty_raw_row(configuration, backendName, analysis, method)
    row.update({
        'run_id': runId,
        'replicate_id': evaluationSeed,
        'configuration_id': f'd{dimension}:{condition}',
        'evaluation_seed': evaluationSeed,
        'dataset_seed': datasetSeed,
        'perturbation_seed': perturbationSeed,
        'start_seed': startSeed,
        'chain_seed': chainSeed,
        'dimension': dimension,
        'surrogate_condition': condition,
        'chain_count': chainCount,
        'burnin': burnin,
        'retained_iterations': retained,
        'tempering': TEMPERING,
        'status': 'exception',
        'error': f'{type(error).__name__}: {error}',
    })
    return row


def gaussian_components(
        backend, problem, condition, perturbationSeed=42_000):
    zero = np.zeros(problem.precision.shape[0])
    target = backend_gaussian(backend, zero, problem.covariance).density
    surrogateMean, surrogatePrecision = surrogate_parameters(
        zero,
        problem.precision,
        problem.eigenvalues,
        problem.eigenvectors,
        perturbationSeed,
        condition,
    )
    surrogateCovariance = np.linalg.inv(surrogatePrecision)
    surrogate = backend_gaussian(
        backend, surrogateMean, surrogateCovariance
    )
    return target, surrogate, surrogatePrecision


def logistic_components(backend, problem, condition, perturbationSeed):
    target = LogisticPosterior(
        problem.features, problem.responses, ALPHA_PRIOR
    )
    surrogateMean, surrogatePrecision = surrogate_parameters(
        problem.mapCoordinate,
        problem.hessian,
        problem.eigenvalues,
        problem.eigenvectors,
        perturbationSeed,
        condition,
    )
    surrogate = backend_gaussian(
        backend, surrogateMean, np.linalg.inv(surrogatePrecision)
    )
    return target, surrogate, surrogatePrecision


def logistic_starts(problem, seed, chainCount):
    randomGenerator = np.random.default_rng(seed)
    noise = randomGenerator.normal(
        size=(chainCount, problem.hessian.shape[0])
    )
    covarianceCholesky = np.linalg.cholesky(
        np.linalg.inv(problem.hessian)
    )
    return problem.mapCoordinate + noise @ covarianceCholesky.T


def run_ablation_cell(
        configuration, backend, analysis, method, runId, evaluationSeed,
        dimension, condition, gammaOverL, chainCount, burnin, retained):
    seeds = ablation_seeds(evaluationSeed, dimension)
    problem = generate_logistic_problem(seeds['dataset'], dimension)
    setupStarted = time.perf_counter()
    target, surrogate, surrogatePrecision = logistic_components(
        backend, problem, condition, seeds['perturbation']
    )
    starts = logistic_starts(problem, seeds['start'], chainCount)
    if method == LOCALISED:
        gamma = gammaOverL * problem.smoothness
        sampler = make_dart_sampler(
            target,
            backend,
            surrogate,
            surrogatePrecision,
            gamma,
        )
    elif method == UNLOCALISED:
        sampler = UnlocalisedGaussianSurrogate(
            target, TEMPERING, surrogate
        )
    else:
        raise ValueError(f'Unknown ablation method: {method}')
    setupSeconds = time.perf_counter() - setupStarted
    metrics = run_sampler(
        sampler,
        backend,
        starts,
        problem.eigenvectors,
        problem.hessian,
        burnin,
        retained,
        seeds['chain'],
    )
    return metrics_row(
        configuration,
        backend.name,
        analysis,
        method,
        runId,
        evaluationSeed,
        seeds['dataset'],
        seeds['start'],
        seeds['chain'],
        dimension,
        condition,
        chainCount,
        burnin,
        retained,
        gammaOverL if method == LOCALISED else 0.0,
        '',
        metrics,
        perturbationSeed=seeds['perturbation'],
        setupSeconds=setupSeconds,
    )


def select_ablation_gamma(
        configuration, backend, initialRows=None, checkpointCallback=None):
    rows = list(initialRows or ())
    allowedPilotKeys = {
        (pilotSeed, dimension, condition, LOCALISED, gammaOverL)
        for gammaOverL in GAMMA_CANDIDATES
        for pilotSeed in configuration.ablationPilotSeeds
        for dimension in configuration.ablationDimensions
        for condition in configuration.ablationConditions
    }
    observedPilotKeys = {
        (
            int(row['evaluation_seed']),
            int(row['dimension']),
            row['surrogate_condition'],
            row['method'],
            row['gamma_over_l'],
        )
        for row in rows
        if row['analysis'] == LOCALISATION_PILOT
    }
    if len(observedPilotKeys) != len(rows) \
            or not observedPilotKeys.issubset(allowedPilotKeys):
        raise ValueError('Checkpoint contains unexpected pilot rows.')
    expectedCandidateRows = (
        len(configuration.ablationDimensions)
        * len(configuration.ablationConditions)
        * len(configuration.ablationPilotSeeds)
    )
    for gammaOverL in GAMMA_CANDIDATES:
        existingRows = [
            row for row in rows
            if row['analysis'] == LOCALISATION_PILOT
            and row['gamma_over_l'] == gammaOverL
        ]
        expectedKeys = {
            (pilotSeed, dimension, condition, LOCALISED)
            for pilotSeed in configuration.ablationPilotSeeds
            for dimension in configuration.ablationDimensions
            for condition in configuration.ablationConditions
        }
        actualKeys = {
            (
                int(row['evaluation_seed']),
                int(row['dimension']),
                row['surrogate_condition'],
                row['method'],
            )
            for row in existingRows
        }
        if len(existingRows) == expectedCandidateRows \
                and actualKeys == expectedKeys:
            continue
        if existingRows:
            raise ValueError(
                f'Checkpoint gamma/L={gammaOverL:g} is partial.'
            )
        for pilotIndex, pilotSeed in enumerate(configuration.ablationPilotSeeds):
            for dimension in configuration.ablationDimensions:
                seeds = ablation_seeds(pilotSeed, dimension)
                for condition in configuration.ablationConditions:
                    print(  # noqa: T201
                        f'[{configuration.name}] ablation pilot={pilotIndex} '
                        f'd={dimension} condition={condition} '
                        f'gamma/L={gammaOverL}'
                    )
                    runId = f'pilot_{pilotIndex}:gamma_{gammaOverL:g}'
                    try:
                        row = run_ablation_cell(
                            configuration,
                            backend,
                            LOCALISATION_PILOT,
                            LOCALISED,
                            runId,
                            pilotSeed,
                            dimension,
                            condition,
                            gammaOverL,
                            configuration.ablationPilotChains,
                            configuration.ablationPilotBurnin,
                            configuration.ablationPilotRetained,
                        )
                    except Exception as error:
                        row = failure_row(
                            configuration,
                            backend.name,
                            LOCALISATION_PILOT,
                            LOCALISED,
                            runId,
                            pilotSeed,
                            seeds['dataset'],
                            seeds['start'],
                            seeds['chain'],
                            dimension,
                            condition,
                            configuration.ablationPilotChains,
                            configuration.ablationPilotBurnin,
                            configuration.ablationPilotRetained,
                            error,
                            perturbationSeed=seeds['perturbation'],
                        )
                        row['gamma_over_l'] = gammaOverL
                    rows.append(row)
        if checkpointCallback is not None:
            checkpointCallback(rows)

    scores = {}
    for gammaOverL in GAMMA_CANDIDATES:
        candidateRows = [
            row for row in rows
            if row['gamma_over_l'] == gammaOverL
            and is_usable(row)
        ]
        if len(candidateRows) != (
                len(configuration.ablationDimensions)
                * len(configuration.ablationConditions)
                * len(configuration.ablationPilotSeeds)):
            scores[gammaOverL] = -np.inf
            continue
        grouped = {}
        for row in candidateRows:
            grouped.setdefault(row['configuration_id'], []).append(
                float(row[PRIMARY_ABLATION_METRIC])
            )
        configurationMeans = np.asarray([
            np.mean(values) for values in grouped.values()
        ])
        scores[gammaOverL] = float(np.median(configurationMeans))
    selectedGamma = max(GAMMA_CANDIDATES, key=lambda value: scores[value])
    if not np.isfinite(scores[selectedGamma]):
        selectedGamma = None
    boundary = selectedGamma is not None and selectedGamma in (
        GAMMA_CANDIDATES[0], GAMMA_CANDIDATES[-1]
    )
    for row in rows:
        selected = (
            selectedGamma is not None
            and row['gamma_over_l'] == selectedGamma
        )
        row['selected_for_evaluation'] = selected
        row['boundary_selection'] = bool(selected and boundary)
    return selectedGamma, rows, scores


def run_ablation(
        configuration, backend, gammaOverL, initialRows=None,
        checkpointCallback=None):
    rows = list(initialRows or ())
    allowedEvaluationKeys = {
        (evaluationSeed, dimension, condition, method)
        for evaluationSeed in configuration.ablationEvaluationSeeds
        for dimension in configuration.ablationDimensions
        for condition in configuration.ablationConditions
        for method in (LOCALISED, UNLOCALISED)
    }
    observedEvaluationKeys = {
        (
            int(row['evaluation_seed']),
            int(row['dimension']),
            row['surrogate_condition'],
            row['method'],
        )
        for row in rows
        if row['analysis'] == LOCALISATION_ABLATION
    }
    if len(observedEvaluationKeys) != len(rows) \
            or not observedEvaluationKeys.issubset(allowedEvaluationKeys):
        raise ValueError('Checkpoint contains unexpected evaluation rows.')
    expectedSeedRows = (
        len(configuration.ablationDimensions)
        * len(configuration.ablationConditions)
        * 2
    )
    for runIndex, evaluationSeed in enumerate(
            configuration.ablationEvaluationSeeds):
        existingRows = [
            row for row in rows
            if row['analysis'] == LOCALISATION_ABLATION
            and int(row['evaluation_seed']) == evaluationSeed
        ]
        expectedKeys = {
            (dimension, condition, method)
            for dimension in configuration.ablationDimensions
            for condition in configuration.ablationConditions
            for method in (LOCALISED, UNLOCALISED)
        }
        actualKeys = {
            (
                int(row['dimension']),
                row['surrogate_condition'],
                row['method'],
            )
            for row in existingRows
        }
        if len(existingRows) == expectedSeedRows \
                and actualKeys == expectedKeys:
            continue
        if existingRows:
            raise ValueError(
                f'Checkpoint evaluation seed {evaluationSeed} is '
                'partial.'
            )
        for dimension in configuration.ablationDimensions:
            seeds = ablation_seeds(evaluationSeed, dimension)
            for condition in configuration.ablationConditions:
                for method in (LOCALISED, UNLOCALISED):
                    print(  # noqa: T201
                        f'[{configuration.name}] ablation run={runIndex} '
                        f'd={dimension} condition={condition} method={method}'
                    )
                    try:
                        row = run_ablation_cell(
                            configuration,
                            backend,
                            LOCALISATION_ABLATION,
                            method,
                            runIndex,
                            evaluationSeed,
                            dimension,
                            condition,
                            gammaOverL,
                            configuration.ablationChains,
                            configuration.ablationBurnin,
                            configuration.ablationRetained,
                        )
                    except Exception as error:
                        row = failure_row(
                            configuration,
                            backend.name,
                            LOCALISATION_ABLATION,
                            method,
                            runIndex,
                            evaluationSeed,
                            seeds['dataset'],
                            seeds['start'],
                            seeds['chain'],
                            dimension,
                            condition,
                            configuration.ablationChains,
                            configuration.ablationBurnin,
                            configuration.ablationRetained,
                            error,
                            perturbationSeed=seeds['perturbation'],
                        )
                        row['gamma_over_l'] = (
                            gammaOverL if method == LOCALISED else 0.0
                        )
                    row['selected_for_evaluation'] = (
                        method == LOCALISED
                    )
                    row['boundary_selection'] = bool(
                        method == LOCALISED
                        and gammaOverL in (
                            GAMMA_CANDIDATES[0], GAMMA_CANDIDATES[-1]
                        )
                    )
                    rows.append(row)
        if checkpointCallback is not None:
            checkpointCallback(rows)
    return rows


def is_usable(row):
    return row['status'] == 'ok'


def checkpoint_json_ready(value):
    if isinstance(value, np.ndarray):
        return checkpoint_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return checkpoint_json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        marker = 'nan'
        if np.isposinf(value):
            marker = '+inf'
        elif np.isneginf(value):
            marker = '-inf'
        return {NONFINITE_FLOAT_KEY: marker}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): checkpoint_json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [checkpoint_json_ready(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f'Checkpoint value is not JSON serialisable: {type(value).__name__}'
    )


def checkpoint_json_restore(value):
    if isinstance(value, list):
        return [checkpoint_json_restore(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {NONFINITE_FLOAT_KEY}:
            markers = {'nan': np.nan, '+inf': np.inf, '-inf': -np.inf}
            return markers[value[NONFINITE_FLOAT_KEY]]
        return {
            key: checkpoint_json_restore(item)
            for key, item in value.items()
        }
    return value


def checkpoint_identity(configuration, backend, benchmarkSlug):
    return checkpoint_json_ready({
        'benchmark': benchmarkSlug,
        'configuration': asdict(configuration),
        'backend': backend.name,
    })


def write_checkpoint(path, identity, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'identity': identity,
        'payload': payload,
    }
    temporaryPath = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', newline='\n', delete=False,
                dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp') \
                as stream:
            temporaryPath = Path(stream.name)
            json.dump(
                checkpoint_json_ready(document), stream,
                allow_nan=False, indent=2, sort_keys=True,
            )
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporaryPath, path)
    except Exception:
        if temporaryPath is not None:
            temporaryPath.unlink(missing_ok=True)
        raise


def load_checkpoint(path, identity):
    path = Path(path)
    try:
        with path.open(encoding='utf-8') as stream:
            document = checkpoint_json_restore(json.load(stream))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f'Invalid checkpoint: {path}.') from error
    if document.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f'Unsupported checkpoint: {path}.')
    if document.get('identity') != checkpoint_json_ready(identity):
        raise RuntimeError(f'Checkpoint does not match this benchmark: {path}.')
    return document['payload']


def remove_checkpoint(path):
    Path(path).unlink(missing_ok=True)


def write_results(outputDirectory, benchmarkSlug, configuration, backend, rows):
    outputDirectory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = outputDirectory / (
        f'{benchmarkSlug}_{configuration.name}_{backend.name}_{timestamp}.csv'
    )
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
