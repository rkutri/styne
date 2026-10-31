"""Compare three adaptive MH proposals for a correlated binomial-logit model.

    y_i ~ Binomial(10, sigmoid(X_i @ beta)), beta ~ N(0, 6.25 I_3).

X has three covariates with population correlation 0.98 and unequal column
scales. All methods adapt a covariance multiplier toward acceptance 0.30.
Warm-up learns the proposals; fixed-proposal production draws determine ESS.
JAX uses compiled trajectories, with production compilation excluded from timing.

Run: python examples/05_adaptation_v2.py --backend numpy
Quick check: python examples/05_adaptation_v2.py --backend jax --smoke
"""
from time import perf_counter
from typing import Any, NamedTuple

import numpy as np
from scipy.special import expit

from styne.backend import infer_backend
from styne.mcmc import Adaptation, MetropolisedRandomWalk
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.mcmc.method.mrw import MRWProposal
from styne.model import LinearForwardMap
from styne.parameter import Vector
from styne.statistics import (
    BinomialResponse, Data, DiagonalCovarianceMatrix, Gaussian, IIDCovarianceMatrix,
    RegressionLikelihood, UnnormalisedPosterior, WelfordAccumulator,
)
from styne.utility import (
    FullCovarianceAdaptation, RobbinsMonroAdaptation, integrated_autocorrelation,
)


class DiagonalAdaptationState(NamedTuple):
    marginalVariance: Any
    moments: Any


class DiagonalCovarianceAdaptation(Adaptation):
    """Learn marginal variances from retained positions, including rejections."""

    def __init__(self, interval=25, nugget=1e-4):
        self._interval = interval
        self._nugget = nugget

    def initial_state(self, parameter, proposalMethod):
        backend = infer_backend(parameter.coordinate)
        metadata = backend.metadata(parameter.coordinate)
        covariance = proposalMethod.covariance
        variance = backend.asarray(covariance.scaling * covariance.marginalVariance,
                                   dtype=metadata.dtype, device=metadata.device)
        return DiagonalAdaptationState(
            variance, WelfordAccumulator.initial_record(parameter.coordinate),
        )

    def prepare_proposal(self, proposalMethod, state):
        return MRWProposal(DiagonalCovarianceMatrix(state.marginalVariance))

    def update_state(self, state, nextState, transitionData):
        moments = WelfordAccumulator.update_record(
            state.moments, nextState.parameter.coordinate,
        )
        backend = infer_backend(state.marginalVariance)
        metadata = backend.metadata(state.marginalVariance)
        count = backend.asarray(moments.nData, dtype=metadata.dtype,
                                device=metadata.device)
        variance = moments.m2 / backend.namespace.maximum(count - 1, 1) + self._nugget
        due = (moments.nData >= 2) & (moments.nData % self._interval == 0)
        variance = backend.namespace.where(due, variance, state.marginalVariance)
        return DiagonalAdaptationState(variance, moments)


class CovarianceScaleState(NamedTuple):
    shapeState: Any
    scaleState: Any


class CovarianceScaleAdaptation(Adaptation):
    """Combine periodic covariance learning with per-step Robbins-Monro scaling.

    Start from an IID proposal. The scale multiplies its covariance shape.
    Full covariance learning retains its Cholesky factor between updates.
    """

    def __init__(self, covariance, interval=25, nugget=1e-4,
                 targetAcceptance=0.30):
        if covariance == 'diagonal':
            self._shapeRule = DiagonalCovarianceAdaptation(interval, nugget)
        elif covariance == 'full':
            self._shapeRule = FullCovarianceAdaptation(interval, nugget)
        else:
            raise ValueError("covariance must be 'diagonal' or 'full'.")
        self._scaleRule = RobbinsMonroAdaptation(targetAcceptance=targetAcceptance)

    def initial_state(self, parameter, proposalMethod):
        backend = infer_backend(parameter.coordinate)
        metadata = backend.metadata(parameter.coordinate)
        # Unit shape separates the initial proposal variance from its geometry.
        unit = backend.asarray(1.0, dtype=metadata.dtype, device=metadata.device)
        shapeTemplate = MRWProposal(IIDCovarianceMatrix(parameter.dimension, unit))
        return CovarianceScaleState(
            self._shapeRule.initial_state(parameter, shapeTemplate),
            self._scaleRule.initial_state(parameter, proposalMethod),
        )

    def prepare_proposal(self, proposalMethod, state):
        shape = self._shapeRule.prepare_proposal(proposalMethod, state.shapeState)
        scale = infer_backend(state.scaleState.logScale).namespace.exp(
            state.scaleState.logScale,
        )
        return MRWProposal(shape.covariance.with_scaling(scale))

    def update_state(self, state, nextState, transitionData):
        return CovarianceScaleState(
            self._shapeRule.update_state(state.shapeState, nextState, transitionData),
            self._scaleRule.update_state(state.scaleState, nextState, transitionData),
        )


cif __name__ == '__main__':
    if __package__:
        from .runtime import as_numpy, configure_backend, parse_arguments
    else:
        from runtime import as_numpy, configure_backend, parse_arguments

    arguments = parse_arguments(__doc__)
    backend, _ = configure_backend(arguments.backend)
    nWarmup, nSamples = (200, 400) if arguments.smoke else (5000, 20000)

    # Generate one data set shared by all three samplers.
    rng = np.random.default_rng(2026)
    nObservations, dimension, correlation, nTrials = 400, 3, 0.98, 10
    columnScales = np.array([1.0, 2.0, 0.5])
    trueBeta = np.array([1.0, -0.6, 0.8])
    common = rng.normal(size=(nObservations, 1))
    independent = rng.normal(size=(nObservations, dimension))
    features = (np.sqrt(correlation) * common
                + np.sqrt(1 - correlation) * independent) * columnScales
    features -= features.mean(axis=0)
    features = features.astype('float32')
    successes = rng.binomial(nTrials, expit(features @ trueBeta))

    nativeFeatures = backend.asarray(features, dtype='float32')
    data = Data(nObservations, nativeFeatures)
    data.measurement = backend.asarray(successes, dtype='float32')
    likelihood = RegressionLikelihood(data, LinearForwardMap(nativeFeatures),
                                      BinomialResponse(nTrials))
    prior = Gaussian(IIDCovarianceMatrix(dimension, backend.asarray(6.25)),
                     Vector(backend.zeros(dimension, dtype='float32')))
    target = UnnormalisedPosterior(prior, likelihood)

    print(f'Binomial-logit: n={nObservations}, trials={nTrials}, d={dimension}')
    print(f'True coefficients: {trueBeta}')
    print('Observed predictor correlations:')
    print(np.array2string(np.corrcoef(features, rowvar=False), precision=3))
    print(f'Adaptive warm-up: {nWarmup}; fixed-proposal production: {nSamples}')
    print('Timing excludes adaptation and JAX production compilation.')
    if arguments.smoke:
        print('Smoke run: short chains check execution, not comparative efficiency.')

    methods = [
        ('MH + scale', RobbinsMonroAdaptation(targetAcceptance=0.30)),
        ('MH + diagonal + scale', CovarianceScaleAdaptation('diagonal')),
        ('MH + full + scale', CovarianceScaleAdaptation('full')),
    ]
    results = {}
    for index, (name, adaptation) in enumerate(methods):
        seed = 31 + index
        start = Vector(backend.zeros(dimension, dtype='float32'))
        sampler = MetropolisedRandomWalk(
            target, IIDCovarianceMatrix(dimension, backend.asarray(0.01)),
            AcceptanceRateDiagnostics(), rng=backend.random_state(seed),
            adaptation=adaptation,
        )

        # Freeze after warm-up so ESS describes a fixed proposal kernel.
        if backend.capabilities.transformedLoops:
            warm = sampler.compiled_trajectory(nWarmup)(start, backend.random_state(seed))
            as_numpy(warm.final.coordinate)
            fixed = sampler.freeze_adaptation(warm.finalState, warm.rng)
            execute = fixed.compiled_trajectory(nSamples)
            initial = fixed.initial_state(warm.final)
            # Compile and synchronise, then time the same pure inputs and draws.
            untimed = execute(initial, warm.rng)
            as_numpy(untimed.coordinates)
            began = perf_counter()
            production = execute(initial, warm.rng)
            samples = as_numpy(production.coordinates)
            seconds = perf_counter() - began
            acceptance = float(as_numpy(production.sums['accepted'])) / nSamples
        else:
            sampler.run(nWarmup, start)
            fixed = sampler.freeze_adaptation()
            began = perf_counter()
            fixed.run(nSamples, sampler.lastState)
            samples = as_numpy(backend.namespace.stack(fixed.chain.trajectory))[1:]
            seconds = perf_counter() - began
            acceptance = fixed.diagnostics.global_acceptance_rate()

        ess = np.array([nSamples / integrated_autocorrelation(samples[:, coordinate])
                        for coordinate in range(dimension)])
        factor = as_numpy(fixed.proposalCovariance.to_cholesky())
        results[name] = {
            'acceptance': acceptance,
            'ess': ess,
            'seconds': seconds,
            'posteriorMean': samples.mean(axis=0),
            'proposalCovariance': factor @ factor.T,
        }

    # ESS is coordinate-wise; ESS/s uses the smallest coordinate ESS.
    print(f'{"Method":<25} {"Accept":>7} {"ESS beta1":>10} {"ESS beta2":>10} '
          f'{"ESS beta3":>10} {"Min ESS":>9} {"Seconds":>9} {"Min ESS/s":>10}')
    for name, result in results.items():
        ess = result['ess']
        acceptance = result['acceptance']
        seconds = result['seconds']
        minimum = np.min(ess)
        print(f'{name:<25} {acceptance:7.3f} '
              f'{ess[0]:10.1f} {ess[1]:10.1f} {ess[2]:10.1f} '
              f'{minimum:9.1f} {seconds:9.3f} {minimum / seconds:10.1f}')

    print('Posterior means:')
    for name, result in results.items():
        posteriorMean = np.array2string(result['posteriorMean'], precision=3)
        print(f'{name:<25} {posteriorMean}')
