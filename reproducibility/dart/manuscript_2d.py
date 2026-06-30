"""Hierarchical SGLMM DART demonstration."""

from dataclasses import dataclass
from math import ceil
import os
from contextlib import nullcontext
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

import numpy as np
import scipy.stats.qmc as qmc
from scipy.fft import dctn, idctn
from numpy.random import default_rng
from pathlib import Path

from manuscript_boilerplate import hasMatplotlib, hasJoblib, plt, joblib

import logging
from styne import enable_logging

from styne.gp.gaussianprocess import GaussianProcess
from styne.gp.dnautility import DNACoarseFinePartition
from styne.mcmc.diagnostics import PersistentAcceptanceRateDiagnostics
from styne.mcmc.method.gibbs import GibbsBuilder
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.mrw import RobbinsMonroMRWFactory
from styne.model.sglmm import SGLMM
from styne.model.trend import ConstantTrend
from styne.parameter.block import BlockParameter
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.bayes import HierarchicalBayes
from styne.statistics.conditional import MetropolisWithinGibbsConditional
from styne.statistics.data import Data
from styne.statistics.hierarchical import (
    SGLMMHyperConditionalDensity, SGLMMLatentConditional
)
from styne.statistics.likelihood import SGLMMLikelihood
from styne.statistics.pc import JointMaternPCPrior
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.response import PoissonResponse
from styne.statistics.stationary import MaternCovariance2D
from styne.utility.grid import Grid, UniformGrid


DEBUG = False
CLAMP_HYPER = False

GAMMA = 5e-3
THETA = 0.8
COARSE_ROOT = 'mala'
FINE_PCN_STEP = 0.1

CAPTURE_THRESHOLD = 0.95
MIN_LATENT_ACCEPTANCE = 0.005
COVERAGE_SPREAD_TOLERANCE = 0.05

NUM_SEEDS = 3
MASTER_SEED = 2026
GROUND_TRUTH_SEED = 42
TARGET_BAND_SAMPLES = 2000
CREDIBLE_QUANTILES = (0.025, 0.975)


@dataclass
class Smoothness:
    key: str
    nu: float
    rho: float
    sigmaSq: float = 0.75
    rhoInit: float = 0.12
    sigmaInit: float = 0.85


SMOOTHNESSES = [
    Smoothness(key='smooth', nu=2.5, rho=0.10, sigmaSq=0.75,
               rhoInit=0.12, sigmaInit=0.85),
    Smoothness(key='rough', nu=1.5, rho=0.05, sigmaSq=0.75,
               rhoInit=0.06, sigmaInit=0.85),
]


@dataclass
class Config:
    nObservations: int = 1024
    trendOffset: float = 0.5
    groundTruthResolution: int = 500
    predictionResolution: int = 100
    captureGridResolution: int = 256
    nSteps: int = 1_100_000
    nBurnIn: int = 100_000
    nSubchain: int = 6
    surrogateCrankUp: int = 10_000
    nyquistFactor: float = 2.0
    coarseResolutionMin: int = 5
    coarseResolutionMax: int = 120
    hyperparameterMarginalVariance: float = 4.0

    @staticmethod
    def build():
        config = Config()
        if DEBUG:
            config.nSteps = 10000
            config.nBurnIn = 2000
        return config


def compute_fine_resolution(nObservations, rho, config):

    nyquist = ceil(config.nyquistFactor * np.sqrt(nObservations))
    covarianceFloor = ceil(1.0 / rho)

    return max(nyquist, covarianceFloor)


def build_ground_truth(smoothness, config):

    rng = default_rng(GROUND_TRUTH_SEED)

    nObservations = config.nObservations

    sobol = qmc.Sobol(d=2, scramble=True, seed=rng)
    coordinates = sobol.random(nObservations)

    truthGP = GaussianProcess.dna(
        MaternCovariance2D(smoothness.rho, smoothness.nu, smoothness.sigmaSq),
        q=config.groundTruthResolution, d=2
    )
    realisation = truthGP.sampler.generate_realisation(rng=rng)

    truthGP.sites = Grid(coordinates)
    truthGP.parameter.coordinate = realisation.coordinate

    etaSites = truthGP.at_sites() + config.trendOffset
    counts = PoissonResponse().simulate(etaSites, rng=rng).coordinate

    resolution = config.predictionResolution

    truthGP.sites = UniformGrid((0., 1., resolution), (0., 1., resolution))
    truthGP.parameter.coordinate = realisation.coordinate
    truthField = (truthGP.at_sites() + config.trendOffset).reshape(
        resolution, resolution)

    captureResolution = config.captureGridResolution

    truthGP.sites = UniformGrid(
        (0., 1., captureResolution), (0., 1., captureResolution))
    truthGP.parameter.coordinate = realisation.coordinate

    captureField = (truthGP.at_sites() + config.trendOffset).reshape(
        captureResolution, captureResolution)

    return coordinates, counts, truthField, captureField

def measure_coarse_resolution(smoothness, captureField, config):
    """
        measure fraction of total L2 energy captured by the coarse
        resolution surrogate. Based on a simple Parseval identity
        argument.
    """

    energy = dctn(captureField, norm='ortho') ** 2
    energy[0, 0] = 0.0
    totalEnergy = float(energy.sum())

    if totalEnergy <= 0:
        raise AssertionError(f'[{smoothness.key}] degenerate truth')

    cumulative = energy.cumsum(axis=0).cumsum(axis=1)

    chosen, capture = None, float('nan')
    limit = captureField.shape[0] - 1
    for coarseResolution in range(
            config.coarseResolutionMin, config.coarseResolutionMax + 1):

        index = min(coarseResolution, limit)
        capture = float(cumulative[index, index]) / totalEnergy

        if capture >= CAPTURE_THRESHOLD:
            chosen = coarseResolution
            break

    if chosen is None:

        chosen = config.coarseResolutionMax
        print(f'[{smoothness.key}] WARNING: threshold {CAPTURE_THRESHOLD:.2f} '
              f'unmet by coarse resolution {chosen} (capture {capture:.3f}).')

    residual = float(np.sqrt(max(0.0, 1.0 - capture)))
    print(f'[{smoothness.key}] measured coarse resolution {chosen} '
          f'(capture {capture:.3f} of L2 energy; relative residual {residual:.3f}).')

    return chosen, capture


def run_demonstration(smoothness, seedIndex, coarseResolution, coordinates,
                      counts, config, resultsDirectory, gamma=GAMMA,
                      theta=THETA, innerThreads=1, writeRaw=True,
                      clampHyper=False):

    limiter = (threadpool_limits(innerThreads)
               if threadpool_limits is not None else nullcontext())
    fineResolution = compute_fine_resolution(
        config.nObservations, smoothness.rho, config)

    sites = Grid(coordinates)
    data = Data(1, sites.to_array())
    data.measurement = counts.reshape(-1, 1)

    smoothnessIndex = next(
        index for index, candidate in enumerate(SMOOTHNESSES)
        if candidate.key == smoothness.key)
    sequence = np.random.SeedSequence(
        MASTER_SEED,
        spawn_key=(smoothnessIndex, seedIndex, coarseResolution,
                   config.nObservations))
    children = sequence.spawn(2)
    initRNG = default_rng(children[0])
    kernelRNG = default_rng(children[1])

    with limiter:

        if clampHyper:
            constructionRho = smoothness.rho
            constructionSigma = float(np.sqrt(smoothness.sigmaSq))

        else:
            constructionRho = smoothness.rhoInit
            constructionSigma = smoothness.sigmaInit

        gp = GaussianProcess.dna(
            MaternCovariance2D(constructionRho, smoothness.nu,
                               constructionSigma ** 2),
            q=fineResolution, d=2)

        predictor = SGLMM(gp, sites, trend=ConstantTrend(config.trendOffset))
        likelihood = SGLMMLikelihood(data, predictor, PoissonResponse())

        latentTarget = RadonNikodym(gp.measure, likelihood)
        pcPrior = JointMaternPCPrior(
            rho0=0.0258, alphaRho=0.15, sigma0=1.5, alphaSigma=0.15)

        coarseGP = GaussianProcess.dna(
            gp.covarianceFunction, q=coarseResolution, d=2)
        coarsePredictor = SGLMM(
            coarseGP, sites, trend=ConstantTrend(config.trendOffset))
        coarseSurrogate = RadonNikodym(
            coarseGP.measure,
            SGLMMLikelihood(data, coarsePredictor, PoissonResponse()))

        partition = DNACoarseFinePartition(gp, coarseResolution, d=2)
        finePrior = partition.fine_measure()

        latentInit = gp.sampler.generate_realisation(rng=initRNG)

        factory = DARTFactory(COARSE_ROOT)
        factory.rng = kernelRNG
        factory.target = latentTarget
        factory.surrogate = [coarseSurrogate]
        factory.partition = partition
        factory.finePrior = finePrior
        factory.finePCNStep = FINE_PCN_STEP
        factory.regularisation = [gamma]
        factory.tempering = [theta]
        factory.nChain = [config.nSubchain]
        factory.crankUpSteps = config.surrogateCrankUp
        factory.spectralWeights = coarseGP.engine.spectralWeights
        factory.crankUpInitialState = latentInit
        factory.subDiagnostics = PersistentAcceptanceRateDiagnostics

        latentMCMC = factory.create()
        localisedDensity = factory.localisedDensity
        latentInit = factory.crankedState

        latentConditional = SGLMMLatentConditional(
            latentTarget, gp, coarseGP=coarseGP, partition=partition,
            finePrior=finePrior, localisedDensity=localisedDensity)
        latentMCMC.target = latentConditional
        latentTransition = MetropolisWithinGibbsConditional(
            latentMCMC, blockIdx=0, nSteps=1)

        hyperTarget = SGLMMHyperConditionalDensity(
            pcPrior, gp, predictor, likelihood, None)
        hyperInit = Vector(np.log([constructionRho, constructionSigma]))
        jointState = BlockParameter([latentInit.clone(), hyperInit.clone()])
        hyperTarget.condition_on(jointState)
        hyperFactory = RobbinsMonroMRWFactory()
        hyperFactory.rng = kernelRNG
        hyperFactory.target = hyperTarget
        hyperFactory.proposalCovariance = IIDCovarianceMatrix(
            2, config.hyperparameterMarginalVariance)
        hyperFactory.adaptDecay = 0.6
        hyperFactory.adaptOffset = 150
        hyperMCMC = hyperFactory.create()
        hyperTransition = MetropolisWithinGibbsConditional(
            hyperMCMC, blockIdx=1, nSteps=1)

        transitions = ([latentTransition] if clampHyper
                       else [latentTransition, hyperTransition])
        hierarchicalBayes = HierarchicalBayes(transitions, root=gp.measure)

        builder = GibbsBuilder()
        builder.model = hierarchicalBayes
        builder.rng = kernelRNG

        sampler = builder.build()
        sampler.storeChain = False

        resolution = config.predictionResolution
        fidelityPredictor = predictor.create_predictor(
            UniformGrid((0., 1., resolution), (0., 1., resolution)))

        thinningStep = max(
            1, (config.nSteps - config.nBurnIn) // TARGET_BAND_SAMPLES)
        fieldSamples = []
        hyperTrajectory = []

        for step, state in enumerate(sampler.stream_run(
                config.nSteps, jointState, progress=True,
                description=f'  {smoothness.key} seed {seedIndex}')):

            hyperTrajectory.append(state.block(1).coordinate.copy())

            if step >= config.nBurnIn and (
                    step - config.nBurnIn) % thinningStep == 0:

                predictor.reset()
                predictor.interpolate(state.block(0))
                fieldSamples.append(np.asarray(fidelityPredictor.mean()))

        latentAcceptance = latentMCMC.diagnostics.global_acceptance_rate()
        hyperAcceptance = (
            float('nan') if clampHyper
            else hyperMCMC.diagnostics.global_acceptance_rate())
        coarseAcceptance = (
            latentMCMC.subsamplers[0].diagnostics.global_acceptance_rate())

    fieldSamples = np.stack(fieldSamples, axis=0)
    hyper = np.exp(np.array(hyperTrajectory)[config.nBurnIn:])

    if writeRaw:

        np.savez_compressed(
            resultsDirectory / f'{smoothness.key}_seed{seedIndex}_raw.npz',
            fieldSamples=fieldSamples, hyper=hyper,
            coarseResolution=coarseResolution, fineResolution=fineResolution,
            latentAcceptance=latentAcceptance, hyperAcceptance=hyperAcceptance,
            coarseAcceptance=coarseAcceptance)

    return dict(
        smoothness=smoothness.key, seedIndex=seedIndex,
        coarseResolution=coarseResolution, fineResolution=fineResolution,
        latentAcceptance=latentAcceptance, hyperAcceptance=hyperAcceptance,
        coarseAcceptance=coarseAcceptance)


def _credible_band(samples):
    lower, upper = np.quantile(samples, CREDIBLE_QUANTILES, axis=0)
    return lower, upper


def pointwise_coverage(truthField, lower, upper):
    return float(np.mean((truthField >= lower) & (truthField <= upper)))


def aggregate_seeds(smoothness, groundTruth, coarseResolution, resultsDirectory):

    resolution = int(np.sqrt(groundTruth['truthField'].size))
    truthField = groundTruth['truthField']

    perSeedMeans, perSeedCoverage, hyperSamples, pooled = [], [], [], []

    latentAcceptances, coarseAcceptances, hyperAcceptances = [], [], []

    for seedIndex in range(NUM_SEEDS):

        raw = np.load(
            resultsDirectory / f'{smoothness.key}_seed{seedIndex}_raw.npz')

        samples = raw['fieldSamples']

        pooled.append(samples)

        seedMean = samples.mean(axis=0).reshape(resolution, resolution)
        seedStd = samples.std(axis=0).reshape(resolution, resolution)
        perSeedMeans.append(seedMean)
        lower, upper = _credible_band(samples)

        perSeedCoverage.append(pointwise_coverage(
            truthField, lower.reshape(resolution, resolution),
            upper.reshape(resolution, resolution)))

        hyperSamples.append(raw['hyper'])

        latentAcceptances.append(float(raw['latentAcceptance']))
        coarseAcceptances.append(float(raw['coarseAcceptance']))
        hyperAcceptances.append(float(raw['hyperAcceptance']))

    pooled = np.concatenate(pooled, axis=0)
    postMean = pooled.mean(axis=0).reshape(resolution, resolution)
    postStd = pooled.std(axis=0).reshape(resolution, resolution)

    lowerPooled, upperPooled = _credible_band(pooled)
    lower = lowerPooled.reshape(resolution, resolution)
    upper = upperPooled.reshape(resolution, resolution)

    coveragePooled = pointwise_coverage(truthField, lower, upper)

    del pooled

    # postStd is pooled across seeds
    absError = np.abs(postMean - truthField)

    perSeedCoverage = np.array(perSeedCoverage)

    meanStack = np.stack(perSeedMeans, axis=0)

    return dict(
        nu=groundTruth['nu'], rho=groundTruth['rho'],
        sigmaSq=groundTruth['sigmaSq'], N=groundTruth['N'],
        q=groundTruth['q'], coarseResolution=coarseResolution,
        capture=groundTruth['capture'], truthField=truthField,
        postMean=postMean, postStd=postStd, lower=lower, upper=upper,
        coveragePooled=coveragePooled, perSeedCoverage=perSeedCoverage,
        coverageAcrossSeedSpread=float(perSeedCoverage.std()),
        absError=absError,
        meanSpreadPerSite=float(meanStack.std(axis=0).mean()),
        hyper=np.concatenate(hyperSamples, axis=0),
        latentAcceptances=latentAcceptances,
        coarseAcceptances=coarseAcceptances,
        hyperAcceptances=hyperAcceptances,
        gamma=GAMMA, theta=THETA, beta=FINE_PCN_STEP)


def main():

    enable_logging(logging.INFO)

    config = Config.build()
    if CLAMP_HYPER:
        print('[DIAGNOSTIC] hyperparameters clamped at truth; latent block only.')

    FORCE = True
    REAGGREGATE = True

    resultsDirectory = Path(__file__).parent / 'results'
    resultsDirectory.mkdir(parents=True, exist_ok=True)
    figuresDirectory = Path(__file__).parent.parent / 'figures'
    figuresDirectory.mkdir(parents=True, exist_ok=True)
    cachePath = (Path(__file__).parent / 'joblib_caches'
                 / f'{Path(__file__).stem}.joblib')
    if hasJoblib:
        cachePath.parent.mkdir(parents=True, exist_ok=True)

    if hasJoblib and cachePath.exists() and not FORCE and not REAGGREGATE:
        perSmoothness = joblib.load(cachePath)['perSmoothness']
        print('Loaded aggregated summaries from cache.')
    else:
        groundTruths, chosenResolution = {}, {}
        for smoothness in SMOOTHNESSES:
            coordinates, counts, truthField, captureField = build_ground_truth(
                smoothness, config)
            coarseResolution, capture = measure_coarse_resolution(
                smoothness, captureField, config)
            fineResolution = compute_fine_resolution(
                config.nObservations, smoothness.rho, config)
            print(f'[{smoothness.key}] nu={smoothness.nu}, rho={smoothness.rho}, '
                  f'sigma^2={smoothness.sigmaSq}, N={config.nObservations}, '
                  f'q={fineResolution}, coarse={coarseResolution}.')
            groundTruths[smoothness.key] = dict(
                coordinates=coordinates, counts=counts, truthField=truthField,
                capture=capture, nu=smoothness.nu, rho=smoothness.rho,
                sigmaSq=smoothness.sigmaSq, N=config.nObservations,
                q=fineResolution)
            chosenResolution[smoothness.key] = coarseResolution

        if FORCE:
            if not hasJoblib:
                raise RuntimeError('joblib required for the parallel driver')
            innerThreads = max(
                1, (os.cpu_count() or 1) // (NUM_SEEDS * len(SMOOTHNESSES)))
            cells = [(smoothness, seedIndex) for smoothness in SMOOTHNESSES
                     for seedIndex in range(NUM_SEEDS)]
            joblib.Parallel(n_jobs=len(cells))(
                joblib.delayed(run_demonstration)(
                    smoothness, seedIndex, chosenResolution[smoothness.key],
                    groundTruths[smoothness.key]['coordinates'],
                    groundTruths[smoothness.key]['counts'], config,
                    resultsDirectory, innerThreads=innerThreads,
                    clampHyper=CLAMP_HYPER)
                for smoothness, seedIndex in cells)

        perSmoothness = {}
        for smoothness in SMOOTHNESSES:
            summary = aggregate_seeds(
                smoothness, groundTruths[smoothness.key],
                chosenResolution[smoothness.key], resultsDirectory)
            perSmoothness[smoothness.key] = summary
            minimumAcceptance = min(summary['latentAcceptances'])
            if minimumAcceptance < MIN_LATENT_ACCEPTANCE:
                print(f'[{smoothness.key}] WARNING: latent acceptance '
                      f'{minimumAcceptance:.3f} below {MIN_LATENT_ACCEPTANCE}; '
                      f'the chain is not mixing.')
            if summary['coverageAcrossSeedSpread'] > COVERAGE_SPREAD_TOLERANCE:
                print(f'[{smoothness.key}] WARNING: coverage spread '
                      f'{summary["coverageAcrossSeedSpread"]:.3f} across seeds; '
                      f'not converged.')

        if hasJoblib:
            joblib.dump({'perSmoothness': perSmoothness}, cachePath, compress=3)

    if hasMatplotlib:
        plot_fidelity(perSmoothness, figuresDirectory)
        plot_hyperparameter_posteriors(perSmoothness, figuresDirectory)
    else:
        print('Warning: matplotlib not installed; skipping plots.')
    print_summary(perSmoothness)


def _style_axes(axis):
    axis.axis('off')


def _load_style():
    try:
        plt.style.use(Path(__file__).parent / 'manuscript.mplstyle')
    except Exception:
        pass


def plot_fidelity(perSmoothness, figuresDirectory):
    import matplotlib.colors as mcolors

    plt.close('all')
    _load_style()
    keys = [smoothness.key for smoothness in SMOOTHNESSES]
    resolution = perSmoothness[keys[0]]['truthField'].shape[0]
    grid = UniformGrid((0., 1., resolution), (0., 1., resolution))
    gridX, gridY = np.meshgrid(grid.xAxis, grid.yAxis, indexing='ij')

    figure, axes = plt.subplots(len(keys), 3, figsize=(16, 4.5 * len(keys)))
    if len(keys) == 1:
        axes = axes[None, :]
    for row, key in enumerate(keys):
        summary = perSmoothness[key]
        truthField = summary['truthField']
        postMean = summary['postMean']
        postStd = summary['postStd']
        shared = np.linspace(
            min(truthField.min(), postMean.min()),
            max(truthField.max(), postMean.max()), 25)
        stdFloor = np.floor(float(postStd.min()) * 10) / 10
        stdMin = stdFloor if stdFloor > 0 else round(float(postStd.min()), 2)
        stdMax = max(stdMin + 0.4, np.ceil(float(postStd.max()) * 10) / 10)

        rawTicks = np.logspace(np.log10(stdMin), np.log10(stdMax), 5)
        stdTicks = []
        for v in rawTicks:
            rounded = round(float(v), 1)
            if stdTicks and rounded <= stdTicks[-1]:
                rounded = stdTicks[-1] + 0.1
            stdTicks.append(rounded)
        stdMax = max(stdMax, stdTicks[-1])

        uncertaintyLevels = np.logspace(np.log10(stdMin), np.log10(stdMax), 25)

        imageTruth = axes[row, 0].contourf(
            gridX, gridY, truthField, levels=shared, cmap='RdBu_r')
        linearTicks = np.linspace(shared[0], shared[-1], 5)
        cbarTruth = plt.colorbar(imageTruth, ax=axes[row, 0], ticks=linearTicks)
        cbarTruth.ax.set_yticklabels([f'{val:.1f}' for val in linearTicks])
        axes[row, 0].set_title(
            rf'Ground truth $\eta$ ({key})'
        )
        imageMean = axes[row, 1].contourf(
            gridX, gridY, postMean, levels=shared, cmap='RdBu_r')
        cbarMean = plt.colorbar(imageMean, ax=axes[row, 1], ticks=linearTicks)
        cbarMean.ax.set_yticklabels([f'{val:.1f}' for val in linearTicks])
        axes[row, 1].set_title(
            rf'Posterior mean')
        imageStd = axes[row, 2].contourf(
            gridX, gridY, np.clip(postStd, stdMin, stdMax),
            levels=uncertaintyLevels,
            norm=mcolors.LogNorm(vmin=stdMin, vmax=stdMax),
            cmap='viridis')
        cbarStd = plt.colorbar(imageStd, ax=axes[row, 2], ticks=stdTicks)
        cbarStd.ax.set_yticklabels([f'{val:.1f}' for val in stdTicks])
        axes[row, 2].set_title('Posterior std. dev.')

        for column in range(3):
            _style_axes(axes[row, column])
    plt.tight_layout()
    plt.savefig(
        figuresDirectory / 'manuscript_2d_fidelity.pdf', bbox_inches='tight')
    print(f'[PLOT] fidelity -> '
          f'{figuresDirectory / "manuscript_2d_fidelity.pdf"}')


def plot_hyperparameter_posteriors(perSmoothness, figuresDirectory):
    plt.close('all')
    _load_style()
    keys = [smoothness.key for smoothness in SMOOTHNESSES]
    figure, axes = plt.subplots(len(keys), 2, figsize=(11, 3.6 * len(keys)))
    if len(keys) == 1:
        axes = axes[None, :]
    for row, key in enumerate(keys):
        summary = perSmoothness[key]
        rhoSamples = summary['hyper'][:, 0]
        sigmaSamples = summary['hyper'][:, 1]
        axes[row, 0].hist(
            rhoSamples, bins=60, density=True, color='steelblue', alpha=0.7)
        axes[row, 0].axvline(
            summary['rho'], color='black', linestyle='--', lw=1.2)
        axes[row, 0].set_title(rf'$\rho$ posterior ({key})')
        axes[row, 0].set_xlabel(r'$\rho$')
        axes[row, 1].hist(
            sigmaSamples, bins=60, density=True, color='indianred', alpha=0.7)
        axes[row, 1].axvline(
            np.sqrt(summary['sigmaSq']), color='black', linestyle='--', lw=1.2)
        axes[row, 1].set_title(rf'$\sigma$ posterior ({key})')
        axes[row, 1].set_xlabel(r'$\sigma$')
        for column in range(2):
            axes[row, column].spines['top'].set_visible(False)
            axes[row, column].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        figuresDirectory / 'manuscript_2d_hyper.pdf', bbox_inches='tight')
    print(f'[PLOT] hyper -> {figuresDirectory / "manuscript_2d_hyper.pdf"}')


def print_summary(perSmoothness):
    print(f"\n{'=' * 97}")
    print(f"{'Section 5.2 demonstration summary':^97}")
    print(f"{'=' * 97}")
    print(f"{'smooth':>8} | {'nu':>4} | {'rho':>5} | {'N':>6} | {'q':>4} "
          f"| {'coarse':>6} | {'capture':>7} | {'cov':>6} | {'cov sd':>7} "
          f"| {'gamma':>8} | {'theta':>6} | {'beta':>6}")
    print(f"{'-' * 97}")
    for key, summary in perSmoothness.items():
        print(f"{key:>8} | {summary['nu']:>4} | {summary['rho']:>5} "
              f"| {summary['N']:>6} | {summary['q']:>4} "
              f"| {summary['coarseResolution']:>6} | {summary['capture']:>7.3f} "
              f"| {summary['coveragePooled']:>6.3f} "
              f"| {summary['coverageAcrossSeedSpread']:>7.4f} "
              f"| {summary['gamma']:>8.4g} | {summary['theta']:>6.2f} "
              f"| {summary['beta']:>6.2f}")
    print(f"{'=' * 97}")
    for key, summary in perSmoothness.items():
        latentAcceptance = ', '.join(
            f'{value:.3f}' for value in summary['latentAcceptances'])
        coarseAcceptance = ', '.join(
            f'{value:.3f}' for value in summary['coarseAcceptances'])
        hyperAcceptance = ', '.join(
            f'{value:.3f}' for value in summary['hyperAcceptances'])
        rhoMean = float(summary['hyper'][:, 0].mean())
        sigmaMean = float(summary['hyper'][:, 1].mean())
        print(f"[{key}] latent acc=[{latentAcceptance}]; "
              f"coarse acc=[{coarseAcceptance}]; hyper acc=[{hyperAcceptance}].")
        print(f"[{key}] hyper post mean: rho={rhoMean:.3f} "
              f"(truth {summary['rho']:.3f}), sigma={sigmaMean:.3f} "
              f"(truth {np.sqrt(summary['sigmaSq']):.3f}); "
              f"mean spread={summary['meanSpreadPerSite']:.4f}.")
    print(f"{'=' * 82}\n")


if __name__ == '__main__':
    main()
