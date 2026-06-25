#!/usr/bin/env python3
import csv
import itertools
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats.qmc as qmc
from numpy.random import default_rng

from manuscript_boilerplate import hasMatplotlib, plt

from styne import enable_logging
import logging
from styne.gp.dnautility import DNACoarseFinePartition
from styne.gp.gaussianprocess import GaussianProcess
from styne.mcmc.diagnostics import PersistentAcceptanceRateDiagnostics
from styne.mcmc.method.gibbs import GibbsBuilder
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.mlda import MLDAFactory
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
from styne.utility.grid import Grid
from styne.utility.postprocessing import integrated_autocorrelation


@dataclass
class Config:
    nObservationCounts = [40, 160, 640, 2560]
    ellTrue: float = 0.1
    sigmaTrue: float = np.sqrt(0.75)
    smoothness: float = 2.5
    trendOffset: float = 0.5

    groundTruthResolution: int = 500
    predictionResolution: int = 100

    nSteps: int = 20_000
    nBurnIn: int = 5_000
    nSubchain: int = 50

    surrogateCrankUp: int = 10_000
    coarseRoot: str = 'pcn'

    ellInit: float = 0.2
    sigmaInit: float = 0.5

    nyquistFactor: float = 2.0
    coarseResolutionMultiplier: float = 5.0
    coarseResolutionExponent: float = 0.25


def compute_fine_resolution(nObservations: int, config: Config) -> int:
    nyquist = ceil(config.nyquistFactor * np.sqrt(nObservations))
    covarianceFloor = ceil(2. / config.ellTrue)
    return max(nyquist, covarianceFloor)


def compute_coarse_resolution(fineResolution: int, config: Config) -> int:
    multiplier = config.coarseResolutionMultiplier
    exponent = config.coarseResolutionExponent
    return max(int(multiplier * (fineResolution ** exponent)), 5)


def run_method(
    config: Config, method: str, data: Data, sites: Grid,
    randomGenerator: np.random.Generator,
    fineResolution: int, coarseResolution: int,
    gamma: Optional[float], tempering: Optional[float]
) -> dict:
    initCov = MaternCovariance2D(
        config.ellInit, config.smoothness, config.sigmaInit**2
    )
    gp = GaussianProcess.dna(initCov, q=fineResolution, d=2)

    predictor = SGLMM(gp, sites, trend=ConstantTrend(config.trendOffset))
    likelihood = SGLMMLikelihood(data, predictor, PoissonResponse())
    latentTarget = RadonNikodym(gp.measure, likelihood)
    pcPrior = JointMaternPCPrior(
        rho0=0.05, alphaRho=0.1, sigma0=1.5, alphaSigma=0.1
    )

    coarseGP = GaussianProcess.dna(
        gp.covarianceFunction, q=coarseResolution, d=2
    )
    coarsePredictor = SGLMM(
        coarseGP, sites, trend=ConstantTrend(config.trendOffset)
    )
    coarseSurrogate = RadonNikodym(
        coarseGP.measure,
        SGLMMLikelihood(data, coarsePredictor, PoissonResponse())
    )
    partition = DNACoarseFinePartition(gp, coarseResolution, d=2)
    finePrior = partition.fine_measure()

    latentInit = gp.sampler.generate_realisation(rng=randomGenerator)
    localisedDensity = None

    if method == 'DART':
        factory = DARTFactory(config.coarseRoot)
        factory.target = latentTarget
        factory.surrogate = [coarseSurrogate]
        factory.partition = partition
        factory.finePrior = finePrior
        factory.regularisation = [gamma]
        factory.tempering = [tempering]
        factory.nChain = [config.nSubchain]
        factory.crankUpSteps = config.surrogateCrankUp
        factory.spectralWeights = coarseGP.engine.spectralWeights
        factory.crankUpInitialState = latentInit
        factory.subDiagnostics = PersistentAcceptanceRateDiagnostics
        latentMCMC = factory.create()
        localisedDensity = factory.localisedDensity
        latentInit = factory.crankedState
    elif method == 'MLDA':
        factory = MLDAFactory(config.coarseRoot)
        factory.target = latentTarget
        factory.surrogate = [coarseSurrogate]
        factory.partition = partition
        factory.finePrior = finePrior
        factory.nChain = [config.nSubchain]
        factory.crankUpSteps = config.surrogateCrankUp
        factory.crankUpInitialState = latentInit
        factory.subDiagnostics = PersistentAcceptanceRateDiagnostics
        latentMCMC = factory.create()
        latentInit = factory.crankedState

    latentConditional = SGLMMLatentConditional(
        latentTarget, gp, coarseGP=coarseGP, partition=partition,
        finePrior=finePrior, localisedDensity=localisedDensity
    )
    latentMCMC.target = latentConditional
    latentTransition = MetropolisWithinGibbsConditional(
        latentMCMC, blockIdx=0, nSteps=1
    )

    hyperTarget = SGLMMHyperConditionalDensity(
        pcPrior, gp, predictor, likelihood, None
    )
    hyperInit = Vector(np.log([config.ellInit, config.sigmaInit]))
    jointState = BlockParameter([latentInit.clone(), hyperInit.clone()])
    hyperTarget.condition_on(jointState)

    hyperFactory = RobbinsMonroMRWFactory()
    hyperFactory.target = hyperTarget
    hyperFactory.proposalCovariance = IIDCovarianceMatrix(2, 0.01)
    hyperMCMC = hyperFactory.create()
    hyperTransition = MetropolisWithinGibbsConditional(
        hyperMCMC, blockIdx=1, nSteps=1
    )

    hierarchicalBayes = HierarchicalBayes(
        [latentTransition, hyperTransition], root=gp.measure
    )

    builder = GibbsBuilder()
    builder.model = hierarchicalBayes
    sampler = builder.build()
    sampler.storeChain = False

    hyperTrajectory = []
    essFieldTraces = []

    predSites = Grid(np.array(
        [[i / 21, j / 21] for i in range(1, 21) for j in range(1, 21)]
    ))
    predPredictor = predictor.create_predictor(predSites)

    for n, state in enumerate(
        sampler.stream_run(
            config.nSteps, jointState, progress=True, description="  Sampling"
        )
    ):
        hyperTrajectory.append(state.block(1).coordinate.copy())
        if n >= config.nBurnIn:
            latentState = state.block(0)
            predictor.reset()
            predictor.interpolate(latentState)
            essFieldTraces.append(predPredictor.mean())

    latentAcceptance = latentMCMC.diagnostics.global_acceptance_rate()
    hyperAcceptance = hyperMCMC.diagnostics.global_acceptance_rate()
    coarseAcceptance = (
        latentMCMC.subsamplers[0].diagnostics.global_acceptance_rate()
    )

    return {
        'hyper': hyperTrajectory,
        'essFieldTraces': np.array(essFieldTraces),
        'latentAcceptance': latentAcceptance,
        'hyperAcceptance': hyperAcceptance,
        'coarseAcceptance': coarseAcceptance,
    }


def collect_stats(config: Config, results: dict) -> dict:
    fieldTraces = results['essFieldTraces']
    hyperTrajectory = np.array(results['hyper'])[config.nBurnIn:]
    nSamples = len(fieldTraces)

    essPerSite = []
    for col in range(fieldTraces.shape[1]):
        iat = integrated_autocorrelation(fieldTraces[:, col])
        if np.isfinite(iat) and iat > 0:
            essPerSite.append(nSamples / iat)
        else:
            essPerSite.append(0.0)
    essLatent = min(essPerSite) if essPerSite else 0.0

    essHyper = []
    for j in range(hyperTrajectory.shape[1]):
        iatHyper = integrated_autocorrelation(np.exp(hyperTrajectory[:, j]))
        if np.isfinite(iatHyper) and iatHyper > 0:
            essHyper.append(nSamples / iatHyper)
        else:
            essHyper.append(0.0)

    minESS = min(essLatent, *essHyper)
    essPerEvaluation = minESS / config.nSteps

    return {
        'essPerEval': essPerEvaluation,
        'essSitesMedian': np.median(essPerSite),
        'latentAcceptance': results['latentAcceptance'],
        'hyperAcceptance': results['hyperAcceptance'],
        'coarseAcceptance': results['coarseAcceptance'],
    }


def evaluate_grid(
    config: Config, data: Data, sites: Grid,
    nObservations: int, baseN: int, allCoordinates: np.ndarray,
    allCounts: np.ndarray, writer
) -> tuple:
    fineResolution = compute_fine_resolution(nObservations, config)
    coarseResolution = compute_coarse_resolution(fineResolution, config)

    mldaSeed = 1000 * nObservations + 100 * 2
    dartSeed = 1000 * nObservations + 100 * 3

    print(f"\n--- Running MLDA Baseline ---")
    mldaResults = run_method(
        config, 'MLDA', data, sites, default_rng(mldaSeed),
        fineResolution, coarseResolution, None, None
    )
    mldaStats = collect_stats(config, mldaResults)
    writer.writerow([
        nObservations, 'MLDA', None, None,
        mldaStats['essPerEval'], mldaStats['latentAcceptance'],
        mldaStats['hyperAcceptance'], mldaStats['coarseAcceptance']
    ])
    print(
        f"MLDA Baseline: ESS/Eval={mldaStats['essPerEval']:.2e}, "
        f"Acc(L/H/C)={mldaStats['latentAcceptance']:.2f}/"
        f"{mldaStats['hyperAcceptance']:.2f}/{mldaStats['coarseAcceptance']:.2f}"
    )

    print(f"\n--- Running DART Calibration (gamma=1e-10, tempering=1.0) ---")
    calibResults = run_method(
        config, 'DART', data, sites, default_rng(dartSeed),
        fineResolution, coarseResolution, 1e-10, 1.0
    )
    calibStats = collect_stats(config, calibResults)
    writer.writerow([
        nObservations, 'DART_Calib', 1e-10, 1.0,
        calibStats['essPerEval'], calibStats['latentAcceptance'],
        calibStats['hyperAcceptance'], calibStats['coarseAcceptance']
    ])
    print(
        f"DART Calibration: ESS/Eval={calibStats['essPerEval']:.2e}, "
        f"Acc(L/H/C)={calibStats['latentAcceptance']:.2f}/"
        f"{calibStats['hyperAcceptance']:.2f}/{calibStats['coarseAcceptance']:.2f}"
    )

    gammas = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    temperings = [0.25, 0.5, 1.0]

    bestEss = -1.0
    bestCombination = None

    print(f"\n--- Sweeping DART Parameters ---")
    for g, t in itertools.product(gammas, temperings):
        print(f"Testing gamma={g:.4e}, tempering={t:.2f}")
        dartResults = run_method(
            config, 'DART', data, sites, default_rng(dartSeed),
            fineResolution, coarseResolution, g, t
        )
        dartStats = collect_stats(config, dartResults)
        writer.writerow([
            nObservations, 'DART', g, t,
            dartStats['essPerEval'], dartStats['latentAcceptance'],
            dartStats['hyperAcceptance'], dartStats['coarseAcceptance']
        ])
        if dartStats['essPerEval'] > bestEss:
            bestEss = dartStats['essPerEval']
            bestCombination = (g, t)

    return bestCombination


def plot_fits(
    nValues: np.ndarray, optimalGammas: np.ndarray, optimalTemperings: np.ndarray,
    gammaConstant: float, gammaRate: float,
    baseTempering: float, temperingScalingExponent: float,
    baseN: int
):
    if not hasMatplotlib:
        return
    
    plt.close('all')
    fig, (axGamma, axTemp) = plt.subplots(1, 2, figsize=(12, 5))

    axGamma.plot(nValues, optimalGammas, 'ko', label='Optimal Gamma')
    fitGammas = gammaConstant * (nValues ** gammaRate)
    axGamma.plot(nValues, fitGammas, 'r-', label='Fit')
    axGamma.set_xscale('log')
    axGamma.set_yscale('log')
    axGamma.set_xlabel('Observations (N)')
    axGamma.set_ylabel('Gamma')
    axGamma.legend()
    axGamma.set_title(
        f"Gamma = {gammaConstant:.4e} * N ^ {gammaRate:.3f}"
    )

    axTemp.plot(nValues, optimalTemperings, 'ko', label='Optimal Tempering')
    fitTemperings = baseTempering * ((nValues / baseN) ** temperingScalingExponent)
    axTemp.plot(nValues, fitTemperings, 'b-', label='Fit')
    axTemp.set_xscale('log')
    axTemp.set_ylim(0, 1.2)
    axTemp.set_xlabel('Observations (N)')
    axTemp.set_ylabel('Tempering')
    axTemp.legend()
    axTemp.set_title(
        f"Tempering = {baseTempering:.3f} * (N/{baseN}) ^ {temperingScalingExponent:.3f}"
    )

    outPath = Path(__file__).parent.parent / "figures" / "dart_sweep_fit.pdf"
    outPath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outPath)
    print(f"Saved fit plots to {outPath}")


def main():
    enable_logging(logging.INFO)
    config = Config()

    groundTruthRNG = default_rng(42)
    maxN = 2560
    sobolSampler = qmc.Sobol(d=2, scramble=True, seed=groundTruthRNG)
    allCoordinates = sobolSampler.random(maxN)

    trueCov = MaternCovariance2D(
        config.ellTrue, config.smoothness, config.sigmaTrue**2
    )
    trueGP = GaussianProcess.dna(
        trueCov, q=config.groundTruthResolution, d=2
    )
    trueRealisation = trueGP.sampler.generate_realisation(rng=groundTruthRNG)

    poolSites = Grid(allCoordinates)
    trueGP.sites = poolSites
    trueGP.parameter.coordinate = trueRealisation.coordinate
    zTrueFull = trueGP.at_sites() + config.trendOffset
    allCounts = PoissonResponse().simulate(zTrueFull, rng=groundTruthRNG)

    print(f"[GT] Ground truth generated for {maxN} points.")
    baseN = config.nObservationCounts[0]

    outCsv = Path(__file__).parent / "dart_sweep_results.csv"
    with open(outCsv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'N', 'Method', 'Gamma', 'Tempering', 'ESS_Per_Eval',
            'Latent_Acc', 'Hyper_Acc', 'Coarse_Acc'
        ])

        optimalGammas = []
        optimalTemperings = []

        for nObservations in config.nObservationCounts:
            print(f"\n{'=' * 80}")
            print(f" N = {nObservations}")
            print(f"{'=' * 80}")

            sites = Grid(allCoordinates[:nObservations])
            counts = allCounts.coordinate[:nObservations].reshape(-1, 1)
            data = Data(1, sites.to_array())
            data.measurement = counts

            bestGamma, bestTempering = evaluate_grid(
                config, data, sites, nObservations, baseN,
                allCoordinates, allCounts, writer
            )

            optimalGammas.append(bestGamma)
            optimalTemperings.append(bestTempering)
            
            print(f"Best for N={nObservations}: gamma={bestGamma:.4e}, "
                  f"tempering={bestTempering:.2f}")

    nArray = np.array(config.nObservationCounts)
    gammaArray = np.array(optimalGammas)
    tempArray = np.array(optimalTemperings)

    gammaRate, logGammaConst = np.polyfit(np.log(nArray), np.log(gammaArray), 1)
    gammaConstant = np.exp(logGammaConst)

    ratioArray = nArray / baseN
    tempExp, logBaseTemp = np.polyfit(np.log(ratioArray), np.log(tempArray), 1)
    baseTempering = np.exp(logBaseTemp)

    print(f"\n{'=' * 80}")
    print(f" FITTING RESULTS")
    print(f"{'=' * 80}")
    print(f"gammaConstant: {gammaConstant:.4e}")
    print(f"gammaRate: {gammaRate:.4f}")
    print(f"baseTempering: {baseTempering:.4f}")
    print(f"temperingScalingExponent: {tempExp:.4f}")

    plot_fits(
        nArray, gammaArray, tempArray,
        gammaConstant, gammaRate, baseTempering, tempExp, baseN
    )


if __name__ == '__main__':
    main()
