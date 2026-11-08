#!/usr/bin/env python3
"""
Good-surrogate regime. Four methods compared on ACF, IAT bars,
and posterior mean. Claims 1-3. IAT bars averaged over nAverages independent MCMC runs.
"""

import numpy as np
from numpy.random import default_rng
from scipy.interpolate import interp1d
from pathlib import Path

# pyrefly: ignore [missing-import]
from manuscript_boilerplate import (
    hasMatplotlib, hasJoblib, plt, joblib
)

import logging
from styne import enable_logging
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.bayes import UnnormalisedPosterior
from styne.statistics.response import PoissonResponse
from styne.statistics.data import Data
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.stationary import MaternCovariance1D
from styne.model.sglmm import SGLMM
from styne.model.trend import ConstantTrend
from styne.mcmc.method.pcn import PCNFactory
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.mlda import MLDAFactory
from styne.mcmc.method.mala import MALAFactory
from styne.utility.tuning import PCNTuner, RWTunerConfig, MALATuner, LangevinTunerConfig
from manuscript_style import METHOD_COLORS
from styne.utility.grid import Grid, UniformGrid
from styne.utility.postprocessing import (
    estimate_autocorrelation_function_1d,
    integrated_autocorrelation
)
from styne.statistics.welford import WelfordAccumulator
from styne.gp.dnautility import DNACoarseFinePartition
from styne.gp.direct import DirectExpansion
from styne.gp.dna import DNAFourierExpansion
from styne.parameter.vector import Vector
import re
if hasMatplotlib:
    from matplotlib.lines import Line2D

lengthScale = 0.1
smoothness = 2.5
variance = 1.25
meanIntercept = 1.5
nObservations = 100
nDoFTruth = 500
nDoFFine = 100
nDoFCoarse = 20
nSubSteps = 25
tempering = 0.75
gamma = 0.1
choleskyNugget = 1e-6
randomSeed = 42
nIatLocations = 50

PRODUCTION = True
if PRODUCTION:
    nSteps = 250_000
    nBurninSteps = 50_000
    nAverages = 8
    nCrankUpSteps = 5000
else:
    nSteps = 11_000
    nBurninSteps = 1_000
    nAverages = 2
    nCrankUpSteps = 500

if hasMatplotlib:
    plt.style.use(Path(__file__).parent / 'manuscript.mplstyle')

CACHE = Path(__file__).parent / 'joblib_caches' / f"{Path(__file__).stem}.joblib"
if hasJoblib:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
FORCE = False

PARAMS = dict(
    lengthScale=lengthScale,
    smoothness=smoothness,
    variance=variance,
    meanIntercept=meanIntercept,
    nObservations=nObservations,
    nDoFTruth=nDoFTruth,
    nDofFine=nDoFFine,
    nDoFCoarse=nDoFCoarse,
    nSubSteps=nSubSteps,
    tempering=tempering,
    gamma=gamma,
    nCrankUpSteps=nCrankUpSteps,
    nSteps=nSteps,
    nBurninSteps=nBurninSteps,
    choleskyNugget=choleskyNugget,
    randomSeed=randomSeed,
    nIatLocations=nIatLocations,
    nAverages=nAverages,
)

methodColours = METHOD_COLORS


def style_axes(axis):
    axis.tick_params(direction="out", length=3.0, width=0.8)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        axis.spines[side].set_linewidth(0.8)
    axis.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)


def log_cholesky_conditioning(observationSites):
    """Check for suitable nugget in Cholesky factorisation."""
    s = observationSites.to_array().ravel()
    r = np.abs(s[:, None] - s[None, :])

    scaled = np.sqrt(5.0) * r / lengthScale
    gram = variance * (1.0 + scaled + scaled ** 2 / 3.0) * np.exp(-scaled)
    gram = gram + choleskyNugget * np.eye(len(s))

    print(
        f"  [Cholesky] Matern Gram condition number "
        f"(nugget={choleskyNugget:g}): {np.linalg.cond(gram):.3e}"
    )


def generate_shared_ground_truth(rng):
    """Generate the GP realisation and observation sites once."""

    covarianceFunction = MaternCovariance1D(lengthScale, smoothness, variance)
    trueGaussianProcess = GaussianProcess.dna(
        covarianceFunction, q=nDoFTruth, d=1
    )

    trueRealisation = trueGaussianProcess.sampler.generate_realisation(rng=rng)
    observationSites = Grid(np.sort(rng.uniform(0.0, 1.0, nObservations)))

    return trueRealisation, observationSites


def compute_observations(rng, trueRealisation, observationSites, trend):
    """Generate Poisson observations for a given trend."""

    latentAtObservations = (
        trueRealisation.evaluate(observationSites) + trend.evaluate(observationSites)
    )

    measurements = PoissonResponse().simulate(latentAtObservations, rng=rng)
    data = Data(1, observationSites.to_array())
    data.measurement = measurements.coordinate.reshape(-1, 1)

    return data


def tune_pcn(factory, initialValue, acceptanceGoal=0.3):
    return PCNTuner(factory, initialValue, config=RWTunerConfig(acceptanceGoal)).tune()


def tune_mala(factory, initialValue, acceptanceGoal=0.5):
    return MALATuner(factory, initialValue, config=LangevinTunerConfig(acceptanceGoal)).tune()


def get_sampler_name(mcmc):

    className = mcmc.__class__.__name__

    if className == "MetropolisAdjustedLangevinAlgorithm":
        return "MALA"
    elif className == "PreconditionedCrankNicolson":
        return "pCN"

    elif className == "MultilevelDelayedAcceptanceMCMC":
        return "MLDA"
    elif className == "DART":
        return "DART"

    return className


def get_parametrisation_name(gp):

    if isinstance(gp.expansion, DirectExpansion):
        return "Cholesky"
    elif isinstance(gp.expansion, DNAFourierExpansion):
        return "DNA"

    return "Unknown"





def run_mcmc_chain_online(
    sampler, initialState, predictor, evaluationGrid, iatIndices
):
    """Run MCMC without storing chain, track IAT traces and posterior mean online."""

    sampler.storeChain = False
    
    iatTraces = np.empty((nSteps - nBurninSteps, nIatLocations))
    welford = WelfordAccumulator()
    
    for stepIndex, state in enumerate(
        sampler.stream_run(
            nSteps,
            initialState,
            progress=True,
            description=f"Sampling {get_sampler_name(sampler)}",
        )
    ):
        if stepIndex >= nBurninSteps:
            preparedState = predictor.prepare(state)
            fieldEvaluations = predictor.predict(
                preparedState, evaluationGrid
            )
            welford.update(fieldEvaluations)
            
            iatFieldEvaluations = fieldEvaluations[iatIndices]
            iatTraces[stepIndex - nBurninSteps] = iatFieldEvaluations
            
    return iatTraces, welford


def iat_summary_over_traces(tracesList):
    """Compute spatial IAT summary averaged over multiple runs from online traces."""

    nRuns = len(tracesList)
    bests = np.empty(nRuns)
    worsts = np.empty(nRuns)
    averages = np.empty(nRuns)
    worstIndices = np.empty(nRuns, dtype=int)
    bestIndices = np.empty(nRuns, dtype=int)

    xValues = np.linspace(0.0, 1.0, nIatLocations)

    for runIndex, traces in enumerate(tracesList):
        integratedAutocorrelationTimes = np.empty(nIatLocations)
        for i in range(nIatLocations):
            iat = integrated_autocorrelation(traces[:, i])
            if np.isfinite(iat) and iat > 0:
                integratedAutocorrelationTimes[i] = iat
            else:
                integratedAutocorrelationTimes[i] = np.nan
            
        bests[runIndex] = np.nanmin(integratedAutocorrelationTimes)
        worsts[runIndex] = np.nanmax(integratedAutocorrelationTimes)
        averages[runIndex] = np.nanmean(integratedAutocorrelationTimes)
        worstIndices[runIndex] = int(np.nanargmax(integratedAutocorrelationTimes))
        bestIndices[runIndex] = int(np.nanargmin(integratedAutocorrelationTimes))

    medianWorstRunIndex = int(np.argsort(worsts)[nRuns // 2])
    medianBestRunIndex = int(np.argsort(bests)[nRuns // 2])

    return {
        'best': (np.mean(bests), np.std(bests) / np.sqrt(nRuns)),
        'worst': (np.mean(worsts), np.std(worsts) / np.sqrt(nRuns)),
        'avg': (np.mean(averages), np.std(averages) / np.sqrt(nRuns)),
        'worst_location': xValues[worstIndices[medianWorstRunIndex]],
        'best_location': xValues[bestIndices[medianBestRunIndex]],
        'worst_index': worstIndices[medianWorstRunIndex],
        'best_index': bestIndices[medianBestRunIndex],
        'median_worst_run': medianWorstRunIndex,
        'median_best_run': medianBestRunIndex
    }


def seed_chain(runIndex, methodIndex):
    """Deterministic seed for each MCMC run."""

    chainSeed = randomSeed + 10_000 * (runIndex + 1) + 1_000 * methodIndex
    np.random.seed(chainSeed)


def run_experiment(rng, trueRealisation, observationSites):
    """Four methods on a smooth field, nAverages runs each."""

    trend = ConstantTrend(meanIntercept)
    covarianceFunction = MaternCovariance1D(lengthScale, smoothness, variance)
    data = compute_observations(rng, trueRealisation, observationSites, trend)

    fineGrid = UniformGrid(0.0, 1.0, nDoFTruth + 2)
    trueField = (
        trueRealisation.evaluate(fineGrid) + trend.evaluate(fineGrid)
    )

    choleskyGaussianProcess = GaussianProcess.direct(
        observationSites, covarianceFunction, nugget=choleskyNugget
    )

    log_cholesky_conditioning(observationSites)

    choleskyPredictor = SGLMM(choleskyGaussianProcess, observationSites, trend=trend)
    choleskyTarget = UnnormalisedPosterior(
        choleskyGaussianProcess.measure,
        RegressionLikelihood(data, choleskyPredictor, PoissonResponse())
    )

    dnaGaussianProcess = GaussianProcess.dna(
        covarianceFunction, q=nDoFFine, d=1
    )
    dnaPredictor = SGLMM(dnaGaussianProcess, observationSites, trend=trend)
    dnaTarget = UnnormalisedPosterior(
        dnaGaussianProcess.measure,
        RegressionLikelihood(data, dnaPredictor, PoissonResponse())
    )

    coarseGaussianProcess = GaussianProcess.dna(
        covarianceFunction, q=nDoFCoarse, d=1
    )
    coarseTarget = UnnormalisedPosterior(
        coarseGaussianProcess.measure,
        RegressionLikelihood(
            data, SGLMM(
                coarseGaussianProcess, observationSites, trend=trend
            ),
            PoissonResponse()
        ),
    )

    partition = DNACoarseFinePartition(
        dnaGaussianProcess, qC=nDoFCoarse, d=1
    )
    finePrior = partition.fine_measure()

    evaluationGrid = UniformGrid(0.0, 1.0, 202)
    gridArray = evaluationGrid.to_array().ravel()
    xValues = np.linspace(0.0, 1.0, nIatLocations)
    iatIndices = [int(np.argmin(np.abs(gridArray - x))) for x in xValues]
    
    results = {}

    print("  [Cholesky + MALA]")
    tracesList = []
    welfordsList = []

    testFactory = MALAFactory()
    testFactory.target = choleskyTarget
    testMcmc = tune_mala(testFactory, choleskyGaussianProcess.measure.mean)
    paramName = get_parametrisation_name(choleskyGaussianProcess)
    samplerName = get_sampler_name(testMcmc)
    legendLabel = f"{paramName}({samplerName})"

    for runIndex in range(nAverages):

        print(f"    Run {runIndex + 1}/{nAverages}")

        seed_chain(runIndex, 0)

        factory = MALAFactory()
        factory.target = choleskyTarget

        mcmc = tune_mala(factory, choleskyGaussianProcess.measure.mean)

        print(f"      tuned MALA step size: {getattr(mcmc, 'stepSize', float('nan')):.3e}")

        iatTraces, welford = run_mcmc_chain_online(
            mcmc,
            choleskyGaussianProcess.measure.mean.clone(),
            choleskyPredictor,
            evaluationGrid,
            iatIndices,
        )

        tracesList.append(iatTraces)
        welfordsList.append(welford)

    results["Cholesky"] = {
        "traces": tracesList,
        "welfords": welfordsList,
        "legend_label": legendLabel,
    }

    print("  [DNA + MALA]")

    tracesList = []
    welfordsList = []

    testFactory = MALAFactory()
    testFactory.target = dnaTarget
    testMcmc = tune_mala(testFactory, dnaGaussianProcess.measure.mean)
    paramName = get_parametrisation_name(dnaGaussianProcess)
    samplerName = get_sampler_name(testMcmc)
    legendLabel = f"{paramName}({samplerName})"

    for runIndex in range(nAverages):

        print(f"    Run {runIndex + 1}/{nAverages}")

        seed_chain(runIndex, 1)

        factory = MALAFactory()
        factory.target = dnaTarget

        mcmc = tune_mala(factory, dnaGaussianProcess.measure.mean)

        print(f"      tuned MALA step size: {getattr(mcmc, 'stepSize', float('nan')):.3e}")

        iatTraces, welford = run_mcmc_chain_online(
            mcmc,
            dnaGaussianProcess.measure.mean.clone(),
            dnaPredictor,
            evaluationGrid,
            iatIndices,
        )

        tracesList.append(iatTraces)
        welfordsList.append(welford)

    results["DNA"] = {
        "traces": tracesList,
        "welfords": welfordsList,
        "legend_label": legendLabel,
    }

    print("  [DNA + MLDA]")

    tracesList = []
    welfordsList = []

    testFactory = MLDAFactory("mala")
    testFactory.target = dnaTarget
    testFactory.surrogate = [coarseTarget]
    testFactory.partition = partition
    testFactory.finePrior = finePrior
    testFactory.nChain = [nSubSteps]
    testFactory.surrogateCrankUp = nCrankUpSteps
    testFactory.crankUpInitialState = dnaGaussianProcess.measure.mean.clone()
    testMcmc = testFactory.create()
    paramName = get_parametrisation_name(dnaGaussianProcess)
    samplerName = get_sampler_name(testMcmc)
    legendLabel = f"{paramName}({samplerName})"

    for runIndex in range(nAverages):

        print(f"    Run {runIndex + 1}/{nAverages}")

        seed_chain(runIndex, 2)

        factory = MLDAFactory("mala")
        factory.target = dnaTarget
        factory.surrogate = [coarseTarget]
        factory.partition = partition
        factory.finePrior = finePrior
        factory.nChain = [nSubSteps]
        factory.surrogateCrankUp = nCrankUpSteps
        factory.crankUpInitialState = dnaGaussianProcess.measure.mean.clone()

        mcmc = factory.create()

        iatTraces, welford = run_mcmc_chain_online(
            mcmc,
            dnaGaussianProcess.measure.mean.clone(),
            dnaPredictor,
            evaluationGrid,
            iatIndices,
        )

        tracesList.append(iatTraces)
        welfordsList.append(welford)

    results["MLDA"] = {
        "traces": tracesList,
        "welfords": welfordsList,
        "legend_label": legendLabel,
    }

    print("  [DNA + DART]")

    tracesList = []
    welfordsList = []

    testFactory = DARTFactory("mala")
    testFactory.target = dnaTarget
    testFactory.surrogate = [coarseTarget]
    testFactory.partition = partition
    testFactory.finePrior = finePrior
    testFactory.regularisation = [gamma]
    testFactory.tempering = [tempering]
    testFactory.nChain = [nSubSteps]
    testFactory.surrogateCrankUp = nCrankUpSteps
    testFactory.crankUpInitialState = dnaGaussianProcess.measure.mean.clone()
    testMcmc = testFactory.create()
    paramName = get_parametrisation_name(dnaGaussianProcess)
    samplerName = get_sampler_name(testMcmc)
    legendLabel = f"{paramName}({samplerName})"

    for runIndex in range(nAverages):

        print(f"    Run {runIndex + 1}/{nAverages}")

        seed_chain(runIndex, 3)

        factory = DARTFactory("mala")
        factory.target = dnaTarget
        factory.surrogate = [coarseTarget]
        factory.partition = partition
        factory.finePrior = finePrior
        factory.regularisation = [gamma]
        factory.tempering = [tempering]
        factory.nChain = [nSubSteps]
        factory.surrogateCrankUp = nCrankUpSteps
        factory.crankUpInitialState = dnaGaussianProcess.measure.mean.clone()
        mcmc = factory.create()
        iatTraces, welford = run_mcmc_chain_online(
            mcmc,
            dnaGaussianProcess.measure.mean.clone(),
            dnaPredictor,
            evaluationGrid,
            iatIndices,
        )
        tracesList.append(iatTraces)
        welfordsList.append(welford)

    results["DART"] = {
        "traces": tracesList,
        "welfords": welfordsList,
        "legend_label": legendLabel,
    }

    return results, trueField, fineGrid, observationSites, data, trend


def format_legend_label(label):
    if " + " in label:
        parts = label.split(" + ")
        label = f"{parts[0]}({parts[1]})"
    label = re.sub(r"MLDA\([^)]*\)", "MLDA", label)
    label = re.sub(r"DART\([^)]*\)", "DART", label)
    return label


def plot_figure_a(results, trueField, fineGrid, observationSites, data, trend):
    """Figure A: ACF (left), IAT bars (centre), posterior mean (right)."""
    plt.close('all')
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    evaluationGrid = UniformGrid(0.0, 1.0, 202)
    methodOrder = ["Cholesky", "DNA", "MLDA", "DART"]

    # Reconstruct the GPs here to avoid pickling issues
    covarianceFunction = MaternCovariance1D(lengthScale, smoothness, variance)
    choleskyGaussianProcess = GaussianProcess.direct(
        observationSites, covarianceFunction, nugget=choleskyNugget
    )
    dnaGaussianProcess = GaussianProcess.dna(
        covarianceFunction, q=nDoFFine, d=1
    )

    gpMap = {
        "Cholesky": choleskyGaussianProcess,
        "DNA": dnaGaussianProcess,
        "MLDA": dnaGaussianProcess,
        "DART": dnaGaussianProcess,
    }

    summaries = {}
    for name in methodOrder:
        result = results[name]
        summaries[name] = iat_summary_over_traces(result["traces"])

    axis = axes[0]
    style_axes(axis)
    maxLag = 2000
    for name in methodOrder:
        result = results[name]
        summary = summaries[name]
        medianWorstRunIndex = summary["median_worst_run"]
        medianBestRunIndex = summary["median_best_run"]
        worstIndex = summary["worst_index"]
        bestIndex = summary["best_index"]
        
        seriesWorst = result["traces"][medianWorstRunIndex][:, worstIndex]
        acfWorst = estimate_autocorrelation_function_1d(seriesWorst)
        
        seriesBest = result["traces"][medianBestRunIndex][:, bestIndex]
        acfBest = estimate_autocorrelation_function_1d(seriesBest)
        
        nLags = min(maxLag, len(acfWorst), len(acfBest))
        lags = np.arange(nLags)
        acfWorstCut = acfWorst[:nLags]
        acfBestCut = acfBest[:nLags]
        
        legendLabel = format_legend_label(result.get("legend_label", name))
        axis.plot(
            lags, acfWorstCut, color=methodColours[name],
            linestyle="-", label=legendLabel
        )
        axis.plot(
            lags, acfBestCut, color=methodColours[name],
            linestyle=":", alpha=0.5
        )
        
        # Very transparent shading between best and worst case curves
        axis.fill_between(
            lags, acfWorstCut, acfBestCut,
            color=methodColours[name], alpha=0.08
        )

    axis.axhline(0, color="k", linestyle="--", linewidth=0.5)
    axis.set_xlabel("Lag")
    axis.set_xlim(0, 1500)
    axis.set_ylabel("ACF")
    axis.set_title("Autocorrelation")
    
    styleHandles = [
        Line2D([0], [0], color="gray", linestyle="-", label="Worst"),
        Line2D([0], [0], color="gray", linestyle=":", label="Best"),
    ]
    axis.legend(handles=styleHandles)

    axis = axes[1]
    style_axes(axis)

    categories = ["Best", "Average", "Worst"]
    categoryKeys = ["best", "avg", "worst"]
    nMethods = len(methodOrder)
    nCategories = len(categories)
    barWidth = 0.18
    xPositions = np.arange(nCategories)

    for methodIndex, name in enumerate(methodOrder):
        summary = summaries[name]
        means = [summary[k][0] for k in categoryKeys]
        ses = [summary[k][1] for k in categoryKeys]
        offset = (methodIndex - (nMethods - 1) / 2) * barWidth
        legendLabel = format_legend_label(results[name].get("legend_label", name))
        axis.bar(
            xPositions + offset, means, barWidth * 0.9,
            yerr=ses, capsize=3, error_kw={"elinewidth": 0.8},
            color=methodColours[name], label=legendLabel,
        )

    axis.set_xticks(xPositions)
    axis.set_xticklabels(categories)
    axis.set_xlabel("Location")
    axis.set_ylabel("IAT")
    axis.set_title(f"IAT summary")

    axis = axes[2]
    style_axes(axis)

    dartWelford = results["DART"]["welfords"][0]
    posteriorMean = dartWelford.mean()
    posteriorStandardDeviation = np.sqrt(dartWelford.marginal_variance())

    gridArray = evaluationGrid.to_array().ravel()
    fineArray = fineGrid.to_array().ravel()

    legendLabel = format_legend_label(results["DART"].get("legend_label", "DNA + DART"))
    axis.plot(fineArray, trueField, "k-", label="True field")
    axis.plot(gridArray, posteriorMean, color=methodColours["DART"],
              label=f"Post. mean", alpha=0.9, linestyle='--')
    axis.fill_between(
        gridArray,
        posteriorMean - 1.96 * posteriorStandardDeviation,
        posteriorMean + 1.96 * posteriorStandardDeviation,
        alpha=0.25, color=methodColours["DART"],
        label="95% cr. band",
    )
    observationX = observationSites.to_array().ravel()
    axis.scatter(observationX, np.full_like(observationX, np.min(trueField) - 0.5),
                 marker="|", color="k", alpha=0.6,
                 label="Obs. locations", zorder=5)
    axis.set_xlabel("s")
    axis.set_ylabel("Latent field u(s)")
    axis.set_title(f"Inference {legendLabel}")
    axis.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))

    trueInterp = interp1d(fineArray, trueField, kind='linear')(gridArray)
    inside = (trueInterp >= posteriorMean - 1.96 * posteriorStandardDeviation) & \
             (trueInterp <= posteriorMean + 1.96 * posteriorStandardDeviation)
    coverage = np.mean(inside)
    
    print(f"Spatial coverage: {coverage:.1%}")
    figure.tight_layout(rect=[0, 0.08, 1, 1], w_pad=0.01)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01)
    )
    figuresDir = Path(__file__).parent.parent / "figures"
    figuresDir.mkdir(parents=True, exist_ok=True)
    figurePath = figuresDir / "manuscript_1d_figA.pdf"
    figure.savefig(figurePath, bbox_inches="tight")
    print(f"Saved {figurePath}")
    plt.close(figure)


def main():
    enable_logging(logging.INFO)
    resultsA = None
    if hasJoblib and CACHE.exists() and not FORCE:
        dataCache = joblib.load(CACHE)
        if dataCache.get('params') == PARAMS:
            resultsA = dataCache['resultsA']
            trueFieldA = dataCache['trueFieldA']
            fineGridA = dataCache['fineGridA']
            observationSitesA = dataCache['observationSitesA']
            dataA = dataCache['dataA']
            trendA = dataCache['trendA']
            print("Loaded results from cache.")
        else:
            print("Stale cache detected, recomputing...")

    if resultsA is None:
        if not hasJoblib:
            print("Warning: joblib is not installed. Caching is disabled.")
        rng = default_rng(randomSeed)
        trueRealisation, observationSites = generate_shared_ground_truth(rng)

        print("=== Figure A: good-surrogate regime ===")
        resultsA, trueFieldA, fineGridA, observationSitesA, dataA, trendA = (
            run_experiment(rng, trueRealisation, observationSites)
        )
        if hasJoblib:
            dataCache = {
                'params': PARAMS,
                'resultsA': resultsA,
                'trueFieldA': trueFieldA,
                'fineGridA': fineGridA,
                'observationSitesA': observationSitesA,
                'dataA': dataA,
                'trendA': trendA,
            }
            joblib.dump(dataCache, CACHE, compress=3)

    if hasMatplotlib:
        plot_figure_a(
            resultsA, trueFieldA, fineGridA, observationSitesA, dataA, trendA
        )
    else:
        print("Warning: matplotlib is not installed. Skipping plotting.")


if __name__ == "__main__":
    main()
