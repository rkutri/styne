#!/usr/bin/env python3
"""
Logistic Regression Experiment.

Runs MALA, MRW, MLDA, and DART on a Bayesian logistic regression
problem with Rademacher covariates, following the setup of Dalalyan
(2017) and Dwivedi et al. (2019).

All methods operate in the preconditioned space phi = Sigma_X^{1/2} theta,
where Sigma_X = (1/n) C^T C is the sample covariance of the covariates.
The prior N(0, (alpha Sigma_X)^{-1}) becomes N(0, alpha^{-1} I) after
preconditioning, and the condition number reduces to (0.25n + alpha) / alpha,
independent of the spectrum of the design matrix. With n and alpha fixed,
the gradient-Lipschitz constant L = 0.25 n + alpha and the condition
number kappa = L / alpha are the same at every dimension.
"""

import numpy as np
from scipy.special import expit
from pathlib import Path

from manuscript_boilerplate import (
    hasMatplotlib, hasJoblib, plt, joblib
)

from styne.parameter.vector import Vector
from styne.statistics.data import Data
from styne.statistics.response import BinomialResponse
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.radonnikodym import RadonNikodym
from styne.model.linear import LinearModel
from styne.utility.map import determine_map

from styne.mcmc.method.mala import MALAFactory
from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.method.mlda import MLDAFactory
from styne.mcmc.method.dartdirect import DirectDART
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics, DummyDiagnostics
from styne.utility.tuning import (
    MALATuner, MRWTuner, LangevinTunerConfig, RWTunerConfig
)
from styne.utility.postprocessing import multichain_ess_per_iter
# pyrefly: ignore [missing-import]
from manuscript_style import METHOD_COLORS

randomSeed = 2026
nObservations = 240
alphaPrior = 3.0
tempering = 0.5
convergenceDimension = 16
sweepDimensions = [2, 4, 8, 16]
mldaSubSteps = 10

deterministicStart = False

PRODUCTION = True

if PRODUCTION:
    nConvergenceRuns = 500
    nEssRuns = 8
    nConvergenceSteps = 3_500
    nEssSteps = 500_000
    nAcceptanceBurnin = 5_000
    nAcceptanceRepeats = 100
    nAcceptanceSteps = 5_000
    nGammaPoints = 16
else:
    nConvergenceRuns = 20
    nEssRuns = 4
    nConvergenceSteps = 2_500
    nEssSteps = 5_000
    nAcceptanceBurnin = 100
    nAcceptanceRepeats = 20
    nAcceptanceSteps = 300
    nGammaPoints = 12

CACHE = Path(__file__).parent / 'joblib_caches' / f"{Path(__file__).stem}.joblib"
if hasJoblib:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
FORCE = False

PARAMS = dict(
    randomSeed=randomSeed,
    nObservations=nObservations,
    alphaPrior=alphaPrior,
    tempering=tempering,
    convergenceDimension=convergenceDimension,
    sweepDimensions=sweepDimensions,
    mldaSubSteps=mldaSubSteps,
    deterministicStart=deterministicStart,
    PRODUCTION=PRODUCTION,
    nConvergenceRuns=nConvergenceRuns,
    nEssRuns=nEssRuns,
    nConvergenceSteps=nConvergenceSteps,
    nEssSteps=nEssSteps,
    nAcceptanceBurnin=nAcceptanceBurnin,
    nAcceptanceRepeats=nAcceptanceRepeats,
    nAcceptanceSteps=nAcceptanceSteps,
    nGammaPoints=nGammaPoints,
)

if hasMatplotlib:
    plt.style.use(Path(__file__).parent / 'manuscript.mplstyle')

methodColours = METHOD_COLORS

def style_axes(axis):
    axis.tick_params(direction="out", length=3.0, width=0.8)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        axis.spines[side].set_linewidth(0.8)
    axis.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)

def compute_mean_error(trajectory, trueParam):
    """Mean absolute error per coordinate over the running-mean trajectory."""
    runningMean = np.cumsum(trajectory, axis=0) / np.arange(1, len(trajectory) + 1)[:, None]
    return np.mean(np.abs(runningMean - trueParam), axis=1)

def project_soft(sampler, start, nSteps, softVec):
    """Run a chain and return its post-burn-in projection onto the slowest
    target direction (softVec) as a 1d series, for multi-chain ESS."""
    sampler.run(nSteps, start)
    trajectory = np.array(sampler.chain.trajectory)
    postBurnin = trajectory[int(0.2 * nSteps):]
    return postBurnin @ np.asarray(softVec, dtype=float)

def get_acceptance_rate(factory, start, nBurnin, nSteps):
    factory.diagnostics = AcceptanceRateDiagnostics()
    sampler = factory.create()
    sampler.run(nBurnin, start)
    factory.diagnostics.clear()
    sampler.continue_run(nSteps)
    return factory.diagnostics.global_acceptance_rate()

def setup_model(randomGenerator, d):
    """
    Setup preconditioned posterior for logistic regression.
    """
    covariates = randomGenerator.choice([-1.0, 1.0], size=(nObservations, d))
    covariates = covariates / np.linalg.norm(covariates, axis=1, keepdims=True)

    trueTheta = np.ones(d)
    eta = covariates @ trueTheta
    probs = expit(eta)
    responses = randomGenerator.binomial(1, probs)

    sigmaX = covariates.T @ covariates / nObservations
    eigenvalues, eigenvectors = np.linalg.eigh(sigmaX)
    sigmaXInvSqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    sigmaXSqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

    precondCovariates = covariates @ sigmaXInvSqrt
    truePhi = sigmaXSqrt @ trueTheta

    data = Data(nObservations, precondCovariates)
    data.measurement = responses

    priorCovariance = IIDCovarianceMatrix(d, 1.0 / alphaPrior)
    prior = Gaussian(priorCovariance, mean=Vector(np.zeros(d)))

    linearModel = LinearModel(precondCovariates)
    response = BinomialResponse(n=1)
    likelihood = RegressionLikelihood(data, linearModel, response)

    posterior = RadonNikodym(prior, likelihood)

    # After preconditioning: C_tilde^T C_tilde = n * I
    smoothness = 0.25 * np.max(
        np.linalg.eigvalsh(precondCovariates.T @ precondCovariates)
    ) + alphaPrior
    kappa = smoothness / alphaPrior

    return (
        posterior, prior, data, linearModel, precondCovariates,
        smoothness, alphaPrior, kappa, truePhi
    )


def main():
    randomGenerator = np.random.default_rng(randomSeed)
    samplerGenerator = np.random.default_rng(randomSeed + 1)

    if hasMatplotlib:
        plt.close('all')
        fig = plt.figure(figsize=(16, 4.5))
        gridSpec = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.6, 1., 0.15, 1.0])
        axConvergence = fig.add_subplot(gridSpec[0, 0])
        axESS = fig.add_subplot(gridSpec[0, 2])
        axAcceptance = fig.add_subplot(gridSpec[0, 4])

    results = None
    if hasJoblib and CACHE.exists() and not FORCE:
        dataCache = joblib.load(CACHE)
        if dataCache.get('params') == PARAMS:
            results = dataCache['results']
            print("Loaded results from cache.")
        else:
            print("Stale cache detected, recomputing...")

    if results is None:
        if not hasJoblib:
            print("Warning: joblib is not installed. Caching is disabled.")
        results = {}
        summaryResults = []

        # ---------------------------------------------------------
        # Panel (a): Convergence at d={convergenceDimension}
        # ---------------------------------------------------------
        print("Running Panel (a)...")
        (postConv, priorConv, _, _, _, smoothnessConv, _, kappaConv,
         truePhiConv) = setup_model(randomGenerator, convergenceDimension)
        initStateConv = Vector(np.zeros(convergenceDimension))

        mapConv, precisionMatrixConv = determine_map(postConv, initStateConv)
        eigenvaluesConv, eigenvectorsConv = np.linalg.eigh(precisionMatrixConv)
        inversePrecisionConv = (
            eigenvectorsConv / eigenvaluesConv
        ) @ eigenvectorsConv.T
        surrogateCovarianceConv = DenseCovarianceMatrix(inversePrecisionConv)
        surrogateConv = Gaussian(surrogateCovarianceConv, mean=mapConv)

        malaFactory = MALAFactory()
        malaFactory.target = postConv
        malaFactory.rng = samplerGenerator
        MALATuner(malaFactory, mapConv, config=LangevinTunerConfig(acceptanceGoal=0.55, tolerance=0.01, nTuning=10000)).tune()

        mrwFactory = MRWFactory()
        mrwFactory.target = postConv
        mrwFactory.rng = samplerGenerator
        MRWTuner(mrwFactory, mapConv, config=RWTunerConfig(acceptanceGoal=0.25, tolerance=0.01, nTuning=10000)).tune()

        gammaDart = 0.2 * smoothnessConv
        inversePrecisionDart = (
            eigenvectorsConv / (tempering * eigenvaluesConv + gammaDart)
        ) @ eigenvectorsConv.T
        dartProposalCovariance = DenseCovarianceMatrix(inversePrecisionDart)

        mldaFactory = MLDAFactory("mrw")
        mldaFactory.target = postConv
        mldaFactory.rng = samplerGenerator
        mldaFactory.surrogate = [surrogateConv.density]
        mldaFactory.nChain = [mldaSubSteps]

        errMala = []
        errMrw = []
        errDart = []
        errMlda = []

        for i in range(nConvergenceRuns):
            if deterministicStart:
                startState = Vector(-np.ones(convergenceDimension))
            else:
                startState = priorConv.generate_realisation(rng=randomGenerator)

            mala = malaFactory.create()
            mala.run(nConvergenceSteps, startState)
            errMala.append(
                compute_mean_error(np.array(mala.chain.trajectory), truePhiConv)
            )

            mrw = mrwFactory.create()
            mrw.run(nConvergenceSteps, startState)
            errMrw.append(
                compute_mean_error(np.array(mrw.chain.trajectory), truePhiConv)
            )

            dart = DirectDART(
                postConv, tempering, gammaDart, surrogateConv, dartProposalCovariance,
                DummyDiagnostics(), rng=samplerGenerator
            )
            dart.run(nConvergenceSteps, startState)
            errDart.append(
                compute_mean_error(np.array(dart.chain.trajectory), truePhiConv)
            )

            mlda = mldaFactory.create()
            mlda.run(nConvergenceSteps, startState)
            errMlda.append(
                compute_mean_error(np.array(mlda.chain.trajectory), truePhiConv)
            )

        results['panel_a'] = {
            'errMala': errMala,
            'errMrw': errMrw,
            'errDart': errDart,
            'errMlda': errMlda,
        }

        # ---------------------------------------------------------
        # Panels (b) & (c): Gamma sweeps
        # ---------------------------------------------------------
        print("Running Panels (b) & (c)...")
        panelBcResults = []

        for idx, d in enumerate(sweepDimensions):
            print(f"  Dimension {d}")
            (post, prior, _, _, _, smoothness, _, kappa,
             truePhi) = setup_model(randomGenerator, d)
            initState = Vector(np.zeros(d))
            mapState, precisionMatrix = determine_map(post, initState)
            eigenvalues, eigenvectors = np.linalg.eigh(precisionMatrix)
            inversePrecision = (eigenvectors / eigenvalues) @ eigenvectors.T
            surrogateCovariance = DenseCovarianceMatrix(inversePrecision)
            surrogate = Gaussian(surrogateCovariance, mean=mapState)

            gammas = np.logspace(-2, 1, nGammaPoints) * smoothness

            softVec = eigenvectors[:, 0]

            acceptanceStarts = [prior.generate_realisation(rng=randomGenerator)
                         for _ in range(nAcceptanceRepeats)]
            essStarts = [prior.generate_realisation(rng=randomGenerator)
                         for _ in range(nEssRuns)]

            acceptanceRates = []
            for gamma in gammas:
                inversePrecisionGamma = (
                    eigenvectors / (tempering * eigenvalues + gamma)
                ) @ eigenvectors.T
                proposalCovariance = DenseCovarianceMatrix(inversePrecisionGamma)

                rates = []
                for start in acceptanceStarts:
                    diagnostics = AcceptanceRateDiagnostics()
                    dart = DirectDART(
                        post, tempering, gamma, surrogate, proposalCovariance, diagnostics,
                        rng=samplerGenerator
                    )
                    dart.run(nAcceptanceBurnin, start)
                    diagnostics.clear()
                    dart.continue_run(nAcceptanceSteps)
                    rates.append(diagnostics.global_acceptance_rate())
                acceptanceRates.append(np.mean(rates))

            essPerGamma = []
            for gamma in gammas:
                inversePrecisionGamma = (
                    eigenvectors / (tempering * eigenvalues + gamma)
                ) @ eigenvectors.T
                proposalCovariance = DenseCovarianceMatrix(inversePrecisionGamma)
                chains = []
                for start in essStarts:
                    sampler = DirectDART(
                        post, tempering, gamma, surrogate, proposalCovariance,
                        DummyDiagnostics(), rng=samplerGenerator
                    )
                    chains.append(project_soft(sampler, start, nEssSteps, softVec))
                essPerGamma.append(multichain_ess_per_iter(np.array(chains)))

            malaBaselineFactory = MALAFactory()
            malaBaselineFactory.target = post
            malaBaselineFactory.rng = samplerGenerator
            MALATuner(
                malaBaselineFactory, mapState, config=LangevinTunerConfig(acceptanceGoal=0.55, tolerance=0.01, nTuning=10000)
            ).tune()

            malaChains = []
            for start in essStarts:
                malaChains.append(
                    project_soft(malaBaselineFactory.create(), start, nEssSteps, softVec)
                )
            malaEss = multichain_ess_per_iter(np.array(malaChains))

            optimalIndex = np.nanargmax(essPerGamma)
            gammaOpt = gammas[optimalIndex]
            summaryResults.append({
                "d": d,
                "L": smoothness,
                "kappa": kappa,
                "h_mala": malaBaselineFactory.stepSize,
                "gamma_opt": gammaOpt,
                "gamma_opt_over_L": gammaOpt / smoothness,
                "ess_dart": essPerGamma[optimalIndex],
                "ess_mala": malaEss,
            })

            panelBcResults.append({
                "d": d,
                "gammas": gammas,
                "smoothness": smoothness,
                "essPerGamma": essPerGamma,
                "acceptanceRates": acceptanceRates,
                "malaEss": malaEss
            })

        results['panel_bc'] = panelBcResults
        results['summary'] = summaryResults
        if hasJoblib:
            dataCache = {'params': PARAMS, 'results': results}
            joblib.dump(dataCache, CACHE, compress=3)

    # Plotting
    if hasMatplotlib:
        errMala = results['panel_a']['errMala']
        errMrw = results['panel_a']['errMrw']
        errDart = results['panel_a']['errDart']
        errMlda = results['panel_a']['errMlda']
        convergenceSteps = np.arange(1, len(errMala[0]) + 1)

        axConvergence.plot(
            convergenceSteps, np.mean(errMala, axis=0),
            color=methodColours["MALA"], label="MALA"
        )
        axConvergence.plot(
            convergenceSteps, np.mean(errMrw, axis=0),
            color=methodColours["MRW"], label="MRW"
        )
        axConvergence.plot(
            convergenceSteps, np.mean(errDart, axis=0),
            color=methodColours["DART"], label="DART"
        )
        axConvergence.plot(
            convergenceSteps, np.mean(errMlda, axis=0),
            color=methodColours["MLDA"], label="MLDA"
        )

        style_axes(axConvergence)
        axConvergence.set_box_aspect(1)
        axConvergence.set_xscale('log')
        axConvergence.set_xlim(1, nConvergenceSteps)
        axConvergence.set_yscale('linear')
        axConvergence.set_xlabel("iteration")
        axConvergence.set_ylabel("error")
        axConvergence.set_title(f"Convergence ($d={convergenceDimension}$)")
        axConvergence.legend(
            loc="center left",
            bbox_to_anchor=(0.85, 0.65),
            borderaxespad=0.0
        )

        cmap = plt.get_cmap("plasma")
        for idx, efficiencyResult in enumerate(results['panel_bc']):
            d = efficiencyResult['d']
            gammas = efficiencyResult['gammas']
            smoothness = efficiencyResult['smoothness']
            essPerGamma = efficiencyResult['essPerGamma']
            acceptanceRates = efficiencyResult['acceptanceRates']
            malaEss = efficiencyResult['malaEss']
            color = cmap(idx / (len(sweepDimensions) - 1) * 0.8)

            axESS.axhline(
                malaEss, color=color, linestyle="--",
                linewidth=0.7, label="MALA baseline" if idx == 0 else "_nolegend_"
            )
            axESS.plot(
                gammas / smoothness, essPerGamma,
                color=color, label=f"$d={d}$"
            )

            axAcceptance.plot(
                gammas / smoothness, acceptanceRates,
                color=color, label=f"$d={d}$"
            )

        style_axes(axESS)
        axESS.set_box_aspect(1)
        axESS.set_xscale('log')
        axESS.set_yscale('log')
        axESS.set_xlabel("$\\gamma / L$")
        axESS.set_ylabel("ESS / iteration")
        axESS.set_ylim(2e-3, 1e-0)
        axESS.set_title("Efficiency")

        style_axes(axAcceptance)
        axAcceptance.set_box_aspect(1)
        axAcceptance.set_xscale('log')
        axAcceptance.set_xlabel("$\\gamma / L$")
        axAcceptance.set_ylabel("Acceptance Rate")
        axAcceptance.set_title("Mobility")

        handles, labels = axESS.get_legend_handles_labels()
        malaHandles = [h for h, l in zip(handles, labels) if l == "MALA baseline"]
        malaLabels = [l for l in labels if l == "MALA baseline"]
        dimHandles = [h for h, l in zip(handles, labels) if l != "MALA baseline"]
        dimLabels = [l for l in labels if l != "MALA baseline"]

        axESS.legend(
            malaHandles, malaLabels,
            loc="lower left",
        )

        axAcceptance.legend(
            dimHandles, dimLabels,
            loc="center left",
            bbox_to_anchor=(1.05, 0.7),
            borderaxespad=0.0
        )

        fig.subplots_adjust(
            left=0.02,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.2
        )
        figuresDir = Path(__file__).parent.parent / "figures"
        figuresDir.mkdir(parents=True, exist_ok=True)
        figPath = figuresDir / "manuscript_quadratic.pdf"
        fig.savefig(figPath, bbox_inches="tight")
        print(f"Saved {figPath}")
        plt.close(fig)
    else:
        print("Warning: matplotlib is not installed. Skipping plotting.")

    # Print Summary Table
    print("\n" + "=" * 72)
    print(f"{'d':>4} | {'L':>6} | {'kappa':>6} | {'gamma_opt':>10} | "
          f"{'gamma_opt/L':>12} | {'ESS DART':>9} | {'ESS MALA':>9}")
    print("-" * 72)
    for result in results['summary']:
        print(f"{result['d']:4d} | {result['L']:6.1f} | {result['kappa']:6.1f} | "
              f"{result['gamma_opt']:10.3f} | {result['gamma_opt_over_L']:12.3f} | "
              f"{result['ess_dart']:9.4f} | {result['ess_mala']:9.4f}")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
