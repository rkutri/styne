import pytest

import numpy as np

import styne.utility.postprocessing as ac

from numpy.random import default_rng

from tests.testSetup import GaussianTargetDensity
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.diagnostics import *
from styne.parameter.scalar import Scalar


@pytest.mark.parametrize("Diagnostics",
                         [DummyDiagnostics, AcceptanceRateDiagnostics,
                          FullDiagnostics])
def test_metropolishastings_initialisation(Diagnostics):

    tgtMean = Scalar(1.)
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    diagnostics = Diagnostics()

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics)

    assert isinstance(mc.target, type(tgtDensity))
    assert mc.chain.trajectory == []


@pytest.mark.parametrize("Diagnostics",
                         [DummyDiagnostics, AcceptanceRateDiagnostics,
                          FullDiagnostics])
def test_step(Diagnostics):

    tgtMean = Scalar(0.)
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    diagnostics = Diagnostics()

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics)

    state = Scalar(2.)
    nextState, transition, _ = mc.step(mc.evaluate_state(state), mc._rng)

    assert transition.current.parameter is state
    selected = transition.proposal if transition.outcome else transition.state
    np.testing.assert_allclose(
        nextState.parameter.coordinate, selected.coordinate
    )


@pytest.mark.parametrize("Diagnostics",
                         [DummyDiagnostics, AcceptanceRateDiagnostics,
                          FullDiagnostics])
def test_run_chain(Diagnostics):

    rng = default_rng(18)

    tgtMean = Scalar(1.5)
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    diagnostics = Diagnostics()

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics, rng=rng)

    tgtMean = Scalar(0.)
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics, rng=rng)

    nSteps = 2000
    initState = Scalar(-3.)
    mc.run(nSteps, initState)

    assert len(mc.chain.trajectory) == nSteps + 1

    states = np.array(mc.chain.trajectory).ravel()

    # postprocessing
    burnin = 200

    thinning = ac.sokal_heuristic(
        ac.estimate_autocorrelation_function_1d(states[burnin:]), 5
    )

    mcSamples = states[burnin::thinning]

    # estimate mean
    meanSample = np.mean(mcSamples)
    meanState = np.mean(states)

    # estimate variance
    sampleVar = np.var(mcSamples)
    stateVar = np.var(states)

    # test moments
    MTOL = 0.5
    VTOL = 0.5

    assert np.abs(tgtMean.coordinate.item() - meanSample) < MTOL
    assert np.abs(tgtMean.coordinate.item() - meanState) < 2. * MTOL

    assert np.abs(tgtVar - sampleVar) < VTOL
    assert np.abs(tgtVar - stateVar) < VTOL


def test_run_zero_steps():
    tgtDensity = GaussianTargetDensity(Scalar(0.), 1.)
    proposalCov = IIDCovarianceMatrix(1, 0.5)
    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, DummyDiagnostics())
    
    initState = Scalar(2.0)
    mc.run(0, initState)
    
    assert len(mc.chain.trajectory) == 1
    assert np.allclose(mc.chain.trajectory[0], initState.coordinate)


def test_metropolis_within_gibbs():
    from styne.model.trend import ConstantTrend
    from styne.model.sglmm import SGLMM
    from styne.statistics.likelihood import SGLMMLikelihood
    from styne.statistics.response import PoissonResponse
    from styne.statistics.data import Data
    from styne.gp.gaussianprocess import GaussianProcess
    from styne.statistics.stationary import MaternCovariance1D
    from styne.mcmc.method.pcn import PCNFactory
    from styne.utility.tuning import PCNTuner
    from styne.statistics.conditional import MetropolisWithinGibbsConditional
    from styne.mcmc.method.gibbs import GibbsBuilder
    from styne.statistics.bayes import HierarchicalBayes
    from styne.utility.grid import Grid
    from styne.statistics.radonnikodym import RadonNikodym
    from styne.parameter.block import BlockParameter

    # Simple 1D Poisson setup
    sites = Grid(np.linspace(0, 1, 10))
    data = Data(1, sites.to_array())
    data.measurement = np.ones((10, 1))


    
    covFcn = MaternCovariance1D(0.2, 1.5, 1.0)
    gp = GaussianProcess.dna(covFcn, q=20, d=1)
    predictor = SGLMM(gp, sites, trend=ConstantTrend(0.0))

    likelihood = SGLMMLikelihood(data, predictor, PoissonResponse())
    
    latentTarget = RadonNikodym(gp.measure, likelihood)
    rng = default_rng(42)
    latentInit = gp.measure.generate_realisation(rng=rng)
    
    latentFactory = PCNFactory()
    latentFactory.target = latentTarget
    latentFactory.beta = 0.5
    latentFactory.rng = rng
    mcmc = latentFactory.create()


    
    # Simple block conditional
    cond = MetropolisWithinGibbsConditional(mcmc, blockIdx=0, nSteps=2)
    joint = HierarchicalBayes(conditionals=[cond], root=gp.measure)
    
    builder = GibbsBuilder()
    builder.model = joint
    builder.rng = rng
    gibbs = builder.build()
    
    nSteps = 5
    initState = BlockParameter([latentInit])
    gibbs.run(nSteps, initState)
    
    assert len(gibbs.chain.block(0).trajectory) == nSteps + 1


def test_metropolis_within_gibbs_does_not_mutate_template_sampler():
    """Each conditioned inner sampler is isolated from the template."""
    from styne.model.trend import ConstantTrend
    from styne.model.sglmm import SGLMM
    from styne.statistics.likelihood import SGLMMLikelihood
    from styne.statistics.response import PoissonResponse
    from styne.statistics.data import Data
    from styne.gp.gaussianprocess import GaussianProcess
    from styne.statistics.stationary import MaternCovariance1D
    from styne.mcmc.method.pcn import PCNFactory
    from styne.statistics.conditional import MetropolisWithinGibbsConditional
    from styne.mcmc.method.gibbs import GibbsBuilder
    from styne.statistics.bayes import HierarchicalBayes
    from styne.utility.grid import Grid
    from styne.statistics.radonnikodym import RadonNikodym
    from styne.parameter.block import BlockParameter

    sites = Grid(np.linspace(0, 1, 10))
    data = Data(1, sites.to_array())
    data.measurement = np.ones((10, 1))

    covFcn = MaternCovariance1D(0.2, 1.5, 1.0)
    gp = GaussianProcess.dna(covFcn, q=20, d=1)
    predictor = SGLMM(gp, sites, trend=ConstantTrend(0.0))
    likelihood = SGLMMLikelihood(data, predictor, PoissonResponse())
    latentTarget = RadonNikodym(gp.measure, likelihood)
    rng = default_rng(43)
    latentInit = gp.measure.generate_realisation(rng=rng)

    latentFactory = PCNFactory()
    latentFactory.target = latentTarget
    latentFactory.beta = 0.5
    latentFactory.rng = rng
    mcmc = latentFactory.create()

    nStepsPerSweep = 3
    cond = MetropolisWithinGibbsConditional(
        mcmc, blockIdx=0, nSteps=nStepsPerSweep
    )
    mcmc.storeChain = True
    joint = HierarchicalBayes(conditionals=[cond], root=gp.measure)

    builder = GibbsBuilder()
    builder.model = joint
    builder.rng = rng
    gibbs = builder.build()

    nGibbsSweeps = 4
    initState = BlockParameter([latentInit])
    gibbs.run(nGibbsSweeps, initState)

    assert len(mcmc.chain.trajectory) == 0




if __name__ == "__main__":
    pytest.main()
