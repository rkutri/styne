import pytest
import numpy as np
from styne.parameter.vector import Vector
from styne.statistics.gaussian import Gaussian
from styne.statistics.bayes import UnnormalisedPosterior
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse
from styne.statistics.data import Data
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.mlda import MLDAFactory
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from tests.conftest import MockIdentityForwardMap

def setup_mcmc_factory(factory_class, dim, gamma=None, tempering=1.0):
    """Common setup for MLDA and DART factories on a Gaussian posterior."""
    
    # 1. Prior
    priorCov = IIDCovarianceMatrix(dim, 1.0)
    prior = Gaussian(priorCov, Vector(np.zeros(dim)))
    
    # 2. Likelihood (Fine)
    data = Data(1, np.zeros((1, dim)))
    data.measurement = np.zeros((dim, 1))
    
    # Identity model + Gaussian noise (diagonal covariance diag(1))
    model = MockIdentityForwardMap(dim)
    noise = GaussianResponse(IIDCovarianceMatrix(dim, 1.0))
    
    fineLikelihood = RegressionLikelihood(data, model, noise)
    
    target = UnnormalisedPosterior(prior, fineLikelihood)
    
    # 3. Surrogate (Coarse - same as fine for testing equivalence)
    surrogate = UnnormalisedPosterior(prior, fineLikelihood)
    
    factory = factory_class(root="pcn")
    factory.target = target
    factory.surrogate = [surrogate]
    factory.nChain = [10]
    factory.diagnostics = AcceptanceRateDiagnostics()
    factory.root.beta = 0.5
    
    if factory_class.__name__ == "DARTFactory":
        factory.regularisation = [gamma if gamma is not None else 1e-6]
        factory.tempering = [tempering]
        
    return factory

def test_dart_mlda_equivalence_limit():
    """Verify DART matches MLDA when gamma -> 0 and tempering = 1."""
    dim = 2
    nSteps = 200
    seed = 42
    
    # MLDA
    mldaFactory = setup_mcmc_factory(MLDAFactory, dim)
    mldaSampler = mldaFactory.create()
    
    # DART (gamma very small, tempering 1.0)
    dartFactory = setup_mcmc_factory(DARTFactory, dim, gamma=1e-10, tempering=1.0)
    dartSampler = dartFactory.create()
    
    # Use deterministic seed via global RNG (as run() doesn't take rng)
    np.random.seed(seed)
    mldaSampler.run(nSteps, Vector(np.zeros(dim)))
    mldaAcc = mldaSampler.diagnostics.global_acceptance_rate()
    
    np.random.seed(seed)
    dartSampler.run(nSteps, Vector(np.zeros(dim)))
    dartAcc = dartSampler.diagnostics.global_acceptance_rate()
    
    assert dartAcc > 0.05
    # DART uses RatioEstimator which adds its own sub-chain sampling.
    assert np.abs(dartAcc - mldaAcc) < 0.2

def test_dart_stability_parameters():
    """Ensure DART diagnostics are stable as gamma and tempering vary."""
    dim = 2
    nSteps = 100
    
    # Test a range of gammas and tempering values
    gammas = [1e-4, 1e-2, 0.1]
    temperings = [0.8, 1.0]
    
    lastAcc = None
    
    for g in gammas:
        for t in temperings:
            factory = setup_mcmc_factory(DARTFactory, dim, gamma=g, tempering=t)
            sampler = factory.create()
            
            sampler.run(nSteps, Vector(np.zeros(dim)))
            acc = sampler.diagnostics.global_acceptance_rate()
            
            assert acc > 0.0, f"DART failed with gamma={g}, tempering={t}"
            
            if lastAcc is not None:
                assert np.abs(acc - lastAcc) < 0.9
            lastAcc = acc

def test_dart_preconditioning():
    """Verify that root pCN preconditions with the prior for RadonNikodym targets."""
    dim = 2
    factory = setup_mcmc_factory(DARTFactory, dim, gamma=0.1)
    sampler = factory.create()
    
    root_mcmc = sampler._proposalMethod.measure.mcmc
    root_proposal = root_mcmc._proposalMethod
    
    # Check that root proposal reference Measure matches the prior domain dimension
    # Use .density.domainDimension for AbsolutelyContinuousProbabilityMeasure
    assert root_proposal.referenceMeasure.density.domainDimension == dim
    assert root_proposal.referenceMeasure.covariance.scaling == 1.0
