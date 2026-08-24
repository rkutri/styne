import pytest
import numpy as np
from styne.parameter.vector import Vector
from styne.statistics.gaussian import GaussianDensity
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.mcmc.method.dart import DARTFactory, DART
from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.transition import TransitionData


@pytest.mark.parametrize(
    ("root", "invalid_setting", "valid_setting"),
    [
        ("pcn", "proposalCovariance", "beta"),
        ("mrw", "beta", "proposalCovariance"),
        ("mala", "beta", "stepSize"),
        ("pmala", "proposalCovariance", "beta"),
    ],
)
def test_dart_root_rejects_unsupported_settings(
        root, invalid_setting, valid_setting):
    """Root factories must not silently accept settings for other methods."""
    factory = DARTFactory(root=root)

    with pytest.raises(AttributeError, match=valid_setting):
        setattr(factory.root, invalid_setting, 1.0)


def test_dart_factory_minimal():
    """Verify that DARTFactory builds a valid 1-level sampler."""
    dim = 2
    kappa = 1.0
    h = 0.5
    
    # Target and Surrogate
    modelMean = Vector(np.zeros(dim))
    targetCov = IIDCovarianceMatrix(dim, float(kappa))
    target = GaussianDensity(targetCov, modelMean)
    
    surrogCov = IIDCovarianceMatrix(dim, float(2.0 * kappa))
    surrogate = GaussianDensity(surrogCov, modelMean)
    
    # Factory setup
    factory = DARTFactory(root="mrw")
    factory.target = target
    factory.surrogate = [surrogate]
    factory.regularisation = [h]
    factory.nChain = [10]
    factory.burnin = 2
    factory.thinning = 1
    factory.diagnostics = DummyDiagnostics()
    
    # Root proposal setup
    factory.root.proposalCovariance = IIDCovarianceMatrix(dim, 0.1)
    
    sampler = factory.create()
    
    assert isinstance(sampler, DART)
    assert sampler.target == target
    # The surrogate density in the sampler should be a LocalisedSurrogateDensity
    assert sampler._proposalMethod.density.regularisation == h

def test_dart_factory_multi_level():
    """Verify that DARTFactory builds a nested sampler for multiple levels."""
    dim = 2
    
    modelMean = Vector(np.zeros(dim))
    target = GaussianDensity(IIDCovarianceMatrix(dim, 1.0), modelMean)
    
    surr1 = GaussianDensity(IIDCovarianceMatrix(dim, 2.0), modelMean)
    surr2 = GaussianDensity(IIDCovarianceMatrix(dim, 4.0), modelMean)
    
    factory = DARTFactory(root="mrw")
    factory.target = target
    factory.surrogate = [surr2, surr1] # Level 0 is most coarse
    factory.regularisation = [0.1, 0.5]
    factory.nChain = [10, 20]
    factory.diagnostics = DummyDiagnostics()
    factory.root.proposalCovariance = IIDCovarianceMatrix(dim, 0.1)
    
    sampler = factory.create()
    
    assert isinstance(sampler, DART)
    # Level 2 (main) has surrogate Measure from Level 1
    # Check that surrogateMeasure's MCMC is Level 1's DART
    surrMeasure = sampler._proposalMethod.measure
    assert isinstance(surrMeasure.mcmc, DART)
    
    # Level 1's surrogate Measure has MCMC from root (MRW)
    rootMeasure = surrMeasure.mcmc._proposalMethod.measure
    from styne.mcmc.metropolishastings import MetropolisHastings
    assert isinstance(rootMeasure.mcmc, MetropolisHastings)
    assert not isinstance(rootMeasure.mcmc, DART)

def test_dart_factory_inconsistent_lengths():
    """Verify that DARTFactory raises ValueError for mismatched parameter list lengths."""
    factory = DARTFactory()
    
    # Use a real density to pass super()._validate()
    dim = 2
    factory.target = GaussianDensity(IIDCovarianceMatrix(dim, 1.0), Vector(np.zeros(dim)))
    
    factory.surrogate = [factory.target, factory.target]
    factory.regularisation = [1.0]
    factory.nChain = [10, 10]
    
    with pytest.raises(ValueError, match="Inconsistent parameter lengths"):
        factory.create()


def test_partitioned_dart_acceptance_invariant():
    """Assert partitioned DART _log_mh_ratio equals target + correction invariant."""
    from styne.gp.gaussianprocess import GaussianProcess
    from styne.gp.dnautility import DNACoarseFinePartition
    from styne.statistics.stationary import MaternCovariance1D
    from styne.utility.partition import IndependentPartitionDensity

    gp = GaussianProcess.dna(MaternCovariance1D(0.2, 2.5, 1.0), q=6, d=1)
    partition = DNACoarseFinePartition(gp, 2, 1)
    coarseDens = partition.coarse_measure().density
    fineDens = partition.fine_measure().density
    target = IndependentPartitionDensity(partition, [coarseDens, fineDens])

    factory = DARTFactory(root='mrw')
    factory.target = target
    factory.surrogate = [coarseDens]
    factory.partition = partition
    factory.finePrior = partition.fine_measure()
    factory.regularisation = [0.1]
    factory.burnin = 0
    factory.thinning = 1
    factory.nChain = [20]
    coarseDim = coarseDens.domainDimension
    factory.root.proposalCovariance = IIDCovarianceMatrix(coarseDim, 0.1)

    sampler = factory.create()
    rng = np.random.default_rng(42)
    state = gp.measure.generate_realisation(seed=42)
    trans, _ = sampler._proposalMethod.propose(state, rng)

    rule = partition.rule
    stateC = Vector(rule.extract(0, trans.state.coordinate))
    proposalC = Vector(rule.extract(0, trans.proposal.coordinate))
    stateF = Vector(rule.extract(1, trans.state.coordinate))
    proposalF = Vector(rule.extract(1, trans.proposal.coordinate))

    surrDens = sampler._proposalMethod.density
    logDiffSurr = (surrDens.evaluate_log_surrogate(proposalC)
                   - surrDens.evaluate_log_surrogate(stateC))
    ratioEst = sampler._proposalMethod._coarseProposal.correction
    logRatioEst = ratioEst.log_ratio_estimate(
        stateC,
        proposalC,
        trans.auxiliary["surrogateTrajectory"],
        trans.auxiliary["proposalTrajectory"],
    )
    print(f"Computed logRatioEst: {logRatioEst}")
    assert logRatioEst != 0.0
    coarseCorrection = -logDiffSurr - logRatioEst

    logDiffFinePrior = (fineDens.evaluate_log(proposalF)
                        - fineDens.evaluate_log(stateF))
    fineCorrection = -logDiffFinePrior

    logDiffTarget = (target.evaluate_log(trans.proposal)
                     - target.evaluate_log(trans.state))
    expected = logDiffTarget + coarseCorrection + fineCorrection
    evaluatedTransition = TransitionData(
        current=sampler.evaluate_state(trans.state),
        proposed=sampler.evaluate_state(trans.proposal),
        auxiliary=trans.auxiliary,
    )
    assert np.isclose(sampler._log_mh_ratio(evaluatedTransition), expected)


def test_partitioned_dart_pcn_fine_edge_case():
    """Verify beta=1 pCN fine draw matches independence prior realization."""
    from styne.gp.gaussianprocess import GaussianProcess
    from styne.gp.dnautility import DNACoarseFinePartition
    from styne.statistics.stationary import MaternCovariance1D
    from styne.mcmc.method.pcn import PCNProposal

    gp = GaussianProcess.dna(MaternCovariance1D(0.2, 2.5, 1.0), q=6, d=1)
    partition = DNACoarseFinePartition(gp, 2, 1)
    finePrior = partition.fine_measure()

    assert np.all(finePrior.mean.coordinate == 0.0)

    kernel = PCNProposal(finePrior, 1.0)
    kernelState = finePrior.generate_realisation(seed=1)

    rngKernel = np.random.default_rng(42)
    rngPrior = np.random.default_rng(42)

    prop = kernel.propose(kernelState, rngKernel)[0].proposal
    draw = finePrior.generate_realisation(rng=rngPrior)

    np.testing.assert_array_equal(prop.coordinate, draw.coordinate)
