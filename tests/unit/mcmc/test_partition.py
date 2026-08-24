import numpy as np

from numpy.random import default_rng

from styne.gp.gaussianprocess import GaussianProcess
from styne.gp.dnautility import DNACoarseFinePartition
from styne.model.sglmm import SGLMM
from styne.statistics.data import Data
from styne.statistics.likelihood import SGLMMLikelihood
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.response import PoissonResponse
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import Grid
from styne.utility.partition import IndependentPartitionDensity

from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.method.mala import MALAFactory
from styne.mcmc.method.pcn import PCNFactory
from styne.mcmc.method.pmala import PMALAFactory

from styne.statistics.covariance import IIDCovarianceMatrix

def setup_target():
    DIM = 1
    gtResolution = 200
    dnaResolution = 50
    coarseResolution = 10
    nObs = 100
    ell = 0.1
    nu = 2.5
    margVar = 1.0
    trueIntensity = 5.0
    
    rng = default_rng(42)

    covFcn = MaternCovariance1D(ell, nu, margVar)

    trueGP = GaussianProcess.dna(covFcn, q=gtResolution, d=1)
    obsSites = Grid(np.sort(rng.uniform(0., 1., nObs)))
    trueRealisation = trueGP.sampler.generate_realisation(rng=rng)
    
    groundTruth = trueRealisation.evaluate(obsSites)
    poissonResponse = PoissonResponse()
    y = poissonResponse.simulate(groundTruth + np.log(trueIntensity), rng=rng)

    data = Data(1, obsSites.to_array())
    data.measurement = y.coordinate.reshape(-1, 1)

    dnaPrior = GaussianProcess.dna(covFcn, q=dnaResolution, d=1)
    dnaInit = dnaPrior.measure.mean.clone()

    partition = DNACoarseFinePartition(dnaPrior, coarseResolution, d=DIM)
    
    coarseGP = GaussianProcess.dna(covFcn, q=coarseResolution, d=DIM)
    
    coarsePredictor = SGLMM(coarseGP, obsSites)
    coarseSurrogate = RadonNikodym(
        partition.coarse_measure(),
        SGLMMLikelihood(data, coarsePredictor, poissonResponse)
    )

    fineSurrogate = partition.fine_measure()

    surrogate = IndependentPartitionDensity(partition, [coarseSurrogate, fineSurrogate.density])
    
    return surrogate, dnaInit


def _test_sampler(name, factory, init_state, n_steps=200):
    print(f"\n--- Testing {name} ---")
    sampler = factory.create()
    state = init_state.with_coordinate(
        np.random.randn(init_state.dimension) * 0.01
    )

    sampler.run(n_steps, state)
    acc_rate = sampler.diagnostics.global_acceptance_rate()
    print(f"Acceptance rate: {acc_rate:.3f}")
    
    # Check if chain moved
    diff = np.linalg.norm(sampler.chain.trajectory[-1] - sampler.chain.trajectory[0])
    print(f"Distance moved from initial state: {diff:.3e}")
    return sampler


if __name__ == "__main__":
    target, init_state = setup_target()

    from styne.parameter.vector import Vector

    # Diagnostic: Check initial log-density
    print(f"Initial log-density: {target.evaluate_log(init_state):.3f}")

    # MRW
    f_mrw = MRWFactory()
    f_mrw.target = target
    f_mrw.proposalCovariance = IIDCovarianceMatrix(init_state.dimension, 1e-6) # Smaller noise
    s_mrw = _test_sampler("MRW", f_mrw, init_state)

    # MALA
    f_mala = MALAFactory()
    f_mala.target = target
    f_mala.stepSize = 1e-4
    _test_sampler("MALA", f_mala, init_state)

    # pCN on the coarseSurrogate directly
    print("\n--- Testing pCN on CoarseSurrogate directly ---")
    f_pcn = PCNFactory()
    f_pcn.target = target._densities[0] # The RadonNikodym coarse surrogate
    f_pcn.beta = 0.1
    coarse_init = Vector(target._partition._rule.extract(0, init_state.coordinate))
    _test_sampler("pCN (Coarse)", f_pcn, coarse_init)

    # PMALA on the coarseSurrogate directly
    print("\n--- Testing PMALA on CoarseSurrogate directly ---")
    f_pmala = PMALAFactory()
    f_pmala.target = target._densities[0]
    f_pmala.beta = 0.01
    _test_sampler("PMALA (Coarse)", f_pmala, coarse_init)


    # Final check: does log-density vary?
    print("\nChecking log-density variation in MRW trajectory:")
    for i in [0, 10, 50, 100]:
        param = init_state.with_coordinate(s_mrw.chain.trajectory[i])
        print(f"Step {i:3d}: log-density = {target.evaluate_log(param):.6f}")
