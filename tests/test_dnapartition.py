import numpy as np
from numpy.random import default_rng

from styne.gp.gaussianprocess import GaussianProcess
from styne.gp.dnautility import DNACoarseFinePartition
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid
from styne.utility.partition import IndependentPartitionDensity


def test_partitioned_parameter_evaluation():
    # Setup baseline DNA GP
    covFcn = MaternCovariance1D(0.2, 2.5, 1.0)
    q = 12
    qC = 4
    d = 1
    
    dnaGP = GaussianProcess.dna(covFcn, q=q, d=d)
    sites = UniformGrid(0., 1., 30)
    dnaGP.sites = sites
    
    # Setup partition
    partition = DNACoarseFinePartition(dnaGP, qC, d)
    coarseMeas = partition.coarse_measure()
    fineMeas = partition.fine_measure()
    
    # 1. Dimension alignment check
    assert coarseMeas.density.domainDimension + fineMeas.density.domainDimension == dnaGP.parameterDimension
    
    rng = default_rng(42)
    coarseSample = coarseMeas.draw(rng)
    fineSample = fineMeas.draw(rng)
    
    # 2. Field evaluation linearity and exact reflection check
    mergedParam = partition._rule.merge([coarseSample.coordinate, fineSample.coordinate])
    
    # Full merged evaluation
    evalMerged = dnaGP.at_sites(mergedParam)
    
    # Coarse only evaluation
    coarsePadded = partition._rule.merge([coarseSample.coordinate, np.zeros_like(fineSample.coordinate)])
    evalCoarseOnly = dnaGP.at_sites(coarsePadded)
    
    # Fine only evaluation
    finePadded = partition._rule.merge([np.zeros_like(coarseSample.coordinate), fineSample.coordinate])
    evalFineOnly = dnaGP.at_sites(finePadded)
    
    np.testing.assert_allclose(evalMerged, evalCoarseOnly + evalFineOnly, 
                               err_msg="Merged parameter field evaluation should equal sum of zero-padded evaluations.")


def test_independent_partition_density_equivalence():
    # Setup baseline DNA GP
    covFcn = MaternCovariance1D(0.2, 2.5, 1.0)
    q = 12
    qC = 4
    d = 1
    
    dnaGP = GaussianProcess.dna(covFcn, q=q, d=d)
    
    # Setup partition
    partition = DNACoarseFinePartition(dnaGP, qC, d)
    coarseMeas = partition.coarse_measure()
    fineMeas = partition.fine_measure()
    
    rng = default_rng(42)
    coarseSample = coarseMeas.draw(rng)
    fineSample = fineMeas.draw(rng)
    
    # Merge
    mergedCoord = partition._rule.merge([coarseSample.coordinate, fineSample.coordinate])
    mergedParam = dnaGP.parameter.clone()
    mergedParam.coordinate = mergedCoord
    
    # Independent Partition Density
    partDens = IndependentPartitionDensity(
        partition, [coarseMeas.density, fineMeas.density]
    )
    
    # Log Density Value
    valPart = partDens.evaluate_log(mergedParam)
    valFull = dnaGP.measure.density.evaluate_log(mergedParam)
    
    np.testing.assert_allclose(valPart, valFull, 
                               err_msg="Partitioned log-density evaluation should be identical to joint evaluation.")

    # Log Density Gradient
    gradPart = partDens.evaluate_log_gradient(mergedParam)
    gradFull = dnaGP.measure.density.evaluate_log_gradient(mergedParam)

    np.testing.assert_allclose(gradPart, gradFull,
                               err_msg="Partitioned log-gradient evaluation should be identical to joint evaluation.")
