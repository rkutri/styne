import numpy as np
import unittest
from styne.gp.gaussianprocess import GaussianProcess
from styne.model.sglmm import SGLMM
from styne.statistics.stationary import MaternCovariance1D
from styne.statistics.response import PoissonResponse
from styne.statistics.likelihood import SGLMMLikelihood
from styne.statistics.data import Data
from styne.statistics.hierarchical import SGLMMHyperConditional
from styne.parameter.vector import Vector
from styne.parameter.block import BlockParameter

class TestHierarchicalLogic(unittest.TestCase):
    def test_whitened_hyper_conditional(self):
        """
        Verify that SGLMMHyperConditionalDensity correctly uses the whitened GP.
        """
        # Setup
        axis = np.linspace(0, 1, 100)
        from styne.utility.grid import UniformGrid
        sites = UniformGrid(0, 1, 100)
        
        ell, var = 0.2, 1.5
        covFcn = MaternCovariance1D(ell, 1.5, 1.0)
        gp = GaussianProcess.dna(covFcn, q=50, d=1)
        
        predictor = SGLMM(gp, sites)
        response = PoissonResponse()
        data = Data(1, sites.to_array())
        data.measurement = np.ones((100, 1))
        likelihood = SGLMMLikelihood(data, predictor, response)
        
        from styne.statistics.pc import JointMaternPCPrior
        pcPrior = JointMaternPCPrior(0.5, 0.05, 3.0, 0.05)
        
        hyperCond = SGLMMHyperConditional(pcPrior, gp, predictor, likelihood, hyperIdx=1)
        
        # State: [latent, log-hyper]
        latent = Vector(np.zeros(gp.parameterDimension))
        hyper = Vector(np.log([0.3, 1.2])) # log(rho), log(sigma)
        state = BlockParameter([latent, hyper])
        
        # This triggers condition_on which builds the SGLMMHyperConditionalDensity
        hyperCond.condition_on(state)
        density = hyperCond.density
        
        # Evaluate log-density at current hyperstate
        logp = density.evaluate_log(hyper)
        
        self.assertTrue(np.isfinite(logp))
        
        # Verify that the GP measure covariance is now Identity (Non-centered)
        from styne.statistics.covariance import IIDCovarianceMatrix
        self.assertIsInstance(gp.measure.covariance, IIDCovarianceMatrix)
        self.assertEqual(gp.measure.covariance.scaling, 1.0)

    def test_hyper_conditional_recomputes_for_new_parameters(self):
        """
        Sequential evaluations use their supplied hyperparameters.
        """
        axis = np.linspace(0, 1, 100)
        from styne.utility.grid import UniformGrid
        sites = UniformGrid(0, 1, 100)
        
        ell, var = 0.2, 1.0
        covFcn = MaternCovariance1D(ell, 1.5, var)
        gp = GaussianProcess.dna(covFcn, q=50, d=1)
        
        predictor = SGLMM(gp, sites)
        response = PoissonResponse()
        data = Data(1, sites.to_array())
        data.measurement = np.ones((100, 1))
        likelihood = SGLMMLikelihood(data, predictor, response)
        
        from styne.statistics.pc import JointMaternPCPrior
        pcPrior = JointMaternPCPrior(0.5, 0.05, 3.0, 0.05)
        
        hyperCond = SGLMMHyperConditional(pcPrior, gp, predictor, likelihood, hyperIdx=1)
        
        latent = Vector(np.ones(gp.parameterDimension))
        hyper1 = Vector(np.log([0.2, 1.0])) # log(rho), log(sigma)
        hyper2 = Vector(np.log([0.2, 10000.0])) # significantly different variance
        state1 = BlockParameter([latent, hyper1])
        
        hyperCond.condition_on(state1)
        density = hyperCond.density
        self.assertFalse(hasattr(density, "_cachedProposalParams"))
        
        logp1 = density.evaluate_log(hyper1)
        eval1 = predictor(latent).copy()
        
        logp2 = density.evaluate_log(hyper2)
        eval2 = predictor(latent).copy()
        
        self.assertNotEqual(logp1, logp2)
        self.assertTrue(np.allclose(eval1, eval2))

if __name__ == '__main__':
    unittest.main()
