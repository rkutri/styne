import unittest
import numpy as np
from styne.gp.dna import DNAFourierEngine
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid

class TestDNAJacobian(unittest.TestCase):

    def test_apply_jacobian_multi_direction_1d(self):
        """Verify 1D DNA engine handles multiple directions in apply_jacobian."""
        q = 50
        engine = DNAFourierEngine(q, d=1)
        sites = UniformGrid(0., 1., 20)
        engine.set_sites(sites)
        
        # Build covariance to initialize weights
        covFcn = MaternCovariance1D(0.2, 1.5, 1.0)
        engine.build_covariance(covFcn)
        
        n_total = engine.spectralWeights.size
        n_dir = 5
        
        # Matrix input: (spectral_dim, n_directions)
        v = np.random.randn(n_total, n_dir)
        
        try:
            res = engine.apply_jacobian(v, None)
            self.assertEqual(res.shape, (len(sites), n_dir))
        except ValueError as e:
            self.fail(f"apply_jacobian failed with multiple directions: {e}")

    def test_apply_adjoint_jacobian_multi_direction_1d(self):
        """Verify 1D DNA engine handles multiple directions in apply_adjoint_jacobian."""
        q = 50
        engine = DNAFourierEngine(q, d=1)
        sites = UniformGrid(0., 1., 20)
        engine.set_sites(sites)
        
        # Build covariance to initialize weights
        covFcn = MaternCovariance1D(0.2, 1.5, 1.0)
        engine.build_covariance(covFcn)
        
        n_total = engine.spectralWeights.size
        n_dir = 5
        
        # Matrix input: (n_sites, n_directions)
        w = np.random.randn(len(sites), n_dir)
        
        try:
            res = engine.apply_adjoint_jacobian(w, None)
            self.assertEqual(res.shape, (n_total, n_dir))
        except ValueError as e:
            self.fail(f"apply_adjoint_jacobian failed with multiple directions: {e}")

    def test_apply_jacobian_multi_direction_2d(self):
        """Verify 2D DNA engine handles multiple directions in apply_jacobian."""
        q = 10
        engine = DNAFourierEngine(q, d=2)
        sites = UniformGrid((0., 1., 5), (0., 1., 5))
        engine.set_sites(sites)
        
        # Mock 2D covariance evaluation for weights
        class MockCov2D:
            def evaluate_fourier(self, xi):
                return np.ones(len(xi))
        
        engine.build_covariance(MockCov2D())
        
        n_total = engine.spectralWeights.size
        n_dir = 3
        
        v = np.random.randn(n_total, n_dir)
        
        try:
            res = engine.apply_jacobian(v, None)
            self.assertEqual(res.shape, (len(sites), n_dir))
        except ValueError as e:
            self.fail(f"apply_jacobian failed in 2D with multiple directions: {e}")

    def test_apply_adjoint_jacobian_multi_direction_2d(self):
        """Verify 2D DNA engine handles multiple directions in apply_adjoint_jacobian."""
        q = 10
        engine = DNAFourierEngine(q, d=2)
        sites = UniformGrid((0., 1., 5), (0., 1., 5))
        engine.set_sites(sites)
        
        class MockCov2D:
            def evaluate_fourier(self, xi):
                return np.ones(len(xi))
        
        engine.build_covariance(MockCov2D())
        
        n_total = engine.spectralWeights.size
        n_dir = 3
        
        w = np.random.randn(len(sites), n_dir)
        
        try:
            res = engine.apply_adjoint_jacobian(w, None)
            self.assertEqual(res.shape, (n_total, n_dir))
        except ValueError as e:
            self.fail(f"apply_adjoint_jacobian failed in 2D with multiple directions: {e}")

if __name__ == '__main__':
    unittest.main()
