import unittest
import numpy as np
from styne.gp.dna import DNAFourierExpansion
from styne.utility.grid import UniformGrid

class TestDNAJacobian(unittest.TestCase):

    def test_directional_derivative_multi_direction_1d(self):
        """Verify 1D DNA expansion handles multiple directions."""
        q = 50
        expansion = DNAFourierExpansion(q, d=1)
        sites = UniformGrid(0., 1., 20)
        evaluation = expansion.bind(sites)
        n_total = expansion.dimension
        n_dir = 5
        
        # Batched input: (n_directions, spectral_dim)
        v = np.random.randn(n_dir, n_total)
        
        try:
            res = evaluation.directional_derivative(np.zeros(n_total), v)
            self.assertEqual(res.shape, (n_dir, len(sites)))
        except ValueError as e:
            self.fail(f"directional derivative failed with multiple directions: {e}")

    def test_adjoint_derivative_multi_cotangent_1d(self):
        """Verify 1D DNA expansion handles multiple cotangents."""
        q = 50
        expansion = DNAFourierExpansion(q, d=1)
        sites = UniformGrid(0., 1., 20)
        evaluation = expansion.bind(sites)
        n_total = expansion.dimension
        n_dir = 5
        
        # Batched input: (n_directions, n_sites)
        w = np.random.randn(n_dir, len(sites))
        
        try:
            res = evaluation.adjoint_derivative(np.zeros(n_total), w)
            self.assertEqual(res.shape, (n_dir, n_total))
        except ValueError as e:
            self.fail(f"adjoint derivative failed with multiple cotangents: {e}")

    def test_directional_derivative_multi_direction_2d(self):
        """Verify 2D DNA expansion handles multiple directions."""
        q = 10
        expansion = DNAFourierExpansion(q, d=2)
        sites = UniformGrid((0., 1., 5), (0., 1., 5))
        evaluation = expansion.bind(sites)
        n_total = expansion.dimension
        n_dir = 3
        
        v = np.random.randn(n_dir, n_total)
        
        try:
            res = evaluation.directional_derivative(np.zeros(n_total), v)
            self.assertEqual(res.shape, (n_dir, len(sites)))
        except ValueError as e:
            self.fail(f"directional derivative failed in 2D: {e}")

    def test_adjoint_derivative_multi_cotangent_2d(self):
        """Verify 2D DNA expansion handles multiple cotangents."""
        q = 10
        expansion = DNAFourierExpansion(q, d=2)
        sites = UniformGrid((0., 1., 5), (0., 1., 5))
        evaluation = expansion.bind(sites)
        n_total = expansion.dimension
        n_dir = 3
        
        w = np.random.randn(n_dir, len(sites))
        
        try:
            res = evaluation.adjoint_derivative(np.zeros(n_total), w)
            self.assertEqual(res.shape, (n_dir, n_total))
        except ValueError as e:
            self.fail(f"adjoint derivative failed in 2D: {e}")

if __name__ == '__main__':
    unittest.main()
