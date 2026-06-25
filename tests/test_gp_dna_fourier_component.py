import numpy as np
import pytest

from numpy.random import default_rng

from styne.gp.dna import (
    BC, BoundaryCondition,
    DNAFourierComponentRealisation,
)


# ---- helpers ----

def _bc1d(bcType):
    return BoundaryCondition((bcType,))


def _bc2d(bcX, bcY):
    return BoundaryCondition((bcX, bcY))


# ---- DNAFourierComponentRealisation API (1D) ----

class TestDNAFourierComponentRealisation1D:

    def setup_method(self):
        self.q = 5

    def test_neumann_dimension(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.NEUMANN), self.q)
        assert r.dimension == self.q + 1

    def test_dirichlet_dimension(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.DIRICHLET), self.q)
        assert r.dimension == self.q

    def test_coefficient_round_trip(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.NEUMANN), self.q)
        coeff = np.arange(self.q + 1, dtype=float)
        r.coefficient = coeff
        np.testing.assert_allclose(r.coefficient, coeff)

    def test_evaluate_interior_shape_neumann(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.NEUMANN), self.q)
        r.coefficient = np.zeros(self.q + 1)
        assert r.evaluate_interior().shape == (self.q + 2,)

    def test_evaluate_interior_shape_dirichlet(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.DIRICHLET), self.q)
        r.coefficient = np.zeros(self.q)
        assert r.evaluate_interior().shape == (self.q + 2,)

    def test_dirichlet_zero_at_endpoints(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.DIRICHLET), self.q)
        r.coefficient = default_rng(0).standard_normal(self.q)
        vals = r.evaluate_interior()
        assert abs(vals[0]) < 1e-14
        assert abs(vals[-1]) < 1e-14

    def test_zero_coefficient_zero_field_neumann(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.NEUMANN), self.q)
        r.coefficient = np.zeros(self.q + 1)
        np.testing.assert_allclose(r.evaluate_interior(), 0.)

    def test_zero_coefficient_zero_field_dirichlet(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.DIRICHLET), self.q)
        r.coefficient = np.zeros(self.q)
        np.testing.assert_allclose(r.evaluate_interior(), 0.)

    def test_evaluate_raises(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.NEUMANN), self.q)
        r.coefficient = np.zeros(self.q + 1)
        with pytest.raises(NotImplementedError):
            r.evaluate(np.linspace(0.1, 0.9, 15))

    def test_clone_independence(self):
        r = DNAFourierComponentRealisation(_bc1d(BC.NEUMANN), self.q)
        r.coefficient = np.ones(self.q + 1)
        clone = r.clone()
        clone.coefficient = np.zeros(self.q + 1)
        np.testing.assert_allclose(r.coefficient, 1.)


# ---- DNAFourierComponentRealisation API (2D) ----

class TestDNAFourierComponentRealisation2D:

    def setup_method(self):
        self.q = 4

    def test_block_sizes(self):
        q = self.q
        assert DNAFourierComponentRealisation(
            _bc2d(BC.NEUMANN, BC.NEUMANN),
            q).dimension == (
            q + 1) ** 2
        assert DNAFourierComponentRealisation(
            _bc2d(BC.DIRICHLET, BC.NEUMANN), q).dimension == q * (q + 1)
        assert DNAFourierComponentRealisation(
            _bc2d(BC.NEUMANN, BC.DIRICHLET),
            q).dimension == (
            q + 1) * q
        assert DNAFourierComponentRealisation(
            _bc2d(BC.DIRICHLET, BC.DIRICHLET),
            q).dimension == q ** 2

    def test_evaluate_interior_shape(self):
        q = self.q
        for bc in BoundaryCondition.all_combinations(2):
            r = DNAFourierComponentRealisation(bc, q)
            r.coefficient = np.zeros(r.dimension)
            assert r.evaluate_interior().shape == (q + 2, q + 2)

    def test_dirichlet_x_zero_boundary(self):
        r = DNAFourierComponentRealisation(
            _bc2d(BC.DIRICHLET, BC.NEUMANN), self.q)
        r.coefficient = default_rng(1).standard_normal(r.dimension)
        vals = r.evaluate_interior()
        np.testing.assert_allclose(vals[0, :], 0., atol=1e-13)
        np.testing.assert_allclose(vals[-1, :], 0., atol=1e-13)

    def test_dirichlet_y_zero_boundary(self):
        r = DNAFourierComponentRealisation(
            _bc2d(BC.NEUMANN, BC.DIRICHLET), self.q)
        r.coefficient = default_rng(2).standard_normal(r.dimension)
        vals = r.evaluate_interior()
        np.testing.assert_allclose(vals[:, 0], 0., atol=1e-13)
        np.testing.assert_allclose(vals[:, -1], 0., atol=1e-13)

    def test_zero_coefficient_zero_field(self):
        q = self.q
        for bc in BoundaryCondition.all_combinations(2):
            r = DNAFourierComponentRealisation(bc, q)
            r.coefficient = np.zeros(r.dimension)
            np.testing.assert_allclose(r.evaluate_interior(), 0.)

    def test_evaluate_raises(self):
        r = DNAFourierComponentRealisation(
            _bc2d(BC.NEUMANN, BC.NEUMANN), self.q)
        r.coefficient = np.zeros(r.dimension)
        pts = np.stack(
            [np.linspace(0.1, 0.9, 8),
             np.linspace(0.1, 0.9, 8)],
            axis=1)
        with pytest.raises(NotImplementedError):
            r.evaluate(pts)
