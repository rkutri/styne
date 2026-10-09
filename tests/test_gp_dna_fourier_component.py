import numpy as np

from numpy.random import default_rng

from styne.gp.dna import (
    BC, BoundaryCondition,
    DNAFourierComponentExpansion,
)


# ---- helpers ----

def bc1d(bcType):
    return BoundaryCondition((bcType,))


def bc2d(bcX, bcY):
    return BoundaryCondition((bcX, bcY))


# ---- DNAFourierComponentExpansion API (1D) ----

class TestDNAFourierComponentExpansion1D:

    def setup_method(self):
        self.q = 5

    def test_neumann_dimension(self):
        r = DNAFourierComponentExpansion(bc1d(BC.NEUMANN), self.q)
        assert r.dimension == self.q + 1

    def test_dirichlet_dimension(self):
        r = DNAFourierComponentExpansion(bc1d(BC.DIRICHLET), self.q)
        assert r.dimension == self.q

    def test_has_no_coefficient_state(self):
        r = DNAFourierComponentExpansion(bc1d(BC.NEUMANN), self.q)
        for name in ("coefficient", "project", "clone"):
            assert not hasattr(r, name)

    def test_evaluate_interior_shape_neumann(self):
        r = DNAFourierComponentExpansion(bc1d(BC.NEUMANN), self.q)
        assert r.evaluate_interior(
            np.zeros(self.q + 1)
        ).shape == (self.q + 2,)

    def test_evaluate_interior_shape_dirichlet(self):
        r = DNAFourierComponentExpansion(bc1d(BC.DIRICHLET), self.q)
        assert r.evaluate_interior(
            np.zeros(self.q)
        ).shape == (self.q + 2,)

    def test_dirichlet_zero_at_endpoints(self):
        r = DNAFourierComponentExpansion(bc1d(BC.DIRICHLET), self.q)
        vals = r.evaluate_interior(
            default_rng(0).standard_normal(self.q)
        )
        assert abs(vals[0]) < 1e-14
        assert abs(vals[-1]) < 1e-14

    def test_zero_coefficient_zero_field_neumann(self):
        r = DNAFourierComponentExpansion(bc1d(BC.NEUMANN), self.q)
        np.testing.assert_allclose(
            r.evaluate_interior(np.zeros(self.q + 1)), 0.
        )

    def test_zero_coefficient_zero_field_dirichlet(self):
        r = DNAFourierComponentExpansion(bc1d(BC.DIRICHLET), self.q)
        np.testing.assert_allclose(
            r.evaluate_interior(np.zeros(self.q)), 0.
        )

    def test_evaluate_at_coordinates(self):
        r = DNAFourierComponentExpansion(bc1d(BC.NEUMANN), self.q)
        result = r.evaluate(
            np.zeros(self.q + 1), np.linspace(0.1, 0.9, 15)
        )
        assert result.shape == (15,)

    def test_multiple_coefficients_do_not_mutate_expansion(self):
        r = DNAFourierComponentExpansion(bc1d(BC.NEUMANN), self.q)
        first = r.evaluate_native(np.ones(self.q + 1))
        r.evaluate_native(np.zeros(self.q + 1))
        np.testing.assert_allclose(
            first, r.evaluate_native(np.ones(self.q + 1))
        )


# ---- DNAFourierComponentExpansion API (2D) ----

class TestDNAFourierComponentExpansion2D:

    def setup_method(self):
        self.q = 4

    def test_block_sizes(self):
        q = self.q
        assert DNAFourierComponentExpansion(
            bc2d(BC.NEUMANN, BC.NEUMANN),
            q).dimension == (
            q + 1) ** 2
        assert DNAFourierComponentExpansion(
            bc2d(BC.DIRICHLET, BC.NEUMANN), q).dimension == q * (q + 1)
        assert DNAFourierComponentExpansion(
            bc2d(BC.NEUMANN, BC.DIRICHLET),
            q).dimension == (
            q + 1) * q
        assert DNAFourierComponentExpansion(
            bc2d(BC.DIRICHLET, BC.DIRICHLET),
            q).dimension == q ** 2

    def test_evaluate_interior_shape(self):
        q = self.q
        for bc in BoundaryCondition.all_combinations(2):
            r = DNAFourierComponentExpansion(bc, q)
            assert r.evaluate_interior(
                np.zeros(r.dimension)
            ).shape == (q + 2, q + 2)

    def test_dirichlet_x_zero_boundary(self):
        r = DNAFourierComponentExpansion(
            bc2d(BC.DIRICHLET, BC.NEUMANN), self.q)
        vals = r.evaluate_interior(
            default_rng(1).standard_normal(r.dimension)
        )
        np.testing.assert_allclose(vals[0, :], 0., atol=1e-13)
        np.testing.assert_allclose(vals[-1, :], 0., atol=1e-13)

    def test_dirichlet_y_zero_boundary(self):
        r = DNAFourierComponentExpansion(
            bc2d(BC.NEUMANN, BC.DIRICHLET), self.q)
        vals = r.evaluate_interior(
            default_rng(2).standard_normal(r.dimension)
        )
        np.testing.assert_allclose(vals[:, 0], 0., atol=1e-13)
        np.testing.assert_allclose(vals[:, -1], 0., atol=1e-13)

    def test_zero_coefficient_zero_field(self):
        q = self.q
        for bc in BoundaryCondition.all_combinations(2):
            r = DNAFourierComponentExpansion(bc, q)
            np.testing.assert_allclose(
                r.evaluate_interior(np.zeros(r.dimension)), 0.
            )

    def test_evaluate_at_coordinates(self):
        r = DNAFourierComponentExpansion(
            bc2d(BC.NEUMANN, BC.NEUMANN), self.q)
        pts = np.stack(
            [np.linspace(0.1, 0.9, 8),
             np.linspace(0.1, 0.9, 8)],
            axis=1)
        result = r.evaluate(np.zeros(r.dimension), pts)
        assert result.shape == (8,)
