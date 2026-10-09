import numpy as np

from styne.gp.dna import (
    DNAFourierComponentExpansion,
    DNAFourierExpansion,
)
from styne.gp.dnautility import BC, BoundaryCondition


def test_component_has_no_coefficient_or_scratch_state():
    component = DNAFourierComponentExpansion(
        BoundaryCondition((BC.NEUMANN,)), 50
    )
    coefficient = np.random.default_rng(123).standard_normal(
        component.dimension
    )

    first = component.evaluate_native(coefficient)
    component.evaluate_native(np.zeros(component.dimension))

    np.testing.assert_array_equal(
        first, component.evaluate_native(coefficient)
    )
    assert "_coeff" not in vars(component)
    assert "_scratch" not in vars(component)


def test_full_expansion_has_no_coefficient_or_output_buffer_state():
    expansion = DNAFourierExpansion(20, 1)
    coefficient = np.random.default_rng(124).standard_normal(
        expansion.dimension
    )

    first = expansion.evaluate_native(coefficient)
    expansion.evaluate_native(np.zeros(expansion.dimension))

    np.testing.assert_array_equal(
        first, expansion.evaluate_native(coefficient)
    )
    assert "_param" not in vars(expansion)
    assert "_nativeBuffer" not in vars(expansion)


def test_full_expansion_matches_sum_of_static_components():
    expansion = DNAFourierExpansion(10, 1)
    coefficient = np.random.default_rng(125).standard_normal(
        expansion.dimension
    )

    expected = sum(
        component.evaluate_native(coefficient[coefficientSlice])
        for component, coefficientSlice in zip(
            expansion._components, expansion._slices
        )
    )
    expected *= 2. ** -0.5

    np.testing.assert_allclose(
        expansion.evaluate_native(coefficient), expected,
        rtol=1e-15, atol=1e-15,
    )


def test_single_and_batch_evaluation_share_one_expansion():
    q = 10
    expansion = DNAFourierExpansion(q, 1)
    rng = np.random.default_rng(127)
    batch = rng.standard_normal((3, expansion.dimension))

    batchResult = expansion.evaluate_native(batch)

    assert batchResult.shape == (3, q + 2)
    np.testing.assert_allclose(
        batchResult,
        np.stack([expansion.evaluate_native(row) for row in batch]),
        rtol=1e-15, atol=1e-15,
    )
