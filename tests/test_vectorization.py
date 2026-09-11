import numpy as np

from styne.gp.dna import DNAFourierExpansion
from styne.utility.grid import UniformGrid


def test_dna_expansion_batch_1d():
    q = 10
    nBatch = 5
    rng = np.random.default_rng(44)

    expansion = DNAFourierExpansion(q, d=1)
    theta = rng.standard_normal(expansion.dimension)
    assert expansion.evaluate_native(theta).shape == (q + 2,)

    thetas = rng.standard_normal((nBatch, expansion.dimension))
    actual = expansion.evaluate_native(thetas)

    assert actual.shape == (nBatch, q + 2)
    np.testing.assert_allclose(
        actual,
        np.stack([expansion.evaluate_native(theta) for theta in thetas]),
        rtol=1e-13,
        atol=1e-13,
    )


def test_bound_dna_expansion_batch_evaluation():
    expansion = DNAFourierExpansion(10, d=1)
    evaluation = expansion.bind(UniformGrid(0.1, 0.9, 8))
    thetas = np.random.default_rng(46).standard_normal(
        (4, expansion.dimension)
    )

    actual = evaluation.evaluate(thetas)

    assert actual.shape == (4, 8)
    np.testing.assert_allclose(
        actual,
        np.stack([evaluation.evaluate(theta) for theta in thetas]),
        rtol=1e-13,
        atol=1e-13,
    )
