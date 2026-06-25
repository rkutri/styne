import itertools
import numpy as np

from styne.gp.dna import DNAFourierEngine
from styne.gp.dnautility import BC, BoundaryCondition


def _expected_multiplier(q, alpha, d, nu, ell):
    factor = 2.0 * nu + d
    l2 = ell**2
    out = []
    for bc in BoundaryCondition.all_combinations(d):
        ranges = []
        for j in range(d):
            r = range(q[j] + 1) if bc[j] == BC.NEUMANN else range(1, q[j] + 1)
            ranges.append(list(r))
        for modes in itertools.product(*ranges):
            omega2 = sum((np.pi * m / alpha[j])**2 for j, m in enumerate(modes))
            out.append(0.5 * (d - factor * (l2 * omega2) / (2.0 * nu + l2 * omega2)))
    return np.array(out)


def test_multiplier_rectangular():
    q, alpha, d = (24, 10), (1.0, 1.7), 2
    nu, ell = 1.5, 0.3
    eng = DNAFourierEngine(q, d, alpha)
    got = np.asarray(eng.compute_log_length_multiplier(nu=nu, lengthScale=ell))
    exp = _expected_multiplier(q, alpha, d, nu, ell)
    assert got.shape == exp.shape
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)
