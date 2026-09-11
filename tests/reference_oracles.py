import itertools

import numpy as np

from styne.gp.dna import BC, BoundaryCondition


def axis_synthesis_matrix(q, boundaryCondition):
    """Direct DNA sine/cosine basis on the native grid."""
    gridIndices = np.arange(q + 2)
    if boundaryCondition == BC.NEUMANN:
        modes = np.arange(q + 1)
        basis = np.cos(np.pi * np.outer(gridIndices, modes) / (q + 1))
        basis[:, 1:] *= np.sqrt(2.0)
        return basis

    modes = np.arange(1, q + 1)
    return np.sqrt(2.0) * np.sin(
        np.pi * np.outer(gridIndices, modes) / (q + 1)
    )


def dna_synthesis_matrix(q, d):
    """Closed-form matrix for the complete DNA native-grid synthesis."""
    q = (q,) * d if np.isscalar(q) else tuple(q)
    blocks = []
    for boundary in BoundaryCondition.all_combinations(d):
        axisMatrices = [
            axis_synthesis_matrix(q[axis], boundary[axis])
            for axis in range(d)
        ]
        block = axisMatrices[0]
        for axisMatrix in axisMatrices[1:]:
            block = np.kron(block, axisMatrix)
        blocks.append(block)
    return 2.0 ** (-d / 2.0) * np.concatenate(blocks, axis=1)


def dna_adjoint_synthesis(q, d, residual, spectralWeights=None):
    """Retained closed-form VJP oracle for DNA spectral synthesis."""
    result = dna_synthesis_matrix(q, d).T @ np.asarray(residual).ravel()
    if spectralWeights is not None:
        result = result * np.asarray(spectralWeights)
    return result


def compute_log_length_multiplier(q, alpha, d, nu, lengthScale):
    """Retained analytic oracle for the DNA log-lengthscale gradient."""
    q = (q,) * d if np.isscalar(q) else tuple(q)
    alpha = (alpha,) * d if np.isscalar(alpha) else tuple(alpha)
    factor = 2.0 * nu + d
    lengthScaleSquared = lengthScale**2
    multipliers = []

    for boundary in BoundaryCondition.all_combinations(d):
        modeRanges = []
        for axis in range(d):
            if boundary[axis] == BC.NEUMANN:
                modeRanges.append(range(q[axis] + 1))
            else:
                modeRanges.append(range(1, q[axis] + 1))

        for modes in itertools.product(*modeRanges):
            frequencySquared = sum(
                (np.pi * mode / alpha[axis]) ** 2
                for axis, mode in enumerate(modes)
            )
            ratio = lengthScaleSquared * frequencySquared / (
                2.0 * nu + lengthScaleSquared * frequencySquared
            )
            multipliers.append(0.5 * (d - factor * ratio))

    return np.asarray(multipliers)
