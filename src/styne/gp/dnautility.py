import itertools
from enum import Enum

import numpy as np

from styne.utility.partition import PartitionRule, Partition
from styne.statistics.gaussian import Gaussian
from styne.parameter.vector import Vector
from styne.statistics.covariance import DiagonalCovarianceMatrix


class BC(Enum):
    """
    Boundary condition type for one axis of a DNA Fourier component.

    NEUMANN corresponds to a cosine expansion (b_i = 0 in the paper's
    notation). DIRICHLET corresponds to a sine expansion (b_i = 1). Content
    here was previously inline comments on the enum values, moved in per
    your instruction to catch these.
    """

    NEUMANN = 0
    DIRICHLET = 1


class BoundaryCondition:
    r"""
    Boundary condition vector $b \in \{BC\}^d$ for a single DNA component.

    Parameters
    ----------
    bcs : tuple
        One `BC` value per spatial dimension.
    """

    def __init__(self, bcs: tuple):
        self._bcs = tuple(bcs)

    @property
    def d(self) -> int:
        return len(self._bcs)

    def __getitem__(self, i: int) -> BC:
        return self._bcs[i]

    def block_size(self, q) -> int:
        q = (q,) * self.d if np.isscalar(q) else tuple(q)
        size = 1
        for bc_i, q_i in zip(self._bcs, q):
            size *= (q_i + 1) if bc_i == BC.NEUMANN else q_i
        return size

    @classmethod
    def all_combinations(cls, d: int) -> list:
        return [cls(bcs)
                for bcs in itertools.product([BC.NEUMANN, BC.DIRICHLET], repeat=d)]


class DNACoarseFineSplit(PartitionRule):
    """Bookkeeping for the coarse/fine partition of DNA spectral coefficients.

    Parameters
    ----------
    q : int | tuple
        Full DNA resolution (interior DOF per axis).
    qC : int | tuple
        Coarse resolution threshold; modes with stored index < threshold are coarse.
    d : int
        Spatial dimension (1 or 2).
    """

    def __init__(self, q, qC, d: int):

        q = (q,) * d if np.isscalar(q) else tuple(q)
        qC = (qC,) * d if np.isscalar(qC) else tuple(qC)

        if any(qc_j >= q_j for qc_j, q_j in zip(qC, q)):
            raise ValueError(f"qC {qC} must be less than q {q} on every axis.")

        self._q = q
        self._qC = qC
        self._d = d

        coarseIdx = []
        fineIdx = []
        offset = 0

        for bc in BoundaryCondition.all_combinations(d):
            blockSize = bc.block_size(q)

            if d == 1:
                nCoarse = qC[0] + 1 if bc[0] == BC.NEUMANN else qC[0]
                coarseIdx.extend(range(offset, offset + nCoarse))
                fineIdx.extend(range(offset + nCoarse, offset + blockSize))
            else:
                nCoarseX = qC[0] + 1 if bc[0] == BC.NEUMANN else qC[0]
                nCoarseY = qC[1] + 1 if bc[1] == BC.NEUMANN else qC[1]
                nModeX = q[0] + 1 if bc[0] == BC.NEUMANN else q[0]
                nModeY = q[1] + 1 if bc[1] == BC.NEUMANN else q[1]

                for i in range(nModeX):
                    for j in range(nModeY):
                        idx = offset + i * nModeY + j
                        if i < nCoarseX and j < nCoarseY:
                            coarseIdx.append(idx)
                        else:
                            fineIdx.append(idx)

            offset += blockSize

        self._indices = [
            np.array(coarseIdx, dtype=int),
            np.array(fineIdx, dtype=int)
        ]

    @property
    def numComponents(self) -> int:
        return 2

    def indices(self, idx: int) -> np.ndarray:
        return self._indices[idx]


class DNACoarseFinePartition(Partition):
    """
    Partition object corresponding to the DNA coarse-fine mode split.

    Parameters
    ----------
    dnaGP : GaussianProcess
        Source GP the partition is built from. Duck-typed, not enforced by
        `isinstance`, checked at construction only via `hasattr(dnaGP,
        'resolution')`.
    qC : int | tuple
        Coarse resolution threshold, same inference caveat as `q` elsewhere.
    d : int
        Spatial dimension (1 or 2).
    """

    def __init__(self, dnaGP, qC, d: int):

        if not hasattr(dnaGP, 'resolution'):
            raise AttributeError("GaussianProcess must expose 'resolution' for DNA partition.")

        super().__init__(
            DNACoarseFineSplit(dnaGP.resolution, qC, d),
            dnaGP.function(np.zeros(dnaGP.parameterDimension)),
        )
        self._measure = dnaGP.measure

    def coarse_measure(self):
        return self._component_measure(0)

    def fine_measure(self):
        return self._component_measure(1)

    def _component_measure(self, idx: int):

        cov = self._measure.covariance
        if not isinstance(cov, DiagonalCovarianceMatrix):
            raise TypeError("Extraction only supported for DiagonalCovarianceMatrix")

        compVar = self._rule.extract(idx, cov.marginalVariance)
        compMean = self._rule.extract(idx, self._measure.mean.coordinate)

        return Gaussian(DiagonalCovarianceMatrix(compVar), Vector(compMean))
