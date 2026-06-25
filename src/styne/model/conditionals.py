import numpy as np

from styne.statistics.likelihood import LikelihoodInterface
from styne.statistics.likelihood import SGLMMLikelihood
from styne.statistics.data import Data
from styne.gp.dnautility import DNACoarseFineSplit
from styne.parameter.vector import Vector


class ConditionalLatentLikelihood(LikelihoodInterface):
    """SGLMM likelihood with one block of the latent field fixed.

    Parameters
    ----------
    fixedBlock : int
        0 → block 0 (coarse) is fixed, free argument is zetaF.
        1 → block 1 (fine) is fixed, free argument is zetaC.
    """

    def __init__(self, fullLikelihood: SGLMMLikelihood, split: DNACoarseFineSplit,
                 fixedBlock: int):
        self._fullLikelihood = fullLikelihood
        self._split = split
        self._fixedBlock = fixedBlock

        fixedDim = split.coarseDimension if fixedBlock == 0 else split.fineDimension
        self._fixedCoord = np.zeros(fixedDim)

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self) -> int:
        return (self._split.fineDimension if self._fixedBlock == 0
                else self._split.coarseDimension)

    @property
    def data(self) -> Data:
        return self._fullLikelihood.data

    def condition_on_fine(self, zetaF: Vector) -> None:
        self._fixedCoord = zetaF.coordinate.copy()

    def condition_on_coarse(self, zetaC: Vector) -> None:
        self._fixedCoord = zetaC.coordinate.copy()

    def _assemble(self, freeCoord: np.ndarray) -> Vector:
        if self._fixedBlock == 0:
            fullCoord = self._split.assemble(self._fixedCoord, freeCoord)
        else:
            fullCoord = self._split.assemble(freeCoord, self._fixedCoord)
        return Vector(fullCoord)

    def evaluate_log(self, zetaFree: Vector) -> float:
        return self._fullLikelihood.evaluate_log(
            self._assemble(zetaFree.coordinate)
        )

    def evaluate_log_gradient(self, zetaFree: Vector) -> np.ndarray:
        fullParam = self._assemble(zetaFree.coordinate)
        fullGrad = self._fullLikelihood.evaluate_log_gradient(fullParam)
        return (fullGrad[self._split.fineIndices] if self._fixedBlock == 0
                else fullGrad[self._split.coarseIndices])
