from __future__ import annotations

import numpy as np

from styne.parameter.parameter import Parameter


class BlockParameter(Parameter):
    """
    Direct sum of an ordered list of parameters, exposed as a single flat
    parameter. The blocking structure is fixed at construction. Supports batch 
    trajectory coordinates (nBatch, nTotal) by broadcasting along axis 0.

    Parameters
    ----------
    blocks : list[Parameter]
        Ordered list of component parameters. Block order is fixed at
        construction and used for both flattening and un-flattening the
        coordinate.
    names : dict, optional
        Maps names to block indices for `__getitem__` lookup by name
        instead of position. Empty if not given.
    """

    def __init__(self, blocks: list[Parameter], names: dict = None):
        self._blocks = list(blocks)
        self._dims = [b.dimension for b in self._blocks]
        self._names = names or {}

    @property
    def nBlocks(self) -> int:
        return len(self._blocks)

    def block(self, idx: int) -> Parameter:
        """
        The parameter at block index `idx`.

        Parameters
        ----------
        idx : int

        Returns
        -------
        Parameter
        """
        return self._blocks[idx]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._blocks[key]
        if not self._names:
            raise KeyError("BlockParameter has no named blocks.")
        return self._blocks[self._names[key]]

    @property
    def names(self) -> dict:
        return dict(self._names)

    @property
    def dimension(self) -> int:
        return sum(self._dims)

    @property
    def coordinate(self) -> np.ndarray:
        """
        Flattened coordinate across all blocks.

        If every block's own coordinate is 1D, this is a plain concatenation.
        If any block is 2D (a batch of trajectories, shape `(nBatch, blockDim)`),
        every block is broadcast to that batch size first, then concatenated
        along axis 1, so a 1D block is tiled across the batch rather than
        raising a shape error.
        """
        # Check if any block has a 2D coordinate (batch)
        coords = [b.coordinate for b in self._blocks]
        isBatch = any(c.ndim == 2 for c in coords)
        
        if not isBatch:
            return np.concatenate(coords)
            
        # For batching, ensure everything is 2D and concatenate on axis 1
        nBatch = max(c.shape[0] if c.ndim == 2 else 1 for c in coords)
        standardized = []
        for c in coords:
            if c.ndim == 1:
                # Tile 1D parameter across the batch
                standardized.append(np.tile(c, (nBatch, 1)))
            else:
                standardized.append(c)
        return np.concatenate(standardized, axis=1)

    @coordinate.setter
    def coordinate(self, value: np.ndarray) -> None:
        # Expected size: total dimension for 1D, or nBatch * dimension for 2D
        # For 2D, we compare value.shape[1] with self.dimension
        currentDim = value.shape[1] if value.ndim == 2 else value.size
        
        if currentDim != self.dimension:
            raise ValueError(
                f"Coordinate dimensionality {currentDim} does not match "
                f"BlockParameter dimension {self.dimension}."
            )
        
        offset = 0
        for block, dim in zip(self._blocks, self._dims):
            # uses ellipsis for axis-agnostic slicing (offset is always on the parameter axis)
            sl = [slice(None)] * value.ndim
            sl[-1] = slice(offset, offset + dim)
            block.coordinate = value[tuple(sl)]
            offset += dim

    def clone(self) -> BlockParameter:
        """
        Return an independent copy, cloning every block.

        Returns
        -------
        BlockParameter
        """
        return BlockParameter(
            [b.clone() for b in self._blocks],
            dict(self._names)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockParameter):
            return NotImplemented
        if self.nBlocks != other.nBlocks:
            return False
        return all(b1 == b2 for b1, b2 in zip(self._blocks, other._blocks))
