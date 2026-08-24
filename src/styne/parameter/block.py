from __future__ import annotations

import numpy as np

from styne.backend import infer_backend
from styne.parameter.parameter import Parameter, as_coordinate


class BlockParameter(Parameter):
    """Immutable direct sum of ordered parameters.

    Coordinates use a common backend, dtype, and device. Leading batch
    dimensions are broadcast before concatenation along the trailing parameter
    axis.
    """

    def __init__(self, blocks: list[Parameter], names: dict = None):
        if not blocks:
            raise ValueError("BlockParameter requires at least one block.")
        if not all(isinstance(block, Parameter) for block in blocks):
            raise TypeError("BlockParameter blocks must be Parameters.")

        self._blocks = tuple(blocks)
        self._dimensions = tuple(
            block.dimension for block in self._blocks
        )
        self._names = dict(names) if names is not None else {}
        self._backend = self._validate_backend()

    @property
    def nBlocks(self) -> int:
        return len(self._blocks)

    def block(self, index: int) -> Parameter:
        return self._blocks[index]

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
        return sum(self._dimensions)

    @property
    def coordinate(self):
        coordinates = tuple(
            block.coordinate for block in self._blocks
        )
        try:
            batchShape = np.broadcast_shapes(*(
                coordinate.shape[:-1] for coordinate in coordinates
            ))
        except ValueError as error:
            raise ValueError(
                "BlockParameter batch dimensions are not broadcastable."
            ) from error

        broadcastCoordinates = tuple(
            self._backend.namespace.broadcast_to(
                coordinate, batchShape + (dimension,)
            )
            for coordinate, dimension in zip(
                coordinates, self._dimensions
            )
        )
        return self._backend.namespace.concatenate(
            broadcastCoordinates, axis=-1
        )

    def with_coordinate(self, coordinate) -> BlockParameter:
        coordinate = as_coordinate(coordinate)
        if coordinate.shape[-1] != self.dimension:
            raise ValueError(
                "BlockParameter coordinate must have trailing dimension "
                f"{self.dimension}; got {coordinate.shape}."
            )

        blocks = []
        offset = 0
        for block, dimension in zip(self._blocks, self._dimensions):
            blockCoordinate = coordinate[
                ..., offset:offset + dimension
            ]
            blocks.append(block.with_coordinate(blockCoordinate))
            offset += dimension
        return type(self)(blocks, self._names)

    def _validate_backend(self):
        coordinates = tuple(
            block.coordinate for block in self._blocks
        )
        backend = infer_backend(*coordinates)
        metadata = tuple(
            backend.metadata(coordinate) for coordinate in coordinates
        )
        reference = metadata[0]
        if any(
                item.dtype != reference.dtype
                or item.device != reference.device
                for item in metadata[1:]):
            raise ValueError(
                "Block coordinates must share dtype and device."
            )
        return backend
