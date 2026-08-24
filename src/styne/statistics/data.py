import numpy as np


class Data:
    """
    Container for experimental or simulated data.

    Parameters
    ----------
    dimension : int
        Dimension of each measurement vector.
    design : np.ndarray
        Experimental design matrix or input locations.
    """

    def __init__(self, dimension: int, design: np.ndarray):
        self._dimension = int(dimension)
        self._design = np.asarray(design)
        self._measurement = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def design(self) -> np.ndarray:
        return self._design

    @property
    def measurement(self) -> np.ndarray:
        return self._measurement

    @measurement.setter
    def measurement(self, measurement: np.ndarray) -> None:

        m = np.asarray(measurement, dtype=float)
        m = np.atleast_2d(m)

        if m.ndim != 2:
            raise ValueError(
                "Measurement must be a 2D array of shape (nData, dimension).")

        if m.shape[1] != self._dimension:
            raise ValueError(
                f"Measurement shape {m.shape} incompatible with "
                f"dimension {self._dimension}."
            )

        self._measurement = m

    @property
    def size(self) -> int:
        if self._measurement is None:
            raise AttributeError("Measurement not yet set.")
        return self._measurement.shape[0]
