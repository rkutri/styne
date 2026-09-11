import numpy as np

from styne.backend import BackendInferenceError, infer_backend


def as_array(value):
    """Preserve registered backend arrays; convert Python values to NumPy."""
    try:
        infer_backend(value)
    except BackendInferenceError:
        return np.asarray(value)
    return value


class Data:
    """
    Container for experimental or simulated data.

    Parameters
    ----------
    dimension : int
        Dimension of each measurement vector.
    design : array-like
        Experimental design matrix or input locations. Registered backend
        arrays are retained without conversion.
    """

    def __init__(self, dimension: int, design):
        self._dimension = int(dimension)
        self._design = as_array(design)
        self._measurement = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def design(self):
        return self._design

    @property
    def measurement(self):
        return self._measurement

    @measurement.setter
    def measurement(self, measurement) -> None:

        m = as_array(measurement)
        if m.ndim == 1:
            m = m.reshape(1, -1)

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
