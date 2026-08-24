from styne.statistics.measure import ProbabilityMeasure
from styne.parameter.parameter import Parameter


class DiracMeasure(ProbabilityMeasure):
    """
    Point-mass probability measure at a fixed `Parameter` location.
    """

    def __init__(self):
        self._location = None

    def with_location(self, location: Parameter):
        result = type(self)()
        result._location = location
        return result

    @property
    def location(self) -> Parameter:
        return self._location

    @location.setter
    def location(self, location: Parameter):
        self._location = location

    @property
    def domainType(self) -> type:
        return type(self._location)

    @property
    def domainDimension(self) -> int:

        if self._location is None:
            raise RuntimeError("Location not set.")
        return self._location.dimension

    def sample(self, randomState) -> tuple[Parameter, object]:
        """Return the fixed location without consuming random state."""
        return self._location, randomState
