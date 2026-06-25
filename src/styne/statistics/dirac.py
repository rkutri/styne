from styne.statistics.measure import ProbabilityMeasure
from styne.parameter.parameter import Parameter


class DiracMeasure(ProbabilityMeasure):

    def __init__(self):
        self._location = None

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

    def draw(self, rng) -> Parameter:
        return self._location
