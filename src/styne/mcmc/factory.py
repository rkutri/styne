from abc import ABC, abstractmethod
from typing import Optional
from numpy.random import Generator

from styne.mcmc.diagnostics import AcceptanceRateDiagnostics, ChainDiagnostics
from styne.statistics.interface import DensityInterface
from styne.mcmc.metropolishastings import MetropolisHastings


class MHFactory(ABC):
    """Abstract factory for constructing MetropolisHastings samplers.

    Subclasses implement `_validate` (calling `super()._validate()` first)
    and `_create_sampler`. The public entry point is `create()`, which
    runs validation and delegates to `_create_sampler`.

    Set `target` and optionally `diagnostics` before calling `create()`.
    If `diagnostics` is not set, an `AcceptanceRateDiagnostics` instance
    is created automatically.
    """

    def __init__(self):
        self._target: DensityInterface = None
        self._diagnostics: ChainDiagnostics = None
        self._rng: Optional[Generator] = None

    @property
    def target(self) -> DensityInterface:
        return self._target

    @target.setter
    def target(self, density: DensityInterface):
        self._target = density

    @property
    def diagnostics(self) -> ChainDiagnostics:
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, dgn: ChainDiagnostics):
        self._diagnostics = dgn

    @property
    def rng(self) -> Optional[Generator]:
        return self._rng

    @rng.setter
    def rng(self, r: Optional[Generator]):
        self._rng = r


    def _validate(self):
        if self._target is None:
            raise ValueError("Target not set.")
        if not isinstance(self._target, DensityInterface):
            raise TypeError("Target must implement DensityInterface.")

    @abstractmethod
    def _create_sampler(self) -> MetropolisHastings:
        """Construct the sampler from validated config on self."""
        ...

    def create(self) -> MetropolisHastings:
        """Validate configuration and build the sampler."""
        self._validate()
        if self._diagnostics is None:
            self._diagnostics = AcceptanceRateDiagnostics()
        return self._create_sampler()
