from dataclasses import dataclass
from typing import Callable, Iterable

from styne.backend.interface import Backend


class BackendRegistryError(ValueError):
    """Base class for backend registration and discovery errors."""


class BackendNotFoundError(BackendRegistryError):
    """Raised when a named backend has not been registered."""


class BackendUnavailableError(ImportError):
    """Raised when an optional backend dependency is not installed."""


class BackendInferenceError(BackendRegistryError):
    """Raised when an array does not identify one registered backend."""


class MixedBackendError(BackendInferenceError):
    """Raised when one operation receives arrays from different backends."""


@dataclass
class _Registration:
    loader: Callable[[], Backend]
    arrayTypes: tuple[type, ...]
    arrayModules: tuple[str, ...]
    backend: Backend | None = None


class BackendRegistry:
    """Registry that discovers array backends through lazy registrations."""

    def __init__(self):
        self._registrations: dict[str, _Registration] = {}

    def register(
            self, name: str, loader: Callable[[], Backend], *,
            arrayTypes: Iterable[type] = (),
            arrayModules: Iterable[str] = ()) -> None:
        if name in self._registrations:
            raise BackendRegistryError(
                f"Backend {name!r} is already registered."
            )

        typeHints = tuple(arrayTypes)
        moduleHints = tuple(arrayModules)
        if not typeHints and not moduleHints:
            raise BackendRegistryError(
                f"Backend {name!r} must declare an array type or module."
            )
        if not callable(loader):
            raise TypeError("Backend loader must be callable.")
        if any(not isinstance(arrayType, type) for arrayType in typeHints):
            raise TypeError("Backend arrayTypes must contain only types.")
        if any(
                not isinstance(module, str) or not module
                for module in moduleHints):
            raise TypeError(
                "Backend arrayModules must contain non-empty strings."
            )

        self._registrations[name] = _Registration(
            loader=loader,
            arrayTypes=typeHints,
            arrayModules=moduleHints,
        )

    def get_backend(self, name: str) -> Backend:
        try:
            registration = self._registrations[name]
        except KeyError as error:
            available = ", ".join(sorted(self._registrations)) or "none"
            raise BackendNotFoundError(
                f"Backend {name!r} is not registered. "
                f"Available backends: {available}."
            ) from error

        if registration.backend is None:
            backend = registration.loader()
            if not isinstance(backend, Backend):
                raise TypeError(
                    f"Loader for backend {name!r} did not return a Backend."
                )
            if backend.name != name:
                raise BackendRegistryError(
                    f"Backend loader registered as {name!r} returned "
                    f"backend {backend.name!r}."
                )
            registration.backend = backend
        return registration.backend

    def infer_backend(self, *arrays) -> Backend:
        if not arrays:
            raise BackendInferenceError(
                "At least one array is required for inference."
            )

        backendNames = [self._infer_name(array) for array in arrays]
        distinctNames = sorted(set(backendNames))
        if len(distinctNames) != 1:
            names = ", ".join(distinctNames)
            raise MixedBackendError(
                f"Arrays from mixed backends are not supported: {names}."
            )
        return self.get_backend(distinctNames[0])

    def _infer_name(self, array) -> str:
        matches = [
            name for name, registration in self._registrations.items()
            if self._matches_hint(array, registration)
        ]
        if not matches:
            arrayType = type(array)
            raise BackendInferenceError(
                "No backend is registered for array type "
                f"{arrayType.__module__}.{arrayType.__qualname__}."
            )
        if len(matches) > 1:
            arrayType = type(array)
            raise BackendInferenceError(
                f"Array type {arrayType.__qualname__} matches multiple "
                "backends: "
                f"{', '.join(sorted(matches))}."
            )

        name = matches[0]
        if not self.get_backend(name).is_array(array):
            arrayType = type(array)
            raise BackendInferenceError(
                f"Backend {name!r} does not recognise hinted array type "
                f"{arrayType.__module__}.{arrayType.__qualname__}."
            )
        return name

    @staticmethod
    def _matches_hint(array, registration: _Registration) -> bool:
        if (
                registration.arrayTypes
                and isinstance(array, registration.arrayTypes)):
            return True

        arrayModule = type(array).__module__
        return any(
            arrayModule == module or arrayModule.startswith(f"{module}.")
            for module in registration.arrayModules
        )


_registry = BackendRegistry()


def register_backend(
        name: str, loader: Callable[[], Backend], *,
        arrayTypes: Iterable[type] = (),
        arrayModules: Iterable[str] = ()) -> None:
    _registry.register(
        name,
        loader,
        arrayTypes=arrayTypes,
        arrayModules=arrayModules,
    )


def get_backend(name: str) -> Backend:
    return _registry.get_backend(name)


def infer_backend(*arrays) -> Backend:
    return _registry.infer_backend(*arrays)
