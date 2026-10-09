"""Backend interfaces and lazy discovery."""

from styne.backend.interface import (
    ArrayNamespace,
    Backend,
    BackendCapabilities,
    BackendCapabilityError,
    BackendMetadata,
)
from styne.backend.registry import (
    BackendInferenceError,
    BackendNotFoundError,
    BackendRegistry,
    BackendRegistryError,
    BackendUnavailableError,
    MixedBackendError,
    get_backend,
    infer_backend,
    register_backend,
)


def load_numpy_backend():
    from styne.backend.numpy import NumPyBackend

    return NumPyBackend()


def load_jax_backend():
    try:
        from styne.backend.jax import JAXBackend
    except ModuleNotFoundError as error:
        if error.name != "jax":
            raise
        raise BackendUnavailableError(
            "JAX backend requires the 'jax' extra. "
            "Install it with 'pip install styne[jax]'."
        ) from error

    return JAXBackend()


def load_pytorch_backend():
    try:
        from styne.backend.pytorch import PyTorchBackend
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise BackendUnavailableError(
            "PyTorch backend requires the 'torch' extra. "
            "Install it with 'pip install styne[torch]'."
        ) from error

    return PyTorchBackend()


register_backend(
    "numpy",
    load_numpy_backend,
    arrayModules=("numpy",),
)
register_backend(
    "jax",
    load_jax_backend,
    arrayModules=("jax", "jaxlib"),
)
register_backend(
    "pytorch",
    load_pytorch_backend,
    arrayModules=("torch",),
)


__all__ = [
    "ArrayNamespace",
    "Backend",
    "BackendCapabilities",
    "BackendCapabilityError",
    "BackendMetadata",
    "BackendInferenceError",
    "BackendNotFoundError",
    "BackendRegistry",
    "BackendRegistryError",
    "BackendUnavailableError",
    "MixedBackendError",
    "get_backend",
    "infer_backend",
    "register_backend",
]
