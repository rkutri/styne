import subprocess
import sys

import pytest

from styne.backend import (
    ArrayNamespace,
    Backend,
    BackendCapabilityError,
    BackendInferenceError,
    BackendMetadata,
    BackendNotFoundError,
    BackendRegistry,
    BackendRegistryError,
    MixedBackendError,
)


class _FirstArray:
    pass


class _SecondArray:
    pass


class _FirstBackend(Backend):
    @property
    def name(self):
        return "first"

    @property
    def namespace(self):
        return ArrayNamespace()

    def is_array(self, value):
        return isinstance(value, _FirstArray)

    def metadata(self, array):
        return BackendMetadata(dtype="float64", device="cpu")


class _SecondBackend(_FirstBackend):
    @property
    def name(self):
        return "second"

    def is_array(self, value):
        return isinstance(value, _SecondArray)


def test_backend_package_does_not_import_optional_frameworks():
    script = (
        "import sys; import styne.backend; "
        "assert 'jax' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("backendName", "frameworkName"),
    (("jax", "jax"), ("pytorch", "torch")),
)
def test_optional_backends_report_their_missing_extra(
        backendName, frameworkName):
    script = """
import builtins
import sys

originalImport = builtins.__import__
frameworkName = sys.argv[2]

def block_framework(name, *args, **kwargs):
    if name == frameworkName or name.startswith(frameworkName + "."):
        raise ModuleNotFoundError(name=frameworkName)
    return originalImport(name, *args, **kwargs)

builtins.__import__ = block_framework

from styne.backend import BackendUnavailableError, get_backend

try:
    get_backend(sys.argv[1])
except BackendUnavailableError as error:
    command = f"pip install styne[{frameworkName}]"
    assert command in str(error)
else:
    raise AssertionError("Missing backend dependency was not reported.")
"""

    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            backendName,
            frameworkName,
        ],
        check=True,
    )


def test_registry_loads_backend_lazily_and_caches_it():
    registry = BackendRegistry()
    loads = []

    def load_backend():
        loads.append(None)
        return _FirstBackend()

    registry.register("first", load_backend, arrayTypes=(_FirstArray,))

    assert loads == []
    backend = registry.get_backend("first")
    assert registry.get_backend("first") is backend
    assert len(loads) == 1


def test_registry_infers_backend_from_registered_array_types():
    registry = BackendRegistry()
    registry.register("first", _FirstBackend, arrayTypes=(_FirstArray,))

    backend = registry.infer_backend(_FirstArray(), _FirstArray())

    assert backend.name == "first"
    assert backend.metadata(_FirstArray()) == BackendMetadata(
        dtype="float64", device="cpu"
    )


def test_registry_uses_module_hints_without_loading_backend_eagerly():
    registry = BackendRegistry()
    loads = []

    def load_backend():
        loads.append(None)
        return _FirstBackend()

    registry.register(
        "first",
        load_backend,
        arrayModules=(__name__,),
    )
    assert loads == []

    assert registry.infer_backend(_FirstArray()).name == "first"
    assert len(loads) == 1


def test_registry_rejects_mixed_backends():
    registry = BackendRegistry()
    registry.register("first", _FirstBackend, arrayTypes=(_FirstArray,))
    registry.register("second", _SecondBackend, arrayTypes=(_SecondArray,))

    with pytest.raises(MixedBackendError, match="first, second"):
        registry.infer_backend(_FirstArray(), _SecondArray())


def test_registry_reports_unknown_backends_and_arrays():
    registry = BackendRegistry()
    registry.register("first", _FirstBackend, arrayTypes=(_FirstArray,))

    with pytest.raises(
            BackendNotFoundError, match="Available backends: first"):
        registry.get_backend("missing")
    with pytest.raises(BackendInferenceError, match="builtins.object"):
        registry.infer_backend(object())
    with pytest.raises(BackendInferenceError, match="At least one array"):
        registry.infer_backend()


def test_registry_validates_registration_and_loaded_backend():
    registry = BackendRegistry()

    with pytest.raises(BackendRegistryError, match="must declare"):
        registry.register("first", _FirstBackend)

    registry.register("first", lambda: object(), arrayTypes=(_FirstArray,))
    with pytest.raises(TypeError, match="did not return a Backend"):
        registry.get_backend("first")


def test_unsupported_capabilities_raise_backend_specific_error():
    backend = _FirstBackend()

    with pytest.raises(BackendCapabilityError, match="'first'.*compilation"):
        backend.compile(lambda value: value)


def test_namespace_has_no_implicit_delegation():
    namespace = ArrayNamespace()

    assert not hasattr(namespace, "undeclared_operation")
    with pytest.raises(NotImplementedError):
        namespace.exp(_FirstArray())
