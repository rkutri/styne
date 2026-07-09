"""
Curated top-level surface of styne.

The main extension and interaction points resolve directly on the
package, e.g. `styne.Model`, `styne.Parameter`, `styne.MetropolisHastings`.
Everything else keeps its documented subpackage path (`styne.gp`,
`styne.statistics`, `styne.mcmc`, `styne.utility`, ...).
"""
import importlib
import logging
from importlib.metadata import PackageNotFoundError, version

from styne.utility.logconfig import enable_logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("styne")
except PackageNotFoundError:
    # source checkout without installed distribution metadata
    __version__ = "0+unknown"

_SYMBOL_TO_MODULE = {
    "Model": "styne.model.model",
    "DifferentiableModel": "styne.model.model",
    "Parameter": "styne.parameter.parameter",
    "MetropolisHastings": "styne.mcmc.metropolishastings",
}

__all__ = ["enable_logging", "__version__", *_SYMBOL_TO_MODULE]


def __getattr__(name):
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
