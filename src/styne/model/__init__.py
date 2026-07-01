"""Curated public surface of styne.model."""
import importlib

_SYMBOL_TO_MODULE = {
    "LinearModel": "styne.model.linear",
    "SGLMM": "styne.model.sglmm",
    "SGLMMPredictor": "styne.model.sglmm",
    "ConstantTrend": "styne.model.trend",
    "LinearTrend": "styne.model.trend",
}

__all__ = list(_SYMBOL_TO_MODULE.keys())


def __getattr__(name):
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
