"""Curated public surface of styne.gp.

Deep imports still work; this is the documented path.
"""
import importlib

_SYMBOL_TO_MODULE = {
    "GaussianProcess": "styne.gp.gaussianprocess",
    "GPSampler": "styne.gp.gaussianprocess",
    "DirectExpansion": "styne.gp.direct",
    "DirectGPPredictor": "styne.gp.direct",
    "BSplineGPPredictor": "styne.gp.bspline",
    "BC": "styne.gp.dna",
    "BoundaryCondition": "styne.gp.dna",
    "DNAFourierComponentExpansion": "styne.gp.dna",
    "DNAFourierExpansion": "styne.gp.dna",
    "DNAGPPredictor": "styne.gp.dna",
    "DNACoarseFineSplit": "styne.gp.dnautility",
    "DNACoarseFinePartition": "styne.gp.dnautility",
    "CirculantEmbeddingEngine1D": "styne.gp.ce",
    "CirculantEmbeddingEngine2D": "styne.gp.ce",
    "ApproximateCirculantEmbeddingEngine1D": "styne.gp.ce",
    "ApproximateCirculantEmbeddingEngine2D": "styne.gp.ce",
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
