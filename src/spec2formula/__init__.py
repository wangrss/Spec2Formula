"""Spec2Formula standalone inference."""
__version__ = "0.1.0"

def __getattr__(name):
    if name == "Predictor":
        from .predictor import Predictor
        return Predictor
    raise AttributeError(name)

