"""SKFitter: symmetry-resolved Slater-Koster tight-binding models fitted to DFT band structures.

    from skfitter import SKFitter             # the fitting engine (skfitter.core)
    from skfitter import load_model           # a saved model (skfitter.model)
    import skfitter.properties as P            # Berry-phase and band properties of a model

Command line: ``skfitter --help``.
"""
__version__ = "1.0.0rc5"
__all__ = ["SKFitter", "load_model", "__version__"]


def __getattr__(name):                        # lazy, so that ``import skfitter.model`` stays light
    if name == "SKFitter":
        from .core import SKFitter
        return SKFitter
    if name == "load_model":
        from .model import load_model
        return load_model
    raise AttributeError(f"module 'skfitter' has no attribute {name!r}")
