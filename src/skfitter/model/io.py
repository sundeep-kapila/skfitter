"""Reading any SKFitter model file: ``load_model(path)`` returns the right class for the file.

    from skfitter.model import load_model
    model = load_model("graphene_model.json")       # SKModel (prayog-sk-model), TBModel (prayog-tb-model) or
                                                    # ModifiedModel (prayog-modified-model: base + modifiers)
                                                    # or ContinuumModel (prayog-continuum-model: k·p on a grid)

Files may be gzip-compressed (``.json.gz``), as some illustrative projects keep them.
"""
from __future__ import annotations

import gzip
import json

from .model import TB_FORMAT, TBModel

SK_FORMAT = "prayog-sk-model"
MODIFIED_FORMAT = "prayog-modified-model"
CONTINUUM_FORMAT = "prayog-continuum-model"
FORMATS = (SK_FORMAT, TB_FORMAT, MODIFIED_FORMAT, CONTINUUM_FORMAT)


def read_json(path: str) -> dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def model_from_dict(data: dict):
    """The model a file's contents describe: an SKModel, a TBModel, or a ModifiedModel (a base model with
    modifiers, which are applied again here)."""
    if not isinstance(data, dict):
        raise ValueError("A model file holds one JSON object.")
    fmt = data.get("format")
    if fmt == TB_FORMAT:
        return TBModel.from_dict(data)
    if fmt == SK_FORMAT:
        from .sk import SKModel                     # SKModel builds its blocks with SKFitter's code (lazy import)
        return SKModel.from_dict(data)
    if fmt == MODIFIED_FORMAT:
        from .modifiers import from_dict
        return from_dict(data)
    if fmt == CONTINUUM_FORMAT:
        from .continuum import ContinuumModel        # a k·p band on a grid (phase SC3)
        return ContinuumModel.from_dict(data)
    raise ValueError(f"Not an SKFitter model file (format {fmt!r}; expected one of {', '.join(FORMATS)}).")


def load_model(path: str):
    """Read a model file (``<seed>_model.json``, optionally gzipped)."""
    return model_from_dict(read_json(path))
