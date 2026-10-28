"""skfitter.model: SKFitter's common tight-binding model type, shared by every module that makes or uses a model.

    from skfitter.model import load_model, check_model
    model = load_model("graphene_model.json")    # an SKModel (SKFitter fit) or a TBModel (numbers only)
    model.bloch(k)                               # H(k); model.onsite(i), model.hopping(i, j, d), model.bonds, ...
    numbers = model.to_tb_model()                # an SK model as numbers only (TBModel of kind "sk")
    m = apply(model, [{"kind": "zeeman", "B": [0, 0, 10]}])         # fields and gap changes (modifiers.py)
    from skfitter.model import bands; bands.band_edges(m)                 # band edges, gap, field sweeps (bands.py)

* ``TBModel``: numbers only (on-site and hopping blocks, optional overlaps, orbital centres) - file
  ``prayog-tb-model`` version 2;
* ``SKModel``: an SKFitter fit with its parameters (file ``prayog-sk-model``); its blocks come from SKFitter's
  own code, so it needs the ``skfitter`` module;
* ``TightBindingModel`` / ``check_model``: the interface both offer, and a consistency check;
* ``apply`` / ``ModifiedModel``: fields and gap changes on any model (modifiers.py); ``bands``: band edges, gaps,
  valleys and field sweeps; ``textbook``: small models for teaching; ``python -m skfitter.model``: the band-edge tools
  from the command line.

Only numpy is needed (no Kwant, no Qt); see docs/COMMON_MODEL.md.
"""
from .io import FORMATS, MODIFIED_FORMAT, SK_FORMAT, load_model, model_from_dict, read_json
from .model import KINDS, TB_FORMAT, TB_VERSION, TBModel
from .modifiers import G_ELECTRON, MU_B, ModifiedModel, apply, sublattice_mass
from .protocol import TightBindingModel, check_model
from .continuum import CONTINUUM_FORMAT, ContinuumModel, fit_continuum
from .sk import FORMAT_VERSION as SK_VERSION, MODEL_SUFFIX, SKModel, model_path

__all__ = ["TBModel", "SKModel", "ModifiedModel", "ContinuumModel", "fit_continuum", "CONTINUUM_FORMAT", "TightBindingModel", "check_model", "load_model",
           "model_from_dict", "read_json", "apply", "sublattice_mass", "MU_B", "G_ELECTRON", "model_path",
           "MODEL_SUFFIX", "TB_FORMAT", "TB_VERSION", "SK_FORMAT", "SK_VERSION", "MODIFIED_FORMAT", "FORMATS",
           "KINDS"]
__version__ = "0.1.0"
