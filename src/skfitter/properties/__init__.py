"""skfitter.properties: physical properties of any common tight-binding model (``skfitter.model``).

    from skfitter.model import load_model
    from skfitter import properties as P
    model = load_model("hbn_model.json")
    P.calculators()                               # what is installed, by family
    P.eligible(model)                             # {calculator id: [issues]} - empty list: it qualifies
    r = P.run(model, "geometric.chern", {"nk": 60})
    P.save_result(r, "results/")

Families (sub-packages; only the installed ones are listed): ``electronic`` (bands with weights, DOS, band edges,
effective masses, model comparison) and ``geometric`` (Berry curvature, spin Berry curvature, Wilson loops, Chern
numbers, hybrid Wannier centres with Chern and Z2 invariants, anomalous and spin Hall conductivity). ``boltzmann``, ``optical`` and ``magnetic`` follow (see the Properties
module design note). Every calculator declares its requirements, typed inputs and cost; the native (numpy) engine is
the reference.
"""
from __future__ import annotations

from .core.calculator import Calculator, Cost, Issue, Requires, Result
from .core.runner import FORMAT, load_result, prepare, run, run_many, save_result

__version__ = "0.1.0"
FAMILIES = {
    "electronic": "Electronic structure",
    "geometric": "Berry phase and topology",
    "boltzmann": "Semiclassical transport",
    "optical": "Optics and opto-electronics",
    "magnetic": "Magnetic and field response",
}
_REGISTRY: dict = {}


def register(cls):
    """Class decorator: add a calculator to the registry (its id is family.name)."""
    inst = cls()
    if "." not in inst.id:
        raise ValueError(f"calculator id {inst.id!r} is not family.name")
    inst.family = inst.id.split(".", 1)[0]
    if inst.family not in FAMILIES:
        raise ValueError(f"unknown family {inst.family!r}")
    _REGISTRY[inst.id] = inst
    return cls


_LOADED = False


def _load():
    global _LOADED
    if not _LOADED:            # a flag, not "registry empty": importing one family module directly must not hide the others
        from . import electronic, geometric  # noqa: F401 - registers the calculators
        _LOADED = True
    return _REGISTRY


def get(calc_id: str) -> Calculator:
    reg = _load()
    if calc_id not in reg:
        raise ValueError(f"unknown property {calc_id!r} (installed: {', '.join(sorted(reg))})")
    return reg[calc_id]


def calculators(family: str | None = None) -> list:
    return [c for c in _load().values() if family is None or c.family == family]


def families() -> list:
    present = {c.family for c in _load().values()}
    return [{"id": f, "title": t, "installed": f in present} for f, t in FAMILIES.items()]


def registry() -> dict:
    """Everything the web form needs: families and the calculators with their inputs and requirements."""
    from . import engines
    return {"families": families(), "calculators": [c.describe() for c in _load().values()],
            "engines": engines.status()}


def eligible(model) -> dict:
    """{calculator id: [issue dicts]} for every installed calculator (no error-level issue: it qualifies)."""
    return {c.id: [i.to_dict() for i in c.issues(model)] for c in _load().values()}


__all__ = ["Calculator", "Cost", "Issue", "Requires", "Result", "FORMAT", "run", "run_many", "prepare",
           "save_result", "load_result", "get", "calculators", "families", "registry", "eligible", "register"]
