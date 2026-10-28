"""The calculator interface: one class per property. What it declares (requirements, inputs, cost, results) is used
by the runner, the server (checks, plan limits), the web form and the help.

    class MyProperty(Calculator):
        id = "electronic.something"; title = "..."; version = 1
        requires = Requires(dims=(2, 3), spinful=True)
        inputs = {"mesh": Mesh("k-mesh", (24, 24, 24))}
        def run(self, ctx, model, p) -> Result: ...
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Issue:
    """Why a model does not suit a calculator (level "error": refused; "warning": runs, with the warning)."""
    level: str
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        return {"level": self.level, "message": self.message, "fix": self.fix}


@dataclass
class Requires:
    dims: tuple = (1, 2, 3)            # allowed numbers of periodic directions
    spinful: bool = False              # a spin-doubled basis (spin operators)
    layer_xy: bool = False             # a 2D-type cell with a1, a2 in the xy plane (k-plane maps, valleys)
    hexagonal: bool = False            # a hexagonal 2D zone (K and K')
    broken_trs: bool = False           # warn when time reversal is not broken (the property then vanishes)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Cost:
    kpoints: int
    matrix: int
    seconds: float

    def to_dict(self) -> dict:
        return {"kpoints": int(self.kpoints), "matrix": int(self.matrix), "seconds": round(float(self.seconds), 2)}


@dataclass
class Result:
    """What a calculator returns: typed views (curve, bands, map, table) plus summary rows and notes."""
    views: list = field(default_factory=list)
    summary: list = field(default_factory=list)          # [(label, value text)]
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    data: dict = field(default_factory=dict)             # extra numbers for scripts and tests

    def curve(self, title, x, x_label, series, y_label, **extra):
        """series: [(name, values)] or [{name, values, ...}]."""
        s = []
        for item in series:
            s.append(item if isinstance(item, dict) else {"name": item[0], "values": item[1]})
        self.views.append({"type": "curve", "title": title, "x": {"label": x_label, "values": x}, "series": s,
                           "y_label": y_label, **extra})

    def bands(self, title, x, energies, ticks, y_label="E − E_F (eV)", **extra):
        self.views.append({"type": "bands", "title": title, "x": x, "energies": energies, "ticks": ticks,
                           "y_label": y_label, **extra})

    def map(self, title, kx, ky, values, label, **extra):
        self.views.append({"type": "map", "title": title, "kx": kx, "ky": ky, "values": values, "label": label,
                           **extra})

    def table(self, title, headers, rows, **extra):
        self.views.append({"type": "table", "title": title, "headers": headers, "rows": rows, **extra})


class Calculator:
    id = "family.name"
    family = ""
    title = ""
    version = 1
    help = ""
    formula = ""                       # LaTeX of the main formula, for the teaching panel
    requires = Requires()
    inputs: dict = {}
    engines = ("native",)

    # -- declared behaviour -------------------------------------------------------------------------------------
    def issues(self, model) -> list:
        """The requirements checked against the model (no heavy computation)."""
        from skfitter.model.modifiers import periodic_axes
        out = []
        r = self.requires
        d = len(periodic_axes(model))
        if d not in r.dims:
            out.append(Issue("error", f"{self.title} needs a {' or '.join(f'{x}D' for x in r.dims)} model; this one "
                                      f"is periodic in {d} direction{'s' if d != 1 else ''}."))
        if r.spinful and not model.spinful:
            out.append(Issue("error", f"{self.title} needs the spin operator: this model has no spin degree of freedom.",
                             "Use a fit with spin-orbit coupling or magnetic orbitals, or make the model spinful "
                             "(the 'spinful' modifier)."))
        if r.layer_xy or r.hexagonal:
            from ..services import kspace
            try:
                if r.hexagonal:
                    kspace.valleys(model)
                else:
                    kspace.check_xy_layer(model, self.title)
            except ValueError as exc:
                out.append(Issue("error", str(exc)))
        if r.broken_trs and not _breaks_trs(model):
            out.append(Issue("warning", f"The model looks time-reversal symmetric (not magnetic, no Zeeman field): "
                                        f"the {self.title[0].lower() + self.title[1:]} then vanishes."))
        return out

    def check(self, model, params: dict) -> dict:
        """The inputs checked and filled with their defaults (ValueError for a bad value or an unknown key)."""
        params = dict(params or {})
        extra = set(params) - set(self.inputs)
        if extra:
            raise ValueError(f"{self.id}: unknown input(s) {', '.join(sorted(extra))}")
        return {name: inp.check(params.get(name), name) for name, inp in self.inputs.items()}

    def cost(self, model, p: dict) -> Cost:
        return Cost(1, _dim(model), 0.01)

    def run(self, ctx, model, p: dict) -> Result:  # pragma: no cover - every calculator overrides it
        raise NotImplementedError

    def describe(self) -> dict:
        return {"id": self.id, "family": self.family, "title": self.title, "version": self.version, "help": self.help,
                "formula": self.formula, "requires": self.requires.to_dict(), "engines": list(self.engines),
                "inputs": {k: v.schema() for k, v in self.inputs.items()}}


def _dim(model) -> int:
    return int(sum(model.norbs(i) for i in range(model.n_sites)))


def _breaks_trs(model) -> bool:
    if getattr(model, "is_magnetic", False):
        return True
    for m in getattr(model, "modifiers", None) or []:
        if m.get("kind") == "zeeman" and any(abs(float(x)) > 0 for x in m.get("B", [])):
            return True
    return False


def seconds_per_k(D: int, factor: float = 1.0) -> float:
    """Rough time of one diagonalisation plus a few matrix products (numpy, one core), in seconds."""
    return factor * (2e-5 + 3e-9 * D ** 3 + 2e-8 * D ** 2)


def lin(a) -> list:
    return np.asarray(a, float).tolist()
