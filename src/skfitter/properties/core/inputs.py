"""Typed inputs of a calculator. Each checks and fills a value and describes itself for the generated web form
(``schema()``). A value of None (or a missing key) takes the default."""
from __future__ import annotations

import math

AXES = ("x", "y", "z")


class Input:
    kind = "input"

    def __init__(self, label: str, default=None, help: str = "", advanced: bool = False):
        self.label, self.default, self.help, self.advanced = label, default, help, advanced

    def check(self, value, name: str):
        return self.default if value is None else value

    def schema(self) -> dict:
        out = {"kind": self.kind, "label": self.label, "default": self.default, "help": self.help}
        if self.advanced:
            out["advanced"] = True
        return out


class Number(Input):
    kind = "number"

    def __init__(self, label, default, unit: str = "", min=None, max=None, integer: bool = False, **kw):
        super().__init__(label, default, **kw)
        self.unit, self.min, self.max, self.integer = unit, min, max, integer

    def check(self, value, name):
        v = self.default if value is None else value
        try:
            v = int(v) if self.integer else float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: a{'n integer' if self.integer else ' number'}") from None
        if not math.isfinite(v):
            raise ValueError(f"{name}: a finite number")
        if self.min is not None and v < self.min:
            raise ValueError(f"{name}: at least {self.min}")
        if self.max is not None and v > self.max:
            raise ValueError(f"{name}: at most {self.max}")
        return v

    def schema(self):
        return {**super().schema(), "unit": self.unit, "min": self.min, "max": self.max, "integer": self.integer}


class Choice(Input):
    kind = "choice"

    def __init__(self, label, default, options, **kw):
        super().__init__(label, default, **kw)
        self.options = [(o, o) if isinstance(o, str) else tuple(o) for o in options]

    def check(self, value, name):
        v = self.default if value is None else value
        if v not in [o[0] for o in self.options]:
            raise ValueError(f"{name}: one of {', '.join(str(o[0]) for o in self.options)}")
        return v

    def schema(self):
        return {**super().schema(), "options": [{"value": o[0], "label": o[1]} for o in self.options]}


class Axis(Choice):
    def __init__(self, label, default="z", **kw):
        super().__init__(label, default, [(a, a) for a in AXES], **kw)


class Bool(Input):
    kind = "bool"

    def check(self, value, name):
        return bool(self.default if value is None else value)


class Mesh(Input):
    """Three k-point counts (non-periodic directions are sampled once whatever is given)."""
    kind = "mesh"

    def __init__(self, label, default=(40, 40, 40), max_per_axis: int = 2000, **kw):
        super().__init__(label, list(default), **kw)
        self.max = max_per_axis

    def check(self, value, name):
        v = self.default if value is None else value
        if isinstance(v, (int, float)):
            v = [v] * 3
        try:
            v = [int(x) for x in v]
        except (TypeError, ValueError):
            raise ValueError(f"{name}: three integers") from None
        if len(v) != 3 or min(v) < 1 or max(v) > self.max:
            raise ValueError(f"{name}: three integers from 1 to {self.max}")
        return v


class EnergyGrid(Input):
    """{min, max, n} in eV relative to E_F; E_F = 0 is always added to the grid by the calculators that report it."""
    kind = "energy_grid"

    def __init__(self, label, default=(-2.0, 2.0, 201), **kw):
        super().__init__(label, {"min": default[0], "max": default[1], "n": default[2]}, **kw)

    def check(self, value, name):
        v = {**self.default, **(value or {})}
        try:
            v = {"min": float(v["min"]), "max": float(v["max"]), "n": int(v["n"])}
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"{name}: min, max (eV) and n") from None
        if not 1 <= v["n"] <= 100001 or (v["n"] > 1 and not v["max"] > v["min"]) or v["max"] < v["min"]:
            raise ValueError(f"{name}: max > min (max = min for one point) and 1 <= n <= 100001")
        return v


class NumberList(Input):
    kind = "number_list"

    def __init__(self, label, default, unit="", min=None, max_len=1001, **kw):
        super().__init__(label, list(default) if default is not None else None, **kw)
        self.unit, self.min, self.max_len = unit, min, max_len

    def check(self, value, name):
        v = self.default if value is None else value
        if v is None:
            return None
        try:
            v = [float(x) for x in v]
        except (TypeError, ValueError):
            raise ValueError(f"{name}: a list of numbers") from None
        if not v or len(v) > self.max_len or any(not math.isfinite(x) for x in v) or \
                (self.min is not None and min(v) < self.min):
            raise ValueError(f"{name}: 1 to {self.max_len} numbers" + (f" >= {self.min}" if self.min is not None else ""))
        return v

    def schema(self):
        return {**super().schema(), "unit": self.unit, "min": self.min}


class Vector(Input):
    kind = "vector"

    def __init__(self, label, default=(0.0, 0.0, 0.0), unit="", **kw):
        super().__init__(label, list(default) if default is not None else None, **kw)
        self.unit = unit

    def check(self, value, name):
        v = self.default if value is None else value
        if v is None:
            return None
        try:
            v = [float(x) for x in v]
        except (TypeError, ValueError):
            raise ValueError(f"{name}: [x, y, z]") from None
        if len(v) != 3 or not all(math.isfinite(x) for x in v):
            raise ValueError(f"{name}: [x, y, z]")
        return v

    def schema(self):
        return {**super().schema(), "unit": self.unit}


class Bands(Input):
    """Which bands: "all", "occupied" (the n_occupied lowest), "valence+conduction" (the top valence and bottom
    conduction band), or a list of band indices (0 = lowest)."""
    kind = "bands"
    WORDS = ("all", "occupied", "edges")

    def check(self, value, name):
        v = self.default if value is None else value
        if isinstance(v, str):
            if v not in self.WORDS:
                raise ValueError(f"{name}: {', '.join(self.WORDS)} or a list of band indices")
            return v
        try:
            v = sorted({int(x) for x in v})
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {', '.join(self.WORDS)} or a list of band indices") from None
        if not v or v[0] < 0:
            raise ValueError(f"{name}: band indices from 0")
        return v


class KPath(Input):
    """"auto" (the standard path of the lattice) or {"points": [[label, [k1, k2, k3]] | null, ...]} (null: a
    jump); the spacing comes from the calculator's own ``dk`` input."""
    kind = "kpath"

    def check(self, value, name):
        v = self.default if value is None else value
        if v in ("auto", None):
            return "auto"
        pts = v.get("points") if isinstance(v, dict) else v
        out = []
        try:
            for p in pts:
                if p is None:
                    out.append(None)
                    continue
                lab, k = p
                k = [float(x) for x in k]
                if len(k) != 3:
                    raise ValueError
                out.append([str(lab), k])
        except (TypeError, ValueError):
            raise ValueError(f"{name}: \"auto\" or a list of [label, [k1, k2, k3]] (fractional), null for a jump") from None
        if sum(1 for p in out if p is not None) < 2:
            raise ValueError(f"{name}: at least two points")
        return {"points": out}


class Files(Input):
    """Model files (paths for scripts; the server puts its stored files' paths here)."""
    kind = "files"

    def __init__(self, label, max_files: int = 6, **kw):
        super().__init__(label, [], **kw)
        self.max_files = max_files

    def check(self, value, name):
        v = self.default if value is None else value
        if isinstance(v, str):
            v = [v]
        try:
            v = [str(x) for x in v]
        except TypeError:
            raise ValueError(f"{name}: a list of model files") from None
        if len(v) > self.max_files:
            raise ValueError(f"{name}: at most {self.max_files} models")
        return v
