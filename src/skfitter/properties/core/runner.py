"""Running calculators: requirements, inputs, cost, the run itself, and the result file ``prayog-property-result``.

    from skfitter.properties import run
    r = run(model, "geometric.ahc", {"mesh": [40, 40, 1]})      # dict: the result file's contents
    save_result(r, "out/")                                     # result.json + one CSV per view

Several calculators in one job share one Bloch kernel (H(R) gathered once, recent eigenstates cached): pass the same
``kernel`` (see ``run_many``).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time

import numpy as np

FORMAT, VERSION = "prayog-property-result", 1
CONVENTIONS = {
    "energies": "eV relative to the model's E_F (0)",
    "bloch_phase": "exp(i k·(R + c_j − c_i)), orbital centres c (SK orbitals on their atoms)",
    "non_orthogonal": "Löwdin-orthogonalised H = S^-1/2 H S^-1/2",
    "berry_curvature": "Ω_n = −2 Im Σ_{m≠n} <n|∂_a H|m><m|∂_b H|n>/(E_n − E_m)², Å²; Ω_z = Ω_xy",
    "ahc": "σ_ab = −(e²/ħ) (1/V N_k) Σ_k Σ_n f_n Ω_n,ab, S/cm (V: cell volume, vacuum included)",
    "shc": "σ^s_ab = +(e²/ħ)(1/2)(1/V N_k) Σ f_n Ω^s_n,ab with J_a = ½{σ_s, ∂_a H}, (ħ/e) S/cm",
}


class Cancelled(RuntimeError):
    pass


class Context:
    """What a calculator gets besides the model and its inputs: the shared kernel, progress, cancellation, a log."""

    def __init__(self, kernel, progress=None, cancelled=None, log=None, engine="native"):
        self.kernel = kernel
        self.engine = engine
        self._progress = progress
        self._cancelled = cancelled
        self.log = log or (lambda *a, **k: None)

    def progress(self, fraction: float, message: str = ""):
        if self._progress:
            self._progress(max(0.0, min(1.0, float(fraction))), message)

    def check(self):
        if self._cancelled is not None and self._cancelled():
            raise Cancelled("Stopped by the user")


def model_hash(model) -> str:
    try:
        payload = json.dumps(model.to_dict(), sort_keys=True, default=str).encode()
    except Exception:  # noqa: BLE001 - informational only
        payload = repr(model).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _versions(engine: str = "native") -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__}
    if engine != "native":
        from .. import engines
        out[engine] = engines.status().get(engine, {}).get("version") or ""
    try:
        import skfitter.model
        out["skfitter.model"] = getattr(skfitter.model, "__version__", "")
    except ImportError:
        pass
    return out


def prepare(model, calc_id: str, params: dict | None = None):
    """(calculator, checked inputs, issues, cost); raises ValueError when the model does not qualify."""
    from .. import get
    calc = get(calc_id)
    issues = calc.issues(model)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        msg = "; ".join(e.message + (f" ({e.fix})" if e.fix else "") for e in errors)
        raise ValueError(f"{calc.title}: {msg}")
    p = calc.check(model, params or {})
    return calc, p, issues, calc.cost(model, p)


def run(model, calc_id: str, params: dict | None = None, engine: str = "native", kernel=None, progress=None,
        cancelled=None, log=None) -> dict:
    """Run one calculator on a model; returns the contents of a prayog-property-result file (lists only)."""
    from ..services.bloch import Kernel
    calc, p, issues, cost = prepare(model, calc_id, params)
    if engine not in calc.engines:
        raise ValueError(f"{calc.title}: engine {engine!r} is not available (engines: {', '.join(calc.engines)})")
    if engine != "native":
        from .. import engines
        engines.check(engine)
        cost = engines.adjust_cost(engine, calc.id, cost)
    ctx = Context(kernel if kernel is not None else Kernel(model), progress, cancelled, log, engine)
    t0 = time.time()
    res = calc.run(ctx, model, p)
    seconds = time.time() - t0
    return {
        "format": FORMAT, "version": VERSION,
        "calculator": {"id": calc.id, "version": calc.version, "title": calc.title, "family": calc.family},
        "engine": engine, "inputs": _plain(p),
        "model": {"hash": model_hash(model), "kind": getattr(model, "kind", "tb"),
                  "provenance": _plain(getattr(model, "provenance", [])),
                  "modifiers": _plain(list(getattr(model, "modifiers", None) or [])),
                  "orbitals": int(ctx.kernel.D), "spinful": bool(model.spinful)},
        "conventions": CONVENTIONS, "versions": _versions(engine), "seconds": round(seconds, 3),
        "cost_estimate": cost.to_dict(), "warnings": [i.message for i in issues] + list(res.warnings),
        "notes": list(res.notes), "summary": [[str(a), str(b)] for a, b in res.summary],
        "views": _plain(res.views), "data": _plain(res.data),
    }


def run_many(model, jobs: list, **kw) -> list:
    """[(calc_id, params), ...] with one shared kernel."""
    from ..services.bloch import Kernel
    k = Kernel(model)
    return [run(model, cid, p, kernel=k, **kw) for cid, p in jobs]


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, complex):
        return [x.real, x.imag]
    return x


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:50] or "view"


def save_result(result: dict, out_dir: str, stem: str | None = None) -> dict:
    """Write result.json and one CSV per curve / bands / table view; returns {"json": path, "csv": [paths]}."""
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or result["calculator"]["id"].replace(".", "_")
    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    csvs = []
    for n, v in enumerate(result.get("views", [])):
        rows, head = None, None
        if v["type"] == "curve":
            head = [v["x"]["label"]] + [s["name"] for s in v["series"]]
            rows = zip(v["x"]["values"], *[s["values"] for s in v["series"]])
        elif v["type"] == "bands":
            E = np.asarray(v["energies"], float)
            head = ["k (1/Å)"] + [f"band {b}" for b in range(E.shape[1])]
            rows = ([x] + list(e) for x, e in zip(v["x"], E))
        elif v["type"] == "table":
            head, rows = v["headers"], v["rows"]
        elif v["type"] == "map":
            head = ["kx (1/Å)", "ky (1/Å)", v["label"]]
            rows = zip(v["kx"], v["ky"], v["values"])
        if rows is None:
            continue
        p = os.path.join(out_dir, f"{stem}_{n + 1}_{_slug(v['title'])}.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for r in rows:
                w.writerow(r)
        csvs.append(p)
    return {"json": path, "csv": csvs}


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("format") != FORMAT:
        raise ValueError(f"Not a property result (format {d.get('format')!r}).")
    if int(d.get("version", 0)) > VERSION:
        raise ValueError("This property result was written by a newer program.")
    return d
