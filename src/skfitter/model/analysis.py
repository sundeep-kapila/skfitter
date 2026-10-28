"""Band-edge analyses as one request -> one JSON-ready answer (phase M2c), for the web pages and scripts.

    from skfitter.model.analysis import check_request, analyse
    req = check_request({"analysis": "field", "sweep": {"B_min": 0, "B_max": 10, "n": 6,
                                                        "direction": "out-of-plane"}})
    out = analyse(model, req)            # rows, slopes (g factors), valleys, direction ... lists and floats only

Analyses (``analysis``):

* ``edges``  band edges (VBM, CBM, gap, direct gap) and valley edges, optionally at a field ``B`` = [x, y, z] (T)
* ``bands``  bands along the default path, zoomed on the edges (``window`` eV around them, default 0.3), with
             spins; with a field ``B`` also the bands in the field (same path) - the "band zoom" plot
* ``field``  edges, gap and valley edges against |B| along one direction (``sweep``: B_min, B_max, n, direction
             "out-of-plane" | "in-plane" | "in-plane-2" | [x, y, z]); slopes dE/dB and g factors
* ``angle``  edges and gap at a fixed |B| (``B``, tesla) as the field turns from out of plane (0°) to in plane
             (90°) (``angles``: min, max, n, phi)
* ``efield`` edges, gap and valley edges against a uniform electric field across a 2D material (``sweep``:
             E_min, E_max in V/Å, n): a gate-tunable gap (bilayer graphene); slopes in meV per V/Å

Common keys: ``valleys`` ("auto": K and K' of a hexagonal 2D lattice; {} none; {name: k_frac}), ``g``, ``spin``,
``orbital`` (the Zeeman terms), ``mesh`` (k-mesh of the edge search), ``n_valence`` (needed for a metal).
"""
from __future__ import annotations

import numpy as np

from . import bands as B
from .modifiers import G_ELECTRON, apply

ANALYSES = ("edges", "bands", "field", "angle", "efield")
DIRECTIONS = ("out-of-plane", "in-plane", "in-plane-2")
MAX_MESH = 48
MAX_POINTS = 2001


def _vec3(v, what):
    try:
        a = [float(x) for x in v]
    except (TypeError, ValueError):
        raise ValueError(f"{what}: [x, y, z]") from None
    if len(a) != 3 or not np.all(np.isfinite(a)):
        raise ValueError(f"{what}: [x, y, z]")
    return a


def check_request(req: dict) -> dict:
    """The request with its defaults filled in; ValueError for anything malformed."""
    if not isinstance(req, dict):
        raise ValueError("analysis request: an object")
    a = req.get("analysis", "edges")
    if a not in ANALYSES:
        raise ValueError(f"analysis: one of {', '.join(ANALYSES)}")
    extra = set(req) - {"analysis", "B", "sweep", "angles", "valleys", "g", "spin", "orbital", "mesh",
                        "n_valence", "window"}
    if extra:
        raise ValueError(f"analysis request: unknown field(s) {', '.join(sorted(extra))}")
    out = {"analysis": a, "g": float(req.get("g", G_ELECTRON)), "spin": bool(req.get("spin", True)),
           "orbital": bool(req.get("orbital", True))}
    try:
        out["mesh"] = int(req.get("mesh") or 24)
    except (TypeError, ValueError):
        raise ValueError("mesh: an integer") from None
    if not 4 <= out["mesh"] <= MAX_MESH:
        raise ValueError(f"mesh: 4 to {MAX_MESH} k-points per direction")
    nv = req.get("n_valence")
    out["n_valence"] = None if nv in (None, "", 0) else int(nv)
    v = req.get("valleys", "auto")
    if v != "auto":
        if not isinstance(v, dict):
            raise ValueError('valleys: "auto", {} or {name: [k1, k2, k3]} (fractional)')
        v = {str(k): _vec3(kf, f"valley {k}") for k, kf in v.items()}
    out["valleys"] = v
    if a in ("edges", "bands"):
        out["B"] = _vec3(req.get("B", [0, 0, 0]), "B (tesla)")
    if a == "bands":
        w = req.get("window", 0.3)
        out["window"] = float(w)
        if not 0 < out["window"] <= 10:
            raise ValueError("window: 0 to 10 eV around the band edges")
    if a == "field":
        s = dict(req.get("sweep") or {})
        try:
            sw = {"B_min": float(s.get("B_min", 0.0)), "B_max": float(s.get("B_max", 10.0)),
                  "n": int(s.get("n", 11)), "direction": s.get("direction", "out-of-plane")}
        except (TypeError, ValueError):
            raise ValueError("sweep: B_min, B_max (tesla), n and direction") from None
        if not (2 <= sw["n"] <= MAX_POINTS) or sw["B_max"] == sw["B_min"]:
            raise ValueError(f"sweep: B_max != B_min and 2 <= n <= {MAX_POINTS}")
        if not (isinstance(sw["direction"], str) and sw["direction"] in DIRECTIONS):
            if isinstance(sw["direction"], str):
                raise ValueError(f"sweep: direction is one of {', '.join(DIRECTIONS)} or [x, y, z]")
            sw["direction"] = _vec3(sw["direction"], "sweep direction")
            if np.linalg.norm(sw["direction"]) == 0:
                raise ValueError("sweep: the direction [x, y, z] is not zero")
        out["sweep"] = sw
    if a == "efield":
        s = dict(req.get("sweep") or {})
        try:
            sw = {"E_min": float(s.get("E_min", 0.0)), "E_max": float(s.get("E_max", 0.3)), "n": int(s.get("n", 7))}
        except (TypeError, ValueError):
            raise ValueError("sweep: E_min, E_max (V/Å) and n") from None
        if not (2 <= sw["n"] <= MAX_POINTS) or sw["E_max"] == sw["E_min"]:
            raise ValueError(f"sweep: E_max != E_min and 2 <= n <= {MAX_POINTS}")
        if max(abs(sw["E_min"]), abs(sw["E_max"])) > 5.0:
            raise ValueError("sweep: |E| up to 5 V/Å")
        out["sweep"] = sw
    if a == "angle":
        out["B"] = float(req.get("B", 10.0))
        s = dict(req.get("angles") or {})
        try:
            an = {"min": float(s.get("min", 0.0)), "max": float(s.get("max", 90.0)), "n": int(s.get("n", 7)),
                  "phi": float(s.get("phi", 0.0))}
        except (TypeError, ValueError):
            raise ValueError("angles: min, max, n (degrees from the normal) and phi") from None
        if not (2 <= an["n"] <= MAX_POINTS) or an["max"] == an["min"]:
            raise ValueError(f"angles: max != min and 2 <= n <= {MAX_POINTS}")
        out["angles"] = an
    return out


def points(req: dict) -> int:
    """How many models a request diagonalises on the full mesh (for plan limits)."""
    return {"field": lambda: req["sweep"]["n"], "efield": lambda: req["sweep"]["n"],
            "angle": lambda: req["angles"]["n"]}.get(req["analysis"],
                                                                                         lambda: 1)()


def _plain(x):
    """numpy -> lists and floats, recursively."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _zeeman(req, Bvec):
    return [{"kind": "zeeman", "B": list(Bvec), "g": req["g"], "spin": req["spin"], "orbital": req["orbital"]}]


def usable_valleys(model, req, n_valence, warnings):
    """The valleys of the request that are valleys of this model (others dropped with a warning)."""
    v = B.suggest_valleys(model) if req["valleys"] == "auto" else dict(req["valleys"])
    ok = {}
    for name, kf in v.items():
        try:
            B.valley_edges(model, {name: kf}, n_valence=n_valence)
            ok[name] = kf
        except ValueError as exc:
            warnings.append(f"valley {name} left out: {exc}")
    return ok


def analyse(model, req: dict, log=None) -> dict:
    """Run a checked request (check_request) on ``model``; the answer holds only lists, floats and strings."""
    req = check_request(req)
    warnings = list(getattr(model, "warnings", []) or [])
    axes = B.plane_axes(model)
    nv0 = req["n_valence"] if req["n_valence"] else B.count_valence(model)
    valleys = usable_valleys(model, req, nv0, warnings)
    out = {"analysis": req["analysis"], "request": req, "n_valence": nv0, "valleys": valleys,
           "two_d": bool(axes["two_d"]), "normal": axes["normal"], "in_plane_1": axes["in_plane_1"],
           "warnings": warnings}
    a = req["analysis"]
    if a in ("edges", "bands"):
        Bvec = req["B"]
        on = bool(np.any(np.asarray(Bvec) != 0))
        m = apply(model, _zeeman(req, Bvec)) if on else model
        nv = B._valence_for(model, m, nv0)
        out["edges"] = B.band_edges(m, n_valence=nv, mesh=req["mesh"])
        if valleys:
            out["valley_edges"] = B.valley_edges(m, valleys, n_valence=nv)
        if a == "bands":
            e0 = B.band_edges(model, n_valence=nv0, mesh=req["mesh"])
            lo, hi = e0["vbm"]["E"] - req["window"], e0["cbm"]["E"] + req["window"]
            path = B.kpath(model, B.default_path(model))
            for key, mm in (("bands", model),) + ((("bands_field", m),) if on else ()):
                bs = B.bands(mm, path, window=(lo, hi))
                out[key] = {"x": bs["x"], "energies": bs["energies"], "band_index": bs["band_index"],
                            "spin": bs.get("spin")}
            out["ticks"] = [[float(x), lab] for x, lab in path["ticks"]]
            out["window"] = [lo, hi]
            out["edges_zero_field"] = e0
        out["B"] = Bvec
    elif a == "field":
        sw = req["sweep"]
        vals = np.linspace(sw["B_min"], sw["B_max"], sw["n"]).tolist()
        r = B.field_sweep(model, vals, direction=sw["direction"], g=req["g"], spin=req["spin"],
                          orbital=req["orbital"], n_valence=nv0, valleys=valleys or None, mesh=req["mesh"])
        out.update({"x": "B", "rows": r["rows"], "slopes": r["slopes"], "direction": r["direction"]})
    elif a == "efield":
        sw = req["sweep"]
        vals = np.linspace(sw["E_min"], sw["E_max"], sw["n"]).tolist()
        r = B.efield_sweep(model, vals, n_valence=nv0, valleys=valleys or None, mesh=req["mesh"])
        out.update({"x": "E", "rows": r["rows"], "slopes": r["slopes"], "direction": r["direction"],
                    "position_matrices": r["position_matrices"]})
        if not r["position_matrices"]:
            warnings.append("the model has no position matrices: the field acts on the orbital centres only "
                            "(a Wannier model from a run with its _tb.dat adds the interband terms)")
        warnings.append("no screening: the field is the one inside the material (a gate field is screened)")
    else:
        an = req["angles"]
        vals = np.linspace(an["min"], an["max"], an["n"]).tolist()
        r = B.direction_sweep(model, req["B"], angles=vals, phi=an["phi"], g=req["g"], spin=req["spin"],
                              orbital=req["orbital"], n_valence=nv0, valleys=valleys or None, mesh=req["mesh"])
        out.update({"x": "theta_deg", "rows": r["rows"], "B": r["B"], "phi_deg": r["phi_deg"]})
    return _plain(out)
