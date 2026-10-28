"""Band edges and their field dependence from the command line (the web plots come with phase M2c).

    python -m skfitter.model edges  MODEL [--modify MODS.json] [--n-valence N]
    python -m skfitter.model field  MODEL --B 0,2,4,6,8,10 [--direction out-of-plane|in-plane|in-plane-2|x,y,z]
                               [--valley K=2/3,1/3,0 --valley "K'=1/3,2/3,0"] [--csv gap_vs_B.csv]
    python -m skfitter.model angle  MODEL --B 10 [--angles 0,15,30,45,60,75,90] [--phi 0] [--valley ...] [--csv ...]

MODEL is a model file (any kind; .json or .json.gz) or ``textbook:mos2`` (monolayer MoS2, three-band model).
Options common to field / angle: --g (default 2.0023), --no-spin, --no-orbital, --n-valence, --mesh, --modify (a JSON
list of modifiers applied first, e.g. a scissor or a sublattice mass). Energies are printed in meV.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction

from . import bands as B
from .io import load_model
from .modifiers import G_ELECTRON, apply


def _model(spec: str, modify: str | None):
    if spec.startswith("textbook:"):
        from . import textbook
        name = spec.split(":", 1)[1]
        if name != "mos2":
            raise SystemExit("textbook models: mos2")
        m = textbook.mos2_three_band()
    else:
        m = load_model(spec)
    if modify:
        with open(modify, encoding="utf-8") as fh:
            m = apply(m, json.load(fh))
    return m


def _vector(text: str):
    return [float(Fraction(x)) for x in text.split(",")]


def _valleys(items, model_spec):
    out = {}
    for it in items or []:
        name, _, v = it.partition("=")
        out[name] = _vector(v)
    if not out and model_spec == "textbook:mos2":
        from .textbook import MOS2_VALLEYS
        out = dict(MOS2_VALLEYS)
    return out or None


def _table(res, x):
    rows = res["rows"]
    keys = ["vbm", "cbm", "gap"] + [k for k in rows[0] if k.endswith((":vb", ":cb", ":gap"))]
    print(f"{x:>10} " + " ".join(f"{k:>12}" for k in keys) + "   (meV)")
    for r in rows:
        print(f"{r[x]:>10.4g} " + " ".join(f"{1000 * r[k]:>12.4f}" for k in keys))
    if res.get("slopes"):
        print("\nslope dE/dB (meV/T) and g = slope / mu_B:")
        for k in keys:
            if k in res["slopes"]:
                s = res["slopes"][k]
                print(f"  {k:>12}: {s['slope_meV_per_T']:>10.5f}   g = {s['g']:>8.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m skfitter.model", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("edges", "field", "angle"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--modify")
        p.add_argument("--n-valence", type=int)
        p.add_argument("--mesh", type=int, default=24)
        if name != "edges":
            p.add_argument("--B", required=True)
            p.add_argument("--g", type=float, default=G_ELECTRON)
            p.add_argument("--no-spin", action="store_true")
            p.add_argument("--no-orbital", action="store_true")
            p.add_argument("--valley", action="append")
            p.add_argument("--csv")
        if name == "field":
            p.add_argument("--direction", default="out-of-plane")
        if name == "angle":
            p.add_argument("--angles", default="0,15,30,45,60,75,90")
            p.add_argument("--phi", type=float, default=0.0)
    a = ap.parse_args(argv)
    m = _model(a.model, a.modify)
    if a.cmd == "edges":
        e = B.band_edges(m, n_valence=a.n_valence, mesh=a.mesh)
        print(json.dumps(e, indent=1))
        return 0
    common = dict(g=a.g, spin=not a.no_spin, orbital=not a.no_orbital, n_valence=a.n_valence,
                  valleys=_valleys(a.valley, a.model), mesh=a.mesh)
    if a.cmd == "field":
        d = a.direction if a.direction in ("out-of-plane", "in-plane", "in-plane-1", "in-plane-2") else \
            _vector(a.direction)
        res = B.field_sweep(m, _vector(a.B), direction=d, **common)
        print(f"field direction {[round(x, 4) for x in res['direction']]} (2D: {B.plane_axes(m)['two_d']})")
        _table(res, "B")
    else:
        res = B.direction_sweep(m, float(a.B), angles=_vector(a.angles), phi=a.phi, **common)
        print(f"|B| = {res['B']} T, theta from the normal (0 = out of plane, 90 = in plane), phi = {res['phi_deg']}")
        _table(res, "theta_deg")
    if a.csv:
        B.write_csv(res["rows"], a.csv)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
