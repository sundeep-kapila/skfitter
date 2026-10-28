"""Small textbook models in the common model type, for teaching, examples and tests.

* ``mos2_three_band()``  monolayer MoS2 in the three-band model of Liu, Shan, Yao, Yao and Xiao, Phys. Rev. B 88,
  085433 (2013): Mo d_z2, d_xy, d_x2-y2 on the triangular lattice, nearest-neighbour hoppings (GGA set) and the
  spin-orbit term (lambda / 2) L_z sigma_z. Direct gap 1.66 eV at K and K'; the valence band at K is
  (d_x2-y2 ± i d_xy)/√2 (orbital moment m = ±2), the conduction band d_z2 (m = 0) - the textbook case of the
  valley Zeeman effect. (The parameters are quoted from memory; check them against the paper before quoting
  numbers from this model.)
* ``haldane(t1, t2, phi, M)``  the Haldane model on the honeycomb lattice (Haldane, PRL 61, 2015 (1988)): Chern
  number ±1 for |M| < 3√3 |t2 sin φ|, 0 beyond; M alone gives the valley Hall insulator (hBN-like).
* ``kane_mele(t1, lam_so, lam_v)``  the Kane-Mele model without Rashba coupling (Kane and Mele, PRL 95, 226801
  (2005)): two time-reversed Haldane copies (S_z conserved), a quantum spin Hall insulator for |lam_v| < 3√3 lam_so.
* ``graphene(t1, a)``  p_z graphene (the Haldane model with t2 = M = 0): Dirac cones at K and K'.
* ``cubic_s(t, a)``  one s orbital on a simple cubic lattice: E = -2t Σ cos(k a), effective mass ħ²/(2 t a²) at Γ.
* ``atom(orbitals, lam)``  one atom in a large empty cell, spinful, with on-site spin-orbit coupling lambda L.S:
  Zeeman and Landé-factor exercises.

``CATALOGUE`` lists the models with their parameters (label, default, unit, range) for a form; ``build(name,
params)`` checks the parameters against it and makes the model (the Properties "Textbook model" source and its
illustrative projects).
"""
from __future__ import annotations

import numpy as np

from .model import TBModel
from .orbitals import L_matrices

LIU_GGA_MOS2 = {"a": 3.190, "e1": 1.046, "e2": 2.104, "t0": -0.184, "t1": 0.401, "t2": 0.507, "t11": 0.218,
                "t12": 0.338, "t22": 0.057, "lambda": 0.073}


def mos2_three_band(soc: bool = True, shift: float = -0.8, params: dict | None = None) -> TBModel:
    """Monolayer MoS2, three-band model (module docstring). ``shift`` (eV) moves all bands so that E_F = 0 lies in
    the gap (the default puts it near mid-gap)."""
    p = {**LIU_GGA_MOS2, **(params or {})}
    a = p["a"]
    E1 = np.array([[p["t0"], p["t1"], p["t2"]], [-p["t1"], p["t11"], p["t12"]], [p["t2"], -p["t12"], p["t22"]]])

    def U(th):                     # rotation of (d_z2, d_xy, d_x2-y2) about z: (d_xy, d_x2-y2) turn by 2 theta
        c, s = np.cos(2 * th), np.sin(2 * th)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    lat = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20.0]])
    hops = {(0, 0, R): U(np.radians(th)) @ E1 @ U(np.radians(th)).T          # R1 and its C3 images
            for th, R in ((0, (1, 0, 0)), (120, (-1, 1, 0)), (240, (0, -1, 0)))}
    onsite = np.diag([p["e1"] + shift, p["e2"] + shift, p["e2"] + shift]).astype(complex)
    labels = ["dz2", "dxy", "dx2y2"]
    if soc:
        Lz = L_matrices(labels)[2]
        onsite = np.kron(np.eye(2), onsite) + 0.5 * p["lambda"] * np.kron(np.diag([1.0, -1.0]), Lz)
        hops = {k: np.kron(np.eye(2), v) for k, v in hops.items()}
        labels = [x + "↑" for x in labels] + [x + "↓" for x in labels]
    return TBModel(lattice=lat, species=["Mo"], frac_coords=[[0, 0, 0]], site_orbitals=[labels], onsite=[onsite],
                   hoppings=hops, spinful=soc, soc=soc, kind="tb", orbital_l=[[2] * len(labels)],
                   source={"textbook": "MoS2 three-band (Liu et al. 2013, GGA)"},
                   provenance=[{"step": "textbook model", "name": "mos2_three_band", "soc": soc}])


MOS2_VALLEYS = {"K": [2 / 3, 1 / 3, 0.0], "K'": [1 / 3, 2 / 3, 0.0]}


def atom(orbitals=("px", "py", "pz"), lam: float = 0.0, cell: float = 30.0, element: str = "X") -> TBModel:
    """One atom with the given orbitals (SKFitter labels of one shell), spinful, on-site lambda L.S."""
    labels = list(orbitals)
    L = L_matrices(labels)
    sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    h = 0.5 * lam * sum(np.kron(sig[a], L[a]) for a in range(3))
    from .orbitals import FAMILY_L, family_of
    ell = [FAMILY_L[family_of(x)[0]] for x in labels] * 2
    return TBModel(lattice=np.eye(3) * cell, species=[element], frac_coords=[[0, 0, 0]],
                   site_orbitals=[[x + "↑" for x in labels] + [x + "↓" for x in labels]], onsite=[h], hoppings={},
                   spinful=True, soc=lam != 0, orbital_l=[ell], kind="tb",
                   provenance=[{"step": "textbook model", "name": "atom", "orbitals": labels, "lambda": lam}])


HONEYCOMB_A = 2.46


def _honeycomb(a):
    lat = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20.0]])
    return lat, [[0, 0, 0], [1 / 3, 1 / 3, 0]]


NN = ((0, 0, 0), (-1, 0, 0), (0, -1, 0))                         # A -> B + R: the three nearest neighbours
NNN = ((1, 0, 0), (-1, 1, 0), (0, -1, 0))                       # A -> A + R, counter-clockwise (120° apart)


def haldane(t1: float = -2.7, t2: float = 0.1, phi: float = np.pi / 2, M: float = 0.0, a: float = HONEYCOMB_A) -> TBModel:
    """Haldane model (module docstring): spinless p_z on A, B; t2 e^{+iφ} along NNN (counter-clockwise) on A and
    e^{-iφ} on B; mass +M on A, -M on B."""
    lat, frac = _honeycomb(a)
    hops = {(0, 1, R): np.array([[t1]], complex) for R in NN}
    for R in NNN:
        hops[(0, 0, R)] = np.array([[t2 * np.exp(1j * phi)]])
        hops[(1, 1, R)] = np.array([[t2 * np.exp(-1j * phi)]])
    return TBModel(lattice=lat, species=["C", "C"], frac_coords=frac, site_orbitals=[["pz"], ["pz"]],
                   onsite=[np.array([[M]], complex), np.array([[-M]], complex)], hoppings=hops, kind="tb",
                   orbital_l=[[1], [1]], is_magnetic=True,
                   source={"textbook": "Haldane model"}, provenance=[{"step": "textbook model", "name": "haldane",
                                                                       "t1": t1, "t2": t2, "phi": phi, "M": M}])


def kane_mele(t1: float = -2.7, lam_so: float = 0.06, lam_v: float = 0.0, a: float = HONEYCOMB_A) -> TBModel:
    """Kane-Mele model without Rashba (module docstring): i lam_so ν σ_z on the NNN bonds, staggered lam_v. Spinful
    (each site: p_z up, p_z down); S_z is conserved."""
    lat, frac = _honeycomb(a)
    sz = np.diag([1.0, -1.0])
    hops = {(0, 1, R): t1 * np.eye(2, dtype=complex) for R in NN}
    for R in NNN:
        hops[(0, 0, R)] = 1j * lam_so * sz
        hops[(1, 1, R)] = -1j * lam_so * sz
    return TBModel(lattice=lat, species=["C", "C"], frac_coords=frac, site_orbitals=[["pz↑", "pz↓"]] * 2,
                   onsite=[lam_v * np.eye(2, dtype=complex), -lam_v * np.eye(2, dtype=complex)], hoppings=hops,
                   spinful=True, soc=lam_so != 0, kind="tb", orbital_l=[[1, 1], [1, 1]],
                   source={"textbook": "Kane-Mele model"},
                   provenance=[{"step": "textbook model", "name": "kane_mele", "t1": t1, "lam_so": lam_so,
                                "lam_v": lam_v}])


def graphene(t1: float = -2.7, a: float = HONEYCOMB_A) -> TBModel:
    """p_z graphene: nearest-neighbour hopping t1 only (Dirac cones at K and K', no gap)."""
    lat, frac = _honeycomb(a)
    hops = {(0, 1, R): np.array([[t1]], complex) for R in NN}
    return TBModel(lattice=lat, species=["C", "C"], frac_coords=frac, site_orbitals=[["pz"], ["pz"]],
                   onsite=[np.zeros((1, 1), complex), np.zeros((1, 1), complex)], hoppings=hops, kind="tb",
                   orbital_l=[[1], [1]], source={"textbook": "p_z graphene"},
                   provenance=[{"step": "textbook model", "name": "graphene", "t1": t1}])


def cubic_s(t: float = 1.0, a: float = 3.0, onsite: float = 0.0) -> TBModel:
    """One s orbital on a simple cubic lattice with hopping -t (module docstring)."""
    hops = {(0, 0, R): np.array([[-t]], complex) for R in ((1, 0, 0), (0, 1, 0), (0, 0, 1))}
    return TBModel(lattice=np.eye(3) * a, species=["X"], frac_coords=[[0, 0, 0]], site_orbitals=[["s"]],
                   onsite=[np.array([[onsite]], complex)], hoppings=hops, kind="tb", orbital_l=[[0]],
                   provenance=[{"step": "textbook model", "name": "cubic_s", "t": t, "a": a}])


# == the catalogue: what a form offers and build() accepts =========================================================
def _p(label, default, unit="", lo=None, hi=None, kind="number"):
    return {"label": label, "default": default, "unit": unit, "min": lo, "max": hi, "kind": kind}


CATALOGUE = {
    "haldane": {"title": "Haldane model", "about": "Spinless p_z on the honeycomb lattice with complex next-nearest-"
                "neighbour hopping t2 e^{±iφ} and a sublattice mass ±M: a Chern insulator (C = ∓1) for |M| < 3√3 "
                "|t2 sin φ|, trivial beyond.",
                "params": {"t1": _p("Nearest-neighbour hopping t1", -2.7, "eV", -10, 10),
                           "t2": _p("Next-nearest-neighbour hopping t2", 0.1, "eV", -2, 2),
                           "phi_deg": _p("Phase φ", 90.0, "°", -360, 360),
                           "M": _p("Sublattice mass M", 0.0, "eV", -5, 5)}},
    "kane_mele": {"title": "Kane-Mele model", "about": "Two time-reversed Haldane copies (spin up and down, S_z "
                  "conserved): a quantum spin Hall insulator for |λ_v| < 3√3 λ_SO.",
                  "params": {"t1": _p("Nearest-neighbour hopping t1", -2.7, "eV", -10, 10),
                             "lam_so": _p("Intrinsic spin-orbit λ_SO", 0.06, "eV", -1, 1),
                             "lam_v": _p("Staggered potential λ_v", 0.0, "eV", -5, 5)}},
    "graphene": {"title": "p_z graphene", "about": "Nearest-neighbour p_z graphene: Dirac cones at K and K' (add "
                 "fields and masses with Modify model).",
                 "params": {"t1": _p("Nearest-neighbour hopping t1", -2.7, "eV", -10, 10)}},
    "mos2_three_band": {"title": "MoS2 three-band model (Liu et al.)", "about": "Monolayer MoS2: Mo d_z2, d_xy, "
                        "d_x2-y2 with spin-orbit coupling - valley Berry curvature and spin-valley locking.",
                        "params": {"soc": _p("Spin-orbit coupling", True, kind="bool"),
                                   "shift": _p("Energy shift (E_F in the gap)", -0.8, "eV", -5, 5)}},
    "cubic_s": {"title": "Simple cubic s band", "about": "One s orbital on a simple cubic lattice: E = -2t Σ cos(k a), "
                "effective mass ħ²/(2 t a²) at Γ.",
                "params": {"t": _p("Hopping t", 1.0, "eV", 0.01, 10), "a": _p("Lattice constant a", 3.0, "Å", 1, 20)}},
}
_BUILD = {"haldane": lambda p: haldane(p["t1"], p["t2"], np.radians(p["phi_deg"]), p["M"]),
          "kane_mele": lambda p: kane_mele(p["t1"], p["lam_so"], p["lam_v"]),
          "graphene": lambda p: graphene(p["t1"]),
          "mos2_three_band": lambda p: mos2_three_band(soc=p["soc"], shift=p["shift"]),
          "cubic_s": lambda p: cubic_s(p["t"], p["a"])}


def check(name: str, params: dict | None = None) -> dict:
    """The parameters of textbook model ``name`` with the defaults filled in (ValueError for anything else)."""
    if name not in CATALOGUE:
        raise ValueError(f"Unknown textbook model {name!r} (models: {', '.join(CATALOGUE)}).")
    spec = CATALOGUE[name]["params"]
    params = dict(params or {})
    extra = set(params) - set(spec)
    if extra:
        raise ValueError(f"{CATALOGUE[name]['title']}: unknown parameter(s) {', '.join(sorted(extra))}.")
    out = {}
    for k, s in spec.items():
        v = params.get(k, s["default"])
        if s["kind"] == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{s['label']}: true or false.")
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                raise ValueError(f"{s['label']}: a number.")
            v = float(v)
            if (s["min"] is not None and v < s["min"]) or (s["max"] is not None and v > s["max"]):
                raise ValueError(f"{s['label']}: between {s['min']} and {s['max']} {s['unit']}.".rstrip())
        out[k] = v
    return out


def build(name: str, params: dict | None = None) -> TBModel:
    """Textbook model ``name`` with ``params`` (checked; see CATALOGUE); its provenance records them."""
    p = check(name, params)
    m = _BUILD[name](p)
    m.source = {**(m.source or {}), "textbook": CATALOGUE[name]["title"], "textbook_model": name, "params": p}
    return m
