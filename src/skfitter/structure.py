"""
The crystal structure, neighbour shells and bond-resolved parameters of an SKFitter model, as plain data for a viewer.

Qt-free and plot-library-free: the web app draws these with three.js; the desktop program can draw the same data.
Everything is built from the ``meta`` record of the DFT data (``skfitter.workflow.prepare_project``: lattice, sites,
elements, moments) and a fit set-up ``cfg`` - no DFT files needed - and uses SKFitter's own neighbour code
(``skfitter.neighbour_shells`` / ``cutoff_from_shells`` / ``neighbour_pairs`` / ``pair_hopping_keys``) so what is
shown is exactly what the fit uses:

    structure_view(meta)                   the cell: lattice, a b c α β γ, space group, sites, colours, shells, bonds
    neighbour_view(meta, cfg)              the bonds a model with cfg["n_neighbors"] shells includes, per site
    param_map(meta, cfg, params)           on-site values per site, hopping integrals V0, η, V(r) per bond

Shells are the distinct atom-atom distances (to 0.001 Å) over the whole cell, the same list for every site; the
cutoff is shell N + 0.1 Å, so a distance just above shell N can be included too (flagged).
"""
from __future__ import annotations

import os
import warnings
from functools import lru_cache

import numpy as np

from skfitter import core as sk

BOND_SYMBOL = {"s": "σ", "p": "π", "d": "δ", "f": "φ"}
_ORB_TOKENS = ("s*", "pz", "px", "s", "p", "d", "f")
FAMILY = {"s": "s", "s*": "s", "p": "p", "d": "d", "f": "f"}
MAX_SHELLS_LISTED = 16          # shells listed (and bonds returned) by structure_view
VACUUM_GAP = 6.0                # Å: an empty slab this thick along a lattice vector counts as vacuum


# ---------------------------------------------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------------------------------------------
def structure_from_meta(meta: dict):
    from pymatgen.core import Lattice, Structure
    sites = meta["sites"]
    return Structure(Lattice(np.asarray(meta["lattice"], float)), [s["symbol"] for s in sites],
                     [s["frac"] for s in sites])


@lru_cache(maxsize=1)
def _colour_table() -> dict:
    try:
        import pymatgen.core as pc
        from monty.serialization import loadfn
        for folder in (os.path.dirname(pc.__file__), os.path.join(os.path.dirname(pc.__file__), "..", "vis")):
            path = os.path.join(folder, "ElementColorSchemes.yaml")
            if os.path.isfile(path):
                return loadfn(path)
    except Exception:  # noqa: BLE001
        pass
    return {}


def element_colour(symbol: str, scheme: str = "VESTA") -> str:
    rgb = (_colour_table().get(scheme) or {}).get(symbol) or (_colour_table().get("Jmol") or {}).get(symbol)
    if not rgb:
        return "#9aa0a6"
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb[:3]))


def element_radius(symbol: str) -> float:
    """Atomic radius (Å) for drawing: pymatgen's empirical radius, else the calculated one, else 1.2 Å."""
    from pymatgen.core import Element
    try:
        e = Element(symbol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            radii = (e.atomic_radius, e.atomic_radius_calculated)
        for r in radii:
            if r is not None:
                return round(float(r), 3)
    except Exception:  # noqa: BLE001
        pass
    return 1.2


def site_labels(meta: dict) -> list:
    """'C1', 'C2', 'B1', ...: element + running number per element (the numbering the GUIs use)."""
    count, out = {}, []
    for s in meta["sites"]:
        count[s["symbol"]] = count.get(s["symbol"], 0) + 1
        out.append(f"{s['symbol']}{count[s['symbol']]}")
    return out


def shown_moments(meta: dict) -> list:
    """The initial moments (VASP MAGMOM) worth showing: those of a spin-polarised run, or of a non-collinear run whose
    MAGMOM sets real moments - not VASP's default of 1 μB on every atom of a non-magnetic spin-orbit run."""
    m = [None if v is None else float(v) for v in (meta.get("moments") or [])]
    if not m:
        return []
    if meta.get("magnetic"):
        return m
    if meta.get("soc") and any(v is not None and abs(v) >= 0.5 for v in m) and not all(v == 1.0 for v in m):
        return m
    return []


def symmetry(structure, symprec: float = 1e-3) -> dict:
    """Space group and equivalent sites as SKFitter finds them (equal on-site parameters for equivalent sites)."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        ds = sga.get_symmetry_dataset()
        eq = getattr(ds, "equivalent_atoms", None)
        if eq is None and isinstance(ds, dict):
            eq = ds["equivalent_atoms"]
        intl = getattr(ds, "international", None) or (ds["international"] if isinstance(ds, dict) else "?")
        num = getattr(ds, "number", None) or (ds["number"] if isinstance(ds, dict) else None)
        return {"symbol": str(intl), "number": None if num is None else int(num),
                "crystal_system": str(sga.get_crystal_system()), "equivalent": [int(v) for v in eq]}
    except Exception:  # noqa: BLE001  (no space group: every site on its own)
        return {"symbol": "?", "number": None, "crystal_system": "", "equivalent": list(range(len(structure)))}


def vacuum_axes(lattice, frac) -> list:
    """Lattice vectors (0, 1, 2) along which the cell has an empty slab of at least VACUUM_GAP Å (2D / slab cells)."""
    out = []
    lengths = np.linalg.norm(lattice, axis=1)
    inv_h = 1.0 / np.linalg.norm(np.linalg.inv(lattice).T, axis=1)      # layer spacing along each axis
    for ax in range(3):
        f = np.sort(np.mod(np.asarray(frac)[:, ax], 1.0))
        gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))
        if gaps.max() * inv_h[ax] >= VACUUM_GAP and lengths[ax] > VACUUM_GAP:
            out.append(ax)
    return out


def _angle(u, v) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def _image(lattice_inv, cart_i, cart_j, d_vec) -> list:
    """The lattice translation R (whole numbers) of a bond: r_j + R = r_i + d."""
    t = (np.asarray(cart_i) + np.asarray(d_vec) - np.asarray(cart_j)) @ lattice_inv
    return [int(v) for v in np.rint(t)]


def _shell_index(shells, dist) -> int:
    """1-based shell number of a distance (shells = the rounded distinct distances)."""
    r = round(float(dist), sk.SHELL_DECIMALS)
    i = int(np.searchsorted(shells, r - 0.5 * 10 ** -sk.SHELL_DECIMALS))
    return i + 1


def site_orbitals(meta: dict, cfg: dict) -> list:
    """Orbitals per site as SKFitter assigns them (orbital_config by element)."""
    oc = cfg.get("orbital_config") or {}
    out = []
    for s in meta["sites"]:
        o = oc.get(s["symbol"], [])
        out.append([o] if isinstance(o, str) else list(o))
    return out


def integral_label(key: str) -> str:
    """'v_Bi_pz-Mn_dz2_pds_0' -> 'Bi pz–Mn dz2 σ', 'v_Mn_d-Mn_d_dds' -> 'Mn d–Mn d σ', 'v_C_s-C_s*_ss*s' -> 'C s–C s* σ'."""
    p = sk.parse_integral_key(key) or sk.parse_integral_key(f"v_{key}")
    if p is None:
        return key
    names = {sk.group_tag(x): x for x in sk.SKFitter.SUB_ORBITALS}
    m = p["label"][-1]
    return (f"{p['el1']} {names.get(p['tag1'], p['tag1'])}–{p['el2']} {names.get(p['tag2'], p['tag2'])} "
            f"{BOND_SYMBOL.get(m, m)}")


def _bond_vector(structure, b) -> np.ndarray:
    """The Cartesian vector of bond b (from site i to site j in the image b["image"])."""
    lat = np.asarray(structure.lattice.matrix, float)
    cart = np.asarray(structure.cart_coords, float)
    return cart[b["j"]] + np.asarray(b["image"], float) @ lat - cart[b["i"]]


def _bond_type(key: str) -> str:
    return sk.bond_type(key)


# ---------------------------------------------------------------------------------------------------------------
#  the views
# ---------------------------------------------------------------------------------------------------------------
def _bonds(structure, cutoff, shells, labels):
    """Undirected bonds within ``cutoff`` (each once): {i, j, image, dist, shell}; plus the directed pair list."""
    pairs, d0 = sk.neighbour_pairs(structure, cutoff)
    lat = np.asarray(structure.lattice.matrix, float)
    inv = np.linalg.inv(lat)
    cart = np.asarray(structure.cart_coords, float)
    seen, bonds = set(), []
    for p in pairs:
        img = _image(inv, cart[p["i"]], cart[p["j"]], p["d_vec"])
        key = (p["i"], p["j"], tuple(img))
        rev = (p["j"], p["i"], tuple(-v for v in img))
        p["image"] = img
        p["shell"] = _shell_index(shells, p["dist"])
        if rev in seen or key in seen:
            continue
        seen.add(key)
        bonds.append({"i": int(p["i"]), "j": int(p["j"]), "image": img, "dist": float(p["dist"]),
                      "shell": p["shell"], "label": f"{labels[p['i']]}–{labels[p['j']]}"})
    bonds.sort(key=lambda b: (b["dist"], b["i"], b["j"], b["image"]))
    return bonds, pairs, d0


def structure_view(meta: dict, symprec: float = 1e-3, max_shells: int = MAX_SHELLS_LISTED) -> dict:
    """The cell of the DFT calculation for the structure tab: lattice and its parameters, space group, sites (with
    Cartesian coordinates, equivalence classes, moments), element colours / radii, the first neighbour shells and
    the bonds out to shell ``max_shells`` (each with its shell number)."""
    st = structure_from_meta(meta)
    lat = np.asarray(st.lattice.matrix, float)
    a, b, c = (float(v) for v in np.linalg.norm(lat, axis=1))
    sym = symmetry(st, symprec)
    labels = site_labels(meta)
    shells = sk.neighbour_shells(st)
    n_list = int(min(max_shells, len(shells)))
    bonds, _, d0 = _bonds(st, shells[n_list - 1] + sk.SHELL_MARGIN, shells, labels) if n_list else ([], [], None)
    bonds = [b for b in bonds if b["shell"] <= n_list]
    moments = shown_moments(meta)
    vac = vacuum_axes(lat, st.frac_coords)
    elements = []
    for e in meta["elements"]:
        idx = [i for i, s in enumerate(meta["sites"]) if s["symbol"] == e]
        elements.append({"symbol": e, "Z": int(meta["sites"][idx[0]]["Z"]), "count": len(idx),
                         "color": element_colour(e), "radius": element_radius(e)})
    eq = sym["equivalent"]
    sites = []
    for i, s in enumerate(meta["sites"]):
        m = float(moments[i]) if i < len(moments) and moments[i] is not None else None
        sites.append({"index": i, "label": labels[i], "symbol": s["symbol"],
                      "frac": [round(float(v), 6) for v in st.frac_coords[i]],
                      "cart": [round(float(v), 6) for v in st.cart_coords[i]],
                      "eq": int(eq[i]), "eq_label": labels[int(eq[i])], "moment": m})
    return {
        "formula": st.composition.reduced_formula, "n_sites": len(st), "n_elements": len(elements),
        "lattice": np.round(lat, 6).tolist(),
        "reciprocal": np.round(np.asarray(st.lattice.reciprocal_lattice.matrix), 6).tolist(),
        "abc": [round(a, 5), round(b, 5), round(c, 5)],
        "angles": [round(_angle(lat[1], lat[2]), 4), round(_angle(lat[0], lat[2]), 4), round(_angle(lat[0], lat[1]), 4)],
        "volume": round(float(abs(np.linalg.det(lat))), 4), "density": round(float(st.density), 4),
        "spacegroup": {k: v for k, v in sym.items() if k != "equivalent"},
        "n_inequivalent": len(set(eq)), "vacuum_axes": vac,
        "dimensionality": "3D" if not vac else ("2D" if len(vac) == 1 else ("1D" if len(vac) == 2 else "0D")),
        "elements": elements, "sites": sites,
        "shells": [round(float(v), 4) for v in shells[:n_list]], "d0": None if d0 is None else float(d0),
        "bonds": bonds, "dft_code": meta.get("dft_code", "vasp"),
        "moment_label": meta.get("moment_label"),      # e.g. "starting magnetisation" (a fraction, not μB)
    }


def neighbour_view(meta: dict, cfg: dict) -> dict:
    """The bonds a model with ``cfg["n_neighbors"]`` shells includes (the fitter's neighbour list), per site and
    shell, with the hopping integrals each bond brings in, and the two things to know about the global shells."""
    st = structure_from_meta(meta)
    labels = site_labels(meta)
    sym = symmetry(st, float(cfg.get("symprec", 1e-3)))
    shells = sk.neighbour_shells(st)
    requested = int(cfg.get("n_neighbors", 2))
    n = min(requested, len(shells))
    cutoff = float(sk.cutoff_from_shells(st, requested))
    bonds, pairs, d0 = _bonds(st, cutoff, shells, labels)
    orbs = site_orbitals(meta, cfg)
    els = [s["symbol"] for s in meta["sites"]]
    pair_shells = sk.pair_shell_distances(pairs, els)
    all_keys = set()
    for b in bonds:
        vec = _bond_vector(st, b)
        ei, ej = els[b["i"]], els[b["j"]]
        b["pair"] = sk.element_pair(ei, ej)
        b["pair_shell"] = sk.shell_index(pair_shells[b["pair"]], b["dist"])[0]
        keys = sorted(sk.pair_hopping_keys(ei, orbs[b["i"]], ej, orbs[b["j"]], vec)
                      | sk.pair_hopping_keys(ej, orbs[b["j"]], ei, orbs[b["i"]], -vec))
        b["integrals"] = [{"key": k, "label": integral_label(k)} for k in keys]
        all_keys.update(keys)
    per_site = []
    for i in range(len(st)):
        mine = [p for p in pairs if p["i"] == i]
        rows = []
        for sh in sorted({p["shell"] for p in mine}):
            ps = [p for p in mine if p["shell"] == sh]
            by_el = {}
            for p in ps:
                e = meta["sites"][p["j"]]["symbol"]
                by_el[e] = by_el.get(e, 0) + 1
            rows.append({"shell": int(sh), "dist": round(float(shells[sh - 1]), 4), "count": len(ps),
                         "by_element": by_el, "beyond": bool(sh > n),
                         "neighbours": [{"j": int(p["j"]), "image": p["image"], "label": labels[p["j"]]} for p in ps]})
        missing = [k for k in range(1, n + 1) if k not in {r["shell"] for r in rows}]
        per_site.append({"index": i, "label": labels[i], "eq": int(sym["equivalent"][i]), "shells": rows,
                         "missing_shells": missing, "n_bonds": len(mine)})
    warn = []
    if requested > len(shells):
        warn.append(f"Only {len(shells)} distinct distances lie within {sk.NEIGHBOUR_SEARCH_RADIUS:g} Å: "
                        f"the model uses {len(shells)} shells, not {requested}.")
    beyond = sorted({p["shell"] for p in pairs if p["shell"] > n})
    if beyond:
        warn.append("The cutoff (shell {} + {:g} Å = {:.3f} Å) also takes in {}: {} Å.".format(
            n, sk.SHELL_MARGIN, cutoff, "shell " + ", ".join(str(s) for s in beyond) if len(beyond) == 1 else
            "shells " + ", ".join(str(s) for s in beyond), ", ".join(f"{shells[s - 1]:.3f}" for s in beyond)))
    gaps = [f"{r['label']} (shell {', '.join(str(k) for k in r['missing_shells'])})"
            for r in per_site if r["missing_shells"] and sym["equivalent"][r["index"]] == r["index"]]
    if gaps:
        shown = gaps if len(gaps) <= 4 else gaps[:3] + [f"{len(gaps) - 3} more sites"]
        warn.append("Shells are counted over the whole cell, so some sites have no neighbour in some shells: "
                        + "; ".join(shown) + ".")
    return {"n_neighbors": requested, "n_shells": n, "cutoff": round(cutoff, 4),
            "d0": float(d0), "shells": [round(float(v), 4) for v in shells[:max(n, 1) + 3]],
            "pair_shells": {k: [float(x) for x in v] for k, v in pair_shells.items()},
            "pair_d0": {k: float(v[0]) for k, v in pair_shells.items()},
            "n_pairs": len(pairs), "bonds": bonds, "sites": per_site,
            "inequivalent": sorted(set(int(v) for v in sym["equivalent"])),
            "hopping_keys": [{"key": k, "label": integral_label(k)} for k in sorted(all_keys)],
            "distance_decay": bool(n > 1), "warnings": warn}


def _scaled(v0, eta, r, d0, decay: bool):
    if v0 is None:
        return None
    if decay and d0 > 1e-4 and eta is not None:
        return float(v0) * (d0 / r) ** float(eta)
    return float(v0)


def param_map(meta: dict, cfg: dict, params: dict) -> dict:
    """The fitted (or any) parameters ``params`` (SKFitter's full sk_dict) placed on the structure: per site its
    on-site energies, spin-orbit λ and exchange splitting Δ (with the site's moment sign), per bond its integrals
    V0 (at d0), η and V(r) = V0 (d0 / r)^η - the value the Hamiltonian uses for that bond - and the overlaps of a
    non-orthogonal model."""
    nv = neighbour_view(meta, cfg)
    st = structure_from_meta(meta)
    eq = symmetry(st, float(cfg.get("symprec", 1e-3)))["equivalent"]
    labels = site_labels(meta)
    orbs = site_orbitals(meta, cfg)
    moments = meta.get("moments") or []
    mag = cfg.get("magnetic_orbitals") or {}
    p = {k: float(v) for k, v in (params or {}).items() if v is not None}
    decay = nv["distance_decay"]
    sites = []
    for i, s in enumerate(meta["sites"]):
        el = s["symbol"]
        onsite = []
        for sub in orbs[i]:
            key = f"e_orbit{int(eq[i])}_{el}_{sub}"
            onsite.append({"orbital": sub, "key": key, "value": p.get(key)})
        fams = []
        for sub in orbs[i]:
            f = sk.SKFitter.FAMILY_MAP.get(sub, sub)
            if f not in fams:
                fams.append(f)
        soc = [{"family": f, "key": f"lambda_{el}_{f}", "value": p[f"lambda_{el}_{f}"]}
               for f in fams if f in ("p", "d", "f") and f"lambda_{el}_{f}" in p]
        sign = -1 if (i < len(moments) and moments[i] is not None and float(moments[i]) < 0) else 1
        exch, seen = [], set()               # one entry per exchange parameter: a family, or a symmetry subshell
        for sub in orbs[i]:
            key = sk.exchange_key(el, sub, mag)
            if key and key in p and key not in seen:
                seen.add(key)
                exch.append({"family": key[len(f"delta_spin_{el}_"):], "key": key, "value": p[key], "sign": sign,
                             "site_value": sign * p[key] / 2.0})
        sites.append({"index": i, "label": labels[i], "symbol": el, "eq_label": labels[int(eq[i])],
                      "onsite": onsite, "soc": soc, "exchange": exch})
    non_orth = bool(cfg.get("non_orthogonal"))
    per_shell = (cfg.get("hopping_mode") or "decay") == "shell"
    for b in nv["bonds"]:
        r = float(b["dist"])
        rows = []
        shell_d = nv["pair_shells"][b["pair"]]          # the shells of the bond's element pair
        n = b["pair_shell"]
        d0 = float(nv["pair_d0"][b["pair"]])
        for it in b["integrals"]:
            k, bt = it["key"], _bond_type(it["key"])
            if per_shell:                     # the bond's own shell integral: V_n of its pair's shell n
                v = p.get(f"{k}_{n}")
                rn = float(shell_d[n])
                eta = p.get(f"eta_{bt}", 2.5) if abs(r - rn) > 1e-5 and r > 1e-6 else None
                val = v if v is None or eta is None else float(v) * (rn / r) ** float(eta)
                row = {**it, "v0": v, "eta": eta, "value": val, "shell_key": f"{k}_{n}"}
                if non_orth:
                    sv = p.get(f"s_{k[2:]}_{n}")
                    row.update(s0=sv, s_eta=None, s_value=sv)
                rows.append(row)
                continue
            v0 = p.get(f"{k}_0", p.get(k))
            eta = p.get(f"eta_{bt}") if decay else None
            row = {**it, "v0": v0, "eta": eta, "value": _scaled(v0, eta, r, d0, decay)}
            if non_orth:
                s0 = p.get(f"s_{k[2:]}_0", p.get(f"s_{k[2:]}"))
                seta = p.get(f"eta_s_{bt}") if decay else None
                row.update(s0=s0, s_eta=seta, s_value=_scaled(s0, seta, r, d0, decay))
            rows.append(row)
        b["integrals"] = rows
    return {**{k: nv[k] for k in ("n_neighbors", "n_shells", "cutoff", "d0", "pair_shells", "pair_d0", "shells", "n_pairs", "warnings",
                                   "distance_decay", "hopping_keys")},
            "bonds": nv["bonds"], "sites": sites, "non_orthogonal": non_orth,
            "hopping_mode": "shell" if per_shell else "decay"}
