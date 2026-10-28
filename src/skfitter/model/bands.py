"""Band structures and band edges of any model, and how they change with a magnetic field (phase M2a).

    from skfitter.model import bands as B
    edges = B.band_edges(model)                      # VBM, CBM, gap (direct or indirect), where they sit, spins
    path = B.kpath(model, [("Γ", [0, 0, 0]), ("K", [1/3, 1/3, 0]), ("M", [0.5, 0, 0]), ("Γ", [0, 0, 0])])
    bs = B.bands(model, path, window=(-0.5, 0.5))   # energies along the path (a zoom window around E_F)
    sweep = B.field_sweep(model, [0, 2, 4, 6, 8, 10], direction="out-of-plane",
                          valleys={"K": [1/3, 1/3, 0], "K'": [-1/3, -1/3, 0]})
    rot = B.direction_sweep(model, 10.0, angles=range(0, 91, 15))   # out of plane (0°) -> in plane (90°)
    B.write_csv(sweep["rows"], "gap_vs_B.csv")      # for plots (the web plots come with phase M2c)

Conventions: energies in eV relative to E_F = 0; k in fractional coordinates of the reciprocal lattice (``k_frac``)
and in 1/Å (``k_cart``, including 2π). The valence bands are the ``n_valence`` lowest bands; by default those below
E_F (+1 meV) in the model without the field, kept fixed through a sweep so that "VBM" always means the same band.
For a 2D material (no hoppings along one lattice direction) k is searched in the plane only, and the field
directions "out-of-plane" / "in-plane" refer to the plane of the layer; for a 3D crystal they refer to z.
Field-induced changes are small (about 0.12 meV per tesla for g = 2): use ``window`` and the sweep tables, not a
full band plot. Spin expectations <σ> of the edge states are given for spinful models.
"""
from __future__ import annotations

import csv

import numpy as np

from .modifiers import E_F_TOL, MU_B, G_ELECTRON, _eig_lowdin, apply, periodic_axes

# == k-space ======================================================================================================


def reciprocal(model) -> np.ndarray:
    """Rows are the reciprocal lattice vectors b_i (1/Å, with 2π)."""
    return 2 * np.pi * np.linalg.inv(np.asarray(model.lattice, float)).T


def kpath(model, points, n_per_inv_A: float = 60.0, min_points: int = 8) -> dict:
    """A path through labelled points: ``points`` = [(label, k_frac), ...]; a ``None`` entry breaks the path (the
    next segment starts at the following point; the tick reads "A|B"). Points per segment ~ its length times
    ``n_per_inv_A`` (at least ``min_points``)."""
    rec = reciprocal(model)
    segs, cur = [], []
    for item in points:
        if item is None:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append((str(item[0]), np.asarray(item[1], float)))
    if cur:
        segs.append(cur)
    kf, x, ticks, x0 = [], [], [], 0.0
    for n_seg, seg in enumerate(segs):
        label, f = seg[0]
        if n_seg == 0:
            ticks.append((x0, label))
        else:
            ticks[-1] = (ticks[-1][0], ticks[-1][1] + "|" + label)
        kf.append(f)
        x.append(x0)
        for label, g in seg[1:]:
            length = float(np.linalg.norm((g - f) @ rec))
            n = max(min_points, int(np.ceil(length * n_per_inv_A)))
            for t in np.linspace(0, 1, n + 1)[1:]:
                kf.append(f + t * (g - f))
                x.append(x0 + t * length)
            x0 += length
            ticks.append((x0, label))
            f = g
    kf = np.array(kf)
    return {"k_frac": kf, "k_cart": kf @ rec, "x": np.array(x), "ticks": ticks}


def default_path(model) -> list:
    """A standard path for the model's lattice (pymatgen's HighSymmKpath when available; Γ-X-M-Γ otherwise),
    restricted to the plane for a 2D material."""
    per = periodic_axes(model)
    hexagonal = suggest_valleys(model)
    if hexagonal:                                       # a hexagonal 2D lattice: Γ-M-K-Γ-K' through both valleys
        m = [0.0, 0.0, 0.0]
        m[per[0]] = 0.5
        return [("Γ", np.zeros(3)), ("M", np.array(m)), ("K", np.array(hexagonal["K"])), ("Γ", np.zeros(3)),
                ("K'", np.array(hexagonal["K'"]))]
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        st = Structure(Lattice(model.lattice), model.species, model.frac_coords)
        hk = HighSymmKpath(st, path_type="setyawan_curtarolo")
        kp = hk.kpath
        prim = np.asarray(hk.prim.lattice.matrix)
        if np.allclose(prim, model.lattice, atol=1e-4):
            out = []
            for seg in kp["path"]:
                pts = [(lab.replace("\\Gamma", "Γ"), np.asarray(kp["kpoints"][lab], float)) for lab in seg]
                if len(per) < 3 and any(abs(p[1][a]) > 1e-8 for p in pts for a in range(3) if a not in per):
                    continue
                if out:
                    out.append(None)
                out += pts
            if out:
                return out
    except Exception:                                   # noqa: BLE001 - fall back to a simple path
        pass
    pts = [("Γ", np.zeros(3))]
    for a in per[:2]:
        v = np.zeros(3)
        v[a] = 0.5
        pts.append(("XYZ"[a], v))
    if len(per) >= 2:
        v = np.zeros(3)
        v[per[0]] = v[per[1]] = 0.5
        pts.append(("M", v))
    pts.append(("Γ", np.zeros(3)))
    return pts


# == eigenstates =================================================================================================
def _spin_ops(model):
    """sigma_x, sigma_y, sigma_z in the model's basis (spin-up orbitals of each site first), or None."""
    if not model.spinful:
        return None
    blocks = []
    for i in range(model.n_sites):
        n = model.norbs(i) // 2
        blocks.append(n)
    D = 2 * sum(blocks)
    ops = [np.zeros((D, D), dtype=complex) for _ in range(3)]
    s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    off = 0
    for n in blocks:
        for a in range(3):
            ops[a][off:off + 2 * n, off:off + 2 * n] = np.kron(s[a], np.eye(n))
        off += 2 * n
    return ops


def eigen(model, k_cart, vectors: bool = False):
    """Eigenvalues (sorted) at one k; with ``vectors`` also the Löwdin-orthonormal eigenvectors (columns)."""
    S = model.bloch(k_cart, overlap=True) if model.non_orthogonal else None
    e, U, _ = _eig_lowdin(model.bloch(k_cart), S)
    return (e, U) if vectors else e


def spin_of(model, U, ops=None) -> np.ndarray:
    """<σ> (n_states, 3) of eigenvectors U (columns). The overlap of an SK model is spin-diagonal, so σ is the
    same in the Löwdin basis."""
    ops = ops if ops is not None else _spin_ops(model)
    if ops is None:
        return None
    return np.array([[float(np.real(np.vdot(U[:, n], op @ U[:, n]))) for op in ops] for n in range(U.shape[1])])


def bands(model, path=None, window=None, spin: bool = True) -> dict:
    """Bands along ``path`` (a kpath(), a list of labelled points, or None for default_path()). ``window`` =
    (Emin, Emax) keeps the bands that enter it (a zoom on the edges)."""
    if path is None or isinstance(path, list):
        path = kpath(model, path or default_path(model))
    ops = _spin_ops(model) if spin else None
    E, S = [], []
    for k in path["k_cart"]:
        e, U = eigen(model, k, vectors=True)
        E.append(e)
        if ops is not None:
            S.append(spin_of(model, U, ops))
    E = np.array(E)
    keep = np.arange(E.shape[1])
    if window is not None:
        lo, hi = window
        keep = np.where((E.max(axis=0) >= lo) & (E.min(axis=0) <= hi))[0]
    out = {"x": path["x"], "k_frac": path["k_frac"], "ticks": path["ticks"], "band_index": keep.tolist(),
           "energies": E[:, keep], "window": window}
    if ops is not None:
        out["spin"] = np.array(S)[:, keep, :]
    return out


# == band edges ==================================================================================================
def _grid(model, n: int):
    per = periodic_axes(model)
    axes = [np.arange(n) / n if a in per else np.zeros(1) for a in range(3)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


def count_valence(model, mesh: int = 12, energies=None) -> int:
    """Bands below E_F (+1 meV) on a coarse mesh: the count most k-points agree on (a semimetal's touching points
    and a VBM sitting exactly at E_F do not change it). ValueError for a metal (the count changes over more than a
    quarter of the k-points). ``energies``: optional, eigenvalues (n, D) at Cartesian k-points (n, 3) in one call
    (e.g. a skfitter.properties Kernel's), instead of the model's own Bloch sums one k-point at a time."""
    rec = reciprocal(model)
    grid = _grid(model, mesh)
    if energies is not None:
        counts = np.sum(np.asarray(energies(grid @ rec)) < E_F_TOL, axis=1).astype(int)
    else:
        counts = np.array([int(np.sum(eigen(model, kf @ rec) < E_F_TOL)) for kf in grid])
    values, freq = np.unique(counts, return_counts=True)
    n = int(values[np.argmax(freq)])
    if freq.max() < 0.75 * len(counts):
        raise ValueError("The model is a metal at E_F = 0 (the number of occupied bands changes with k): give "
                         "n_valence for the gap you mean.")
    return n


def _valence_for(base, modified, n_base: int) -> int:
    """The valence count of a modified model: twice the base count when the modifiers made it spinful."""
    return 2 * n_base if modified.spinful and not base.spinful else n_base


def _refine(model, kf, band, sign, per, step, tol=1e-7):
    """Local search for an extremum of ``sign * E_band(k)`` (sign -1: maximum) around ``kf`` (fractional)."""
    rec = reciprocal(model)
    kf = np.array(kf, float)

    def f(x):
        return sign * eigen(model, x @ rec)[band]
    best = f(kf)
    moves = [np.array(v) for v in np.array(np.meshgrid(*[(-1, 0, 1)] * len(per), indexing="ij")).reshape(
        len(per), -1).T if np.any(v)]
    while step > tol:
        improved = False
        for mv in moves:
            x = kf.copy()
            x[per] += step * mv
            v = f(x)
            if v < best - 1e-13:
                best, kf, improved = v, x, True
        if not improved:
            step *= 0.5
    return kf, sign * best


def _edge(model, kf, band, ops):
    rec = reciprocal(model)
    e, U = eigen(model, kf @ rec, vectors=True)
    out = {"E": float(e[band]), "band": int(band), "k_frac": [round(float(x), 7) for x in kf],
           "k_cart": [round(float(x), 7) for x in kf @ rec]}
    if ops is not None:
        out["spin"] = [round(float(x), 6) for x in spin_of(model, U[:, [band]], ops)[0]]
    return out


def band_edges(model, n_valence=None, mesh: int = 24, refine: bool = True) -> dict:
    """The valence band maximum, conduction band minimum, the gap, whether it is direct, and the smallest direct
    gap; each with its k-point (and spin for spinful models). A semimetal (touching bands) has gap 0."""
    nv = n_valence if n_valence is not None else count_valence(model)
    rec = reciprocal(model)
    per = periodic_axes(model)
    grid = _grid(model, mesh)
    E = np.array([eigen(model, kf @ rec) for kf in grid])
    if not 0 < nv < E.shape[1]:
        raise ValueError(f"n_valence must be between 1 and {E.shape[1] - 1}.")
    iv, ic = int(np.argmax(E[:, nv - 1])), int(np.argmin(E[:, nv]))
    idg = int(np.argmin(E[:, nv] - E[:, nv - 1]))
    kv, kc, kd = grid[iv], grid[ic], grid[idg]
    ops = _spin_ops(model)
    if refine and per:
        step = 0.5 / mesh
        kv, _ = _refine(model, kv, nv - 1, -1, per, step)
        kc, _ = _refine(model, kc, nv, +1, per, step)

        def gap_at(x):
            e = eigen(model, x @ rec)
            return e[nv] - e[nv - 1]
        # direct gap: refine the difference with the same search
        kd = np.array(kd, float)
        best, s = gap_at(kd), step
        moves = [np.array(v) for v in np.array(np.meshgrid(*[(-1, 0, 1)] * len(per), indexing="ij")).reshape(
            len(per), -1).T if np.any(v)]
        while s > 1e-7:
            improved = False
            for mv in moves:
                x = kd.copy()
                x[per] += s * mv
                v = gap_at(x)
                if v < best - 1e-13:
                    best, kd, improved = v, x, True
            if not improved:
                s *= 0.5
    vbm, cbm = _edge(model, kv, nv - 1, ops), _edge(model, kc, nv, ops)
    e_d = eigen(model, kd @ rec)
    gap = cbm["E"] - vbm["E"]
    direct = {"E": float(e_d[nv] - e_d[nv - 1]), "k_frac": [round(float(x), 7) for x in kd]}
    same_k = np.linalg.norm(((np.array(vbm["k_frac"]) - np.array(cbm["k_frac"]) + 0.5) % 1.0) - 0.5) < 1e-4
    return {"n_valence": int(nv), "vbm": vbm, "cbm": cbm, "gap": float(max(gap, 0.0)), "overlap": gap < 0,
            "gap_type": "direct" if same_k or abs(direct["E"] - gap) < 1e-6 else "indirect", "direct_gap": direct}


def valley_edges(model, valleys: dict, n_valence=None, radius: float = 0.04, levels: int = 2) -> dict:
    """Band edges in each valley: ``valleys`` = {name: k_frac}; the extrema of the top valence and bottom
    conduction band are searched within ``radius`` (fractional) of each point. For each valley: VB top, CB bottom,
    their gap, where they sit, and the ``levels`` highest valence / lowest conduction energies at those k-points
    (so a spin-split pair shows both members), with spins."""
    nv = n_valence if n_valence is not None else count_valence(model)
    per = periodic_axes(model)
    rec = reciprocal(model)
    ops = _spin_ops(model)
    out = {}
    for name, k0 in valleys.items():
        k0 = np.asarray(k0, float)
        kv, ev = _refine(model, k0, nv - 1, -1, per, radius / 4)
        kc, ec = _refine(model, k0, nv, +1, per, radius / 4)
        for kk in (kv, kc):
            if np.linalg.norm((kk - k0)[per]) > radius * 1.5:
                raise ValueError(f"valley {name}: the band extremum is not within {radius} of {k0.tolist()} "
                                 "(not a valley of this model?)")
        e_v, U_v = eigen(model, kv @ rec, vectors=True)
        e_c, U_c = eigen(model, kc @ rec, vectors=True)
        row = {"vb": _edge(model, kv, nv - 1, ops), "cb": _edge(model, kc, nv, ops), "gap": float(ec - ev),
               "vb_levels": [float(x) for x in e_v[max(0, nv - levels):nv][::-1]],
               "cb_levels": [float(x) for x in e_c[nv:nv + levels]]}
        if ops is not None:
            row["vb_spins"] = spin_of(model, U_v[:, max(0, nv - levels):nv][:, ::-1], ops).round(6).tolist()
            row["cb_spins"] = spin_of(model, U_c[:, nv:nv + levels], ops).round(6).tolist()
        out[name] = row
    return out


# == field directions ============================================================================================
def plane_axes(model) -> dict:
    """Unit vectors: ``normal`` (out of plane: perpendicular to the layer of a 2D material, z for 3D) and two
    in-plane directions ``in_plane_1`` (along the first in-plane lattice vector) and ``in_plane_2``."""
    lat = np.asarray(model.lattice, float)
    per = periodic_axes(model)
    if len(per) == 2:
        n = np.cross(lat[per[0]], lat[per[1]])
        a1 = lat[per[0]]
    else:
        n = np.array([0.0, 0.0, 1.0])
        a1 = np.array([1.0, 0.0, 0.0])
    n = n / np.linalg.norm(n)
    e1 = a1 - (a1 @ n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return {"normal": n, "in_plane_1": e1, "in_plane_2": e2, "two_d": len(per) == 2}


def suggest_valleys(model) -> dict:
    """The K and K' points of a hexagonal 2D lattice (|a1| = |a2| at 60° or 120°), in fractional coordinates;
    {} for any other lattice (give the valleys yourself)."""
    per = periodic_axes(model)
    if len(per) != 2:
        return {}
    a1, a2 = (np.asarray(model.lattice[a], float) for a in per)
    if abs(np.linalg.norm(a1) - np.linalg.norm(a2)) > 1e-3 * np.linalg.norm(a1):
        return {}
    cos = float(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)))

    def put(f1, f2):
        v = [0.0, 0.0, 0.0]
        v[per[0]], v[per[1]] = f1, f2
        return v
    if abs(cos - 0.5) < 1e-3:                                   # 60°
        return {"K": put(2 / 3, 1 / 3), "K'": put(1 / 3, 2 / 3)}
    if abs(cos + 0.5) < 1e-3:                                   # 120°
        return {"K": put(1 / 3, 1 / 3), "K'": put(2 / 3, 2 / 3)}
    return {}


def field_direction(model, direction="out-of-plane", theta=None, phi: float = 0.0) -> np.ndarray:
    """A unit vector: "out-of-plane", "in-plane" (along in_plane_1), "in-plane-2", a Cartesian 3-vector, or the
    angles ``theta`` (degrees from the normal) and ``phi`` (degrees in the plane from in_plane_1)."""
    ax = plane_axes(model)
    if theta is not None:
        t, p = np.radians(float(theta)), np.radians(float(phi))
        v = np.cos(t) * ax["normal"] + np.sin(t) * (np.cos(p) * ax["in_plane_1"] + np.sin(p) * ax["in_plane_2"])
    elif isinstance(direction, str):
        v = {"out-of-plane": ax["normal"], "in-plane": ax["in_plane_1"], "in-plane-1": ax["in_plane_1"],
             "in-plane-2": ax["in_plane_2"]}.get(direction)
        if v is None:
            raise ValueError("direction: 'out-of-plane', 'in-plane', 'in-plane-2' or [x, y, z]")
    else:
        v = np.asarray(direction, float)
    return v / np.linalg.norm(v)


# == sweeps ======================================================================================================
def _row(model, value, nv, valleys, mesh, refine):
    e = band_edges(model, n_valence=nv, mesh=mesh, refine=refine)
    row = {"value": float(value), "vbm": e["vbm"]["E"], "cbm": e["cbm"]["E"], "gap": e["gap"],
           "gap_type": e["gap_type"], "direct_gap": e["direct_gap"]["E"], "vbm_k": e["vbm"]["k_frac"],
           "cbm_k": e["cbm"]["k_frac"]}
    if "spin" in e["vbm"]:
        row["vbm_spin"], row["cbm_spin"] = e["vbm"]["spin"], e["cbm"]["spin"]
    if valleys:
        for name, v in valley_edges(model, valleys, n_valence=nv).items():
            row[f"{name}:vb"], row[f"{name}:cb"], row[f"{name}:gap"] = v["vb"]["E"], v["cb"]["E"], v["gap"]
            row[f"{name}:vb_levels"], row[f"{name}:cb_levels"] = v["vb_levels"], v["cb_levels"]
            if "vb_spins" in v:
                row[f"{name}:vb_spins"], row[f"{name}:cb_spins"] = v["vb_spins"], v["cb_spins"]
    return row


def sweep(model, values, make, n_valence=None, valleys=None, mesh: int = 24, refine: bool = True,
          quantity: str = "value") -> dict:
    """Band edges of ``apply(model, make(v))`` for each v in ``values`` (``make`` returns a list of modifiers).
    The valence count comes from the unmodified model and is kept through the sweep."""
    nv0 = n_valence if n_valence is not None else count_valence(model)
    rows, nv = [], nv0
    for v in values:
        m = apply(model, make(v))
        nv = _valence_for(model, m, nv0)
        r = _row(m, v, nv, valleys, mesh, refine)
        r[quantity] = r.pop("value")
        rows.append(r)
    return {"quantity": quantity, "n_valence": nv, "rows": rows}


def field_sweep(model, B_values, direction="out-of-plane", theta=None, phi=0.0, g: float = G_ELECTRON,
                spin: bool = True, orbital: bool = True, n_valence=None, valleys=None, mesh: int = 24,
                refine: bool = True) -> dict:
    """Band edges, gap and valley edges against the field strength |B| (tesla) along one direction (see
    field_direction). Adds g-factor fits: for every edge and gap, the slope dE/dB (meV/T) and g = slope / mu_B."""
    u = field_direction(model, direction, theta, phi)
    out = sweep(model, B_values, lambda b: [{"kind": "zeeman", "B": (b * u).tolist(), "g": g, "spin": spin,
                                             "orbital": orbital}],
                n_valence=n_valence, valleys=valleys, mesh=mesh, refine=refine, quantity="B")
    out["direction"] = u.tolist()
    out["slopes"] = slopes(out["rows"], "B")
    return out


def efield_sweep(model, E_values, n_valence=None, valleys=None, mesh: int = 24, refine: bool = True) -> dict:
    """Band edges, gap and valley edges against a uniform electric field (V/Å) along the normal of a 2D material
    (the efield modifier, W4d: a gate field across the layer(s), no screening). Adds the slopes (meV per V/Å)."""
    ax = plane_axes(model)
    if not ax["two_d"]:
        raise ValueError("an electric field sweep needs a 2D material (a layer with vacuum across it)")
    u = np.asarray(ax["normal"], float)
    out = sweep(model, E_values, lambda e: [{"kind": "efield", "E": (e * u).tolist()}] if e else [],
                n_valence=n_valence, valleys=valleys, mesh=mesh, refine=refine, quantity="E")
    out["direction"] = u.tolist()
    out["slopes"] = slopes(out["rows"], "E")
    out["position_matrices"] = bool(getattr(model, "position_matrices", None))
    return out


def direction_sweep(model, B: float, angles=range(0, 91, 15), phi: float = 0.0, g: float = G_ELECTRON,
                    spin: bool = True, orbital: bool = True, n_valence=None, valleys=None, mesh: int = 24,
                    refine: bool = True) -> dict:
    """Band edges and gap at a fixed |B| (tesla) as the field turns from out of plane (0°) to in plane (90°),
    in the plane through the normal and the in-plane direction at ``phi``."""
    def make(theta):
        v = B * field_direction(model, theta=theta, phi=phi)
        return [{"kind": "zeeman", "B": v.tolist(), "g": g, "spin": spin, "orbital": orbital}]
    out = sweep(model, list(angles), make, n_valence=n_valence, valleys=valleys, mesh=mesh, refine=refine,
                quantity="theta_deg")
    out["B"], out["phi_deg"] = float(B), float(phi)
    return out


def slopes(rows, x: str) -> dict:
    """Least-squares slope (meV per unit of x) of every scalar energy column, and for x = B the g-factor
    slope / mu_B (meV/T -> dimensionless)."""
    xs = np.array([r[x] for r in rows], float)
    out = {}
    if len(xs) < 2 or np.ptp(xs) == 0:
        return out
    for key in rows[0]:
        vals = [r.get(key) for r in rows]
        if key == x or not all(isinstance(v, float) for v in vals):
            continue
        s = float(np.polyfit(xs, np.array(vals) * 1000.0, 1)[0])
        out[key] = {"slope_meV_per_" + ("T" if x == "B" else x): round(s, 6)}
        if x == "B":
            out[key]["g"] = round(s / (1000.0 * MU_B), 4)
    return out


def write_csv(rows, path: str) -> str:
    """The rows of a sweep as CSV (lists become ';'-joined)."""
    keys = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([";".join(str(x) for x in np.ravel(r[k])) if isinstance(r[k], (list, tuple)) else r[k]
                        for k in keys])
    return path
