"""Electronic structure: bands with orbital / spin weights, DOS and projected DOS, band edges and gaps, effective
masses, and the comparison of several models on one path."""
from __future__ import annotations

import numpy as np

from .. import register
from ..core.calculator import Calculator, Cost, Requires, Result, seconds_per_k
from ..core.inputs import Bool, Choice, EnergyGrid, Files, Mesh, Number, Vector
from ..core.select import DK, N_OCC, PATH, occupied, path_points
from ..services import kspace
from ..services.integrate import cumulative, gaussian_dos, tetrahedron_dos
from ..services.occupation import spin_degeneracy
from ..services.units import HBAR2_OVER_ME

GROUPS = [("none", "none"), ("element", "element"), ("site", "atom"), ("l", "s / p / d / f"),
          ("orbital", "orbital"), ("spin", "spin")]


def _D(model):
    return sum(model.norbs(i) for i in range(model.n_sites))


def _weights(kernel, U, by):
    """{group: (n_k, n_b) weights} from eigenvectors U (n_k, D, D) (Löwdin orbitals for non-orthogonal models)."""
    P = np.abs(U) ** 2                                         # (n_k, D(basis), n_b)
    return {g: P[:, idx, :].sum(axis=1) for g, idx in kernel.orbital_groups(by).items()}


@register
class Bands(Calculator):
    id = "electronic.bands"
    title = "Band structure"
    help = "Bands along a k-path, optionally coloured by element, atom, s/p/d/f, orbital or spin weight."
    formula = r"H(\mathbf k)\,|n\mathbf k\rangle = E_n(\mathbf k)\,|n\mathbf k\rangle"
    inputs = {"path": PATH, "dk": Number("Path spacing", 0.02, unit="1/Å", min=1e-4, max=1.0),
              "weights": Choice("Weights by", "none", GROUPS),
              "emin": Number("Window from", -6.0, unit="eV"), "emax": Number("to", 6.0, unit="eV")}

    def cost(self, model, p):
        D = _D(model)
        n = len(kspace.path(model, path_points(p["path"]), p["dk"])["x"])
        return Cost(n, D, n * seconds_per_k(D))

    def run(self, ctx, model, p):
        K = ctx.kernel
        kp = kspace.path(model, path_points(p["path"]), p["dk"])
        E, U = K.eig(kp["cart"])
        keep = np.where((E.max(axis=0) >= p["emin"]) & (E.min(axis=0) <= p["emax"]))[0]
        res = Result()
        extra = {"band_index": keep.tolist(), "window": [p["emin"], p["emax"]]}
        if p["weights"] != "none":
            extra["weights"] = {g: w[:, keep] for g, w in _weights(K, U, p["weights"]).items()}
        if model.spinful:
            sops = K.spin_operators()
            Uk = U[:, :, keep]
            extra["spin_z"] = np.real(np.einsum("kai,ab,kbi->ki", Uk.conj(), sops[2], Uk))
        res.bands("Band structure", kp["x"], E[:, keep], [[x, lab] for x, lab in kp["ticks"]], **extra)
        res.summary = [("k-points", str(len(kp["x"]))), ("bands shown", f"{len(keep)} of {K.D}"),
                       ("lowest / highest energy", f"{E.min():.3f} / {E.max():.3f} eV")]
        try:
            nv = occupied(model, 0, ctx.kernel)
            if 0 < nv < K.D:
                vb, cb = E[:, nv - 1], E[:, nv]
                res.summary.append(("gap on the path", f"{max(0.0, cb.min() - vb.max()):.4f} eV"))
        except ValueError:
            res.summary.append(("gap on the path", "metal at E_F"))
        res.data = {"energies": E, "x": kp["x"]}
        return res


@register
class DOS(Calculator):
    id = "electronic.dos"
    title = "Density of states"
    help = ("DOS and projected DOS on a uniform mesh, by the linear tetrahedron method (triangles in 2D) or Gaussian "
            "smearing; also the number of electrons per cell below each energy (carrier density relative to E_F).")
    formula = r"g(E) = \frac{1}{N_k}\sum_{n\mathbf k}\delta(E - E_n(\mathbf k))"
    requires = Requires(dims=(1, 2, 3))
    inputs = {"mesh": Mesh("k-mesh", (24, 24, 24)),
              "method": Choice("Method", "tetrahedron", [("tetrahedron", "tetrahedra / triangles"),
                                                         ("gaussian", "Gaussian smearing")]),
              "sigma": Number("Gaussian width", 0.05, unit="eV", min=1e-4),
              "energies": EnergyGrid("Energies", (-4.0, 4.0, 801)),
              "projections": Choice("Projected on", "element", GROUPS)}

    def cost(self, model, p):
        D = _D(model)
        n = int(np.prod(kspace.effective_mesh(model, p["mesh"])))
        return Cost(n, D, n * seconds_per_k(D))

    def run(self, ctx, model, p):
        K = ctx.kernel
        nk = kspace.effective_mesh(model, p["mesh"])
        k = kspace.mesh(model, nk) @ kspace.reciprocal(model)
        ctx.progress(0.1, f"eigenstates on {len(k)} k-points")
        E, U = K.eig(k)
        g = spin_degeneracy(model)
        Eg = np.linspace(p["energies"]["min"], p["energies"]["max"], p["energies"]["n"])

        def dos(weights=None):
            if p["method"] == "tetrahedron":
                return g * tetrahedron_dos(Eg, E, nk, weights)
            return g * gaussian_dos(Eg, E, p["sigma"], weights)
        total = dos()
        series = [("total", total)]
        if p["projections"] != "none":
            for name, w in _weights(K, U, p["projections"]).items():
                series.append((name, dos(w)))
        N = cumulative(Eg, total) + g * np.sum(E < Eg[0]) / len(k)
        N0 = g * np.sum(E < 0.0) / len(k)
        res = Result()
        res.curve("Density of states", Eg, "E − E_F (eV)", series, "states / eV / cell (with spin)", marker_x=0.0)
        res.curve("Electrons per cell below E", Eg, "E − E_F (eV)",
                  [("N(E)", N), ("N(E) − N(E_F): added electrons per cell", N - N0)], "electrons / cell")
        i0 = int(np.argmin(np.abs(Eg)))
        res.summary = [("k-mesh", f"{nk[0]}×{nk[1]}×{nk[2]} ({len(k)} points), {p['method']}"),
                       ("DOS at E_F", f"{total[i0]:.4g} states/eV/cell"),
                       ("electrons per cell below E_F", f"{N0:.4f}"),
                       ("states in the model (with spin)", f"{g * K.D}")]
        res.data = {"E": Eg, "dos": total, "N": N}
        return res


@register
class BandEdges(Calculator):
    id = "electronic.edges"
    title = "Band edges and gap"
    help = "VBM, CBM, the gap (direct or indirect), the smallest direct gap and, for hexagonal 2D lattices, the valleys."
    formula = r"E_g = \min_{\mathbf k} E_c(\mathbf k) - \max_{\mathbf k} E_v(\mathbf k)"
    requires = Requires(dims=(1, 2, 3))
    inputs = {"mesh": Number("Search mesh per direction", 24, integer=True, min=4, max=200), "n_occupied": N_OCC,
              "valleys": Bool("Valleys K, K' (hexagonal 2D)", True)}

    def cost(self, model, p):
        D = _D(model)
        n = p["mesh"] ** len(kspace.periodic_axes(model)) + 400
        return Cost(n, D, n * seconds_per_k(D, 3))

    def run(self, ctx, model, p):
        from skfitter.model import bands as TB
        nv = occupied(model, p["n_occupied"], ctx.kernel)
        e = TB.band_edges(model, n_valence=nv, mesh=p["mesh"])
        res = Result()
        rows = [["VBM", round(e["vbm"]["E"], 6), str(e["vbm"]["k_frac"])],
                ["CBM", round(e["cbm"]["E"], 6), str(e["cbm"]["k_frac"])],
                ["gap (" + ("overlap" if e["overlap"] else e["gap_type"]) + ")", round(e["gap"], 6), ""],
                ["smallest direct gap", round(e["direct_gap"]["E"], 6), str(e["direct_gap"]["k_frac"])]]
        vals = TB.suggest_valleys(model) if p["valleys"] else {}
        ve = {}
        if vals:
            try:
                ve = TB.valley_edges(model, vals, n_valence=nv)
            except ValueError as exc:
                res.warnings.append(str(exc))
        for name, v in ve.items():
            rows.append([f"valley {name}: VB top / CB bottom / gap", f"{v['vb']['E']:.6f} / {v['cb']['E']:.6f} / "
                         f"{v['gap']:.6f}", str(v["vb"]["k_frac"])])
        res.table("Band edges (eV; k fractional)", ["", "E (eV)", "k"], rows)
        res.summary = [("gap", f"{e['gap']:.4f} eV ({e['gap_type']})"), ("VBM", f"{e['vbm']['E']:.4f} eV"),
                       ("CBM", f"{e['cbm']['E']:.4f} eV"), ("valence bands", str(nv))]
        res.data = {"edges": e, "valleys": ve}
        return res


@register
class EffectiveMass(Calculator):
    id = "electronic.masses"
    title = "Effective masses"
    help = ("Effective-mass tensor m_e (∂²E/∂k_a∂k_b)⁻¹ ħ² of a band at a k-point (by default the VBM and CBM), from "
            "second-order perturbation theory on the exact H(k) derivatives (or finite differences of E). Principal "
            "masses and directions; in 2D the in-plane tensor.")
    formula = r"\frac{\partial^2E_n}{\partial k_a\partial k_b} = \langle n|\partial_a\partial_bH|n\rangle + 2\,\mathrm{Re}\sum_{m\neq n}\frac{\langle n|\partial_aH|m\rangle\langle m|\partial_bH|n\rangle}{E_n-E_m}"
    inputs = {"where": Choice("At", "edges", [("edges", "VBM and CBM"), ("point", "a k-point and band")]),
              "k": Vector("k (fractional)", (0.0, 0.0, 0.0)), "band": Number("Band (0 = lowest)", 0, integer=True, min=0),
              "n_occupied": N_OCC,
              "method": Choice("Method", "perturbation", [("perturbation", "perturbation theory"),
                                                          ("finite_difference", "finite differences of E")]),
              "step": Number("Finite-difference step", 0.005, unit="1/Å", min=1e-5, advanced=True)}

    def run(self, ctx, model, p):
        from skfitter.model import bands as TB
        K = ctx.kernel
        rec = kspace.reciprocal(model)
        per = kspace.periodic_axes(model)
        cart_axes = _cart_axes(model, per)
        points = []
        if p["where"] == "edges":
            nv = occupied(model, p["n_occupied"], ctx.kernel)
            e = TB.band_edges(model, n_valence=nv)
            points = [("VBM", np.array(e["vbm"]["k_frac"]), nv - 1), ("CBM", np.array(e["cbm"]["k_frac"]), nv)]
        else:
            if p["band"] >= K.D:
                raise ValueError(f"band: the model has {K.D} bands")
            points = [("point", np.array(p["k"]), p["band"])]
        res = Result()
        rows = []
        for name, kf, n in points:
            k = kf @ rec
            Hd, warn = _hessian(K, k, n, p["method"], p["step"])
            Hs = Hd[np.ix_(cart_axes, cart_axes)]
            inv_m = Hs / HBAR2_OVER_ME                             # m_e / m* tensor
            w, V = np.linalg.eigh(inv_m)
            with np.errstate(divide="ignore"):
                masses = np.where(np.abs(w) > 1e-9, 1.0 / w, np.inf)
            for j in range(len(w)):
                rows.append([name, n, str(np.round(kf, 5).tolist()), round(float(masses[j]), 5),
                             str(np.round(_full(V[:, j], cart_axes), 4).tolist())])
            if warn:
                res.warnings.append(f"{name} (band {n}): {warn}")
            res.data[name] = {"band": int(n), "k_frac": kf, "hessian_eV_A2": Hd, "inv_mass": inv_m, "masses": masses}
        res.table("Principal effective masses (m*/m_e; negative: hole-like curvature)",
                  ["at", "band", "k (fractional)", "m*/m_e", "direction (Cartesian)"], rows)
        res.summary = [(f"{r[0]} m*/m_e along {r[4]}", f"{r[3]}") for r in rows]
        return res


def _cart_axes(model, per):
    """Cartesian axes spanned by the periodic directions (2D layer in xy: [0, 1])."""
    if len(per) == 3:
        return [0, 1, 2]
    lat = np.asarray(model.lattice, float)[per]
    span = np.abs(lat).sum(axis=0)
    return [a for a in range(3) if span[a] > 1e-8][:len(per)] or [0]


def _full(v, axes):
    out = np.zeros(3)
    out[axes] = v
    return out


def _hessian(K, k, n, method, step):
    """∂²E_n/∂k_a∂k_b (eV Å²) at Cartesian k, and a warning text for degenerate bands."""
    E, U = K.eig(k[None, :])
    E, U = E[0], U[0]
    warn = ""
    deg = np.where(np.abs(E - E[n]) < 1e-6)[0]
    if len(deg) > 1:
        warn = (f"degenerate with band(s) {[int(x) for x in deg if x != n]}: the mass of a degenerate band depends on "
                "the direction (warped bands); the values are those of finite differences along the axes.")
        method = "finite_difference"
    if method == "finite_difference":
        h = step
        Hd = np.zeros((3, 3))
        e0 = E[n]
        for a in range(3):
            for b in range(a, 3):
                ea, eb = np.eye(3)[a] * h, np.eye(3)[b] * h
                if a == b:
                    ep, em = K.energies((k + ea)[None])[0][n], K.energies((k - ea)[None])[0][n]
                    Hd[a, a] = (ep - 2 * e0 + em) / h ** 2
                else:
                    f = lambda d: K.energies((k + d)[None])[0][n]  # noqa: E731
                    Hd[a, b] = Hd[b, a] = (f(ea + eb) - f(ea - eb) - f(-ea + eb) + f(-ea - eb)) / (4 * h * h)
        return Hd, warn
    u = U[:, n]
    dH = [K.dH(k[None], a)[0] for a in range(3)]
    V = [U.conj().T @ d @ U for d in dH]
    dE = E[n] - E
    w = np.zeros_like(dE)
    ok = np.abs(dE) > 1e-9
    w[ok] = 1.0 / dE[ok]
    Hd = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            second = float(np.real(u.conj() @ K.d2H(k[None], a, b)[0] @ u))
            pert = 2.0 * float(np.real(np.sum(V[a][n, :] * V[b][:, n] * w)))
            Hd[a, b] = Hd[b, a] = second + pert
    return Hd, warn


@register
class Compare(Calculator):
    id = "electronic.compare"
    title = "Model comparison"
    help = ("The bands of this model and of other models (SK fits, models written out as numbers) on one path, with the RMS difference "
            "of the bands nearest E_F (paired by their position from the gap).")
    formula = r"\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{n,\mathbf k}\left(E^{A}_n(\mathbf k) - E^{B}_{n'}(\mathbf k)\right)^2}"
    inputs = {"models": Files("Other models"), "labels": Files("Their names", advanced=True), "path": PATH,
              "dk": Number("Path spacing", 0.02, unit="1/Å", min=1e-4, max=1.0),
              "n_below": Number("Bands below E_F compared", 2, integer=True, min=0, max=50),
              "n_above": Number("Bands above E_F compared", 2, integer=True, min=0, max=50),
              "emin": Number("Window from", -6.0, unit="eV"), "emax": Number("to", 6.0, unit="eV")}

    def run(self, ctx, model, p):
        from skfitter.model import load_model
        from ..services.bloch import Kernel
        kp = kspace.path(model, path_points(p["path"]), p["dk"])
        others = [load_model(f) for f in p["models"]]
        names = list(p["labels"]) + [f"model {i + 2}" for i in range(len(p["labels"]), len(others))]
        res = Result()
        sets = [("this model", ctx.kernel.energies(kp["cart"]), model)]
        for name, m in zip(names, others):
            if not np.allclose(np.asarray(m.lattice), np.asarray(model.lattice), atol=1e-3):
                res.warnings.append(f"{name}: a different cell - drawn on the same fractional path.")
            sets.append((name, Kernel(m).energies(kp["frac"] @ kspace.reciprocal(m)), m))
        for name, E, _ in sets:
            keep = np.where((E.max(axis=0) >= p["emin"]) & (E.min(axis=0) <= p["emax"]))[0]
            res.bands(name, kp["x"], E[:, keep], [[x, lab] for x, lab in kp["ticks"]], band_index=keep.tolist(),
                      window=[p["emin"], p["emax"]], overlay=True)
        rows = []
        try:
            nv0 = occupied(model, 0, ctx.kernel)
        except ValueError:
            nv0 = None
        for name, E, m in sets[1:]:
            try:
                nv = occupied(m, 0)
            except ValueError:
                nv = None
            if nv0 is None or nv is None:
                rows.append([name, "", "one of the models is a metal: compare the plots"])
                continue
            pairs = [(nv0 - j, nv - j) for j in range(1, p["n_below"] + 1)] + \
                    [(nv0 + j, nv + j) for j in range(p["n_above"])]
            pairs = [(a, b) for a, b in pairs if 0 <= a < sets[0][1].shape[1] and 0 <= b < E.shape[1]]
            if not pairs:
                rows.append([name, "", "no common bands"])
                continue
            d = np.array([sets[0][1][:, a] - E[:, b] for a, b in pairs])
            rows.append([name, round(1000 * float(np.sqrt(np.mean(d ** 2))), 3), f"{len(pairs)} bands paired from the gap"])
        if rows:
            res.table("RMS difference to this model", ["model", "RMS (meV)", "bands"], rows)
            res.summary = [(f"RMS to {r[0]}", f"{r[1]} meV") for r in rows if r[1] != ""]
        return res
