"""Topology of 2D layers: Wilson loops (Berry phases) around K and K', Chern and valley Chern numbers
(Fukui-Hatsugai-Suzuki). Ported from SKFitter's step 4 (same constructions and numbers)."""
from __future__ import annotations

import numpy as np

from .. import register
from ..core.calculator import Calculator, Cost, Requires, Result, seconds_per_k
from ..core.inputs import Number
from ..core.select import BANDS, N_OCC, band_list, occupied
from ..services import kspace

KP = "K'"


def _pi(v) -> str:
    return "n/a (degenerate on the loop)" if not np.isfinite(v) else f"{v / np.pi:+.4f} π"


def _gap_at(kernel, k, nocc):
    e = kernel.energies(np.asarray(k, float)[None, :])[0]
    return float(e[nocc] - e[nocc - 1]) if 0 < nocc < len(e) else float("nan")


def _loop_phases(kernel, centre, radius, n_pts, idx, occ_cols):
    th = 2.0 * np.pi * np.arange(n_pts) / n_pts
    k = np.asarray(centre, float)[None, :] + radius * np.stack([np.cos(th), np.sin(th), 0.0 * th], axis=1)
    _, U = np.linalg.eigh(kernel.H(k))
    Ub = U[:, :, list(idx)]
    M = np.einsum("kai,kaj->kij", Ub.conj(), np.roll(Ub, -1, axis=0))
    d = np.einsum("kii->ki", M)
    mag = np.abs(d).min(axis=0)
    band = -np.angle(np.prod(d / np.maximum(np.abs(d), 1e-300), axis=0))
    band[mag < 0.5] = np.nan
    if occ_cols:
        det = np.linalg.det(M[:, occ_cols][:, :, occ_cols])
        occ = -float(np.angle(np.prod(det / np.maximum(np.abs(det), 1e-300))))
    else:
        occ = 0.0
    return band, occ, mag


def _k_label(model):
    """A point labelled K on the DFT path recorded in the model's source, if any (decides which corner is K)."""
    src = getattr(model, "source", {}) or {}
    for item in (src.get("kpath_points") or []):
        if item is not None and str(item[0]).strip().upper() == "K":
            return item[1]
    return None


@register
class WilsonLoops(Calculator):
    id = "geometric.wilson"
    title = "Wilson loops around K and K'"
    help = ("Berry phases of circles around the valleys of a hexagonal 2D zone against the radius: the flux of Ω_z "
            "through the disk (mod 2π), per band and for the occupied bands together (U(N) Wilson loop).")
    formula = r"\gamma(r) = -\mathrm{Im}\ln\prod_i\langle u(\mathbf k_i)|u(\mathbf k_{i+1})\rangle"
    requires = Requires(dims=(2,), hexagonal=True)
    inputs = {
        "k3": Number("Plane: reduced k3", 0.0),
        "radius": Number("Largest radius (0: 0.4 |K−M|)", 0.0, unit="1/Å", min=0),
        "n_loop": Number("Points per loop", 120, integer=True, min=8, max=5000),
        "n_radii": Number("Number of circles", 30, integer=True, min=1, max=500),
        "bands": BANDS, "n_occupied": N_OCC,
    }

    def cost(self, model, p):
        D = sum(model.norbs(i) for i in range(model.n_sites))
        n = 2 * p["n_loop"] * p["n_radii"]
        return Cost(n, D, n * seconds_per_k(D))

    def run(self, ctx, model, p):
        K = ctx.kernel
        vp = kspace.valleys(model, p["k3"], _k_label(model))
        nocc = occupied(model, p["n_occupied"], ctx.kernel)
        idx = band_list(p["bands"], K.D, lambda: nocc)
        occ_cols = [c for c, i in enumerate(idx) if i < nocc]
        notes = []
        r_max = p["radius"] if p["radius"] > 0 else 0.4 * vp["d_KM"]
        if r_max > 0.95 * vp["d_KM"]:
            notes.append(f"Loop radius clipped to {0.95 * vp['d_KM']:.3g} 1/Å (0.95 × |K−M|) so the disks do not overlap.")
            r_max = 0.95 * vp["d_KM"]
        nl, nr = p["n_loop"], p["n_radii"]
        radii = r_max * np.arange(1, nr + 1) / nr
        out = {}
        for v, name in enumerate(("K", "K'")):
            gb, go, mags = np.empty((nr, len(idx))), np.empty(nr), np.empty((nr, len(idx)))
            for j, r in enumerate(radii):
                ctx.check()
                gb[j], go[j], mags[j] = _loop_phases(K, vp[name], r, nl, idx, occ_cols)
                ctx.progress((v * nr + j + 1) / (2 * nr))
            go = np.unwrap(go)
            for c in range(len(idx)):
                if np.all(np.isfinite(gb[:, c])):
                    gb[:, c] = np.unwrap(gb[:, c])
            out[name] = {"band": gb, "occupied": go, "gap": _gap_at(K, vp[name], nocc)}
        gapless = [n for n in ("K", "K'") if np.isfinite(out[n]["gap"]) and out[n]["gap"] < 1e-3]
        res = Result()
        series = [("γ_occ(K)/π", out["K"]["occupied"] / np.pi), ("γ_occ(K')/π", out["K'"]["occupied"] / np.pi)]
        for name in ("K", "K'"):
            series += [(f"band {i} at {name}", out[name]["band"][:, c] / np.pi) for c, i in enumerate(idx)]
        res.curve("Berry phase of circles around K and K'", radii, "r (1/Å)", series, "γ / π", bands_from=2)
        res.summary = [("loops", f"{nr} circles up to r = {r_max:.4g} 1/Å, {nl} points each"),
                       ("γ(K), occupied bands (largest loop)", _pi(out["K"]["occupied"][-1])),
                       ("γ(K'), occupied bands (largest loop)", _pi(out["K'"]["occupied"][-1])),
                       ("gap at K / K'", f"{out['K']['gap']:.4g} eV / {out[KP]['gap']:.4g} eV")]
        for c, i in enumerate(idx):
            res.summary.append((f"γ(K) / γ(K'), band {i}", f"{_pi(out['K']['band'][-1, c])} / {_pi(out[KP]['band'][-1, c])}"))
        if gapless:
            notes.append("The gap closes at " + " and ".join(gapless) + ": the Berry phase is π on every loop while "
                         "Ω(k) vanishes elsewhere - the flux of a valley is a δ-function at the Dirac point (P·T "
                         "symmetry); break inversion (a sublattice mass) to give the valleys opposite signs.")
        else:
            notes.append("γ(r) is the Berry flux through the disk of radius r (mod 2π, unwrapped along r). Opposite "
                         "signs at K and K' with zero total is the valley Hall situation of a crystal without inversion.")
        res.notes = notes
        res.data = {"radii": radii, "valleys": out, "bands": idx, "occupied_bands": [idx[c] for c in occ_cols],
                    "gapless": gapless, "r_max": r_max}
        return res


@register
class ChernNumbers(Calculator):
    id = "geometric.chern"
    title = "Chern and valley Chern numbers"
    help = ("Chern numbers of each band and of the occupied bands on an nk × nk mesh of a 2D zone by the lattice "
            "method of Fukui, Hatsugai and Suzuki (integers even on a coarse mesh); for a hexagonal zone also the "
            "valley Chern numbers C_K, C_K' (plaquettes split by their nearest zone corner).")
    formula = r"C = \frac{1}{2\pi}\sum_{\square} F_\square,\quad F_\square = \arg\prod_{\partial\square} U_\mu"
    requires = Requires(dims=(2,), layer_xy=True)
    inputs = {
        "nk": Number("Mesh points per direction", 60, integer=True, min=6, max=2000),
        "k3": Number("Plane: reduced k3", 0.0),
        "bands": BANDS, "n_occupied": N_OCC,
    }

    def cost(self, model, p):
        D = sum(model.norbs(i) for i in range(model.n_sites))
        n = p["nk"] ** 2
        return Cost(n, D, n * seconds_per_k(D))

    def run(self, ctx, model, p):
        K = ctx.kernel
        try:
            vp = kspace.valleys(model, p["k3"], _k_label(model))
        except ValueError:
            vp = None
        nk = p["nk"]
        B = kspace.reciprocal(model)
        nocc = occupied(model, p["n_occupied"], ctx.kernel)
        idx = band_list(p["bands"], K.D, lambda: nocc)
        occ_cols = [c for c, i in enumerate(idx) if i < nocc]
        u = (np.arange(nk) + 0.5) / nk
        U1, U2 = np.meshgrid(u, u, indexing="ij")
        frac = np.stack([U1.ravel(), U2.ravel(), np.full(nk * nk, float(p["k3"]))], axis=1)
        ctx.progress(0.05, f"eigenstates on a {nk}×{nk} mesh")
        E, V = K.eig(frac @ B)
        D = E.shape[1]
        E = E.reshape(nk, nk, D)
        V = V.reshape(nk, nk, D, D)[..., idx]
        tau = K.centres
        nxt = []
        for ax, G in ((0, B[0]), (1, B[1])):
            Vn = np.roll(V, -1, axis=ax)
            ph = np.exp(-1j * (tau @ G))[:, None]
            if ax == 0:
                Vn[-1] = ph * V[0]
            else:
                Vn[:, -1] = ph * V[:, 0]
            nxt.append(Vn)
        M = [np.einsum("ijan,ijam->ijnm", V.conj(), Vn) for Vn in nxt]
        orient = 1.0 if np.cross(B[0], B[1])[2] > 0 else -1.0

        def flux(L1, L2):
            L1 = L1 / np.maximum(np.abs(L1), 1e-300)
            L2 = L2 / np.maximum(np.abs(L2), 1e-300)
            return -orient * np.angle(L1 * np.roll(L2, -1, axis=0) * np.conj(np.roll(L1, -1, axis=1)) * np.conj(L2))

        F_band = np.stack([flux(M[0][..., c, c], M[1][..., c, c]) for c in range(len(idx))], axis=-1)
        F_occ = flux(np.linalg.det(M[0][..., occ_cols, :][..., :, occ_cols]),
                     np.linalg.det(M[1][..., occ_cols, :][..., :, occ_cols])) if occ_cols else np.zeros((nk, nk))
        cfrac = np.stack([U1.ravel() + 0.5 / nk, U2.ravel() + 0.5 / nk, np.full(nk * nk, float(p["k3"]))], axis=1)
        cen = kspace.fold_to_bz(model, cfrac @ B)
        if vp is not None:
            dist = np.linalg.norm(cen[:, None, :2] - vp["corners"][None, :, :], axis=2)
            in_K = np.array([vp["corner_valley"][j] == "K" for j in np.argmin(dist, axis=1)]).reshape(nk, nk)
        else:
            in_K = None

        def numbers(F):
            out = {"C": float(F.sum() / (2 * np.pi))}
            if in_K is not None:
                out.update({"C_K": float(F[in_K].sum() / (2 * np.pi)), "C_Kp": float(F[~in_K].sum() / (2 * np.pi))})
            return out
        extra = np.array([vp["K"], vp["K'"]]) if vp is not None else np.zeros((0, 3))
        Ev = K.energies(extra) if len(extra) else np.zeros((0, D))
        Eall = np.concatenate([E.reshape(-1, D), Ev], axis=0)
        touching, min_gap, singular = [], {}, []
        for c, i in enumerate(idx):
            g = []
            if i > 0:
                g.append(np.min(Eall[:, i] - Eall[:, i - 1]))
            if i < D - 1:
                g.append(np.min(Eall[:, i + 1] - Eall[:, i]))
            min_gap[i] = float(min(g)) if g else float("nan")
            if g and min_gap[i] < 1e-3:
                touching.append(i)
            elif np.max(np.abs(F_band[..., c])) > 0.75 * np.pi:
                singular.append(i)
        occ_gap = float(np.min(Eall[:, nocc] - Eall[:, nocc - 1])) if 0 < nocc < D else float("nan")
        occ_touch = bool(occ_cols) and np.isfinite(occ_gap) and occ_gap < 1e-3
        per_band = {i: numbers(F_band[..., c]) for c, i in enumerate(idx)}
        o = numbers(F_occ)
        coarse = singular or (["occupied"] if not occ_touch and np.max(np.abs(F_occ)) > 0.75 * np.pi else [])
        res = Result()
        bad = " (not defined: the bands touch)"
        valley = in_K is not None
        res.summary = [("mesh", f"{nk}×{nk} (Fukui-Hatsugai-Suzuki plaquettes)"),
                       ("Chern number C, occupied bands", f"{o['C']:+.4f}" + (bad if occ_touch else ""))]
        if valley:
            res.summary += [("valley Chern C_K / C_K', occupied", f"{o['C_K']:+.4f} / {o['C_Kp']:+.4f}"),
                            ("valley contrast C_K − C_K', occupied", f"{o['C_K'] - o['C_Kp']:+.4f}"),
                            ("gap at K / K'", f"{_gap_at(K, vp['K'], nocc):.4g} eV / {_gap_at(K, vp[KP], nocc):.4g} eV")]
        res.summary.append(("smallest gap above the occupied bands", f"{occ_gap:.4g} eV"))
        heads = ["band (−1: occupied set)", "C"] + (["C_K", "C_K'", "C_K − C_K'"] if valley else []) + ["smallest gap (eV)"]
        rows = []
        for i, d in list(per_band.items()) + [(-1, o)]:
            row = [i, round(d["C"], 6)]
            if valley:
                row += [round(d["C_K"], 6), round(d["C_Kp"], 6), round(d["C_K"] - d["C_Kp"], 6)]
            row.append(round(min_gap.get(i, occ_gap), 6))
            rows.append(row)
            if i >= 0:
                res.summary.append((f"band {i}: C" + ("  (C_K, C_K')" if valley else ""), f"{d['C']:+.4f}"
                                    + (f"  ({d['C_K']:+.4f}, {d['C_Kp']:+.4f})" if valley else "")
                                    + (bad if i in touching else "")))
        res.table("Chern numbers", heads, rows)
        res.map("Berry flux per plaquette, occupied bands", cen[:, 0], cen[:, 1], F_occ.ravel(), "F (rad)",
                outline=kspace.bz_outline(model).tolist(), symmetric=True)
        if occ_touch or touching:
            res.notes.append("Some bands touch (Dirac points, or Kramers / spin-degenerate pairs): a Chern number needs "
                             "an isolated band (set), so those values depend on the mesh.")
        if coarse:
            res.notes.append("Some plaquettes carry a flux close to ±π: increase nk until the numbers stop changing.")
        if valley and not occ_touch and abs(o["C"]) < 0.05 and abs(o.get("C_K", 0)) > 0.01:
            res.notes.append("C = 0 with C_K = −C_K': a valley Hall insulator. C_K tends to ±½ only for a gap small "
                             "against the bandwidth (massive Dirac cone).")
        res.data = {"per_band": per_band, "occupied": o, "touching": touching, "occupied_touching": occ_touch,
                    "occupied_gap": occ_gap, "coarse": coarse, "bands": idx}
        return res
