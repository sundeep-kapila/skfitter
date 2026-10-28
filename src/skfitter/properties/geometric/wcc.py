"""Hybrid Wannier charge centres (Wilson loops along a reciprocal lattice vector) and the invariants read from them:
the Chern number of a plane (winding of the summed centres) and the Z2 index of a time-reversal-invariant plane
(Soluyanov and Vanderbilt, Phys. Rev. B 83, 235401 (2011); Yu, Qi, Bernevig, Fang and Dai, Phys. Rev. B 84,
075119 (2011)).

For a plane of the zone that holds two reciprocal lattice vectors (loop direction b_l, scan direction b_s; the third
reduced coordinate fixed), the occupied states are followed along a closed loop k_s b_s + t b_l, t = 0 ... 1. The
eigenvalues λ_n of the U(N) Wilson loop give the hybrid Wannier centres x_n = −Im ln λ_n / 2π (fractions of the real
lattice vector a_l), as functions of k_s.

Gauge: the Bloch kernel uses phases exp(ik·(R + c_j − c_i)) (orbitals at their centres c), so H(k + G) = V† H(k) V
with V = diag(exp(iG·c)) and the loop closes with u(k + b_l) = V† u(k). Without this factor the centres would be
shifted by the orbital positions.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..core.calculator import Calculator, Cost, Requires, Result, seconds_per_k
from ..core.inputs import Bool, Choice, Number
from ..core.select import N_OCC, occupied
from ..services import kspace

AXES = {"k1": 0, "k2": 1, "k3": 2}


def _axis(v) -> int:
    if isinstance(v, (int, np.integer)):
        return int(v)
    key = str(v).strip().lower()
    if key in AXES:
        return AXES[key]
    if key in ("x", "y", "z"):
        return "xyz".index(key)
    raise ValueError(f"axis {v!r}: use k1, k2 or k3")


def wilson_centres(kernel, recip, fixed_axis, fixed_value, loop_axis, scan, n_loop, n_occ):
    """Hybrid Wannier centres (len(scan), n_occ) in [0, 1) for the occupied bands 0 ... n_occ-1, and the smallest
    direct gap above them on the loops. scan: reduced coordinates along the third axis."""
    fixed_axis, loop_axis = _axis(fixed_axis), _axis(loop_axis)
    scan_axis = ({0, 1, 2} - {fixed_axis, loop_axis}).pop()
    if fixed_axis == loop_axis:
        raise ValueError("the fixed axis and the loop axis must differ")
    recip = np.asarray(recip, float)
    V = np.exp(1j * (kernel.centres @ recip[loop_axis]))                # u(k + b_l) = V† u(k)
    t = np.arange(n_loop) / n_loop
    out, gap = np.empty((len(scan), n_occ)), np.inf
    for j, s in enumerate(scan):
        f = np.zeros((n_loop, 3))
        f[:, fixed_axis], f[:, scan_axis], f[:, loop_axis] = fixed_value, s, t
        E, U = np.linalg.eigh(kernel.H(f @ recip))
        if n_occ < E.shape[1]:
            gap = min(gap, float((E[:, n_occ] - E[:, n_occ - 1]).min()))
        Uo = U[:, :, :n_occ]
        W = np.eye(n_occ, dtype=complex)
        for i in range(n_loop):
            nxt = Uo[i + 1] if i + 1 < n_loop else V.conj()[:, None] * Uo[0]
            M = Uo[i].conj().T @ nxt
            a, _, b = np.linalg.svd(M)                                   # unitary part: stable for any n_loop
            W = W @ (a @ b)
        out[j] = np.sort(np.mod(-np.angle(np.linalg.eigvals(W)) / (2 * np.pi), 1.0))
    return out, gap


def chern_from_centres(wcc_full):
    """Winding of the summed centres over a full scan 0 ... 1 (last point = first + 1 reciprocal vector)."""
    tot = np.unwrap(2 * np.pi * np.asarray(wcc_full).sum(axis=1)) / (2 * np.pi)
    return float(tot[-1] - tot[0])


def z2_largest_gap(wcc_half):
    """Z2 of a time-reversal-invariant plane from the centres over half the scan (0 ... 1/2): the parity of the
    number of centres a line through the middle of the largest gap jumps over (Soluyanov-Vanderbilt). Returns
    (z2, crossings, smallest largest-gap)."""
    w = np.sort(np.asarray(wcc_half), axis=1)
    n = w.shape[1]

    def mid(row):
        ext = np.concatenate([row, [row[0] + 1.0]])
        g = np.diff(ext)
        i = int(np.argmax(g))
        return float(np.mod(ext[i] + 0.5 * g[i], 1.0)), float(g[i])

    z, g_min, crossings = 0, np.inf, 0
    for j in range(len(w) - 1):
        z0, g0 = mid(w[j])
        z1, _ = mid(w[j + 1])
        g_min = min(g_min, g0)
        # centres of the next row lying between the two gap lines (counted on the circle, shorter arc)
        d = np.mod(z1 - z0 + 0.5, 1.0) - 0.5
        lo, hi = (z0, z0 + d) if d >= 0 else (z0 + d, z0)
        x = np.mod(w[j + 1] - lo, 1.0)
        m = int(np.sum(x < (hi - lo)))
        crossings += m
        z = (z + m) % 2
    return int(z), int(crossings), float(g_min if n else np.nan)


def z2_reference_lines(wcc_half, lines=(0.137, 0.419, 0.781)):
    """Cross-check of the Z2 index: centres followed from one scan point to the next (matched by the shortest moves
    on the circle), and the parity of their crossings of fixed lines x = x0 over half the plane. With time-reversal
    symmetry every line gives the same parity; disagreement means the scan or loop mesh is too coarse."""
    from scipy.optimize import linear_sum_assignment
    w = np.asarray(wcc_half, float)
    out = []
    for x0 in lines:
        n = 0
        for j in range(len(w) - 1):
            a, b = w[j], w[j + 1]
            d = np.mod(b[None, :] - a[:, None] + 0.5, 1.0) - 0.5
            r, c = linear_sum_assignment(np.abs(d))
            for i, k in zip(r, c):
                step = d[i, k]
                if step == 0:
                    continue
                t = np.mod(x0 - a[i], 1.0) if step > 0 else np.mod(a[i] - x0, 1.0)
                if 0 < t <= abs(step):
                    n += 1
        out.append(n % 2)
    return out


@register
class HybridWannierCentres(Calculator):
    id = "geometric.wcc"
    title = "Hybrid Wannier charge centres, Chern and Z2"
    help = ("Wilson loops of the occupied bands along one reciprocal lattice vector, scanned across a plane of the "
            "zone: the hybrid Wannier centres, the Chern number of the plane (their winding) and, for a "
            "time-reversal-invariant plane, the Z2 index (largest-gap method).")
    formula = r"W(k_s)=\prod_i M^{(i,i+1)},\; x_n(k_s)=-\frac{1}{2\pi}\mathrm{Im}\ln\lambda_n"
    requires = Requires(dims=(2, 3))
    inputs = {
        "fixed": Choice("Fixed reduced axis", "k3", ("k1", "k2", "k3")),
        "fixed_value": Number("Value of the fixed coordinate", 0.0),
        "loop": Choice("Loop along", "k2", ("k1", "k2", "k3")),
        "n_loop": Number("Points per loop", 60, integer=True, min=8, max=4000),
        "n_scan": Number("Points across the plane", 41, integer=True, min=3, max=2001),
        "full": Bool("Scan the full plane (0 ... 1; needed for the Chern number)", True),
        "n_occupied": N_OCC,
    }

    def cost(self, model, p):
        D = sum(model.norbs(i) for i in range(model.n_sites))
        n = p["n_loop"] * p["n_scan"]
        return Cost(n, D, n * seconds_per_k(D))

    def run(self, ctx, model, p):
        K = ctx.kernel
        nocc = occupied(model, p["n_occupied"], K)
        rec = kspace.reciprocal(model)
        stop = 1.0 if p["full"] else 0.5
        scan = np.linspace(0.0, stop, int(p["n_scan"]))
        wcc, gap = wilson_centres(K, rec, p["fixed"], p["fixed_value"], p["loop"], scan, int(p["n_loop"]), nocc)
        res = Result()
        half = scan <= 0.5 + 1e-12
        z2, crossings, gmin = z2_largest_gap(wcc[half])
        series = [(f"x_{n}", wcc[:, n]) for n in range(nocc)]
        scan_name = ({"k1", "k2", "k3"} - {p["fixed"], p["loop"]}).pop()
        res.curve("Hybrid Wannier charge centres", scan, f"reduced {scan_name}",
                  series, "x (fraction of the lattice vector)")
        res.summary = [("plane", f"{p['fixed']} = {p['fixed_value']:g}, loops along {p['loop']}, "
                                  f"{len(scan)} × {p['n_loop']} k-points"),
                       ("occupied bands", str(nocc)),
                       ("smallest direct gap above them on the plane", f"{gap:.4g} eV")]
        notes = []
        if p["full"]:
            C = chern_from_centres(wcc)
            res.summary.append(("Chern number (winding of the summed centres)", f"{C:+.3f}"))
        lines = z2_reference_lines(wcc[half])
        agree = all(v == z2 for v in lines)
        res.summary.append(("Z2 (half plane)", f"{z2}" + ("" if agree else "  (not converged)")))
        res.summary.append(("Z2 checks: largest-gap line / fixed lines", f"{z2} / {', '.join(map(str, lines))}"))
        if not agree:
            notes.append("The Z2 estimates disagree: refine the scan (n_scan) and the loops (n_loop), or the bands "
                         "touch on this plane.")
        if gap < 1e-3:
            notes.append("The occupied bands touch the next band on this plane: the centres and invariants are not "
                         "defined there (a metal, or a gap smaller than the mesh resolves).")
        if _magnetic(model):
            notes.append("Z2 needs time-reversal symmetry (Kramers pairs at the time-reversal-invariant momenta). This "
                         "model is magnetic: the Z2 value is meaningful only for a plane protected by another "
                         "antiunitary symmetry; read the Chern number and the centres instead.")
        notes.append("Z2 counts how often a line through the largest gap between the centres is crossed across half "
                     "the plane (Soluyanov-Vanderbilt); a 3D insulator's strong index is (Z2(k3=0) + Z2(k3=1/2)) mod 2.")
        res.notes = notes
        res.data = {"scan": scan, "wcc": wcc, "n_occupied": nocc, "gap": gap, "z2": z2, "crossings": crossings,
                    "z2_lines": lines, "z2_converged": agree,
                    "smallest_largest_gap": gmin, "chern": chern_from_centres(wcc) if p["full"] else None}
        return res


def _magnetic(model) -> bool:
    for attr in ("is_magnetic",):
        if getattr(model, attr, False):
            return True
    return any(m.get("kind") in ("zeeman", "exchange") for m in (getattr(model, "modifiers", None) or []))
