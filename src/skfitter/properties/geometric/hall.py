"""Anomalous and spin Hall conductivity: Fermi-sea Kubo formula on a uniform mesh with adaptive refinement (the
conventions of SKFitter's Berry-phase routines)."""
from __future__ import annotations

import numpy as np

from .. import register
from ..core.calculator import Calculator, Cost, Requires, Result, seconds_per_k
from ..core.inputs import Axis, Bool, Choice, EnergyGrid, Mesh, Number, NumberList
from ..core.select import BANDS, band_list, occupied
from ..services import kspace
from ..services.units import E2_OVER_H, E2_OVER_HBAR, KB_EV, sheet_factor
from . import kubo

REFINE = {
    "refine": Bool("Adaptive refinement of hot spots", True, advanced=True),
    "refine_threshold": Number("Refine where max|Ω| h² exceeds", 1.0, min=0, advanced=True),
    "refine_levels": Number("Refinement levels", 3, integer=True, min=0, max=6, advanced=True),
    "degen_thresh": Number("Degeneracy threshold", 1e-4, unit="eV", min=0, advanced=True),
}


def _grid(p):
    e = p["energies"]
    return np.union1d(np.linspace(e["min"], e["max"], e["n"]), [0.0])


def _summary(E, sigma, units, name):
    i0 = int(np.argmin(np.abs(E)))
    imax = int(np.argmax(np.abs(sigma)))
    return [(f"{name}(E_F = 0)", f"{sigma[i0]:.4g} {units}"),
            (f"largest |{name}|", f"{sigma[imax]:.4g} {units} at E − E_F = {E[imax]:+.3f} eV")]


def _hall_cost(model, p, per_k):
    D = sum(model.norbs(i) for i in range(model.n_sites))
    nk = kspace.effective_mesh(model, p["mesh"])
    n = int(np.prod(nk))
    return Cost(n, D, 1.15 * n * seconds_per_k(D, per_k))


@register
class AnomalousHall(Calculator):
    id = "geometric.ahc"
    title = "Anomalous Hall conductivity"
    help = ("Intrinsic anomalous Hall conductivity against the Fermi energy: the Fermi-sea sum of the Berry curvature "
            "(needs broken time reversal: a magnetic model with spin-orbit coupling, or a Zeeman field). S/cm with the "
            "cell volume; for a 2D material also the sheet value in e²/h.")
    formula = r"\sigma_{\alpha\beta} = -\frac{e^2}{\hbar}\frac{1}{V N_k}\sum_{\mathbf k}\sum_n f_n\,\Omega_{n,\alpha\beta}"
    requires = Requires(dims=(2, 3), broken_trs=True)
    inputs = {
        "mesh": Mesh("k-mesh", (40, 40, 40)), "component": Axis("Component (z: σ_xy)", "z"),
        "energies": EnergyGrid("Fermi energies", (-2.0, 2.0, 200)),
        "smearing": Number("kT", 0.02, unit="eV", min=0), "bands": BANDS, **REFINE,
    }

    def cost(self, model, p):
        return _hall_cost(model, p, 5)

    def run(self, ctx, model, p):
        K = ctx.kernel
        a, b = kubo.PLANE[p["component"]]
        nk = kspace.effective_mesh(model, p["mesh"])
        E = _grid(p)
        sel = None if p["bands"] == "all" else band_list(p["bands"], K.D, lambda: occupied(model, 0, ctx.kernel))
        res = Result()
        rec = kspace.reciprocal(model)
        acc, n_k, info = kubo.fermi_sea_sum(K, rec, nk, a, b, E, p["smearing"], None, 256, p["degen_thresh"],
                                            p["refine"], p["refine_threshold"], p["refine_levels"], bands=sel,
                                            progress=lambda f: ctx.progress(0.95 * f), check=ctx.check)
        volume = abs(np.linalg.det(np.asarray(model.lattice, float)))
        sigma = -E2_OVER_HBAR * 1e8 / volume * acc / n_k
        name = {"x": "σ_yz", "y": "σ_zx", "z": "σ_xy"}[p["component"]]
        series = [(f"{name} (S/cm)", sigma)]
        h = sheet_factor(model)
        if np.isfinite(h):
            sheet = sigma * h * 1e-8 / E2_OVER_H
            series.append({"name": f"{name} sheet (e²/h)", "values": sheet, "axis": 2})
        res.curve(f"Anomalous Hall conductivity {name}", E, "E_F (eV)", series, f"{name} (S/cm)", marker_x=0.0,
                  **({"y2_label": f"{name} sheet (e²/h)"} if np.isfinite(h) else {}))
        res.summary = _summary(E, sigma, "S/cm", name)
        if np.isfinite(h):
            res.summary.append((f"{name} sheet (E_F = 0)", f"{sheet[int(np.argmin(np.abs(E)))]:.4g} e²/h "
                                                           f"(cell height {h:.3f} Å)"))
        kT = p["smearing"]
        res.summary.append(("k-mesh / smearing", f"{nk[0]}×{nk[1]}×{nk[2]}, kT = {kT} eV"
                            + (f"; refined {sum(info)} points" if info else "")))
        res.data = {"efermi": E, "sigma": sigma, "nk": nk, "refined": info}
        return res





@register
class SpinHall(Calculator):
    id = "geometric.shc"
    title = "Spin Hall conductivity"
    help = ("Intrinsic spin Hall conductivity σ^s_αβ (spin s, current α, field β) against the Fermi energy, or at one "
            "Fermi energy against temperature (several shifted meshes give error bars). (ħ/e) S/cm.")
    formula = r"\sigma^{s}_{\alpha\beta} = \frac{e^2}{\hbar}\frac{1}{2}\frac{1}{V N_k}\sum_{\mathbf k}\sum_n f_n\,\Omega^{s}_{n,\alpha\beta}"
    requires = Requires(dims=(2, 3), spinful=True)
    inputs = {
        "spin": Axis("Spin s", "z"), "alpha": Axis("Current α", "x"), "beta": Axis("Field β", "y"),
        "mesh": Mesh("k-mesh", (40, 40, 40)),
        "mode": Choice("Against", "efermi", [("efermi", "Fermi energy"), ("temperature", "temperature")]),
        "energies": EnergyGrid("Fermi energies", (-2.0, 2.0, 200)),
        "smearing": Number("kT", 0.02, unit="eV", min=0),
        "temperatures": NumberList("Temperatures", (5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300), unit="K", min=0),
        "efermi": Number("E_F (temperature scan)", 0.0, unit="eV"),
        "n_shifts": Number("Meshes (temperature scan)", 4, integer=True, min=1, max=32),
        "seed": Number("Random seed of the shifts", 0, integer=True, min=0, advanced=True),
        **REFINE,
    }

    def cost(self, model, p):
        c = _hall_cost(model, p, 5)
        if p["mode"] == "temperature":
            c.seconds *= p["n_shifts"]
        return c

    def run(self, ctx, model, p):
        K = ctx.kernel
        s, a, b = p["spin"], p["alpha"], p["beta"]
        sop = K.spin_operators()[kubo.AXIS[s]]
        nk = kspace.effective_mesh(model, p["mesh"])
        rec = kspace.reciprocal(model)
        volume = abs(np.linalg.det(np.asarray(model.lattice, float)))
        pref = 0.5 * E2_OVER_HBAR * 1e8 / volume
        name = f"σ^{s}_{a}{b}"
        res = Result()
        if p["mode"] == "efermi":
            E = _grid(p)
            acc, n_k, info = kubo.fermi_sea_sum(K, rec, nk, a, b, E, p["smearing"], sop, 256, p["degen_thresh"],
                                                p["refine"], p["refine_threshold"], p["refine_levels"],
                                                progress=lambda f: ctx.progress(0.95 * f), check=ctx.check)
            sigma = pref * acc / n_k
            res.curve(f"Spin Hall conductivity {name}", E, "E_F (eV)", [(f"{name} ((ħ/e) S/cm)", sigma)],
                      f"{name} ((ħ/e) S/cm)", marker_x=0.0)
            res.summary = _summary(E, sigma, "(ħ/e) S/cm", name)
            res.summary.append(("k-mesh / smearing", f"{nk[0]}×{nk[1]}×{nk[2]}, kT = {p['smearing']} eV"))
            res.data = {"efermi": E, "sigma": sigma, "nk": nk, "refined": info}
            return res
        T = np.unique(np.asarray(p["temperatures"], float))
        kT = KB_EV * T
        rng = np.random.RandomState(int(p["seed"]))
        active = np.array([n > 1 for n in nk], float)
        n_sh = int(p["n_shifts"])
        per_mesh = np.empty((n_sh, T.size))
        for i in range(n_sh):
            shift = np.zeros(3) if i == 0 else rng.uniform(0.0, 1.0, 3) * active
            acc, n_k, _ = kubo.fermi_sea_sum(K, rec, nk, a, b, np.full(T.size, p["efermi"]), kT, sop, 256,
                                             p["degen_thresh"], p["refine"], p["refine_threshold"], p["refine_levels"],
                                             shift=shift, progress=lambda f, i=i: ctx.progress((i + f) / n_sh),
                                             check=ctx.check)
            per_mesh[i] = pref * acc / n_k
        sigma = per_mesh.mean(axis=0)
        err = per_mesh.std(axis=0, ddof=1) / np.sqrt(n_sh) if n_sh > 1 else np.full(T.size, np.nan)
        res.curve(f"Spin Hall conductivity {name} against temperature at E_F = {p['efermi']:g} eV", T, "T (K)",
                  [{"name": f"{name} ((ħ/e) S/cm)", "values": sigma, "error": err}]
                  + [(f"mesh {i + 1}", per_mesh[i]) for i in range(n_sh)], f"{name} ((ħ/e) S/cm)", bands_from=1)
        fmt = lambda j: f"{sigma[j]:.4g} ± {err[j]:.2g} (ħ/e) S/cm" if np.isfinite(err[j]) else f"{sigma[j]:.4g}"  # noqa: E731
        res.summary = [(f"{name}(T = {T[0]:g} K)", fmt(0)), (f"{name}(T = {T[-1]:g} K)", fmt(-1)),
                       ("k-meshes", f"{n_sh} of {nk[0]}×{nk[1]}×{nk[2]}")]
        res.notes.append("A temperature dependence smaller than the error bars is not resolved at this k-mesh.")
        res.data = {"temperatures": T, "sigma": sigma, "sigma_err": err, "sigma_meshes": per_mesh}
        return res
