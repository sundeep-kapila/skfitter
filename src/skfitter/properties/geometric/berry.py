"""Berry curvature (path and k-plane map) and spin Berry curvature (path)."""
from __future__ import annotations

import numpy as np

from .. import register
from ..core.calculator import Calculator, Cost, Requires, Result, seconds_per_k
from ..core.inputs import Axis, Bool, Choice, Number
from ..core.select import BANDS, DK, N_OCC, PATH, band_list, occupied, path_points
from ..services import kspace
from ..services.occupation import fermi
from ..services.units import BOHR_ANGSTROM
from . import kubo


def _path_cost(model, p, per_k):
    from ..services.bloch import Kernel
    D = sum(model.norbs(i) for i in range(model.n_sites))
    try:
        n = len(kspace.path(model, path_points(p["path"]), p["dk"])["x"])
    except Exception:  # noqa: BLE001
        n = 1000
    return Cost(n, D, n * seconds_per_k(D, per_k))


@register
class BerryCurvature(Calculator):
    id = "geometric.berry_curvature"
    title = "Berry curvature"
    help = ("Band-resolved Berry curvature Ω_n(k) along a k-path or as a map of the k-plane (2D-type cells), with the "
            "sums over the chosen bands and over the occupied ones. Degenerate groups share their mean (gauge "
            "invariant).")
    formula = r"\Omega^{\gamma}_n = -2\,\mathrm{Im}\sum_{m\neq n}\frac{\langle n|\partial_\alpha H|m\rangle\langle m|\partial_\beta H|n\rangle}{(E_n-E_m)^2}"
    requires = Requires(dims=(2, 3))
    inputs = {
        "mode": Choice("Where", "path", [("path", "along the k-path"), ("map", "map of the k-plane")]),
        "component": Axis("Component Ω_γ (γ = z: Ω_xy)", "z"),
        "bands": BANDS, "n_occupied": N_OCC, "path": PATH, "dk": DK,
        "nk": Number("Map points per direction", 40, integer=True, min=4, max=1000),
        "k3": Number("Map plane: reduced k3", 0.0),
        "average_degenerate": Bool("Average over degenerate bands", True, advanced=True),
        "degen_thresh": Number("Degeneracy threshold", 1e-4, unit="eV", min=0, advanced=True),
    }

    def cost(self, model, p):
        if p["mode"] == "map":
            D = sum(model.norbs(i) for i in range(model.n_sites))
            n = p["nk"] ** 2
            return Cost(n, D, n * seconds_per_k(D, 5))
        return _path_cost(model, p, 5)

    def run(self, ctx, model, p):
        K = ctx.kernel
        n_occ = lambda: occupied(model, p["n_occupied"], ctx.kernel)  # noqa: E731
        idx = band_list(p["bands"], K.D, n_occ)
        try:
            nocc = n_occ()
        except ValueError:
            nocc = None
        occ_cols = [c for c, i in enumerate(idx) if nocc is not None and i < nocc]
        res = Result()
        comp = p["component"]
        if p["mode"] == "path":
            kp = kspace.path(model, path_points(p["path"]), p["dk"])
            k = kp["cart"]
        else:
            kspace.check_xy_layer(model, "A Berry-curvature map")
            frac = kspace.plane(model, p["nk"], p["k3"])
            k = kspace.fold_to_bz(model, frac @ kspace.reciprocal(model))
        ctx.progress(0.05, f"Berry curvature at {len(k)} k-points")
        omega, E = kubo.band_curvature(K, k, idx, comp, p["degen_thresh"], p["average_degenerate"])
        total = omega.sum(axis=1)
        tocc = omega[:, occ_cols].sum(axis=1) if occ_cols else np.zeros(len(k))
        name = {"x": "Ω_x (Ω_yz)", "y": "Ω_y (Ω_zx)", "z": "Ω_z (Ω_xy)"}[comp]
        if p["mode"] == "path":
            res.curve(f"Berry curvature {name} along the path", kp["x"], "k (1/Å)",
                      [("selected bands", total), ("occupied bands", tocc)]
                      + [(f"band {i}", omega[:, c]) for c, i in enumerate(idx)], f"{name} (Å²)",
                      ticks=[[x, lab] for x, lab in kp["ticks"]], pieces=kp["pieces"], bands_from=2)
        else:
            outline = kspace.bz_outline(model)
            res.map(f"Berry curvature {name}, occupied bands", k[:, 0], k[:, 1], tocc, f"{name} (Å²)",
                    outline=outline.tolist(), symmetric=True)
            res.map(f"Berry curvature {name}, selected bands", k[:, 0], k[:, 1], total, f"{name} (Å²)",
                    outline=outline.tolist(), symmetric=True)
        res.summary = [("k-points", f"{len(k)}"), ("bands", _span(idx, K.D)),
                       ("occupied bands", ", ".join(str(idx[c]) for c in occ_cols) or "none"),
                       ("max |Ω| (selected, summed)", f"{np.max(np.abs(total)):.4g} Å²"),
                       ("max |Ω| (occupied, summed)", f"{np.max(np.abs(tocc)):.4g} Å²")]
        if nocc is None:
            res.warnings.append("A metal at E_F = 0: the 'occupied' sum needs n_occupied.")
        res.data = {"bands": idx, "occupied_bands": [idx[c] for c in occ_cols], "omega": omega, "total": total,
                    "total_occupied": tocc, "k_cart": k}
        return res


def _span(idx, D):
    if idx == list(range(idx[0], idx[-1] + 1)):
        return f"{idx[0]}-{idx[-1]} of {D}"
    return f"{idx} of {D}"


@register
class SpinBerryCurvature(Calculator):
    id = "geometric.spin_berry"
    title = "Spin Berry curvature"
    help = ("Spin Berry curvature Ω^s_αβ(k) along the k-path with J_α = ½{σ_s, ∂_α H}: per band and the Fermi-sea sum "
            "(the integrand of the spin Hall conductivity). Kramers-degenerate pairs are summed when the bands come "
            "in complete degenerate pairs.")
    formula = r"\Omega^{s}_{n,\alpha\beta} = -2\,\mathrm{Im}\sum_{m\neq n}\frac{\langle n|J^s_\alpha|m\rangle\langle m|\partial_\beta H|n\rangle}{(E_n-E_m)^2}"
    requires = Requires(dims=(2, 3), spinful=True)
    inputs = {
        "spin": Axis("Spin s", "z"), "alpha": Axis("Current α", "x"), "beta": Axis("Field β", "y"),
        "bands": BANDS, "path": PATH, "dk": DK,
        "efermi": Number("E_F", 0.0, unit="eV"), "smearing": Number("kT", 0.0, unit="eV", min=0),
        "units": Choice("Units", "au", [("au", "atomic units (spin current ħ/2)"), ("A2", "Å² (Pauli)")]),
        "pair_degenerate": Choice("Kramers pairs", "auto", [("auto", "sum when degenerate"), ("yes", "sum"),
                                                            ("no", "single states")], advanced=True),
        "degen_thresh": Number("Degeneracy threshold", 1e-4, unit="eV", min=0, advanced=True),
    }

    def cost(self, model, p):
        return _path_cost(model, p, 5)

    def run(self, ctx, model, p):
        K = ctx.kernel
        idx = band_list(p["bands"], K.D, lambda: occupied(model, 0, ctx.kernel))
        kp = kspace.path(model, path_points(p["path"]), p["dk"])
        ctx.progress(0.05, f"spin Berry curvature at {len(kp['x'])} k-points")
        om, E = kubo.spin_curvature(K, kp["cart"], p["spin"], p["alpha"], p["beta"], p["degen_thresh"])
        conv = 0.5 / BOHR_ANGSTROM ** 2 if p["units"] == "au" else 1.0
        f = fermi(E, p["efermi"], p["smearing"])
        total = conv * np.sum(f * om, axis=1)
        total_sel = conv * np.sum(f[:, idx] * om[:, idx], axis=1)
        om_sel, E_sel = conv * om[:, idx], E[:, idx]
        can_pair = len(idx) % 2 == 0 and idx[0] % 2 == 0 and idx == list(range(idx[0], idx[-1] + 1))
        gap = float(np.abs(E_sel[:, 0::2] - E_sel[:, 1::2]).max()) if can_pair else float("inf")
        mode = p["pair_degenerate"]
        if mode == "yes" and not can_pair:
            raise ValueError(f"Kramers pairs: the bands {idx} are not complete pairs (even count from an even index).")
        paired = (mode == "yes") or (mode == "auto" and can_pair and gap < p["degen_thresh"])
        if paired:
            cols = om_sel[:, 0::2] + om_sel[:, 1::2]
            labels = [f"bands {idx[2 * j]}-{idx[2 * j + 1]}" for j in range(len(idx) // 2)]
        else:
            cols, labels = om_sel, [f"band {i}" for i in idx]
        unit = "a.u." if p["units"] == "au" else "Å²"
        res = Result()
        res.curve(f"Spin Berry curvature Ω^{p['spin']}_{p['alpha']}{p['beta']} along the path", kp["x"], "k (1/Å)",
                  [("Σ f·Ω (all bands)", total), ("Σ f·Ω (selected bands)", total_sel)]
                  + [(lab, cols[:, c]) for c, lab in enumerate(labels)], f"Ω^s ({unit})",
                  ticks=[[x, lab] for x, lab in kp["ticks"]], pieces=kp["pieces"], bands_from=2)
        imax = int(np.argmax(np.abs(total)))
        res.summary = [("k-points", f"{len(total)}"), ("bands", _span(idx, K.D) + (" (Kramers pairs summed)" if paired else "")),
                       ("E_F / kT", f"{p['efermi']:g} eV / {p['smearing']:g} eV"),
                       ("max |Σ f·Ω| (all bands)", f"{total[imax]:.4g} {unit} at k = {kp['x'][imax]:.4f} 1/Å"),
                       ("max |Ω| of a selected band", f"{np.max(np.abs(om_sel)):.4g} {unit}")]
        if np.max(np.abs(om)) < 1e-8:
            res.notes.append("The spin Berry curvature vanishes within numerical noise (no spin-orbit coupling, or a "
                             "symmetry-forbidden component).")
        res.notes.append("Integrated over the zone, the Fermi-sea sum gives the spin Hall conductivity; peaks narrower "
                         "than the path spacing can be missed.")
        res.data = {"bands": idx, "omega": cols, "total": total, "total_selected": total_sel, "paired": paired,
                    "energies": E_sel}
        return res
