"""Continuum (k·p) wire models, discretized on a grid (phase SC3).

A band edge near k0 described by

    H(k) = E0 + s sum_i hbar^2 k_i^2 / (2 m_i)  +  sum_ij beta_ij k_i sigma_j  +  Delta sigma_x
                                                                                   (s = +1 conduction, -1 valence)

- an effective-mass tensor (diagonal, in the lab frame), and a k-linear spin-orbit tensor beta (meV nm): Rashba
(beta antisymmetric in the transverse plane), Dresselhaus, or Te's radial "hedgehog" term near H (beta diagonal) -
is put on a cubic grid of spacing a (finite differences: k_i^2 -> 2 (1 - cos k_i a) / a^2, k_i -> sin(k_i a) / a)
and becomes an ordinary tight-binding model (``TBModel``), so every calculation on a model runs on it
unchanged. A 60 nm wire is a few hundred grid points per slice instead of ~10^5 atoms.

Split pair (``splitting_meV`` = Delta > 0): the two bands of the pair are apart by 2 Delta at k0 - Te's top valence
bands at H (H4 above H5, 2 Delta ~ 0.1 eV), E = E0 + s sum hbar^2 k^2/2m -+ sqrt(Delta^2 + (beta_zz k_z)^2), which has
the camel back of H4 when beta^2 / 2 Delta exceeds hbar^2 / 2 m_z. sigma is then a pseudospin and k0 (H) is not
time-reversal invariant, so the model holds the two valleys (``valleys`` = 2): per grid point the valley-H pair and
its time-reversal partner at H' (H(-k)*), ordered as the spinful models are (the "spin-down" block is the partner),
so the singlet pairing of the superconducting layer pairs H with H'. The g-tensor of the calculations then acts on
that valley index: g_zz splits H from H' (Te's holes: g_zz large, g_xx = g_yy small).

Shapes
------
``wire``  the wire along z (the experiments' frame: Z along the wire), periodic along z with period a; the cross-
          section is a rectangle (width_nm = [Wx, Wy]) or a circle (diameter_nm), with hard walls: grid points at
          spacing a strictly inside the shape (a rectangle of width W holds round(W / a) - 1 points per direction).
``bulk``  one grid point per cell, periodic along x, y and z (for checks and fits).

The g-tensor is kept with the model (``g``) for the superconducting calculations to use; the model's
Hamiltonian has no field in it (B enters through the calculations).

The file format is ``prayog-continuum-model``: the parameters only; the grid model is rebuilt on reading.
"""
from __future__ import annotations

import copy

import numpy as np

from .model import TBModel

CONTINUUM_FORMAT = "prayog-continuum-model"
CONTINUUM_VERSION = 2                     # 2: splitting_meV and valleys (version-1 files read unchanged)
HBAR2_2M0 = 3.80998212                    # hbar^2 / 2 m_e in eV Å^2
HBAR2_2M0_MEV_NM2 = 38.0998212            # the same in meV nm^2
MAX_CROSS_SECTION = 20000
SIGMA = (np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.array([[1, 0], [0, -1]], complex))

DEFAULTS = {
    "name": "continuum wire",
    "band": "conduction",                 # or "valence" (a hole band: E decreases away from k0)
    "E0_meV": 0.0,                        # the band edge relative to E_F
    "masses": [0.1, 0.1, 0.1],            # m_x, m_y, m_z in units of the electron mass
    "soc_meV_nm": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],     # beta_ij: k_i sigma_j
    "splitting_meV": 0.0,                 # Delta: the pair split by 2 Delta at k0 (Delta sigma_x); needs valleys = 2
    "valleys": 1,                         # 1: sigma is the spin; 2: k0 and its time-reversal partner (a pseudospin)
    "g": [[2.0023193, 0, 0], [0, 2.0023193, 0], [0, 0, 2.0023193]],
    "grid_nm": 3.0,
    "shape": "wire",                      # or "bulk"
    "cross_section": "circle",            # wire: "circle" or "rectangle"
    "diameter_nm": 30.0,
    "width_nm": [30.0, 30.0],
    "fit": None,                          # where the parameters came from (fit_continuum), informational
}


def soc_preset(kind: str, strength_meV_nm: float, axis: str = "x") -> list:
    """beta for common k-linear terms (meV nm), with the wire along z:
    "rashba"  alpha (k_z sigma_axis') - a field across the wire (axis = the spin-orbit field direction, x or y)
    "radial"  beta (k . sigma) - Te near H (a hedgehog texture)
    "dresselhaus"  beta (k_x sigma_x - k_y sigma_y)."""
    b = np.zeros((3, 3))
    a = float(strength_meV_nm)
    if kind == "rashba":
        b[2, "xyz".index(axis)] = a
    elif kind == "radial":
        b = a * np.eye(3)
    elif kind == "dresselhaus":
        b[0, 0], b[1, 1] = a, -a
    elif kind != "none":
        raise ValueError(f"soc preset: rashba, radial, dresselhaus or none (not {kind!r})")
    return b.tolist()


def check_params(p: dict) -> dict:
    """The parameters with defaults filled in; ValueError listing what is wrong."""
    q = copy.deepcopy(DEFAULTS)
    for k, v in (p or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"continuum model: unknown parameter {k!r}")
        q[k] = copy.deepcopy(v)
    errs = []
    if q["band"] not in ("conduction", "valence"):
        errs.append("band: conduction or valence")
    try:
        m = np.asarray(q["masses"], float)
        if m.shape != (3,) or not np.all(m > 0):
            errs.append("masses: three positive numbers (units of the electron mass)")
        b = np.asarray(q["soc_meV_nm"], float)
        g = np.asarray(q["g"], float)
        if b.shape != (3, 3) or not np.all(np.isfinite(b)):
            errs.append("soc_meV_nm: a 3x3 matrix (meV nm)")
        if g.shape != (3, 3) or not np.all(np.isfinite(g)):
            errs.append("g: a 3x3 matrix")
        if not float(q["grid_nm"]) > 0:
            errs.append("grid_nm above 0")
        float(q["E0_meV"])
        if not float(q["splitting_meV"]) >= 0:
            errs.append("splitting_meV: 0 or above")
        if int(q["valleys"]) not in (1, 2):
            errs.append("valleys: 1 or 2")
        elif float(q["splitting_meV"]) > 0 and int(q["valleys"]) != 2:
            errs.append("splitting_meV above 0 breaks time reversal within one valley: set valleys = 2 (k0 and its "
                        "partner, e.g. H and H' of Te)")
    except (TypeError, ValueError):
        errs.append("masses, soc_meV_nm, g, grid_nm, E0_meV, splitting_meV, valleys: numbers")
    if q["shape"] not in ("wire", "bulk"):
        errs.append("shape: wire or bulk")
    if q["shape"] == "wire":
        if q["cross_section"] not in ("circle", "rectangle"):
            errs.append("cross_section: circle or rectangle")
        try:
            if q["cross_section"] == "circle" and not float(q["diameter_nm"]) > 0:
                errs.append("diameter_nm above 0")
            if q["cross_section"] == "rectangle" and not (len(q["width_nm"]) == 2 and min(map(float, q["width_nm"])) > 0):
                errs.append("width_nm: [Wx, Wy] above 0")
        except (TypeError, ValueError):
            errs.append("diameter_nm / width_nm: numbers")
    if errs:
        raise ValueError("continuum model: " + "; ".join(errs))
    q["masses"] = [float(x) for x in q["masses"]]
    q["splitting_meV"] = float(q["splitting_meV"])
    q["valleys"] = int(q["valleys"])
    q["soc_meV_nm"] = np.asarray(q["soc_meV_nm"], float).tolist()
    q["g"] = np.asarray(q["g"], float).tolist()
    if q["shape"] == "wire":
        n = len(cross_section_points(q))
        if n == 0:
            raise ValueError("continuum model: the cross-section holds no grid point - a finer grid or a wider wire")
        if n > MAX_CROSS_SECTION:
            raise ValueError(f"continuum model: {n} grid points per slice; at most {MAX_CROSS_SECTION} - a coarser grid")
    return q


def cross_section_points(p: dict) -> list:
    """(x, y) in nm of the grid points of a wire's cross-section (centred on the wire axis)."""
    a = float(p["grid_nm"])
    if p["cross_section"] == "rectangle":
        Wx, Wy = (float(w) for w in p["width_nm"])
        nx, ny = max(0, int(round(Wx / a)) - 1), max(0, int(round(Wy / a)) - 1)
        return [((i - (nx - 1) / 2) * a, (j - (ny - 1) / 2) * a) for j in range(ny) for i in range(nx)]
    R = float(p["diameter_nm"]) / 2
    n = int(np.ceil(R / a)) + 1
    pts = []
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x, y = i * a, j * a
            if x * x + y * y < R * R - 1e-12:
                pts.append((x, y))
    return pts


def _blocks(p: dict):
    """(onsite 2x2, {direction: hopping 2x2 to the neighbour at +a along x, y, z}) in eV."""
    s = 1.0 if p["band"] == "conduction" else -1.0
    a = 10.0 * float(p["grid_nm"])                                   # Å
    c = [s * HBAR2_2M0 / (m * a * a) for m in p["masses"]]           # eV
    beta = 1e-2 * np.asarray(p["soc_meV_nm"], float)                 # meV nm -> eV Å
    eye = np.eye(2, dtype=complex)
    onsite = (1e-3 * float(p["E0_meV"]) + 2 * sum(c)) * eye + 1e-3 * float(p["splitting_meV"]) * SIGMA[0]
    hop = {}
    for d in range(3):
        so = sum(beta[d, j] * SIGMA[j] for j in range(3))
        hop[d] = -c[d] * eye - 1j / (2 * a) * so
    if int(p["valleys"]) == 2:                                       # [valley k0 | its partner: h(-k)*]
        z = np.zeros((2, 2), complex)
        onsite = np.block([[onsite, z], [z, onsite.conj()]])
        hop = {d: np.block([[t, z], [z, t.conj()]]) for d, t in hop.items()}
    return onsite, hop


class ContinuumModel(TBModel):
    """A k·p band discretized on a grid (module docstring); a TBModel in every other respect."""

    def __init__(self, params: dict, source: dict | None = None):
        p = check_params(params)
        self.params = p
        a = 10.0 * float(p["grid_nm"])
        onsite, hop = _blocks(p)
        if p["shape"] == "bulk":
            lattice = np.eye(3) * a
            pos = [[0.0, 0.0, 0.0]]
            hoppings = {(0, 0, tuple(int(d == k) for k in range(3))): hop[d] for d in range(3)}
        else:
            pts = cross_section_points(p)
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            Lx = 10.0 * ((max(xs) - min(xs)) + 4 * float(p["grid_nm"]) + 10.0)       # Å, with room around
            Ly = 10.0 * ((max(ys) - min(ys)) + 4 * float(p["grid_nm"]) + 10.0)
            lattice = np.array([[0.0, 0.0, a], [Lx, 0.0, 0.0], [0.0, Ly, 0.0]])      # a1 along z (the wire)
            cart = np.array([[10.0 * x, 10.0 * y, 0.0] for x, y in pts])
            pos = (cart @ np.linalg.inv(lattice)).tolist()
            index = {(round(x / float(p["grid_nm"])), round(y / float(p["grid_nm"]))): n
                     for n, (x, y) in enumerate(((q[0] - min(xs), q[1] - min(ys)) for q in pts))}
            hoppings = {}
            for (ix, iy), n in index.items():
                hoppings[(n, n, (1, 0, 0))] = hop[2]                  # along the wire (z)
                if (ix + 1, iy) in index:
                    hoppings[(n, index[(ix + 1, iy)], (0, 0, 0))] = hop[0]
                if (ix, iy + 1) in index:
                    hoppings[(n, index[(ix, iy + 1)], (0, 0, 0))] = hop[1]
        n = len(pos)
        two = int(p["valleys"]) == 2
        src = {"continuum": {k: p[k] for k in p if k != "fit"}, **(source or {})}
        if p.get("fit"):
            src["continuum_fit"] = p["fit"]
        super().__init__(lattice=lattice, species=["W"] * n, frac_coords=pos,
                         site_orbitals=[["a", "b"] if two else ["s"]] * n,
                         onsite=[onsite] * n, hoppings=hoppings, spinful=True,
                         soc=bool(np.any(np.asarray(p["soc_meV_nm"])) or p["splitting_meV"] > 0), source=src,
                         kind="continuum", orbital_l=[[0, 0, 0, 0] if two else [0, 0]] * n)

    @property
    def g_tensor(self) -> np.ndarray:
        return np.asarray(self.params["g"], float)

    def to_dict(self) -> dict:
        return {"format": CONTINUUM_FORMAT, "version": CONTINUUM_VERSION, "params": self.params,
                "source": {k: v for k, v in self.source.items() if k not in ("continuum", "continuum_fit")}}

    @classmethod
    def from_dict(cls, d: dict) -> "ContinuumModel":
        if d.get("format") != CONTINUUM_FORMAT:
            raise ValueError(f"Not a continuum model file (format {d.get('format')!r}).")
        if int(d.get("version", 0)) > CONTINUUM_VERSION:
            raise ValueError("Continuum model file from a newer skfitter.")
        return cls(d["params"], source=d.get("source"))

    def to_tb_model(self) -> TBModel:
        return TBModel.from_dict(TBModel.to_dict(self))


def kp_energies(p: dict, k_cart_per_nm) -> np.ndarray:
    """The undiscretized k·p energies (meV) of the valley at k0, at k (1/nm, cartesian):
    E0 + s sum hbar^2 k^2/2m -+ |beta^T k + Delta x|."""
    p = check_params({**p, "shape": "bulk"})
    k = np.asarray(k_cart_per_nm, float)
    s = 1.0 if p["band"] == "conduction" else -1.0
    kin = s * sum(HBAR2_2M0_MEV_NM2 * k[i] ** 2 / p["masses"][i] for i in range(3))
    h = np.asarray(p["soc_meV_nm"], float).T @ k + np.array([p["splitting_meV"], 0.0, 0.0])   # the (pseudo)spin field
    split = float(np.linalg.norm(h))
    return np.array([p["E0_meV"] + kin - split, p["E0_meV"] + kin + split])


# == fitting a band edge of a tight-binding model ====================================================================
SPLIT_MIN_MEV = 0.5                       # a pair split by more than 2 x this at k0 is fitted as a split pair


def _reference(model):
    """(E_ref eV, n_valence or None) for "vbm" / "cbm": a scissor of the model fixes them (its n_valence or its
    mid-gap) - a DFT semimetal has bands on both sides of E_F = 0 near the edge - otherwise E_F = 0."""
    for step in reversed(list(getattr(model, "provenance", None) or [])):
        if isinstance(step, dict) and step.get("step") == "scissor":
            if step.get("n_valence"):
                return 0.0, int(step["n_valence"])
            if step.get("midgap") is not None:
                return float(step["midgap"]), None
    return 0.0, None


def fit_continuum(model, k0_frac, band="auto", dk_per_nm: float = 0.02, base: dict | None = None,
                  n_valence: int | None = None) -> dict:
    """Continuum parameters of a band edge of ``model`` (an SK fit, a Wannier model, any model of the protocol) at
    the fractional k-point ``k0_frac``: E0, the band (conduction / valence from the curvature), the masses along x, y,
    z from the second derivative of the average of the edge's spin pair, and a diagonal k-linear term from its spin
    splitting (|beta_ii| = d(E_+ - E_-)/dk / 2; the sign and the texture's orientation are not determined - a radial
    texture is assumed). ``band``: "auto" (the pair nearest E_F), "cbm" (lowest above E_F), "vbm" (highest below), or
    a band index (the pair band, band + 1 for a spinful model); after a scissor "vbm" / "cbm" follow its valence
    bands, and ``n_valence`` (the number of valence bands) sets them for any model - a DFT semimetal, fitted as it
    is: a scissor moves only the conduction bands, so the valence pair is the same with or without it. A pair split at k0 by more than 2 x 0.5 meV (Te's H4 / H5 at H) is fitted as a split pair: Delta = half the
    splitting, beta_zz from the curvature of the splitting along z ((E+ - E-)^2 / 4 = Delta^2 + beta^2 k^2), two
    valleys; the transverse splitting terms are reported in the fit record, not kept. Returns the parameters (merged
    into ``base``) with a ``fit`` record (residuals of the quadratic fit)."""
    from .bands import eigen, reciprocal
    G = reciprocal(model)
    k0 = np.asarray(k0_frac, float) @ G                                # 1/Å
    e0 = np.sort(np.real(eigen(model, k0)))
    pair = 2 if model.spinful else 1
    ref, nval = (0.0, int(n_valence)) if n_valence else _reference(model)
    e0 = e0 - ref
    if band == "auto":
        b = int(np.argmin(np.abs(e0)))
        if pair == 2:
            b = b - 1 if b % 2 == 1 else b
    elif band == "cbm" and nval:
        b = nval
    elif band == "vbm" and nval:
        b = nval - pair
    elif band == "cbm":
        above = np.where(e0 > 0)[0]
        if not len(above):
            raise ValueError("fit: no band above E_F at this k-point")
        b = int(above[0])
    elif band == "vbm":
        below = np.where(e0 <= 0)[0]
        if not len(below):
            raise ValueError("fit: no band below E_F at this k-point")
        b = int(below[-1]) - (pair - 1)
    else:
        b = int(band)
    b = max(0, min(len(e0) - pair, b))
    dk = float(dk_per_nm) / 10.0                                       # 1/Å
    steps = np.array([-2, -1, 0, 1, 2])
    masses, beta, resid = [], [], []
    E0 = None
    curv = []
    half = (e0[b + 1] - e0[b]) * 1e3 / 2 if pair == 2 else 0.0          # meV
    split = half > SPLIT_MIN_MEV
    beta_par = []
    for d in range(3):
        u = np.eye(3)[d]
        E = np.array([np.sort(np.real(eigen(model, k0 + s * dk * u)))[b:b + pair] for s in steps]) * 1e3   # meV
        E = E - 1e3 * ref
        avg = E.mean(axis=1)
        spl = (E[:, -1] - E[:, 0]) if pair == 2 else np.zeros(len(steps))
        kk = steps * dk * 10.0                                         # k in 1/nm
        coef = np.polyfit(kk, avg, 2)
        curv.append(2 * coef[0])                                       # d2E/dk2 (meV nm^2)
        resid.append(float(np.max(np.abs(np.polyval(coef, kk) - avg))))
        if split:                                                      # (S/2)^2 = c0 + c1 k + c2 k^2
            c2, c1, c0 = np.polyfit(kk, (spl / 2) ** 2, 2)
            beta.append(float(np.sqrt(max(c2 - c1 * c1 / (4 * max(c0, 1e-12)), 0.0))))
            beta_par.append(float(c1 / (2 * np.sqrt(max(c0, 1e-12)))))
        else:
            beta.append(0.5 * (0.5 * (spl[1] + spl[3]) - spl[2]) / (dk * 10.0) if pair == 2 else 0.0)
        E0 = float(avg[2])
    sign = 1.0 if np.mean(curv) > 0 else -1.0
    for c in curv:
        masses.append(float(2 * HBAR2_2M0_MEV_NM2 / max(abs(c), 1e-9)))
    params = dict(base or {})
    fit = {"k0_frac": [float(x) for x in k0_frac], "band_index": b, "pair": pair, "dk_per_nm": dk_per_nm,
           "max_residual_meV": round(max(resid), 6), "curvature_signs": [int(np.sign(c)) for c in curv],
           "reference_eV": ref}
    if split:
        soc = np.zeros((3, 3))
        soc[2, 2] = round(beta[2], 6)
        fit.update({"split_pair": True, "transverse_beta_meV_nm": [round(beta[0], 6), round(beta[1], 6)],
                    "parallel_beta_meV_nm": [round(x, 6) for x in beta_par],
                    "note": "split pair (Te H4 / H5): E0 the pair's centre, Delta half its splitting at k0; masses "
                            "from the average of the pair; beta_zz from the curvature of the splitting along z (sign "
                            "not determined); two valleys (k0 and its time-reversal partner)"})
        params.update({"splitting_meV": round(half, 6), "valleys": 2})
    else:
        soc = np.diag([round(abs(x), 6) for x in beta])
        fit["note"] = ("masses from the spin-averaged curvature; |beta_ii| from the spin splitting (radial texture "
                       "assumed; signs not determined)")
        params.update({"splitting_meV": 0.0, "valleys": 1})
    params.update({"band": "conduction" if sign > 0 else "valence", "E0_meV": round(E0, 6),
                   "masses": [round(m, 6) for m in masses], "soc_meV_nm": soc.tolist(), "fit": fit})
    return params
