#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skfitter.workflow: the steps of a fit as functions
=================================================

DFT loading and caching (VASP and Quantum ESPRESSO), project folders, orbital set-up and default bounds, the fit
run / save / reload jobs (with progress, metric, cancel and log callbacks), the property calculations (Berry
curvature, AHC, SHC, Wilson loops, Chern numbers), result scanning, and the matplotlib figures saved with the
results. The ``skfitter`` command (``skfitter.cli``) is built on these functions; graphical front ends can use the
same functions, so every front end gives the same numbers.

All numerics are done by the ``SKFitter`` class in ``skfitter.core`` (imported lazily by ``get_skfitter()``, so this
module imports without pymatgen until a fit is run).
"""
from __future__ import annotations

import contextlib
import datetime
import glob
import io
import json
import math
import os
import re
import shutil
import time
import warnings

import numpy as np

import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")  # figures are drawn on explicit Figure objects; keeps pyplot (used inside skfitter) off screen

import matplotlib.cm  # noqa: F401,E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
warnings.filterwarnings("ignore", message="POTCAR data with symbol")
warnings.filterwarnings("ignore", message=".*non-interactive.*")

HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================================
#  Look: one validated palette used by every plot (fixed slot order, never cycled)
# =====================================================================================
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("sk_seq", SEQ_BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list("sk_div", [CAT[0], "#f0efec", CAT[7]])
FAMILY_COLOR = {"s": CAT[0], "p": CAT[1], "d": CAT[2], "f": CAT[4]}
FAMILY_LABEL = {"s": "s", "p": "p", "d": "d", "f": "f"}

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
}


def styled():
    """rcParams context for the workflow's own plots (leaves the figures skfitter draws itself untouched)."""
    return matplotlib.rc_context(RC)


E2_OVER_HBAR = 2.434134e-4  # e^2/hbar in siemens
GAMMA = "Γ"


def fit_layout(fig):
    """Constrained layout is solved again at every draw, so titles, labels and legends follow the widget size
    (tight_layout only runs once, at the size the figure had when it was made)."""
    try:
        fig.set_layout_engine("constrained")
    except Exception:  # noqa: BLE001
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001
            pass


class Cancelled(BaseException):
    """Raised inside long computations when the user pressed Cancel.

    Derives from BaseException so that the ``except Exception`` guards inside skfitter (which turn numerical
    failures into a large penalty) cannot swallow it.
    """


# =====================================================================================
#  skfitter access (imported lazily so the window can explain a missing dependency)
# =====================================================================================
_SK = None


def get_skfitter():
    global _SK
    if _SK is None:
        import importlib
        _SK = importlib.import_module("skfitter.core")
    return _SK




def sk_tables():
    cls = get_skfitter().SKFitter
    return cls.SUB_ORBITALS, cls.FAMILY_MAP


# =====================================================================================
#  KPOINTS handling and k-path axis
# =====================================================================================
def _norm_label(s: str) -> str:
    s = s.strip().strip("!#").strip()
    t = s.lower().replace("$", "").replace("\\", "")
    if t in ("g", "gamma", "γ", "ɣ"):
        return GAMMA
    return s.replace("\\Gamma", GAMMA).replace("GAMMA", GAMMA)


def parse_kpoints_file(path: str) -> dict:
    """Parse a VASP line-mode KPOINTS file.

    Returns {'mode', 'npts', 'segments': [{'k0','k1','l0','l1'}]} (fractional reciprocal coordinates).
    Non line-mode files give an empty ``segments`` list.
    """
    out = {"mode": "unknown", "npts": 0, "segments": []}
    try:
        with open(path, "r", errors="ignore", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        return out
    if len(lines) < 6 or not lines[2].lower().startswith("l"):
        return out
    try:
        npts = int(lines[1].split()[0])
    except (ValueError, IndexError):
        return out
    if lines[3].lower().startswith(("c", "k")):  # Cartesian coordinates are not supported
        out.update(mode="cartesian", npts=npts)
        return out
    pts = []
    try:  # same reader as SKFitter, so the plots and the fit agree on the path
        n_seg, raw = get_skfitter().SKFitter._read_kpath_file(path)
        pts = [(list(c), _norm_label(lab)) for c, lab in raw]
        npts = int(n_seg)
        lines = lines[:4]
    except Exception:  # noqa: BLE001
        pts = []
    for ln in lines[4:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            xyz = [float(p) for p in parts[:3]]
        except ValueError:
            continue
        lab = " ".join(t for t in parts[3:] if t not in ("!", "#"))
        pts.append((xyz, _norm_label(lab)))
    segs = [{"k0": pts[i][0], "k1": pts[i + 1][0], "l0": pts[i][1], "l1": pts[i + 1][1],
             "i0": (i // 2) * npts, "i1": (i // 2 + 1) * npts - 1}            # VASP writes both ends of a segment
            for i in range(0, len(pts) - 1, 2)]
    out.update(mode="line", npts=npts, segments=segs)
    return out


def parse_qe_kpath(bands_input, kfrac, lattice, alat=None, log=print) -> dict:
    """k-path of a Quantum ESPRESSO bands run, in the same form as parse_kpoints_file (segments carry the DFT
    indices i0 / i1 of their end points). Uses the K_POINTS crystal_b / tpiba_b card of the pw.x input when it
    matches the k-points; otherwise the straight segments are found from the k-points (mode 'auto')."""
    sk = get_skfitter()
    kfrac = np.asarray(kfrac, float)
    if bands_input and os.path.isfile(bands_input):
        try:
            kp = sk.read_qe_kpath(bands_input)
            if not kp["band_path"]:
                raise ValueError(f"K_POINTS {kp['units']} is not a band path (use crystal_b or tpiba_b)")
            nodes = np.array(kp["nodes"], float)
            if kp["units"].startswith("tpiba"):
                if not alat:
                    raise ValueError("tpiba_b coordinates need alat")
                nodes = nodes @ np.asarray(lattice, float).T / alat
            segs = sk.qe_kpath_segments(nodes, kp["weights"], [_norm_label(x) for x in kp["labels"]], len(kfrac))
            for sg in segs:
                if not (np.allclose(kfrac[sg["i0"]], sg["k0"], atol=1e-4) and np.allclose(kfrac[sg["i1"]], sg["k1"], atol=1e-4)):
                    raise ValueError("its high-symmetry points do not coincide with the k-points of the XML file")
            return {"mode": "line", "npts": 0, "segments": segs}
        except (OSError, ValueError, IndexError) as exc:
            log(f"[skfitter] Could not use the k-path of {os.path.basename(bands_input)} ({exc}); "
                "finding the segments from the k-points.")
    segs = sk.detect_kpath_segments(kfrac, lattice)
    return {"mode": "auto" if segs else "unknown", "npts": 0, "segments": segs}


def _merge_labels(a: str, b: str) -> str:
    return a if a == b or not b else (b if not a else f"{a}|{b}")


def segment_tick_labels(segments) -> list:
    if not segments:
        return []
    labs = [segments[0]["l0"]]
    for i in range(1, len(segments)):
        labs.append(_merge_labels(segments[i - 1]["l1"], segments[i]["l0"]))
    labs.append(segments[-1]["l1"])
    return labs


def collapsed_nodes(segments):
    """Node coordinates/labels of the path with continuous joins merged."""
    nodes, labels = [], []
    for i, sg in enumerate(segments):
        if i == 0:
            nodes.append(list(sg["k0"])); labels.append(sg["l0"])
        elif not np.allclose(nodes[-1], sg["k0"]):
            nodes.append(list(sg["k0"])); labels.append(sg["l0"])
        else:
            labels[-1] = _merge_labels(labels[-1], sg["l0"])
        nodes.append(list(sg["k1"])); labels.append(sg["l1"])
    return nodes, labels


def compute_path_axis(kcart, segments=None, npts=None) -> dict:
    """Cumulative k-distance axis for a band path, collapsing the jumps between segments.

    Segments that carry the DFT indices of their end points (``i0`` / ``i1``: VASP line mode and Quantum ESPRESSO
    crystal_b / tpiba_b paths) place the ticks exactly; otherwise breaks are guessed from repeated points and jumps.
    """
    kcart = np.asarray(kcart, float)
    nk = len(kcart)
    step = np.r_[0.0, np.linalg.norm(np.diff(kcart, axis=0), axis=1)] if nk > 1 else np.zeros(nk)
    nseg = len(segments) if segments else 0
    if nseg and all("i0" in sg and "i1" in sg for sg in segments) and segments[-1]["i1"] == nk - 1:
        brk = [sg["i0"] for p, sg in zip(segments[:-1], segments[1:]) if sg["i0"] != p["i1"]]
        st = step.copy()
        if brk:
            st[brk] = 0.0
        x = np.cumsum(st)
        tick_idx = [segments[0]["i0"]] + [sg["i1"] for sg in segments]
        labels = segment_tick_labels(segments)
    else:
        if nseg and npts and nk == nseg * npts:
            brk = [i * npts for i in range(1, nseg)]
        else:
            pos = step[step > 1e-8]
            med = float(np.median(pos)) if len(pos) else 1.0
            brk = [i for i in range(1, nk) if step[i] < 1e-8 or step[i] > 6.0 * med]
        st = step.copy()
        if brk:
            st[brk] = 0.0
        x = np.cumsum(st)
        tick_idx = [0] + brk + [max(nk - 1, 0)]
        labels = segment_tick_labels(segments) if (nseg and len(brk) == nseg - 1) else None
    ticks = x[tick_idx]
    # merge ticks that fall on the same position (duplicated joint points)
    keep_t, keep_l = [], []
    for j, t in enumerate(ticks):
        if keep_t and abs(t - keep_t[-1]) < 1e-9:
            if labels is not None and labels[j] and labels[j] != keep_l[-1]:
                keep_l[-1] = _merge_labels(keep_l[-1], labels[j])
            continue
        keep_t.append(float(t)); keep_l.append(labels[j] if labels is not None else "")
    return {"x": x, "ticks": np.array(keep_t), "tick_labels": keep_l if labels is not None else None}


def build_path_kpoints(segments, npts, recip):
    """Cartesian k-points along the KPOINTS path (npts per segment) with matching x axis and ticks."""
    Ks, xs, ticks = [], [], [0.0]
    x0 = 0.0
    for i, sg in enumerate(segments):
        k0, k1 = np.asarray(sg["k0"], float), np.asarray(sg["k1"], float)
        t = np.linspace(0.0, 1.0, npts)[:, None]
        kc = (k0 + (k1 - k0) * t) @ recip
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))]
        x = x0 + d
        if i > 0 and np.allclose(segments[i - 1]["k1"], sg["k0"]):
            kc, x = kc[1:], x[1:]
        Ks.append(kc); xs.append(x)
        x0 = x0 + d[-1]
        ticks.append(x0)
    return np.vstack(Ks), np.concatenate(xs), np.array(ticks), segment_tick_labels(segments)


def reciprocal_matrix(lattice) -> np.ndarray:
    """Rows are the reciprocal lattice vectors b_i (with the 2*pi factor), 1/Angstrom."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, float)).T


# =====================================================================================
#  DFT data: load, cache, projections
# =====================================================================================
def orbital_names_for(norb: int):
    if norb == 3:
        return ["s", "p", "d"], {"s": [0], "p": [1], "d": [2]}
    if norb == 4:
        return ["s", "p", "d", "f"], {"s": [0], "p": [1], "d": [2], "f": [3]}
    if norb == 9:
        n = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]
        return n, {"s": [0], "p": [1, 2, 3], "d": [4, 5, 6, 7, 8]}
    if norb == 16:
        n = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2",
             "fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"]
        return n, {"s": [0], "p": [1, 2, 3], "d": [4, 5, 6, 7, 8], "f": list(range(9, 16))}
    n = [f"orb{i}" for i in range(norb)]
    return n, {"all": list(range(norb))}


def read_efermi_from_xml(path: str, tail_bytes: int = 8_000_000):
    """Last <i name="efermi"> value in vasprun.xml, found by scanning the end of the file (it sits in the final <dos> block)."""
    pat = re.compile(rb'<i\s+name="efermi"\s*>\s*([-+0-9.eEdD]+)\s*</i>')
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            for start in (max(0, size - tail_bytes), 0):
                fh.seek(start)
                hits = pat.findall(fh.read())
                if hits:
                    return float(hits[-1].decode().replace("D", "E").replace("d", "e"))
                if start == 0:
                    break
    except (OSError, ValueError):
        pass
    return None


def incar_site_moments(vrun, noncollinear: bool, n_sites: int):
    """The initial moment of every site from MAGMOM (z components of a non-collinear run), else None: the value the
    run used (vasprun <parameters>), else the INCAR's (as SKFitter._vasp_site_moments)."""
    for source in (getattr(vrun, "parameters", None), getattr(vrun, "incar", None)):
        try:
            m = [float(v) for v in np.ravel((source or {}).get("MAGMOM") or [])]
        except (TypeError, ValueError, AttributeError):
            continue
        if noncollinear and len(m) == 3 * n_sites:
            return m[2::3]
        if not noncollinear and len(m) == n_sites:
            return m
    return None


def magnetic_possible(meta: dict) -> bool:
    """Magnetic orbitals (exchange splitting) can be chosen: spin-polarised bands, or a non-collinear run with
    moments in its MAGMOM (one list of spinor bands: magnetism does not show as up / down bands)."""
    return bool(meta.get("magnetic") or (meta.get("soc") and any(abs(v) >= 0.5 for v in meta.get("moments") or [])))


def load_dft(vasprun_path: str, kpoints_path: str, log=print) -> dict:
    """Parse vasprun.xml (with projections when present). Mirrors the k-point selection of SKFitter."""
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.io.vasp import Vasprun
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    def _open(proj, dos=False):
        try:
            return Vasprun(vasprun_path, parse_dos=dos, parse_eigen=True,
                           parse_projected_eigen=proj, parse_potcar_file=False)
        except TypeError:  # very old pymatgen
            return Vasprun(vasprun_path, parse_dos=dos, parse_eigen=True, parse_projected_eigen=proj)

    log(f"[skfitter] Parsing {os.path.basename(vasprun_path)} ...")
    try:
        vrun = _open(True)
    except Exception as exc:  # noqa: BLE001
        log(f"[skfitter] Projected eigenvalues could not be parsed ({exc}); continuing without projections.")
        vrun = _open(False)

    # pymatgen only fills Vasprun.efermi while parsing the <dos> block (skfitter parses it, so it always has one)
    efermi = vrun.efermi
    if efermi is None:
        efermi = read_efermi_from_xml(vasprun_path)
    if efermi is None:
        log("[skfitter] E_F not found by the quick scan; re-reading vasprun.xml with the DOS block ...")
        try:
            vrun = _open(True, dos=True)
        except Exception:  # noqa: BLE001
            vrun = _open(False, dos=True)
        efermi = vrun.efermi
    if efermi is None:
        raise ValueError("vasprun.xml does not contain a Fermi energy (efermi) - the run seems incomplete.")
    efermi = float(efermi)
    structure = vrun.final_structure
    params = getattr(vrun, "parameters", {}) or {}
    soc = bool(params.get("LSORBIT", False) or params.get("LNONCOLLINEAR", False))
    nelect = params.get("NELECT")

    eig = vrun.eigenvalues
    up = np.asarray(eig[Spin.up][:, :, 0]) - efermi
    dn = np.asarray(eig[Spin.down][:, :, 0]) - efermi if Spin.down in eig else None
    magnetic = dn is not None and not np.allclose(up, dn, atol=1e-4)

    recip = structure.lattice.reciprocal_lattice
    kfrac = np.asarray(vrun.actual_kpoints, float)
    kcart = np.asarray(recip.get_cartesian_coords(kfrac), float)
    keep = np.ones(len(kcart), bool)
    weights = getattr(vrun, "kpoints_weights", None)  # same lookup as SKFitter._load_vasp_data
    if weights is not None and len(weights) == len(kcart):
        w = np.asarray(weights)
        if np.any(w == 0) and np.any(w > 0):
            keep &= (w == 0)
    idx = np.where(keep)[0]
    kc = kcart[idx]
    valid = [True]
    for i in range(1, len(kc)):
        dist = np.linalg.norm(kc[i] - kc[i - 1])
        valid.append(not (dist < 1e-5 and i > len(kc) - 5))
    sel = idx[np.asarray(valid)]
    kcart = kcart[sel]

    E = {"up": up[sel], "down": dn[sel] if (dn is not None and magnetic) else None}
    P = {}
    proj = getattr(vrun, "projected_eigenvalues", None)
    if proj:
        for spin, key in ((Spin.up, "up"), (Spin.down, "down")):
            if spin in proj and (key == "up" or magnetic):
                P[key] = np.asarray(proj[spin], dtype=np.float32)[sel]
    norb = P["up"].shape[-1] if P else 0
    orb_names, fam_cols = orbital_names_for(norb) if norb else ([], {})

    sites = [{"symbol": s.specie.symbol, "Z": int(s.specie.Z), "frac": [float(c) for c in s.frac_coords]}
             for s in structure]
    elements = []
    for s in sites:
        if s["symbol"] not in elements:
            elements.append(s["symbol"])
    try:
        spacegroup = SpacegroupAnalyzer(structure, symprec=1e-3).get_space_group_symbol()
    except Exception:  # noqa: BLE001
        spacegroup = "?"
    try:
        st = os.stat(vasprun_path)
        stamp = [st.st_size, st.st_mtime_ns]
    except OSError:
        stamp = [0, 0]

    meta = {
        "formula": structure.composition.reduced_formula, "spacegroup": spacegroup, "efermi": efermi,
        "nelect": None if nelect is None else float(nelect), "soc": soc, "magnetic": bool(magnetic),
        "sites": sites, "elements": elements, "lattice": np.asarray(structure.lattice.matrix).tolist(),
        "nk": int(len(kcart)), "nbands": int(up.shape[1]), "orbital_names": orb_names, "family_cols": fam_cols,
        "has_projection": bool(P), "source_stamp": stamp, "dft_code": "vasp",
        "moments": incar_site_moments(vrun, soc, len(sites)),
    }
    dft = {"meta": meta, "kcart": kcart, "E": E, "P": P}
    finalize_path(dft, kpoints_path)
    log(f"[skfitter] {meta['formula']} | {spacegroup} | {meta['nk']} k-points | {meta['nbands']} bands | "
        f"SOC={soc} | spin-polarised={magnetic} | projections={'yes' if P else 'no'}")
    return dft


# Quantum ESPRESSO real spherical harmonics (projwfc.x order of m) -> the VASP column order used for colouring
#   l=1: pz px py -> VASP s py pz px ;  l=2: dz2 dxz dyz dx2-y2 dxy -> VASP dxy dyz dz2 dxz dx2-y2 ;  l=3 likewise
QE_M_TO_VASP_COL = {0: {1: 0}, 1: {1: 2, 2: 3, 3: 1}, 2: {1: 6, 2: 7, 3: 5, 4: 8, 5: 4},
                    3: {1: 12, 2: 13, 3: 11, 4: 14, 5: 10, 6: 15, 7: 9}}


def qe_projection_array(pj: dict, nsites: int, nk: int, nbands: int, half=None):
    """(nk, nbands, nsites, norb) weights from read_qe_projwfc, in the column layout of VASP (s, py, pz, px, dxy, ...)
    or - for spin-orbit (j, mj) states - per l (s, p, d[, f]). half = 0 / 1 picks the spin-up / spin-down half of
    a file that lists both spins one after the other."""
    proj = pj["proj"]
    if half is not None and proj.shape[1] == 2 * nk:
        proj = proj[:, half * nk:(half + 1) * nk]
    if proj.shape[1] != nk or proj.shape[2] < nbands:
        raise ValueError(f"the projections are for {proj.shape[1]} k-points x {proj.shape[2]} bands, the bands file "
                         f"has {nk} x {nbands}")
    states = [st for st in pj["states"] if st[1] <= 3]
    lmax = max((st[1] for st in states), default=0)
    resolved = all(st[2] is not None for st in states)
    norb = (16 if lmax >= 3 else 9) if resolved else (4 if lmax >= 3 else 3)
    P = np.zeros((nk, nbands, nsites, norb), dtype=np.float32)
    for s, (na, l, m) in enumerate(pj["states"]):
        if l > 3 or not 0 <= na < nsites:
            continue
        col = QE_M_TO_VASP_COL[l].get(m, None) if resolved else l
        if col is None:
            continue
        P[:, :, na, col] += proj[s, :, :nbands]
    return P


def load_dft_espresso(xml_path: str, bands_input: str = None, scf_path: str = None, projwfc_path: str = None,
                      log=print) -> dict:
    """Parse a Quantum ESPRESSO bands run (pw.x XML file; optional pw.x bands input for the k-path labels, SCF
    output for E_F and projwfc.x filproj file for orbital projections) into the same dict as load_dft."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sk = get_skfitter()

    log(f"[skfitter] Parsing {os.path.basename(xml_path)} (Quantum ESPRESSO) ...")
    data = sk.read_qe_xml(xml_path)
    efermi = None
    if scf_path and os.path.isfile(scf_path):
        efermi = sk.read_qe_fermi(scf_path)
        if efermi is None:
            log(f"[skfitter] No Fermi energy found in {os.path.basename(scf_path)}; using the one of the bands XML file.")
        else:
            log(f"[skfitter] E_F = {efermi:.4f} eV from {os.path.basename(scf_path)}")
    if efermi is None:
        efermi = data["efermi"]
    if efermi is None:
        raise ValueError("The XML file holds no Fermi energy - add the SCF output (pw.x stdout or its XML file).")
    efermi = float(efermi)
    if data.get("calculation") and data["calculation"].lower() != "bands":
        log(f"[skfitter] Note: the XML file comes from a calculation='{data['calculation']}' run (expected 'bands').")

    structure = Structure(Lattice(data["lattice"]), data["symbols"], data["cart_coords"], coords_are_cartesian=True)
    up = np.asarray(data["eig_up"], float) - efermi
    dn = None if data["eig_down"] is None else np.asarray(data["eig_down"], float) - efermi
    magnetic = dn is not None and (dn.shape != up.shape or not np.allclose(up, dn, atol=1e-4))
    kcart = np.asarray(structure.lattice.reciprocal_lattice.get_cartesian_coords(data["kpoints_frac"]), float)
    E = {"up": up, "down": dn if magnetic else None}

    sites = [{"symbol": s.specie.symbol, "Z": int(s.specie.Z), "frac": [float(c) for c in s.frac_coords]}
             for s in structure]
    P = {}
    if projwfc_path and os.path.isfile(projwfc_path):
        try:
            nk, nb = up.shape
            pj_up = sk.read_qe_projwfc(projwfc_path)
            P["up"] = qe_projection_array(pj_up, len(sites), nk, nb, half=0)
            if magnetic:
                down_path = re.sub(r"up$", "down", projwfc_path)
                if down_path != projwfc_path and os.path.isfile(down_path):
                    P["down"] = qe_projection_array(sk.read_qe_projwfc(down_path), len(sites), nk, dn.shape[1], half=1)
                elif pj_up["proj"].shape[1] == 2 * nk:
                    P["down"] = qe_projection_array(pj_up, len(sites), nk, dn.shape[1], half=1)
                else:
                    log("[skfitter] No spin-down projections found (…projwfc_down); colouring the spin-down bands is off.")
        except (OSError, ValueError, IndexError) as exc:
            log(f"[skfitter] Orbital projections could not be read from {os.path.basename(projwfc_path)} ({exc}).")
            P = {}
    norb = P["up"].shape[-1] if P else 0
    orb_names, fam_cols = orbital_names_for(norb) if norb else ([], {})
    shells = {}                                    # the projected shells of each element ('4S', '3D', ...)
    if P:
        for atom, _, lab in pj_up.get("labels") or []:
            if 0 <= atom < len(sites):
                shells.setdefault(sites[atom]["symbol"], set()).add(lab)

    elements = []
    for s in sites:
        if s["symbol"] not in elements:
            elements.append(s["symbol"])
    try:
        spacegroup = SpacegroupAnalyzer(structure, symprec=1e-3).get_space_group_symbol()
    except Exception:  # noqa: BLE001
        spacegroup = "?"
    meta = {
        "formula": structure.composition.reduced_formula, "spacegroup": spacegroup, "efermi": efermi,
        "nelect": None if data["nelec"] is None else float(data["nelec"]),
        "soc": bool(data["spinorbit"] or data["noncolin"]), "magnetic": bool(magnetic),
        "sites": sites, "elements": elements, "lattice": np.asarray(structure.lattice.matrix).tolist(),
        "nk": int(len(kcart)), "nbands": int(up.shape[1]), "orbital_names": orb_names, "family_cols": fam_cols,
        "has_projection": bool(P), "source_stamp": files_stamp([xml_path, scf_path, projwfc_path]),
        "dft_code": "espresso", "alat": float(data["alat"]),
        "projected_shells": {el: sorted(v) for el, v in shells.items()},
        "moments": sk.read_qe_site_moments(scf_path, len(sites)) if scf_path and os.path.isfile(scf_path) else None,
    }
    dft = {"meta": meta, "kcart": kcart, "E": E, "P": P}
    finalize_path(dft, bands_input)
    log(f"[skfitter] {meta['formula']} | {spacegroup} | {meta['nk']} k-points | {meta['nbands']} bands | "
        f"SOC={meta['soc']} | spin-polarised={magnetic} | projections={'yes' if P else 'no'}")
    return dft


def files_stamp(paths) -> list:
    """[size, mtime] of every existing file (None entries are skipped); identifies the inputs of a DFT cache."""
    out = []
    for p in paths:
        if not p:
            continue
        try:
            st = os.stat(p)
            out += [st.st_size, st.st_mtime_ns]
        except OSError:
            out += [0, 0]
    return out


def finalize_path(dft: dict, kpath_file: str) -> None:
    """k-path segments and the band-plot axis: kpath_file is the KPOINTS file (VASP) or the pw.x bands input (QE)."""
    meta = dft["meta"]
    if meta.get("dft_code", "vasp") == "espresso":
        lat = np.asarray(meta["lattice"], float)
        kfrac = np.asarray(dft["kcart"], float) @ lat.T / (2.0 * np.pi)
        kp = parse_qe_kpath(kpath_file, kfrac, lat, meta.get("alat"))
    else:
        kp = parse_kpoints_file(kpath_file)
    dft["kpath"] = kp
    ax = compute_path_axis(dft["kcart"], kp["segments"], kp["npts"])
    dft["x"], dft["ticks"], dft["tick_labels"] = ax["x"], ax["ticks"], ax["tick_labels"]


def save_dft_cache(path: str, dft: dict) -> None:
    arrays = {"kcart": dft["kcart"], "E_up": dft["E"]["up"]}
    if dft["E"]["down"] is not None:
        arrays["E_down"] = dft["E"]["down"]
    for k, v in dft["P"].items():
        arrays[f"P_{k}"] = v
    np.savez_compressed(path, meta=json.dumps(dft["meta"]), **arrays)


def load_dft_cache(path: str, stamp_paths, kpath_file: str, dft_code: str = "vasp"):
    """The cached DFT data if it was made from the same input files (stamp_paths: the files that went into it)."""
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("dft_code", "vasp") != dft_code:
            return None
        if isinstance(stamp_paths, str):
            stamp_paths = [stamp_paths]
        if meta.get("source_stamp") != files_stamp(stamp_paths):
            return None
        dft = {"meta": meta, "kcart": z["kcart"],
               "E": {"up": z["E_up"], "down": z["E_down"] if "E_down" in z.files else None},
               "P": {k: z[f"P_{k}"] for k in ("up", "down") if f"P_{k}" in z.files}}
        finalize_path(dft, kpath_file)
        return dft
    except Exception:  # noqa: BLE001
        return None


# =====================================================================================
#  DFT sources: which files a project needs, per DFT code
# =====================================================================================
DFT_CODES = (("VASP", "vasp"), ("Quantum ESPRESSO", "espresso"))
DFT_CODE_NAMES = {"vasp": "VASP", "espresso": "Quantum ESPRESSO"}

# role -> name of the copy kept in the project folder
PROJECT_FILES = {
    "vasp": {"kpoints": "KPOINTS", "vasprun": "vasprun.xml"},
    "espresso": {"qe_xml": "qe_bands.xml", "qe_bands_in": "qe_bands.in", "qe_scf": "qe_scf.out",
                 "qe_projwfc": "qe_projwfc_up"},
}
REQUIRED_FILES = {"vasp": ("kpoints", "vasprun"), "espresso": ("qe_xml",)}
FILE_LABELS = {"kpoints": "KPOINTS", "vasprun": "vasprun.xml", "qe_xml": "pw.x XML (bands run)",
               "qe_bands_in": "Bands input (pw.x)", "qe_scf": "SCF output (E_F)", "qe_projwfc": "projwfc.x filproj"}


def project_dft_code(project_dir: str) -> str:
    """DFT code of a project folder: from skfitter_project.json, else from the files present (VASP if unsure)."""
    code = read_project_file(project_dir).get("dft_code")
    if code in PROJECT_FILES:
        return code
    if os.path.isfile(os.path.join(project_dir, PROJECT_FILES["espresso"]["qe_xml"])) and \
            not os.path.isfile(os.path.join(project_dir, "vasprun.xml")):
        return "espresso"
    return "vasp"


def project_sources(project_dir: str, code: str = None) -> dict:
    """{'code', role: path of the project copy or None} for the DFT files of a project folder."""
    code = code or project_dft_code(project_dir)
    out = {"code": code}
    for role, name in PROJECT_FILES[code].items():
        p = os.path.join(project_dir, name)
        out[role] = p if os.path.isfile(p) else None
    return out


def project_has_dft(project_dir: str, code: str = None) -> bool:
    src = project_sources(project_dir, code)
    return all(src.get(r) for r in REQUIRED_FILES[src["code"]])


def fitter_source_kwargs(src: dict) -> dict:
    """SKFitter keyword arguments that point it at the DFT files in `src` (see project_sources)."""
    if src["code"] == "espresso":
        return {"dft_code": "espresso", "qe_xml_path": src.get("qe_xml"), "qe_bands_input": src.get("qe_bands_in"),
                "qe_scf_path": src.get("qe_scf")}
    return {"dft_code": "vasp", "vasprun_path": src.get("vasprun"), "kpoints_path": src.get("kpoints")}


def projection_weights(dft: dict, spin: str, atoms: str, mode: str, orbital: str = "s"):
    """Weights used for band colouring.

    mode 'family'  -> (nk, nb, n_families), names
    mode 'element' -> (nk, nb, n_elements), names
    mode 'orbital' -> (nk, nb, 1), [orbital]
    ``atoms`` is 'All' or an element symbol.
    """
    meta = dft["meta"]
    P = dft["P"][spin]
    sym = np.array([s["symbol"] for s in meta["sites"]])
    ions = np.arange(len(sym)) if atoms in ("All", "") else np.where(sym == atoms)[0]
    if len(ions) == 0:
        ions = np.arange(len(sym))
    Pi = P[:, :, ions, :]
    if mode == "element":
        names = [e for e in meta["elements"] if e in sym[ions]]
        W = np.stack([P[:, :, np.where(sym == e)[0], :].sum(axis=(2, 3)) for e in names], axis=-1)
        return W, names
    fam = meta["family_cols"]
    if mode == "orbital":
        names_o = meta["orbital_names"]
        if orbital in fam:
            cols = fam[orbital]
        else:
            cols = [names_o.index(orbital)] if orbital in names_o else [0]
        return Pi[:, :, :, cols].sum(axis=(2, 3))[..., None], [orbital]
    names = list(fam.keys())
    W = np.stack([Pi[:, :, :, fam[f]].sum(axis=(2, 3)) for f in names], axis=-1)
    return W, names


def blend_colors(W, colors, dominant: bool):
    """Per-(k, band) RGB from weights. dominant=True: colour of the largest share, pale when the mix is impure."""
    W = np.clip(np.asarray(W, float), 0, None)
    tot = W.sum(-1)
    C = np.array([to_rgb(c) for c in colors])
    share = W / np.maximum(tot, 1e-12)[..., None]
    if dominant:
        top = share.argmax(-1)
        purity = np.take_along_axis(share, top[..., None], -1)[..., 0]
        base = C[top]
        fade = np.clip((purity - 1.0 / max(len(colors), 1)) / (1.0 - 1.0 / max(len(colors), 1) + 1e-9), 0, 1)
        fade = (0.35 + 0.65 * fade)[..., None]
        rgb = 1.0 - (1.0 - base) * fade
    else:
        rgb = share @ C
    grey = np.array(to_rgb(AXIS))
    rgb = np.where((tot < 1e-3)[..., None], grey, rgb)
    return rgb, tot


def band_segments(x, E, rgb, ymin, ymax):
    """LineCollection segments/colours for all bands intersecting the window; zero-length steps are dropped."""
    segs, cols = [], []
    dx = np.diff(x)
    ok = dx > 1e-12
    for b in range(E.shape[1]):
        e = E[:, b]
        if e.max() < ymin or e.min() > ymax:
            continue
        pts = np.stack([x, e], axis=-1)
        s = np.stack([pts[:-1], pts[1:]], axis=1)[ok]
        c = (0.5 * (rgb[:-1, b] + rgb[1:, b]))[ok] if rgb is not None else None
        segs.append(s)
        if c is not None:
            cols.append(c)
    if not segs:
        return np.zeros((0, 2, 2)), np.zeros((0, 3))
    return np.concatenate(segs), (np.concatenate(cols) if cols else np.zeros((0, 3)))


# =====================================================================================
#  Orbital / configuration helpers
# =====================================================================================
ORBITAL_LAYOUT = [
    ("s", ["s", "s*"]),
    ("p", ["p", "pxy", "px", "py", "pz"]),
    ("d sets", ["d", "d_t2g", "d_eg", "d_e1", "d_e2"]),
    ("d single", ["dz2", "dxy", "dyz", "dzx", "dx2y2"]),
    ("f sets", ["f", "f_m0", "f_m1", "f_m2", "f_m3"]),
    ("f cubic", ["f_a2u", "f_t1u", "f_t2u"]),
    ("f single", ["fz3", "fxz2", "fyz2", "fzx2", "fxyz", "fx3", "fy3x2"]),
]
ORBITAL_TIPS = {
    "s": "s orbital", "s*": "excited s* orbital (sp3s* models)", "p": "px, py, pz",
    "pxy": "px, py (in-plane pair; combine with pz for 2D materials)", "px": "px", "py": "py", "pz": "pz",
    "d": "all five d orbitals", "d_t2g": "dxy, dyz, dzx", "d_eg": "dx2-y2, dz2",
    "d_e1": "dyz, dzx", "d_e2": "dxy, dx2-y2 (e.g. MoS2: dz2, d_e2, d_e1)",
    "dz2": "dz2", "dxy": "dxy", "dyz": "dyz", "dzx": "dzx", "dx2y2": "dx2-y2",
    "f": "all seven f orbitals", "f_m0": "fz3 (|m| = 0)", "f_m1": "fxz2, fyz2 (|m| = 1)",
    "f_m2": "fxyz, fz(x2-y2) (|m| = 2)", "f_m3": "fx(x2-3y2), fy(3x2-y2) (|m| = 3)",
    "f_a2u": "xyz (cubic a2u)", "f_t1u": "x(5x2-3r2), y(5y2-3r2), z(5z2-3r2) (cubic t1u)",
    "f_t2u": "x(y2-z2), y(z2-x2), z(x2-y2) (cubic t2u; e.g. Ce, CeO2: f_a2u, f_t1u, f_t2u)",
    "fz3": "z(5z2-3r2)", "fxz2": "x(5z2-r2)", "fyz2": "y(5z2-r2)", "fzx2": "z(x2-y2)", "fxyz": "xyz",
    "fx3": "x(x2-3y2)", "fy3x2": "y(3x2-y2)",
}
PRESETS = {
    "(preset)": None, "s": ["s"], "p": ["p"], "d": ["d"], "s + p": ["s", "p"], "s + d": ["s", "d"],
    "s + p + d": ["s", "p", "d"], "sp3s*": ["s", "p", "s*"], "pz only": ["pz"], "pxy + pz": ["pxy", "pz"],
    "MoS2-type d": ["dz2", "d_e2", "d_e1"],
    "f": ["f"], "s + d + f": ["s", "d", "f"], "p + f": ["p", "f"],
}


def orbitals_conflict(a: str, b: str) -> bool:
    SUB, FAM = sk_tables()
    if a == b:
        return True
    if FAM[a] != FAM[b]:
        return False
    from skfitter.core import groups_overlap          # the cubic f groups are combinations of the tesseral f orbitals
    return groups_overlap(a, b)


def validate_orbital_config(orb_cfg: dict) -> str:
    """'' if the model can be built, otherwise a message: at least one orbital over all elements, and no overlapping
    choices. An element without orbitals is allowed (its atoms are left out of the model) - see
    ``orbital_warning``."""
    if not any(orb_cfg.values()):
        return "Select at least one orbital (for any element)."
    for elem, subs in orb_cfg.items():
        for i, a in enumerate(subs):
            for b in subs[i + 1:]:
                if orbitals_conflict(a, b):
                    return f"For {elem}: '{a}' and '{b}' overlap - choose one of them."
    return ""


def orbital_warning(orb_cfg: dict) -> str:
    """A note (not an error) when some elements have no orbitals: their atoms are not in the tight-binding model."""
    empty = [e for e, subs in orb_cfg.items() if not subs]
    if not empty or not any(orb_cfg.values()):
        return ""
    names = ", ".join(empty)
    return (f"No orbital is chosen for {names}: {'its' if len(empty) == 1 else 'their'} atoms are left out of the "
            "tight-binding model (no on-site energies or hoppings for them).")


def basis_size(orb_cfg: dict, sites: list) -> int:
    SUB, _ = sk_tables()
    total = 0
    for s in sites:
        total += sum(SUB[o]["dim"] for o in orb_cfg.get(s["symbol"], []))
    return total


def is_rare_earth(Z: int) -> bool:
    """Lanthanides (La-Lu) and actinides (Ac-Lr): the elements with an open or just-filled f shell."""
    return 57 <= Z <= 71 or 89 <= Z <= 103


def default_orbitals(Z: int) -> list:
    if is_rare_earth(Z):
        return ["s", "d", "f"]
    return [get_skfitter().get_valence_orbitals_from_Z(Z)]


def _integral(key: str) -> str:
    """The integral label of a hopping or overlap key ('v_Ir_s-Mn_d_sds_0' -> 'sds'; a group pair: the integral of
    the two families), '' for other keys."""
    return get_skfitter().integral_name(key) or ""


def _f_integral(key: str) -> str:
    """'v_Ce_p-Ce_f_pfp_0' / 's_..._ffs_0' -> 'pfp' / 'ffs' for the integrals with an f orbital, else ''."""
    lab = _integral(key)
    return lab if re.match(r"^(s\*?f|pf|df|ff)[spdf]$", lab) else ""


def default_bound(key: str):
    """Same defaults as SKFitter.fit(); f integrals (4f / 5f shells are compact: small couplings) get their own.
    A group-pair integral has the bounds of the integral it splits."""
    fi = _f_integral(key)
    if fi:
        if key.startswith("s_"):
            return (-0.5, 0.5)
        return (-2.0, 2.0) if fi.startswith("ff") else (-4.0, 4.0)
    if key.startswith("lambda_"):
        return (0.0, 1.0)
    if key.startswith("delta_spin_"):
        return (-5.0, 5.0)
    if key.startswith("e_orbit"):
        return (-10.0, 15.0)
    if key.startswith("e_mix"):                  # on-site mixing / crystal field beyond the group energies
        return (-3.0, 3.0)
    if key.startswith("s_"):
        return (-0.8, 0.8)
    if key.startswith("v_"):
        lab = _integral(key)
        if lab in ("pds", "dds", "sss", "s*s*s", "ss*s", "sds", "s*ds"):
            return (-10.0, 2.0)
        if lab in ("pdp", "ddp", "pps"):
            return (-2.0, 8.0)
        if lab == "ddd":
            return (-2.0, 2.0)
        return (-8.0, 8.0)
    if key.startswith("eta_"):
        return (1.0, 5.0)
    return (-8.0, 8.0)


def default_initial(key: str) -> float:
    lo, hi = default_bound(key)
    fi = _f_integral(key)
    if fi:                       # usual Slater-Koster signs, small values: sigma (-1)^(l2+1), pi (-1)^l2 ... with l2 = 3
        mag = {"s": 0.3, "p": 0.15, "d": 0.05, "f": 0.0}[fi[-1]] * (0.3 if fi.startswith("ff") else 1.0)
        sign = {"s": 1.0, "p": -1.0, "d": 1.0, "f": -1.0}[fi[-1]]
        if key.startswith("s_"):                       # overlaps: the opposite sign (H ~ -K S), smaller
            sign, mag = -sign, 0.3 * mag
        return float(np.clip(sign * mag, lo, hi)) + 0.0
    if key.startswith("s_"):
        v = 0.1
    elif key.startswith("delta_spin_"):
        v = 1.0
    elif key.startswith("v_") and _integral(key) in ("pds", "dds", "sds"):
        v = -2.0
    elif key.startswith("v_") and _integral(key) in ("pdp", "ddp"):
        v = 1.0
    else:
        v = 0.0
    return float(np.clip(v, lo, hi))


def param_group(key: str) -> str:
    if key.startswith("e_orbit"):
        return "On-site energy"
    if key.startswith("e_mix"):
        return "On-site mixing"
    if key.startswith("lambda_"):
        return "Spin-orbit"
    if key.startswith("delta_spin_"):
        return "Exchange splitting"
    if key.startswith("s_"):
        return "Overlap"
    if key.startswith("eta_"):
        return "Distance decay"
    return "Hopping"


def config_signature(cfg: dict) -> str:
    keep = {k: cfg.get(k) for k in ("orbital_config", "magnetic_orbitals", "n_bands_below", "n_bands_above",
                                    "n_neighbors", "symprec", "fit_eta", "fixed_eta", "include_soc", "non_orthogonal",
                                    "tb_nelect", "onsite_mixing")}
    if (cfg.get("hopping_mode") or "decay") != "decay":       # absent for the decay law: old signatures stay valid
        keep["hopping_mode"] = cfg["hopping_mode"]
    if cfg.get("soc_orbitals"):                              # likewise: only when given
        keep["soc_orbitals"] = cfg["soc_orbitals"]
    return json.dumps(keep, sort_keys=True, default=str)


MIN_ATOMIC_WEIGHT = 0.2      # a DFT state with all orbital projections together below this is "not atomic-like"
MIN_MODEL_WEIGHT = 0.5       # model_states_only: the least weight on the model's orbitals of a fitted DFT state


def _model_columns(meta: dict, subshells) -> list:
    """Projection columns (meta['orbital_names']) of a site's model subshells; families when only those exist."""
    names = meta.get("orbital_names") or []
    cols = []
    for sub in subshells:
        c = [names.index(x) for x in SUB_TO_PROJ.get(sub, []) if x in names]
        if not c:
            fam = sub[0]
            if fam in names:                                  # family-resolved projections (e.g. spin-orbit QE)
                c = [names.index(fam)]
            elif meta.get("family_cols"):
                c = list(meta["family_cols"].get(fam, []))
        cols += [i for i in c if i not in cols]
    return cols


def model_state_mask(dft: dict, fitter, threshold=None):
    """(n_k, n_bands) bool of the DFT states with at least `threshold` of their weight on the orbitals of the model
    (fitter.site_orbitals); vacuum states and bands of other orbitals are False. The DFT k-points are matched to the
    fitter's. Non-collinear and spin-orbit runs (one projection set for all states) and collinear magnets (spin-up and
    spin-down states merged in energy order, as SKFitter fits them) are both handled. None if there are no
    projections or the data do not match."""
    threshold = MIN_MODEL_WEIGHT if threshold is None else threshold
    proj = dft.get("P") or {}
    meta = dft["meta"]
    if proj.get("up") is None:
        return None

    def weight(P):
        P = np.asarray(P, float)
        if P.shape[2] != len(fitter.site_orbitals):
            return None
        w = np.zeros(P.shape[:2])
        for i, subs in enumerate(fitter.site_orbitals):
            cols = _model_columns(meta, subs)
            if cols:
                w += P[:, :, i, :][:, :, cols].sum(axis=2)
        return w

    w = weight(proj["up"])
    if w is None:
        return None
    if dft["E"].get("down") is not None:                       # collinear magnet: merge up and down by energy
        if proj.get("down") is None:
            return None
        wd = weight(proj["down"])
        if wd is None:
            return None
        E = np.hstack([np.asarray(dft["E"]["up"], float), np.asarray(dft["E"]["down"], float)])
        order = np.argsort(E, axis=1, kind="stable")
        w = np.take_along_axis(np.hstack([w, wd]), order, axis=1)
    kd, kf = np.asarray(dft["kcart"], float), np.asarray(fitter.kpoints, float)
    if len(kd) != len(kf) or not np.allclose(kd, kf, atol=1e-5):
        d = np.linalg.norm(kf[:, None, :] - kd[None, :, :], axis=2)
        j = d.argmin(axis=1)
        if np.any(d[np.arange(len(kf)), j] > 1e-5):
            return None
        w = w[j]
    if w.shape[1] != fitter._all_dft_energies.shape[1]:
        return None
    return w >= threshold


def apply_state_filter(fitter, cfg: dict, project_dir: str, log=print):
    """cfg['model_states_only']: fit only the DFT states of the model's orbitals (SKFitter.restrict_to_states)."""
    if not cfg.get("model_states_only"):
        return
    src = project_sources(project_dir)
    dft = None
    cache = os.path.join(project_dir, "dft_cache.npz")
    if os.path.isfile(cache):
        if src["code"] == "espresso":
            dft = load_dft_cache(cache, [src["qe_xml"], src["qe_scf"], src["qe_projwfc"]], src.get("qe_bands_in"),
                                 "espresso")
        else:
            dft = load_dft_cache(cache, [src["vasprun"]], src.get("kpoints"), "vasp")
    mask = None if dft is None else model_state_mask(dft, fitter)
    if mask is None:
        log("[skfitter] 'Fit only states of the model's orbitals' needs the orbital projections matching the bands "
            "(vasprun.xml with LORBIT, or the projwfc.x file(s)); fitting the plain band window.")
        return
    fitter.restrict_to_states(mask, "the DFT states of the model's orbitals")


def build_fitter(cfg: dict, project_dir: str):
    fitter = _make_fitter(cfg, project_dir)
    apply_state_filter(fitter, cfg, project_dir)
    if float(cfg.get("scissor") or 0.0):
        fitter.apply_scissor(float(cfg["scissor"]))       # cfg['scissor']: shift the DFT conduction bands (eV)
    return fitter


def _make_fitter(cfg: dict, project_dir: str):
    sk = get_skfitter()
    return sk.SKFitter(
        **fitter_source_kwargs(project_sources(project_dir)),
        orbital_config={k: list(v) for k, v in cfg["orbital_config"].items()},
        magnetic_orbitals={k: list(v) for k, v in (cfg.get("magnetic_orbitals") or {}).items()},
        n_bands_below=int(cfg["n_bands_below"]), n_bands_above=int(cfg["n_bands_above"]),
        n_neighbors=int(cfg["n_neighbors"]), symprec=float(cfg["symprec"]),
        fit_eta=bool(cfg["fit_eta"]), fixed_eta=float(cfg["fixed_eta"]),
        include_soc=bool(cfg["include_soc"]), non_orthogonal=bool(cfg["non_orthogonal"]),
        tb_nelect=None if cfg.get("tb_nelect") in (None, "", 0) else float(cfg["tb_nelect"]),
        hopping_mode=cfg.get("hopping_mode") or "decay",
        onsite_mixing=cfg.get("onsite_mixing") or None,
        soc_orbitals={k: list(v) for k, v in cfg["soc_orbitals"].items()} if cfg.get("soc_orbitals") else None,
    )


def describe_fitter(fitter) -> dict:
    keys = fitter._get_param_keys()
    return {
        "param_keys": keys,
        "initial": {k: default_initial(k) for k in keys},
        "bounds": {k: default_bound(k) for k in keys},
        "info": {
            "base_dim": int(fitter.base_dim), "total_dim": int(fitter.total_dim),
            "n_params": len(keys), "spacegroup": str(fitter.spacegroup),
            "n_target_bands": len(fitter.target_indices), "nk": int(len(fitter.kpoints)),
            "soc": bool(fitter.soc), "magnetic": bool(fitter.is_magnetic),
            "tb_nelect": float(fitter.tb_nelect), "n_neighbors": int(fitter.n_neighbors),
            "n_pairs": len(fitter.neighbor_list),
            **({"moments": moments_text(fitter)} if moments_text(fitter) else {}),
        },
    }


def moments_text(fitter) -> str:
    """'Mn1 ↑, Mn2 ↓' when the model's magnetic sites have antiparallel moments (from the VASP MAGMOM), else ''."""
    signs = list(getattr(fitter, "site_moment_signs", None) or [])
    dirs = getattr(fitter, "site_moment_dirs", None)
    if fitter.is_magnetic and dirs is not None and any(abs(d[2]) < 1 - 1e-6 for d in dirs):   # non-collinear
        return ", ".join(f"{fitter.structure[i].specie.symbol}{i + 1} ({', '.join(f'{c:+.2f}' for c in d)})"
                         for i, d in enumerate(dirs) if fitter.structure[i].specie.symbol in fitter.magnetic_orbitals)
    if not (fitter.is_magnetic and any(v < 0 for v in signs)):
        return ""
    return ", ".join(f"{fitter.structure[i].specie.symbol}{i + 1} {'↑' if v > 0 else '↓'}" for i, v in enumerate(signs)
                     if fitter.structure[i].specie.symbol in fitter.magnetic_orbitals)




def all_dft_bands(dft: dict):
    """The (nk, nbands) energy array SKFitter fits against: spin-up (+ spin-down merged and sorted if magnetic)."""
    up, dn = dft["E"]["up"], dft["E"]["down"]
    return np.sort(np.hstack([up, dn]), axis=1) if dn is not None else up


def dft_window(dft: dict, include_soc: bool, n_below: int, n_above: int):
    """Indices of the DFT bands SKFitter would fit (mirrors SKFitter._load_vasp_data). None if NELECT unknown."""
    meta = dft["meta"]
    if meta.get("nelect") is None:
        return None
    E = all_dft_bands(dft)
    nb = E.shape[1]
    polarised = bool(include_soc or meta["magnetic"])
    nocc = int(round(meta["nelect"])) if polarised else int(round(meta["nelect"] / 2.0))
    vbm = max(0, nocc - 1)
    sf = 2 if (include_soc and not meta["magnetic"]) else 1
    start = max(0, vbm - n_below * sf + 1)
    end = min(nb, vbm + n_above * sf + 1)
    return {"indices": list(range(start, end)), "vbm": vbm, "nbands_total": nb}


def kpath_fractional(dft: dict) -> np.ndarray:
    """The k-points of the DFT path in reciprocal-lattice (fractional) coordinates."""
    recip = reciprocal_matrix(dft["meta"]["lattice"])
    return np.asarray(dft["kcart"], float) @ np.linalg.inv(recip)


def kpath_points(dft: dict) -> list:
    """The labelled high-symmetry points of the DFT k-path, first occurrence of each label: [{label, index, frac}].
    A tick labelled 'H|P' (a jump in the path) gives both H (end of one segment) and P (start of the next)."""
    x = np.asarray(dft.get("x", []), float)
    ticks = list(np.asarray(dft.get("ticks", []), float))
    labels = list(dft.get("tick_labels") or [])
    if not len(x) or not ticks:
        return []
    frac = kpath_fractional(dft)
    out, seen = [], set()
    for i, t in enumerate(ticks):
        lab = str(labels[i]) if i < len(labels) and labels[i] is not None else ""
        parts = [q.strip() for q in lab.split("|")] if lab else [""]
        at = [int(j) for j in np.where(np.abs(x - t) < 1e-6)[0]] or [int(np.argmin(np.abs(x - t)))]
        for n, part in enumerate(parts):
            j = at[min(n, len(at) - 1)]
            name = part or f"k{j}"
            if name in seen:
                continue
            seen.add(name)
            out.append({"label": name, "index": j, "frac": [round(float(c), 6) for c in frac[j]]})
    return out


def nearest_kpoint(dft: dict, kfrac) -> tuple:
    """(index, distance in 1/Angstrom) of the DFT path k-point closest to ``kfrac`` (fractional), counting k-points
    that differ by a reciprocal-lattice vector as the same point. Projections exist only on the DFT path."""
    recip = reciprocal_matrix(dft["meta"]["lattice"])
    d = kpath_fractional(dft) - np.asarray(kfrac, float).reshape(1, 3)
    d -= np.round(d)
    dist = np.linalg.norm(d @ recip, axis=1)
    j = int(np.argmin(dist))
    return j, float(dist[j])


def orbital_character(dft: dict, indices, k_index: int, min_share: float = 0.02) -> dict:
    """Orbital character of the fitted DFT bands at one k-point of the path, from the DFT orbital projections.

    ``indices`` are band indices into all_dft_bands (the fitted window: spin-up and spin-down merged and sorted for a
    collinear magnet). Each state's weight is summed over the sites of each element and over the orbitals of each
    family (s, p, d, f): channels '<element> <family>'. Channels below ``min_share`` in every state go into 'other';
    'unassigned' is 1 minus the total (charge outside the projection spheres or basis).
    Returns {k_index, channels, rows: [{band, E, spin, weights, total}]}."""
    meta = dft["meta"]
    P = dft.get("P") or {}
    if not P or not meta.get("has_projection", bool(P)):
        raise ValueError("The DFT data has no orbital projections (QE: run projwfc.x; VASP: LORBIT = 11).")
    k = int(k_index)
    E_up = np.asarray(dft["E"]["up"], float)
    if not 0 <= k < E_up.shape[0]:
        raise ValueError(f"k-point index {k} is outside the DFT path (0-{E_up.shape[0] - 1}).")
    states = [("up", b, float(E_up[k, b])) for b in range(E_up.shape[1])]
    if dft["E"].get("down") is not None:
        E_dn = np.asarray(dft["E"]["down"], float)
        states += [("down", b, float(E_dn[k, b])) for b in range(E_dn.shape[1])]
        states.sort(key=lambda s: s[2])
    sym = [s["symbol"] for s in meta["sites"]]
    elems = [e for e in meta["elements"] if e in sym]
    fam = meta.get("family_cols") or {}
    raw = []
    for idx in indices:
        if not 0 <= int(idx) < len(states):
            continue
        spin, b, e = states[int(idx)]
        Pk = np.asarray(P.get(spin, P.get("up"))[k, b], float)          # (sites, orbital columns)
        w = {}
        for el in elems:
            rows = [i for i, s in enumerate(sym) if s == el]
            for f, cols in fam.items():
                v = float(Pk[np.ix_(rows, cols)].sum()) if cols else 0.0
                if v > 0:
                    w[f"{el} {f}"] = v
        raw.append((int(idx), e, spin, w))
    names = sorted({c for _, _, _, w in raw for c in w})
    keep = [c for c in names if max(w.get(c, 0.0) for _, _, _, w in raw) >= min_share]
    order = [f"{el} {f}" for el in elems for f in fam if f"{el} {f}" in keep]
    out_rows = []
    for idx, e, spin, w in raw:
        ww = {c: round(w.get(c, 0.0), 4) for c in order}
        other = sum(v for c, v in w.items() if c not in keep)
        if other > 1e-4:
            ww["other"] = round(other, 4)
        total = sum(w.values())
        ww["unassigned"] = round(max(0.0, 1.0 - total), 4)
        out_rows.append({"band": idx, "E": round(e, 4), "spin": spin if dft["E"].get("down") is not None else None,
                         "weights": ww, "total": round(total, 4)})
    channels = order + (["other"] if any("other" in r["weights"] for r in out_rows) else []) + ["unassigned"]
    return {"k_index": k, "channels": channels, "rows": out_rows}


def _family_of(label: str) -> str:
    """Orbital family (s, p, d, f) of a model orbital label: 's', 'pz', 'pxy', 'd', 'dz2', 'd_t2g', 'f_a2u', ..."""
    c = str(label).strip().lower()[:1]
    if c not in "spdf":
        raise ValueError(f"orbital label {label!r}: no s, p, d or f family")
    return c


def suggest_orbitals(dft: dict, indices, orbital_config: dict, cutoff: float = 0.1, fermi_weight: float = 1.0,
                     fermi_window: float = 0.5, degen_tol: float = 1e-3, electrons_per_state: float = 1.0,
                     semicore_gap: float = 6.0, min_model_weight: float = 0.0, projected_shells=None) -> dict:
    """Orbital families the model lacks, ranked from the DFT orbital projections of the fitted bands (all k-points).

    Each state (k, band) of the window gets a weight W = 1 + (fermi_weight - 1) exp(-(E/fermi_window)^2), as in the
    fit, and its projections summed per element and family (channels '<element> <family>'), averaged over degenerate
    states. For every channel:
      absolute  - the largest weight in any state of the window;
      weighted  - the largest, over the bands of the window, of the W-weighted average over k;
      missing   - the share of the W-weighted weight the model cannot describe (1 - weight of its own channels) that
                  the channel carries: what adding it would explain;
      average   - its W-weighted average weight over all states of the window;
      centre    - its projection-weighted mean energy (eV from E_F) over the DFT states from ``semicore_gap`` eV below
                  the window upwards (deep semicore states left out): a starting on-site energy;
      electrons - its projected weight in the occupied states of that range (electrons_per_state per state), and
      filled    - that as a fraction of the family's capacity 2(2l+1) per site. A nearly full family (a semicore-like
                  Bi 6s) adds its 2(2l+1) electrons per site to the model's count; a nearly empty one adds none; in
                  between the weight comes from hybridisation and the formal valence decides.
    'unassigned' (1 - the summed projections) is weight no atomic orbital carries: it bounds what can be explained.
    A channel is suggested by a method when its value reaches ``cutoff``. ``min_model_weight`` > 0 (the fit's
    model_states_only) leaves out the states with less weight than that on the model's orbitals, as the fit does.
    ``projected_shells`` ({element: ['3D', '4S', ...]}, the projwfc.x state labels): the valence shells without a
    projection state ('unprojected': e.g. Mn 4p when the pseudopotential has no 4p wavefunction), whose character
    can only show up as unassigned weight, and a note when that weight is a large part of what the model misses.
    Returns a dict for printing and JSON."""
    meta = dft["meta"]
    P = dft.get("P") or {}
    if not P or not meta.get("has_projection", bool(P)):
        raise ValueError("The DFT data has no orbital projections (QE: run projwfc.x; VASP: LORBIT = 11).")
    sym = [s["symbol"] for s in meta["sites"]]
    elems = [e for e in meta["elements"] if e in sym]
    fam = {f: list(c) for f, c in (meta.get("family_cols") or {}).items() if c}
    channels = [(el, f) for el in elems for f in fam]
    rows_of = {el: [i for i, s in enumerate(sym) if s == el] for el in elems}
    E_up = np.asarray(dft["E"]["up"], float)
    spins = [("up", E_up)]
    if dft["E"].get("down") is not None:
        spins.append(("down", np.asarray(dft["E"]["down"], float)))
    nk = E_up.shape[0]
    included = {(el, _family_of(lab)) for el, labs in (orbital_config or {}).items() for lab in labs}
    unknown = sorted({el for el, _ in included} - set(elems))
    if unknown:
        raise ValueError(f"[model] orbitals name elements that are not in the structure: {', '.join(unknown)}")

    def chan(spin, k, b):
        Pk = np.asarray(P.get(spin, P.get("up"))[k, b], float)
        return np.array([Pk[np.ix_(rows_of[el], fam[f])].sum() for el, f in channels])

    idx = [int(i) for i in indices]
    nb = len(idx)
    En = np.full((nk, nb), np.nan)
    Wc = np.zeros((nk, nb, len(channels)))
    for k in range(nk):
        states = sorted(((float(Es[k, b]), sp, b) for sp, Es in spins for b in range(Es.shape[1])), key=lambda t: t[0])
        sel = [states[i] for i in idx if 0 <= i < len(states)]
        e = np.array([t[0] for t in sel])
        w = np.array([chan(sp, k, b) for _, sp, b in sel])
        g = np.concatenate([[0], np.cumsum(np.diff(e) > degen_tol)])            # degenerate groups
        for gi in np.unique(g):
            m = g == gi
            w[m] = w[m].mean(axis=0)
        En[k, :len(e)], Wc[k, :len(e)] = e, w
    ok = np.isfinite(En)
    W = np.where(ok, 1.0 + (max(float(fermi_weight), 1.0) - 1.0) * np.exp(-(np.nan_to_num(En, nan=1e3) / fermi_window) ** 2), 0.0)
    inc = np.array([c in included for c in channels])
    n_dropped = 0
    if min_model_weight > 0:                                    # model_states_only: the fit ignores these states
        drop = ok & (Wc[:, :, inc].sum(axis=2) < float(min_model_weight))
        n_dropped = int(drop.sum())
        W = np.where(drop, 0.0, W)
    total = Wc.sum(axis=2)
    missing = np.clip(1.0 - Wc[:, :, inc].sum(axis=2), 0.0, None)
    unassigned = np.clip(1.0 - total, 0.0, None)
    Wsum = W.sum()
    miss_tot = float((W * missing).sum())
    # band centres over the DFT states from semicore_gap below the window up (both spins)
    e_low = float(np.nanmin(En)) - float(semicore_gap)
    num, den, occ = np.zeros(len(channels)), np.zeros(len(channels)), np.zeros(len(channels))
    for sp, Es in spins:
        Pa = np.asarray(P.get(sp, P.get("up")), float)                      # (k, bands, sites, columns)
        for c, (el, f) in enumerate(channels):
            wc = Pa[:, :, rows_of[el]][:, :, :, fam[f]].sum(axis=(2, 3))
            Eb = Es[:, :wc.shape[1]]
            wv = wc * (Eb >= e_low)                                     # valence and above: no deep semicore states
            num[c] += float((wv * Eb).sum())
            den[c] += float(wv.sum())
            occ[c] += float((wv * (Eb < 0.0)).sum()) / nk * float(electrons_per_state)
    out = []
    for c, (el, f) in enumerate(channels):
        x = Wc[:, :, c]
        per_band = (W * x).sum(axis=0) / np.maximum(W.sum(axis=0), 1e-300)
        jb = int(np.argmax(per_band))
        kk, bb = np.unravel_index(int(np.argmax(np.where(ok & (W > 0), x, -1))), x.shape)
        out.append({"channel": f"{el} {f}", "element": el, "family": f, "in_model": bool(inc[c]),
                    "absolute": float(x[kk, bb]), "absolute_at": {"k_index": int(kk), "band": idx[bb], "E": float(En[kk, bb])},
                    "weighted": float(per_band[jb]), "weighted_band": idx[jb],
                    "average": float((W * x).sum() / Wsum),
                    "missing_share": float((W * x).sum() / miss_tot) if not inc[c] and miss_tot > 0 else None,
                    "centre_eV": float(num[c] / den[c]) if den[c] > 0 else None, "electrons": float(occ[c]),
                    "filled": float(occ[c] / (2 * (2 * "spdf".index(f) + 1) * len(rows_of[el])))})
    cand = [r for r in out if not r["in_model"]]
    cand.sort(key=lambda r: -(r["missing_share"] or 0.0))
    pick = {m: [r["channel"] for r in cand if r[m] >= cutoff] for m in ("absolute", "weighted")}
    unproj = get_skfitter().unprojected_valence_shells(projected_shells) if projected_shells else {}
    unproj = {el: v for el, v in unproj.items() if el in elems}
    un_avg, miss_avg = float((W * unassigned).sum() / Wsum), miss_tot / Wsum
    note = None
    if unproj and un_avg >= 0.03 and un_avg >= 0.5 * miss_avg:
        shells = ", ".join(f"{el} {x}" for el, v in unproj.items() for x in v)
        fams = {(el, x[-1]) for el, v in unproj.items() for x in v}
        new = [f"{el} {f}" for el, f in sorted(fams) if (el, f) not in included]
        have = [f"{el} {f}" for el, f in sorted(fams) if (el, f) in included]
        note = (f"{100 * un_avg:.0f}% of the weight per state is on no projected orbital ({100 * un_avg / max(miss_avg, 1e-12):.0f}% "
                f"of what the model cannot carry). The projections have no {shells} (the pseudopotentials carry no "
                "such wavefunctions), so their character cannot be ranked here and may be what is missing"
                + (f": try adding {', '.join(new)}" if new else "")
                + (f" ({', '.join(have)} already in the model)" if have else "") + ".")
    return {"unprojected": unproj, "unprojected_note": note, "cutoff": float(cutoff), "bands": [idx[0], idx[-1]], "n_k": nk, "fermi_weight": float(fermi_weight),
            "fermi_window": float(fermi_window), "energy_floor": e_low, "channels": out,
            "min_model_weight": float(min_model_weight), "states_left_out": n_dropped, "states": int(ok.sum()), "candidates": [r["channel"] for r in cand],
            "suggested": pick, "missing_average": miss_tot / Wsum, "unassigned_average": float((W * unassigned).sum() / Wsum),
            "model_channels": [f"{el} {f}" for el, f in channels if (el, f) in included]}


def _spin_title(spin: str, magnetic: bool) -> str:
    if not magnetic:
        return "Bands"
    return "Spin up (majority)" if spin == "up" else "Spin down (minority)"


def _path_axes(ax, dft_or_axis, ylim):
    """Common k-path decoration: node lines, labels, Fermi level."""
    ticks = dft_or_axis["ticks"]
    labels = dft_or_axis["tick_labels"]
    for t in ticks:
        ax.axvline(t, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if labels else [""] * len(ticks))
    ax.axhline(0.0, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.set_ylim(*ylim)
    ax.tick_params(length=3)


def draw_dft_bands(fig: Figure, dft: dict, *, color_by="family", atoms="All", orbital="s", window=(-6.0, 6.0),
                   spin_view="side"):
    """DFT band structure coloured by orbital projection.

    color_by: 'family' (s/p/d/f mix), 'element', 'orbital' (weight of one orbital/family on a blue ramp), 'none'.
    spin_view: 'side' (two panels), 'overlay' (up blue / down red), 'up', 'down'  - only used for spin-polarised data.
    """
    meta = dft["meta"]
    x = dft["x"]
    magnetic = meta["magnetic"] and dft["E"]["down"] is not None
    if not magnetic:
        panels = ["up"]
    elif spin_view == "side":
        panels = ["up", "down"]
    elif spin_view in ("up", "down"):
        panels = [spin_view]
    else:
        panels = ["overlay"]
    can_colour = bool(dft["P"]) and color_by != "none" and panels != ["overlay"]
    with styled():
        fig.clear()
        axs = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
        legend_handles, cbar_mappable = [], None
        for ax, spin in zip(axs, panels):
            if spin == "overlay":
                for key, col, lab in (("up", CAT[0], "spin up"), ("down", CAT[7], "spin down")):
                    segs, _ = band_segments(x, dft["E"][key], None, *window)
                    ax.add_collection(LineCollection(segs, colors=col, linewidths=1.3, zorder=2))
                legend_handles = [Line2D([0], [0], color=CAT[0], lw=1.6, label="spin up"),
                                  Line2D([0], [0], color=CAT[7], lw=1.6, label="spin down")]
                ax.set_title("Spin up and spin down", loc="left")
            else:
                E = dft["E"][spin]
                rgb = None
                if can_colour and spin in dft["P"]:
                    if color_by == "family":
                        W, names = projection_weights(dft, spin, atoms, "family")
                        cols = [FAMILY_COLOR.get(n, CAT[3]) for n in names]
                        rgb, _ = blend_colors(W, cols, dominant=False)
                        used = W.reshape(-1, W.shape[-1]).max(axis=0) > 0.02      # only families that carry weight
                        legend_handles = [Line2D([0], [0], color=c, lw=2.2, label=FAMILY_LABEL.get(n, n))
                                          for n, c, u in zip(names, cols, used) if u]
                    elif color_by == "element":
                        W, names = projection_weights(dft, spin, "All", "element")
                        cols = [CAT[i % len(CAT)] for i in range(len(names))]
                        rgb, _ = blend_colors(W, cols, dominant=False)
                        legend_handles = [Line2D([0], [0], color=c, lw=2.2, label=n) for n, c in zip(names, cols)]
                    else:  # a single orbital: weight on a sequential ramp
                        W, _ = projection_weights(dft, spin, atoms, "orbital", orbital)
                        w = np.clip(W[..., 0], 0, None)
                        vmax = max(float(np.percentile(w, 99.5)), 1e-6)
                        rgb = SEQ_CMAP(np.clip(w / vmax, 0, 1))[..., :3]
                        cbar_mappable = matplotlib.cm.ScalarMappable(
                            norm=matplotlib.colors.Normalize(0, vmax), cmap=SEQ_CMAP)
                segs, cols_ = band_segments(x, E, rgb, *window)
                ax.add_collection(LineCollection(segs, colors=cols_ if rgb is not None else INK,
                                                 linewidths=1.5 if rgb is not None else 1.1, zorder=2,
                                                 capstyle="round"))
                ax.set_title(_spin_title(spin, magnetic), loc="left")
            ax.set_xlim(x[0], x[-1])
            _path_axes(ax, dft, window)
        axs[0].set_ylabel("E − E$_F$ (eV)")
        if legend_handles:
            axs[-1].legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(legend_handles),
                           handlelength=1.6, columnspacing=1.2, borderaxespad=0.3)
        if cbar_mappable is not None:
            cb = fig.colorbar(cbar_mappable, ax=list(axs), fraction=0.03, pad=0.02)
            cb.set_label(f"{orbital} weight", color=INK2)
            cb.outline.set_edgecolor(AXIS)
        elif not can_colour and dft["P"] == {} and color_by != "none":
            axs[0].text(0.02, 0.03, "no orbital projections (projwfc.x filproj file not given)"
                        if meta.get("dft_code") == "espresso" else
                        "vasprun.xml has no orbital projections (LORBIT = 11 or 12)",
                        transform=axs[0].transAxes, color=MUTED, fontsize=8)
        fit_layout(fig)


def draw_fit_window(fig: Figure, dft: dict, indices, window=(-6.0, 6.0)):
    """All DFT bands (merged over spin if magnetic), the ones SKFitter would fit highlighted."""
    E = all_dft_bands(dft)
    x = dft["x"]
    sel = set(indices or [])
    with styled():
        fig.clear()
        ax = fig.add_subplot(111)
        others = [b for b in range(E.shape[1]) if b not in sel]
        if others:
            segs, _ = band_segments(x, E[:, others], None, *window)
            ax.add_collection(LineCollection(segs, colors=AXIS, linewidths=0.9, zorder=1))
        if sel:
            segs, _ = band_segments(x, E[:, sorted(sel)], None, *window)
            ax.add_collection(LineCollection(segs, colors=CAT[1], linewidths=1.6, zorder=2))
        ax.set_xlim(x[0], x[-1])
        _path_axes(ax, dft, window)
        ax.set_ylabel("E − E$_F$ (eV)")
        ax.legend(handles=[Line2D([0], [0], color=CAT[1], lw=1.8, label=f"fitted ({len(sel)} bands)"),
                           Line2D([0], [0], color=AXIS, lw=1.2, label="not fitted")], loc="upper right")
        ax.set_title("Bands used in the fit", loc="left")
        fit_layout(fig)


def fit_rmse(fitter) -> float:
    e_tb = fitter.tb_bands[:, fitter.target_tb_band_indices]
    return float(np.sqrt(np.nanmean((e_tb - fitter.target_dft_bands) ** 2)))


def draw_fit(fig: Figure, fitter, title=True):
    """DFT vs tight-binding bands (the fitted window) on an existing figure."""
    e_dft = np.asarray(fitter.target_dft_bands)
    e_tb = np.asarray(fitter.tb_bands)[:, fitter.target_tb_band_indices]
    try:
        segments = fitter._get_kpath_segments()          # KPOINTS or pw.x bands path, with DFT indices
    except Exception:  # noqa: BLE001
        segments = []
    axinfo = compute_path_axis(fitter.kpoints, segments)
    x = axinfo["x"]
    rmse = float(np.sqrt(np.nanmean((e_tb - e_dft) ** 2)))
    with styled():
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(x, e_dft, color=INK, lw=1.3, solid_capstyle="round")
        ax.plot(x, e_tb, color=CAT[1], lw=1.5, ls=(0, (5, 2.5)))
        _path_axes(ax, axinfo, (min(np.nanmin(e_dft), e_tb.min()) - 0.3, max(np.nanmax(e_dft), e_tb.max()) + 0.3))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel("E − E$_F$ (eV)")
        ax.text(0.02, 0.97, f"RMSE = {rmse * 1000:.1f} meV", transform=ax.transAxes, va="top", color=INK2,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
        ax.legend(handles=[Line2D([0], [0], color=INK, lw=1.3, label="DFT"),
                           Line2D([0], [0], color=CAT[1], lw=1.5, ls=(0, (5, 2.5)), label="Tight-binding fit")],
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.3)
        if title:
            ax.set_title("Tight-binding vs DFT" + (" (non-orthogonal)" if fitter.non_orthogonal else " (orthogonal)"),
                         loc="left")
        fit_layout(fig)


def draw_convergence(fig: Figure, xs, ys):
    with styled():
        fig.clear()
        ax = fig.add_subplot(111)
        if len(xs):
            ax.plot(xs, ys, color=CAT[0], lw=1.6)
            ax.set_yscale("log")
        ax.grid(True, axis="y")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("best RMSE (meV)")
        ax.set_title("Fit progress", loc="left")
        fit_layout(fig)


# =====================================================================================
#  Output capture, progress and cancel hooks around SKFitter
# =====================================================================================
class StdoutRelay(io.TextIOBase):
    """File-like object that hands every complete line to ``callback`` (used to capture skfitter's prints)."""

    def __init__(self, callback):
        self._cb, self._buf = callback, ""

    def writable(self):
        return True

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._cb(line.rstrip())
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._cb(self._buf.rstrip())
        self._buf = ""


@contextlib.contextmanager
def captured(callback):
    relay = StdoutRelay(callback)
    try:
        with contextlib.redirect_stdout(relay), contextlib.redirect_stderr(relay):
            yield relay
    finally:
        relay.flush()


DE_STEP = re.compile(r"differential_evolution step (\d+): f\(x\)=\s*([-+0-9.eE]+)")
LS_START = re.compile(r"^\s*start (\d+): (?:loss = ([-+0-9.eE]+)|failed)")


class FitMonitor:
    """Progress, best-so-far tracking and cancellation for SKFitter.fit().

    SKFitter evaluates the loss through ``objective_function`` (differential evolution) and ``_residuals``
    (least squares). Both are wrapped on the *instance* for the duration of the fit; the wrappers count
    evaluations, remember the best parameter vector and raise ``Cancelled`` when the user pressed Cancel.
    """

    def __init__(self, fitter, fit_opts, progress, metric, cancelled, log):
        self.f, self.opts = fitter, fit_opts
        self.progress, self.metric, self.cancelled, self.log = progress, metric, cancelled, log
        self.n_params = len(fitter._get_param_keys())
        self.method = fit_opts.get("method", "ls")
        self.obj_calls = self.res_calls = self.stage_calls = 0
        self.done_starts = 0
        self.best_loss, self.best_x = float("inf"), None
        self._last_emit = 0.0

    # -- helpers -----------------------------------------------------------------
    def _n_points(self):
        idx = getattr(self.f, "_active_idx", None)
        nk = len(idx) if idx is not None else len(self.f.kpoints)
        return max(nk * len(self.f.target_tb_band_indices), 1)

    def _rms_meV(self, loss):
        return 1000.0 * math.sqrt(max(loss, 0.0) / self._n_points())

    def _fraction(self):
        scale = 20.0 * (self.n_params + 1)
        inner = 1.0 - math.exp(-self.stage_calls / scale)
        if self.method == "de":
            total = (int(self.opts.get("maxiter", 100)) + 1) * int(self.opts.get("popsize", 15)) * self.n_params
            de = min(1.0, self.obj_calls / max(total, 1))
            return 0.9 * de if self.res_calls == 0 else 0.9 + 0.09 * inner
        n = max(int(self.opts.get("n_starts", 8)), 1)
        if self.done_starts >= n:                           # polish on the full k-set (only when k_stride > 1)
            return 0.9 + 0.09 * inner
        return 0.9 * (self.done_starts + inner) / n

    def _maybe_emit(self, force=False):
        now = time.monotonic()
        if force or now - self._last_emit > 0.2:
            self._last_emit = now
            self.progress(min(0.99, self._fraction()), self._status())
            if np.isfinite(self.best_loss):
                self.metric(self.obj_calls + self.res_calls, self._rms_meV(self.best_loss))

    def _status(self):
        if self.method == "de" and not self.res_calls:
            return f"differential evolution - {self.obj_calls} evaluations"
        if self.method == "ls":
            n = int(self.opts.get("n_starts", 8))
            return f"least squares - start {min(self.done_starts + 1, n)}/{n}"
        return "least-squares polish"

    def _see(self, x, loss, kind):
        if kind == "obj":
            self.obj_calls += 1
        else:
            self.res_calls += 1
        self.stage_calls += 1
        if loss < self.best_loss and loss < 1e8:
            self.best_loss, self.best_x = loss, np.array(x, dtype=float)
        self._maybe_emit()

    # -- stdout parsing ------------------------------------------------------------
    def line(self, text: str):
        m = LS_START.match(text)
        if m:
            self.done_starts += 1
            self.stage_calls = 0
        m = DE_STEP.search(text)
        if m:
            self.metric(self.obj_calls + self.res_calls, self._rms_meV(float(m.group(2))))
        if self.method == "de" and "Least-squares" in text:
            self.stage_calls = 0
        self.log(text)

    # -- context manager -----------------------------------------------------------
    def __enter__(self):
        f = self.f
        orig_obj, orig_res = f.objective_function, f._residuals

        def obj(x):
            if self.cancelled():
                raise Cancelled()
            v = orig_obj(x)
            self._see(x, v, "obj")
            return v

        def res(x):
            if self.cancelled():
                raise Cancelled()
            r = orig_res(x)
            self._see(x, float(np.dot(r, r)), "res")
            return r

        f.objective_function, f._residuals = obj, res
        return self

    def __exit__(self, *exc):
        self.f.__dict__.pop("objective_function", None)
        self.f.__dict__.pop("_residuals", None)
        self._maybe_emit(force=True)
        return False


class HamProbe:
    """Cancel + progress for the property calculations.

    Every property routine evaluates H(k) through ``_tb_orthogonal_hamiltonians``; wrapping it counts the k-points
    done.
    """

    def __init__(self, fitter, expected, progress, cancelled, label=""):
        self.f, self.expected, self.progress, self.cancelled, self.label = fitter, max(expected, 1), progress, cancelled, label
        self.count, self._last = 0, 0.0
        import threading
        self._lock = threading.Lock()              # the Brillouin-zone sums may evaluate k-blocks on several threads

    def tick(self, n):
        if self.cancelled():
            raise Cancelled()
        with self._lock:
            self.count += n
            now = time.monotonic()
            report = now - self._last > 0.15
            if report:
                self._last = now
            frac = min(0.97, self.count / self.expected)
        if report:
            self.progress(frac, self.label)

    def __enter__(self):
        f = self.f
        orig_sk = f._tb_orthogonal_hamiltonians

        def sk_ham(k_cart, chunk=512):
            self.tick(len(np.atleast_2d(k_cart)))
            return orig_sk(k_cart, chunk)

        f._tb_orthogonal_hamiltonians = sk_ham
        return self

    def __exit__(self, *exc):
        self.f.__dict__.pop("_tb_orthogonal_hamiltonians", None)
        return False


# =====================================================================================
#  Starting values: on-site energies from the DFT orbital projections
# =====================================================================================
_D_COLS = {"dxy": ["dxy"], "dyz": ["dyz"], "dzx": ["dxz"], "dx2y2": ["dx2-y2"], "dz2": ["dz2"]}
SUB_TO_PROJ = {
    "s": ["s"], "s*": ["s"], "px": ["px"], "py": ["py"], "pz": ["pz"], "pxy": ["px", "py"], "p": ["px", "py", "pz"],
    **_D_COLS,
    "d_e1": ["dyz", "dxz"], "d_e2": ["dxy", "dx2-y2"], "d_t2g": ["dxy", "dyz", "dxz"], "d_eg": ["dx2-y2", "dz2"],
    "d": ["dxy", "dyz", "dxz", "dx2-y2", "dz2"],
    # f: VASP's (and SKFitter's) names; the QE projections are mapped onto them (QE_M_TO_VASP_COL)
    "fy3x2": ["fy3x2"], "fxyz": ["fxyz"], "fyz2": ["fyz2"], "fz3": ["fz3"], "fxz2": ["fxz2"], "fzx2": ["fzx2"],
    "fx3": ["fx3"], "f_m0": ["fz3"], "f_m1": ["fyz2", "fxz2"], "f_m2": ["fxyz", "fzx2"], "f_m3": ["fy3x2", "fx3"],
    # cubic f groups: every tesseral f orbital they contain (the lm-resolved projections carry no phases)
    "f_a2u": ["fxyz"], "f_t1u": ["fz3", "fxz2", "fyz2", "fx3", "fy3x2"], "f_t2u": ["fzx2", "fxz2", "fyz2", "fx3", "fy3x2"],
    "f": ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"],
}
_ONSITE_KEY = re.compile(r"^e_orbit(\d+)_([A-Za-z]+)_(.+)$")


def estimate_onsite(dft: dict, fitter, keys, e_window=12.0, clip=True) -> dict:
    """Band centre of each orbital (projection-weighted mean energy) as the on-site guess (``clip``: kept inside the
    default bounds). States count from e_window eV below E_F - or from the bottom of the fitted DFT bands if they
    reach deeper (e.g. the C 2s band of graphene near -20 eV) - up to e_window eV above E_F. States that are hardly
    atomic-like (all projections together < min_total, e.g. the nearly-free-electron vacuum states of a 2D layer
    with vacuum in the cell) are left out: their small weights would otherwise add up and pull the centres up."""
    min_total = MIN_ATOMIC_WEIGHT
    meta = dft["meta"]
    if not dft["P"] or not meta.get("orbital_names"):
        return {}
    names = meta["orbital_names"]
    e_low = -e_window
    fitted = getattr(fitter, "target_dft_bands", None)
    if fitted is not None and np.size(fitted):
        e_low = min(e_low, float(np.nanmin(fitted)) - 0.5)
    sym = np.array([s["symbol"] for s in meta["sites"]])
    eq = list(getattr(fitter, "eq_site_map", range(len(sym))))
    out, s_centre = {}, {}
    for k in keys:
        m = _ONSITE_KEY.match(k)
        if not m:
            continue
        eqid, elem, sub = int(m.group(1)), m.group(2), m.group(3)
        ions = [i for i in range(len(sym)) if sym[i] == elem and int(eq[i]) == eqid]
        if not ions:
            ions = [i for i in range(len(sym)) if sym[i] == elem]
        cols_wanted = SUB_TO_PROJ.get(sub, [])
        cols = [names.index(c) for c in cols_wanted if c in names]
        if not cols and meta.get("family_cols"):                 # families only (LORBIT = 10)
            fam = {"s": "s", "p": "p", "d": "d", "f": "f"}.get(sub[0])
            cols = meta["family_cols"].get(fam, [])
        if not cols:
            continue
        num = den = 0.0
        for spin, P in dft["P"].items():
            E = dft["E"][spin]
            w = P[:, :, ions, :][:, :, :, cols].sum(axis=(2, 3)).astype(float)
            atomic = P.sum(axis=(2, 3)) >= min_total
            w = np.where((E > e_low) & (E < e_window) & atomic, w, 0.0)
            num += float((w * E).sum())
            den += float(w.sum())
        if den > 1e-6:
            centre = num / den
            lo, hi = default_bound(k)
            out[k] = float(np.clip(centre, lo, hi)) if clip else float(centre)
            if sub == "s":
                s_centre[(eqid, elem)] = out[k]
    for k in keys:                                                # s*: no DFT counterpart, sit above s
        m = _ONSITE_KEY.match(k)
        if m and m.group(3) == "s*" and (int(m.group(1)), m.group(2)) in s_centre:
            lo, hi = default_bound(k)
            v = s_centre[(int(m.group(1)), m.group(2))] + 6.0
            out[k] = float(np.clip(v, lo, hi)) if clip else float(v)
    return out


def model_env(fitter, cfg: dict, dft=None) -> dict:
    """The built model as ``skfitter.params.starting_values`` needs it: structure, equivalent sites, orbitals, d0,
    the distance law, and the DFT band centre of each orbital (for aligning a parameter set's energies)."""
    from skfitter import params as SP
    centres = {}
    if dft is not None:
        try:
            centres = estimate_onsite(dft, fitter, fitter._get_param_keys(), clip=False)
        except Exception:  # noqa: BLE001
            centres = {}
    st = fitter.structure
    return SP.model_environment(
        st.lattice.matrix, [site.specie.symbol for site in st], st.frac_coords, list(fitter.eq_site_map),
        cfg["orbital_config"], float(fitter.d0), int(cfg["n_neighbors"]), bool(cfg["fit_eta"]),
        float(cfg["fixed_eta"]), bool(cfg["non_orthogonal"]), centres,
        hopping_mode=getattr(fitter, "hopping_mode", "decay"), shell_distances=getattr(fitter, "shell_distances", None),
        pair_shells=fitter._pair_shells())


def starting_values(fitter, cfg: dict, set_data: dict, dft=None) -> dict:
    """Starting values and bounds of all parameters of ``fitter`` from a parameter set (defaults where the set has
    nothing): {initial, bounds, source, shift, coverage, notes, ...}."""
    from skfitter import params as SP
    keys = fitter._get_param_keys()
    return SP.starting_values(set_data, model_env(fitter, cfg, dft), keys, default_initials(fitter, dft),
                              {k: default_bound(k) for k in keys})


def default_initials(fitter, dft=None) -> dict:
    keys = fitter._get_param_keys()
    init = {k: default_initial(k) for k in keys}
    if getattr(fitter, "hopping_mode", "decay") == "shell":      # per shell: the default falls off as (d0/r_n)^3
        sk = get_skfitter()
        shells = fitter._pair_shells()
        for k in keys:
            p = sk.parse_integral_key(k)
            dist = shells.get(p["pair"]) if p else None
            if p and p["shell"] is not None and dist and p["shell"] < len(dist) and dist[p["shell"]] > 0:
                init[k] = float(init[k] * min(1.0, float(fitter.d0) / dist[p["shell"]]) ** 3)
    if dft is not None:
        try:
            init.update(estimate_onsite(dft, fitter, keys))
        except Exception:  # noqa: BLE001
            pass
    return init


# =====================================================================================
#  Fit: run, save, reload
# =====================================================================================
DEFAULT_FIT_OPTS = {
    "method": "ls", "n_starts": 8, "maxiter": 200, "popsize": 15, "k_stride": 1, "refine": True,
    "refine_max_nfev": 300, "seed": 0,
    "add_weight_at_symmetry_points": False, "symmetry_weight": 10.0, "symmetry_width": 2,
    "add_weight_at_fermi_bands": False, "fermi_weight": 10.0, "fermi_window": 0.5,
}


def fit_kwargs(cfg, opts, initial, bounds, keys) -> dict:
    o = {**DEFAULT_FIT_OPTS, **(opts or {})}
    kw = dict(
        maxiter=int(o["maxiter"]), popsize=int(o["popsize"]),
        initial_params={k: float(initial[k]) for k in keys if k in (initial or {})},
        custom_bounds={k: (float(bounds[k][0]), float(bounds[k][1])) for k in keys if k in (bounds or {})},
        fit_eta=bool(cfg["fit_eta"]), fixed_eta=float(cfg["fixed_eta"]),
        k_stride=int(o["k_stride"]), refine=bool(o["refine"]), refine_max_nfev=int(o["refine_max_nfev"]),
        method=str(o["method"]), n_starts=int(o["n_starts"]),
        seed=None if o.get("seed") in (None, "") else int(o["seed"]),
        add_weight_at_symmetry_points=bool(o["add_weight_at_symmetry_points"]),
        add_weight_at_fermi_bands=bool(o["add_weight_at_fermi_bands"]),
        symmetry_weight=float(o["symmetry_weight"]), symmetry_width=int(o["symmetry_width"]),
        fermi_weight=float(o["fermi_weight"]), fermi_window=float(o["fermi_window"]),
        workers=int(o.get("workers") or 1), tolerance=float(o.get("tolerance") or 1e-10),
        overlap_floor=None if o.get("overlap_floor") in (None, "") else float(o["overlap_floor"]),
    )
    return kw


def error_report(fitter) -> dict:
    try:
        return {k: (None if not np.isfinite(v) else float(v)) for k, v in
                fitter.fit_error_report(fitter.opt_params, 0.5).items()}
    except Exception:  # noqa: BLE001
        return {}


def sidecar_path(project_dir, seed):
    return os.path.join(project_dir, f"{seed}_settings.json")


def read_sidecar(project_dir, seed) -> dict:
    try:
        with open(sidecar_path(project_dir, seed), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def update_sidecar(project_dir, seed, **fields) -> dict:
    data = read_sidecar(project_dir, seed)
    data.update(fields)
    with open(sidecar_path(project_dir, seed), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return data


def save_fit_outputs(fitter, cfg, opts, initial, bounds, project_dir, seed, cancelled=False, log=print) -> dict:
    """<seed>_fit.json / _fit.txt (written by SKFitter.test_and_plot_fit), <seed>_fit.png, <seed>_settings.json."""
    import matplotlib.pyplot as plt
    jpath = os.path.join(project_dir, f"{seed}_fit.json")
    tpath = os.path.join(project_dir, f"{seed}_fit.txt")
    with captured(log):
        rmse = fitter.test_and_plot_fit(save_fig=None, save_param=jpath)
        fitter.test_and_plot_fit(save_fig=None, save_param=tpath)
    plt.close("all")
    ppath = os.path.join(project_dir, f"{seed}_fit.png")
    fig = Figure(figsize=(8, 6))
    draw_fit(fig, fitter)
    fig.savefig(ppath, dpi=300)
    rep = error_report(fitter)
    update_sidecar(project_dir, seed, version=1, seedname=seed, config=cfg, fit_options=opts or {},
                   initial={k: float(v) for k, v in (initial or {}).items()},
                   bounds={k: [float(b[0]), float(b[1])] for k, b in (bounds or {}).items()},
                   saved=datetime.datetime.now().isoformat(timespec="seconds"), rmse_meV=float(rmse) * 1000.0,
                   error_report_meV=rep, stopped_early=bool(cancelled))
    log(f"[skfitter] Saved {os.path.basename(jpath)}, {os.path.basename(tpath)}, {os.path.basename(ppath)}, "
        f"{os.path.basename(sidecar_path(project_dir, seed))}")
    return {"json": jpath, "txt": tpath, "png": ppath, "settings": sidecar_path(project_dir, seed),
            "rmse": float(rmse), "report": rep}


def run_fit_job(cfg, opts, initial, bounds, project_dir, seed, progress, metric, cancelled, log=print,
                dft=None) -> dict:
    with captured(log):
        fitter = build_fitter(cfg, project_dir)
    keys = fitter._get_param_keys()
    if not initial:                                   # no starting values given: sensible defaults + DFT band centres
        initial = default_initials(fitter, dft)
        log("[skfitter] Starting values: on-site energies from the DFT projections" if (dft and dft["P"])
            else "[skfitter] Starting values: defaults (no orbital projections in the DFT data)")
    if not bounds:
        bounds = {k: default_bound(k) for k in keys}
    kw = fit_kwargs(cfg, opts, initial, bounds, keys)
    log(f"[skfitter] Fitting {len(keys)} parameters to {len(fitter.kpoints)} k-points x "
        f"{len(fitter.target_tb_band_indices)} bands ({kw['method'].upper()})")
    mon = FitMonitor(fitter, kw, progress, metric, cancelled, log)
    t0, stopped = time.time(), False
    try:
        with mon, captured(mon.line):
            fitter.fit(**kw)
    except Cancelled:
        stopped = True
        if mon.best_x is None:
            raise RuntimeError("Stopped before any parameters were evaluated.")
        fitter._set_active_k(None)
        fitter.opt_params = fitter._param_vec_to_dict(mon.best_x)
        with captured(log):
            fitter._compute_tb_bands(fitter.opt_params)
        log("[skfitter] Stopped by the user - keeping the best parameters found so far.")
    log(f"[skfitter] Fit finished in {time.time() - t0:.1f} s")
    files = save_fit_outputs(fitter, cfg, kw, initial, bounds, project_dir, seed, stopped, log)
    return {"fitter": fitter, "cfg": cfg, "files": files, "keys": list(keys), "initial": kw["initial_params"],
            "bounds": kw["custom_bounds"], "cancelled": stopped, "rmse": files["rmse"], "report": files["report"]}


def load_fit_job(cfg, project_dir, seed, strict=True, log=print) -> dict:
    """Rebuild the fitter for ``cfg`` and attach the parameters stored in <seed>_fit.json/.txt (no refit)."""
    lines = []

    def tee(t):
        lines.append(t)
        log(t)

    with captured(tee):
        fitter = build_fitter(cfg, project_dir)
        ok = fitter.load_optimised_params(seed, project_dir, strict=strict)
    if not ok:
        why = " ".join(l.replace("[WARN]", "").strip() for l in lines if "[WARN]" in l or "No saved" in l)
        raise RuntimeError(why or f"No usable saved parameters for '{seed}' in the project folder.")
    keys = list(fitter._get_param_keys())
    side = read_sidecar(project_dir, seed)
    return {"fitter": fitter, "cfg": cfg, "keys": keys, "rmse": fit_rmse(fitter), "report": error_report(fitter),
            "initial": side.get("initial", {}), "bounds": {k: tuple(v) for k, v in side.get("bounds", {}).items()},
            "cancelled": bool(side.get("stopped_early", False)), "files": {
                "json": os.path.join(project_dir, f"{seed}_fit.json"), "txt": os.path.join(project_dir, f"{seed}_fit.txt"),
                "png": os.path.join(project_dir, f"{seed}_fit.png")}}


def config_from_settings(project_dir, seed, meta) -> dict:
    """Best-effort set-up from the SETTINGS record of a fit written by a script (no <seed>_settings.json)."""
    sk = get_skfitter()
    path = sk.SKFitter.find_param_file(seed, project_dir)
    if not path:
        return {}
    try:
        _, _, st = sk.SKFitter.read_param_file(path)
    except Exception:  # noqa: BLE001
        return {}
    if not st:
        return {}
    cfg = {"n_neighbors": st.get("n_neighbors"), "include_soc": st.get("soc"),
           "non_orthogonal": st.get("non_orthogonal")}
    orbs = st.get("orbitals") or []
    oc = {}
    for site, o in zip(meta["sites"], orbs):
        oc.setdefault(site["symbol"], list(o))
    cfg["orbital_config"] = oc
    return {k: v for k, v in cfg.items() if v is not None}


def free_parameter_rows(fitter, bounds=None, initial=None) -> list:
    """[(group, key, value, lo, hi, at_bound, initial)] for the free parameters."""
    rows = []
    for k in fitter._get_param_keys():
        v = float(fitter.opt_params[k])
        lo, hi = (bounds or {}).get(k, default_bound(k))
        span = max(hi - lo, 1e-9)
        at = (v - lo) < 1e-3 * span or (hi - v) < 1e-3 * span
        rows.append((param_group(k), k, v, float(lo), float(hi), bool(at), (initial or {}).get(k)))
    order = {g: i for i, g in enumerate(["On-site energy", "Hopping", "Overlap", "Spin-orbit", "Exchange splitting",
                                          "Distance decay"])}
    rows.sort(key=lambda r: (order.get(r[0], 99), r[1]))
    return rows




# =====================================================================================
#  Properties: Berry curvature, AHC, SHC (SK Hamiltonian)
# =====================================================================================
PROPERTY_TITLES = {
    "bc_path": "Berry curvature along the path", "bc_map": "Berry curvature map",
    "bc_wilson": "Wilson loops around K and K'", "bc_chern": "Chern and valley Chern numbers",
    "sbc_path": "Spin Berry curvature along the path",
    "ahc": "Anomalous Hall conductivity", "shc": "Spin Hall conductivity",
    "shc_T": "Spin Hall conductivity vs temperature", "bands": "DFT / TB bands",
    "wcc": "Hybrid Wannier centres, Chern and Z2",
}
SOURCE_TITLES = {"sk": "SK Hamiltonian"}


def property_tag(spec: dict) -> str:
    k, s = spec["kind"], spec["source"]
    if k == "bc":
        if spec["mode"] in BC_VALLEY_MODES:
            return f"{s}_bc_{spec['mode']}"
        return f"{s}_bc_{spec['mode']}_{spec['component']}"
    if k == "ahc":
        return f"{s}_ahc" + ("" if s == "wb" else f"_{spec['component']}")
    if k == "shc":
        if s == "wb":
            return f"{s}_shc"
        d = [_dl(spec[x]) for x in ("spin", "alpha", "beta")]
        if any(v.startswith("[") for v in d):                 # vectors: s111_a1-10_b11-2 (no brackets in file names)
            return f"{s}_shc_s{d[0].strip('[]')}_a{d[1].strip('[]')}_b{d[2].strip('[]')}"
        return f"{s}_shc_{d[0]}_{d[1]}{d[2]}"
    if k == "shc_T":
        return f"{s}_shcT_{spec['spin']}_{spec['alpha']}{spec['beta']}"
    if k == "sbc":
        return f"{s}_sbc_path_{spec['spin']}_{spec['alpha']}{spec['beta']}"
    if k == "wcc":
        return f"{s}_wcc_{spec['fixed']}_{float(spec['fixed_value']):g}_{spec['loop']}"
    return "wb_bands"


def _dl(d):
    """Direction label of a property: 'x' or '[1-10]' for a Cartesian vector."""
    return get_skfitter().direction_label(d)


def property_label(spec: dict) -> str:
    k, s = spec["kind"], spec["source"]
    if k == "bc":
        if spec["mode"] == "wilson":
            return f"Wilson loops around K and K' - {SOURCE_TITLES[s]}"
        if spec["mode"] == "chern":
            return f"Chern and valley Chern numbers - {SOURCE_TITLES[s]}"
        return f"Berry curvature Ω{spec['component']} - {'path' if spec['mode'] == 'path' else 'map'} - {SOURCE_TITLES[s]}"
    if k == "ahc":
        name = {"x": "yz", "y": "zx", "z": "xy"}[spec.get("component", "z")]
        return f"AHC σ{name if s != 'wb' else 'xy'} - {SOURCE_TITLES[s]}"
    if k == "shc":
        return (f"SHC σ^{_dl(spec['spin'])}_{_dl(spec['alpha'])}{_dl(spec['beta'])} - {SOURCE_TITLES[s]}" if s != "wb"
                else f"SHC σ^z_xy - {SOURCE_TITLES[s]}")
    if k == "shc_T":
        return f"SHC vs T σ^{spec['spin']}_{spec['alpha']}{spec['beta']} - {SOURCE_TITLES[s]}"
    if k == "sbc":
        return f"Spin Berry curvature Ω^{spec['spin']}_{spec['alpha']}{spec['beta']} - path - {SOURCE_TITLES[s]}"
    if k == "wcc":
        return (f"Hybrid Wannier centres ({spec['fixed']} = {float(spec['fixed_value']):g}, loops along {spec['loop']}) - "
                f"{SOURCE_TITLES[s]}")
    return "Band comparison - " + SOURCE_TITLES["wb"]


def _headers_from_dat(path, ncols, fallback):
    try:
        with open(path, encoding="utf-8") as fh:
            head = [ln[1:].strip() for ln in fh if ln.startswith("#")]
    except OSError:
        head = []
    for ln in head:
        for seg in ln.split(";"):
            toks = seg.split()
            if len(toks) == ncols:
                return toks
    out = list(fallback)[:ncols]
    out += [f"col{i + 1}" for i in range(len(out), ncols)]
    return out


def table_from_dat(path, kind, max_rows=600) -> dict:
    """Read a .dat written by SKFitter for the results table."""
    try:
        arr = np.atleast_2d(np.loadtxt(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"headers": [], "rows": [], "n_total": 0}
    n = len(arr)
    step = max(1, int(math.ceil(n / max_rows)))
    sub = arr[::step]
    fb = {"bc_path": ["k distance (1/Å)", "Ω total", "Ω occupied"], "bc_map": ["kx (1/Å)", "ky (1/Å)", "Ω total", "Ω occupied"],
          "ahc": ["E_F (eV)", "σ"], "shc": ["E_F (eV)", "σ"],
          "sbc_path": ["k distance (1/Å)", "Ω total (all bands)", "Ω total (fitted bands)"],
          "shc_T": ["T (K)", "kT (eV)", "σ mean", "σ std. err."],
          "bc_wilson": ["r (1/Å)", "γ_occ(K)/π", "γ_occ(K')/π"],
          "bc_chern": ["band (-1: occupied set)", "C", "C_K", "C_K'", "C_K − C_K'", "smallest gap (eV)"],
          "wcc": ["k (reduced)"]}.get(kind, [])
    return {"headers": _headers_from_dat(path, arr.shape[1], fb), "rows": [[f"{v:.6g}" for v in r] for r in sub],
            "n_total": n, "step": step}


def _unlink_quiet(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _close(fig):
    import matplotlib.pyplot as plt
    try:
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


def _sigma_summary(E, sigma, units, name):
    E, sigma = np.asarray(E), np.asarray(sigma)
    i0 = int(np.argmin(np.abs(E)))
    imax = int(np.argmax(np.abs(sigma)))
    return [(f"{name}(E_F = 0)", f"{sigma[i0]:.4g} {units}"),
            (f"largest |{name}|", f"{sigma[imax]:.4g} {units} at E − E_F = {E[imax]:+.3f} eV")]


KP = "K'"
BC_VALLEY_MODES = ("wilson", "chern")      # Berry-curvature displays that are numbers of a 2D hexagonal zone


def _pi(v) -> str:
    return "n/a (degenerate on the loop)" if not np.isfinite(v) else f"{v / np.pi:+.4f} π"


def _run_valley_topology(fitter, spec, png, dat, progress, cancelled):
    """Wilson loops around K and K' (mode 'wilson') or Chern / valley Chern numbers (mode 'chern') of the SK model:
    (summary rows, notes, table kind). Plot and data go to png / dat."""
    k3 = float(spec.get("k3", 0.0))
    notes = []
    if spec["mode"] == "wilson":
        n_loop, n_radii = int(spec.get("wn", 120)), int(spec.get("wnr", 30))
        radius = float(spec.get("wr", 0.0)) or None
        with HamProbe(fitter, 2 * n_loop * n_radii + 2, progress, cancelled, "Wilson loops"):
            fig, _, res = fitter.compute_wilson_loops_sk(k3=k3, radius=radius, n_loop=n_loop, n_radii=n_radii,
                                                         save_fig=png, save_data=dat, show=False)
        _close(fig)
        v = res["valleys"]
        summary = [("loops", f"{n_radii} circles up to r = {res['r_max']:.4g} 1/Å, {res['n_loop']} points each"),
                   ("γ(K), occupied fitted bands (largest loop)", _pi(v["K"]["occupied"][-1])),
                   ("γ(K'), occupied fitted bands (largest loop)", _pi(v[KP]["occupied"][-1])),
                   ("gap at K / K'", f"{v['K']['gap']:.4g} eV / {v[KP]['gap']:.4g} eV")]
        for c, i in enumerate(res["bands"]):
            summary.append((f"γ(K) / γ(K'), band {i}", f"{_pi(v['K']['band'][-1, c])} / {_pi(v[KP]['band'][-1, c])}"))
        notes += res["notes"]
        if res["gapless"]:
            notes.append("The gap closes at " + " and ".join(res["gapless"]) + ": the Berry phase is π on every loop "
                         "while Ω(k) is zero everywhere else - the whole flux of a valley is a δ-function at the Dirac point "
                         "(P·T symmetry). A phase is defined modulo 2π, so +π and −π are the same number here; break "
                         "inversion (a sublattice potential, e.g. graphene on hBN or hBN) to give the valleys opposite signs.")
        else:
            notes.append("γ(r) is the Berry flux through the disk of radius r (mod 2π, unwrapped along r). Opposite signs "
                         "at K and K' with zero total is the valley Hall situation of a crystal without inversion.")
        return summary, notes, "bc_wilson"
    nk = int(spec.get("nk", 60))
    with HamProbe(fitter, nk * nk + 4, progress, cancelled, "Chern numbers"):
        fig, _, res = fitter.compute_chern_numbers_sk(nk=nk, k3=k3, save_fig=png, save_data=dat, show=False)
    _close(fig)
    o = res["occupied"]
    bad = " (not defined: the bands touch)"
    summary = [("mesh", f"{nk}×{nk} (Fukui-Hatsugai-Suzuki plaquettes)"),
               ("Chern number C, occupied fitted bands", f"{o['C']:+.4f}" + (bad if res["occupied_touching"] else "")),
               ("valley Chern C_K / C_K', occupied", f"{o['C_K']:+.4f} / {o['C_Kp']:+.4f}"),
               ("valley contrast C_K − C_K', occupied", f"{o['C_K'] - o['C_Kp']:+.4f}"),
               ("gap at K / K'", f"{res['gap_K']:.4g} eV / {res['gap_Kp']:.4g} eV"),
               ("smallest gap above the occupied bands", f"{res['occupied_gap']:.4g} eV")]
    for i, d in res["per_band"].items():
        summary.append((f"band {i}: C  (C_K, C_K')", f"{d['C']:+.4f}  ({d['C_K']:+.4f}, {d['C_Kp']:+.4f})"
                        + (bad if i in res["touching"] else "")))
    if res["occupied_touching"] or res["touching"]:
        notes.append("Some bands touch (at K/K' for gapless graphene, or Kramers / spin-degenerate pairs): a Chern number "
                     "needs an isolated band (set), so those values are not meaningful (they depend on the mesh). The "
                     "Berry flux of a touching point is a δ-function - see the Wilson loops for its π phase.")
    if res["coarse"]:
        notes.append("Some plaquettes carry a flux close to ±π: the curvature varies on the scale of the mesh. Increase "
                     "the grid nk until the numbers stop changing.")
    if not res["occupied_touching"] and abs(o["C"]) < 0.05 and abs(o["C_K"]) > 0.01:
        notes.append("C = 0 with C_K = −C_K': a valley Hall insulator. C_K tends to ±½ only when the gap is small against "
                     "the bandwidth (massive Dirac cone); a large gap spreads the curvature over the zone and |C_K| < ½.")
    return summary, notes, "bc_chern"


def _run_wcc(fitter, spec, png, dat, cancelled):
    """Hybrid Wannier centres of a plane, its Chern number and Z2 index (skfitter.properties, calculator
    geometric.wcc) on the SK model of the fit: (summary, notes, data, figure). Plot and data go to png / dat."""
    from .model import SKModel
    from .properties.core.runner import run as run_calculator
    model = SKModel.from_fitter(fitter)
    params = {k: spec[k] for k in ("fixed", "fixed_value", "loop", "n_loop", "n_scan", "full", "n_occupied")}
    out = run_calculator(model, "geometric.wcc", params, cancelled=cancelled)
    d = out["data"]
    scan, wcc = np.asarray(d["scan"], float), np.asarray(d["wcc"], float)
    scan_axis = ({"k1", "k2", "k3"} - {spec["fixed"], spec["loop"]}).pop()
    head = (f"hybrid Wannier centres (fractions of a{spec['loop'][1]}), {spec['fixed']} = {float(spec['fixed_value']):g}, "
            f"loops along {spec['loop']}; Z2 = {d['z2']}" + (f", Chern = {d['chern']:+.3f}" if d.get("chern") is not None else "")
            + f"\n{scan_axis} " + " ".join(f"x_{n}" for n in range(wcc.shape[1])))
    np.savetxt(dat, np.column_stack([scan, wcc]), fmt="%.8f", header=head, encoding="utf-8")
    fig = Figure(figsize=(6.4, 4.6))
    with styled():
        ax = fig.add_subplot(111)
        for n in range(wcc.shape[1]):
            ax.plot(scan, wcc[:, n], "o", ms=2.2, color=INK)
        ax.set_xlim(scan[0], scan[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"reduced {scan_axis}")
        ax.set_ylabel(f"centre x (fraction of a{spec['loop'][1]})")
        title = f"{spec['fixed']} = {float(spec['fixed_value']):g}: Z2 = {d['z2']}"
        if d.get("chern") is not None:
            title += f", C = {d['chern']:+.2f}"
        ax.set_title(title, loc="left")
        ax.grid(True)
        fit_layout(fig)
    fig.savefig(png, dpi=150)
    summary = [tuple(r) for r in out["summary"]]
    return summary, list(out["notes"]), d, fig


def run_property(fitter, spec, project_dir, seed, progress, log, cancelled) -> dict:
    """Run one property calculation with the SKFitter methods and write plot + data into the project folder.

    Returns {'tag','label','fig','png','dat','table','summary','notes','spec','elapsed'}.
    """
    kind, src = spec["kind"], spec["source"]
    if src != "sk":
        raise ValueError("Properties are computed from the SK Hamiltonian (source 'sk').")
    tag = property_tag(spec)
    png = os.path.join(project_dir, f"{seed}_{tag}.png")
    dat = os.path.join(project_dir, f"{seed}_{tag}.dat")
    t0 = time.time()
    notes, summary, table_kind = [], [], kind
    fig, res = None, None
    with captured(log):
        # ------------------------------------------------------------------ Berry curvature
        if kind == "bc" and spec["mode"] in BC_VALLEY_MODES:
            if src != "sk":
                raise ValueError("Wilson loops and Chern numbers are computed from the SK Hamiltonian only.")
            summary, notes, table_kind = _run_valley_topology(fitter, spec, png, dat, progress, cancelled)
        elif kind == "bc":
            comp, mode = spec["component"], spec["mode"]
            n_k = (len(fitter._dense_kpath(float(spec["dk"]))["cart"]) if mode == "path" else int(spec["nk"]) ** 2)
            table_kind = "bc_path" if mode == "path" else "bc_map"
            common = dict(mode=mode, component=comp, dk=float(spec["dk"]), nk=int(spec["nk"]), k3=float(spec["k3"]),
                          ylim=spec.get("ylim"), save_fig=png, save_data=dat, show=False,
                          band_offset=spec.get("band_offset"))
            with HamProbe(fitter, 5 * n_k, progress, cancelled, "Berry curvature"):
                fig, _, res = fitter.plot_berry_curvature_sk(**common)
            omega = res["omega"]
            summary = [("k-points", f"{len(omega)}"),
                       ("fitted bands", f"{res['bands'][0]}-{res['bands'][-1]} of {fitter.total_dim}"),
                       ("occupied fitted bands", ", ".join(map(str, res["occupied_bands"])) or "none"),
                       ("max |Ω| (all fitted bands, summed)", f"{np.max(np.abs(res['total'])):.4g} Å²"),
                       ("max |Ω| (occupied, summed)", f"{np.max(np.abs(res['total_occupied'])):.4g} Å²")]
        # ------------------------------------------------------------------ spin Berry curvature
        elif kind == "sbc":
            table_kind = "sbc_path"
            n_k = len(fitter._dense_kpath(float(spec["dk"]))["cart"])
            a, b = spec["alpha"], spec["beta"]
            with HamProbe(fitter, (3 if a == b else 5) * n_k, progress, cancelled, "Spin Berry curvature"):
                fig, _, res = fitter.plot_spin_berry_curvature_sk(
                    spin=spec["spin"], alpha=a, beta=b, dk=float(spec["dk"]), efermi=float(spec.get("efermi", 0.0)),
                    smearing=float(spec.get("smearing", 0.0)), units=spec.get("units", "au"),
                    ylim=spec.get("ylim"), band_offset=spec.get("band_offset"), save_fig=png, save_data=dat, show=False)
            unit = "a.u." if res["units"] == "a.u." else "Å²"
            tot = np.asarray(res["total"])
            imax = int(np.argmax(np.abs(tot)))
            bands = res["bands"]
            summary = [("k-points", f"{len(tot)}"),
                       ("fitted bands", f"{bands[0]}-{bands[-1]} of {fitter.total_dim}"
                        + (" (Kramers pairs summed)" if res["paired"] else "")),
                       ("E_F / kT", f"{res['efermi']:g} eV / {res['smearing']:g} eV"),
                       ("max |Σ f·Ω| (all bands)", f"{tot[imax]:.4g} {unit} at k = {res['x'][imax]:.4f} 1/Å"),
                       ("max |Ω| of a fitted band", f"{np.max(np.abs(res['omega'])):.4g} {unit}")]
            if res.get("negligible"):
                notes.append("The spin Berry curvature vanishes within numerical noise (no spin-orbit coupling, or a "
                             "symmetry-forbidden component).")
            notes.append("Integrated over the Brillouin zone, the Fermi-sea sum (top panel) gives the spin Hall "
                         "conductivity. Sharp peaks sit where spin-orbit coupling splits bands that would otherwise "
                         "touch; a peak narrower than the path spacing can be missed - reduce the spacing to resolve it.")
        # ------------------------------------------------------------------ anomalous Hall
        elif kind == "ahc":
            nk = tuple(int(v) for v in spec["nk"])
            comp = spec.get("component", "z")
            efr = (float(spec["emin"]), float(spec["emax"]))
            nE = int(spec["num_efermi"])
            bands = None if spec.get("bands", "all") == "all" else "fitted"
            common = dict(nk=nk, component=comp, efermi_range=efr, num_efermi=nE, smearing=float(spec["smearing"]),
                          refine=bool(spec.get("refine", True)), save_fig=png, save_data=dat, show=False, bands=bands)
            with HamProbe(fitter, 5 * int(np.prod(nk)) * 1.15, progress, cancelled, "Anomalous Hall conductivity"):
                fig, _, res = fitter.compute_anomalous_hall_conductivity(**common)
            E, sig = res["efermi"], res["sigma"]
            units = res["units"]
            summary = _sigma_summary(E, sig, units, "σ")
            summary.append(("k-mesh / smearing", f"{nk[0]}×{nk[1]}×{nk[2]}, kT = {spec['smearing']} eV"))
        # ------------------------------------------------------------------ spin Hall
        elif kind == "shc":
            nk = tuple(int(v) for v in spec["nk"])
            efr = (float(spec["emin"]), float(spec["emax"]))
            nE = int(spec["num_efermi"])
            if src == "sk":
                a, b = spec["alpha"], spec["beta"]
                per_k = 3 if a == b else 5
                with HamProbe(fitter, per_k * int(np.prod(nk)) * 1.15, progress, cancelled, "Spin Hall conductivity"):
                    fig, _, res = fitter.compute_spin_hall_conductivity(
                        nk=nk, spin=spec["spin"], alpha=a, beta=b, efermi_range=efr, num_efermi=nE,
                        smearing=float(spec["smearing"]), refine=bool(spec.get("refine", True)),
                        save_fig=png, save_data=dat, show=False)
                summary = _sigma_summary(res["efermi"], res["sigma"], res["units"], "σ")
                summary.append(("k-mesh / smearing", f"{nk[0]}×{nk[1]}×{nk[2]}, kT = {spec['smearing']} eV"))
        # ------------------------------------------------------------------ SHC vs temperature
        elif kind == "shc_T":
            nk = tuple(int(v) for v in spec["nk"])
            temps = [float(t) for t in spec["temperatures"]]
            per_k = 3 if spec["alpha"] == spec["beta"] else 5
            with HamProbe(fitter, per_k * int(np.prod(nk)) * 1.15 * int(spec["n_shifts"]), progress, cancelled,
                          "Spin Hall conductivity vs temperature"):
                fig, _, res = fitter.compute_spin_hall_conductivity_vs_temperature(
                    temperatures=temps, nk=nk, spin=spec["spin"], alpha=spec["alpha"], beta=spec["beta"],
                    efermi=float(spec.get("efermi", 0.0)), n_shifts=int(spec["n_shifts"]), refine=bool(spec.get("refine", True)),
                    save_fig=png, save_data=dat, show=False)
            T, s = res["temperatures"], res["sigma"]
            err = res["sigma_err"]
            summary = [(f"σ(T = {T[0]:g} K)", f"{s[0]:.4g} ± {err[0]:.2g} (ħ/e)·S/cm" if np.isfinite(err[0]) else f"{s[0]:.4g}"),
                       (f"σ(T = {T[-1]:g} K)", f"{s[-1]:.4g} ± {err[-1]:.2g} (ħ/e)·S/cm" if np.isfinite(err[-1]) else f"{s[-1]:.4g}"),
                       ("k-meshes", f"{res['n_shifts']} of {nk[0]}×{nk[1]}×{nk[2]}")]
            notes.append("A temperature dependence smaller than the error bars is not resolved at this k-mesh.")
        # ------------------------------------------------------------------ hybrid Wannier centres, Chern, Z2
        elif kind == "wcc":
            summary, notes, res, fig = _run_wcc(fitter, spec, png, dat, cancelled)
            table_kind = "wcc"
        else:
            raise ValueError(f"unknown property {kind!r}")
    if fig is not None:
        _close(fig)
    table = table_from_dat(dat, table_kind) if os.path.isfile(dat) else {"headers": [], "rows": [], "n_total": 0}
    return {"tag": tag, "label": property_label(spec), "fig": fig, "png": png,
            "dat": dat if os.path.isfile(dat) else None, "table": table, "summary": summary, "notes": notes,
            "spec": spec, "elapsed": time.time() - t0, "res": res}


def _draw_sigma(fig: Figure, E, sigma, title, label, colour):
    with styled():
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(E, sigma, color=colour, lw=1.8, label=label)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8, ls=(0, (4, 3)), label="E$_F$ (DFT)")
        i0 = int(np.argmin(np.abs(np.asarray(E))))
        ax.plot([E[i0]], [sigma[i0]], "o", color=INK, ms=5, zorder=5, label=f"value at E$_F$ = {sigma[i0]:.4g}")
        ax.set_xlabel("Fermi energy E (eV)")
        ax.set_xlim(E[0], E[-1])
        ax.grid(True, axis="y")
        ax.legend(loc="best")
        ax.set_title(title, loc="left")
        fit_layout(fig)




PROPERTY_TAG_RE = re.compile(r"^(?P<src>sk|wb|wk)_(?P<what>sbc_path|bc_path|bc_map|bc_wilson|bc_chern|ahc|shcT|shc|wcc|bands)"
                             r"(?:_(?P<rest>.+))?$")


def scan_results(project_dir, seed) -> list:
    """Property plots already stored in the folder: [{'tag','label','png','dat','kind'}]."""
    out = []
    for p in sorted(glob.glob(os.path.join(glob.escape(project_dir), f"{glob.escape(seed)}_*.png")), key=os.path.getmtime):
        tag = os.path.basename(p)[len(seed) + 1:-4]
        m = PROPERTY_TAG_RE.match(tag)
        if not m:
            continue
        kind = {"bc_path": "bc_path", "bc_map": "bc_map", "bc_wilson": "bc_wilson", "bc_chern": "bc_chern", "ahc": "ahc", "shcT": "shc_T", "shc": "shc",
                "sbc_path": "sbc_path", "wcc": "wcc", "bands": "bands"}[m.group("what")]
        d = p[:-4] + ".dat"
        rest = (m.group("rest") or "").replace("_", " ")
        label = f"{PROPERTY_TITLES.get(kind, kind)} - {SOURCE_TITLES[m.group('src')]}" + (f" ({rest})" if rest else "")
        out.append({"tag": tag, "label": label, "png": p, "dat": d if os.path.isfile(d) else None, "kind": kind})
    return out


FIT_FILE_SUFFIXES = ("_fit.json", "_fit.txt", "_params.json", "_params.txt")   # = SKFitter.PARAM_FILE_SUFFIXES


def find_saved_fits(project_dir: str) -> list:
    """Seednames that have a saved fit (SKFitter parameter file) in the folder, newest first."""
    found = {}
    for suf in FIT_FILE_SUFFIXES:
        for p in glob.glob(os.path.join(glob.escape(project_dir), f"*{suf}")):
            seed = os.path.basename(p)[: -len(suf)]
            found[seed] = max(found.get(seed, 0.0), os.path.getmtime(p))
    return [s for s, _ in sorted(found.items(), key=lambda kv: -kv[1])]


PROJECT_FILE = "skfitter_project.json"


def write_project_file(project_dir, **fields):
    path = os.path.join(project_dir, PROJECT_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {"created": datetime.datetime.now().isoformat(timespec="seconds")}
    data.update(fields)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return data


def read_project_file(project_dir) -> dict:
    try:
        with open(os.path.join(project_dir, PROJECT_FILE), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _copy_input(src, dst, progress):
    """Copy one input file into the project folder (skipped if it is the same file, or an identical copy)."""
    if not src:
        return
    if os.path.exists(dst) and os.path.samefile(src, dst):
        return
    if os.path.exists(dst) and not src.endswith(".gz"):                        # same file picked again: keep the copy (and the DFT cache stamp)
        a, b = os.stat(src), os.stat(dst)
        if a.st_size == b.st_size and a.st_mtime_ns == b.st_mtime_ns:
            return
    if src.endswith(".gz") and not dst.endswith(".gz"):
        import gzip
        if os.path.exists(dst) and os.stat(dst).st_mtime_ns == os.stat(src).st_mtime_ns:
            return
        progress(-1, f"Unpacking {os.path.basename(src)} ({os.path.getsize(src) / 1e6:.1f} MB) ...")
        with gzip.open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        os.replace(dst + ".part", dst)
        st = os.stat(src)
        os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
        return
    progress(-1, f"Copying {os.path.basename(src)} ({os.path.getsize(src) / 1e6:.1f} MB) ...")
    shutil.copy2(src, dst)                         # copy2 keeps the modification time, so the copy can be recognised


_LOCKS_HELD = set()


@contextlib.contextmanager
def project_lock(project_dir, timeout=3600.0, stale=7200.0):
    """Hold <project_dir>/.skfitter.lock while the DFT inputs are copied and read (other processes wait). Re-entrant
    within a process; a lock file older than ``stale`` seconds (a crashed run) is taken over."""
    os.makedirs(project_dir, exist_ok=True)
    path = os.path.join(os.path.abspath(project_dir), ".skfitter.lock")
    if path in _LOCKS_HELD:
        yield
        return
    t0 = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > stale:
                    os.remove(path)
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout:
                raise RuntimeError(f"{path} is held by another skfitter run; remove it if no run is active")
            time.sleep(0.5)
    _LOCKS_HELD.add(path)
    try:
        yield
    finally:
        _LOCKS_HELD.discard(path)
        try:
            os.remove(path)
        except OSError:
            pass


def prepare_project(project_dir, code, sources, progress, log) -> dict:
    """Copy the DFT inputs into the project folder and load the data (one process at a time: project_lock)."""
    with project_lock(project_dir):
        return _prepare_project(project_dir, code, sources, progress, log)


def _prepare_project(project_dir, code, sources, progress, log) -> dict:
    """Copy the DFT inputs (sources: role -> path, see PROJECT_FILES) into the project folder and load the data.

    Optional Quantum ESPRESSO files that are left empty are removed from the project copy, so a stale file of an
    earlier pick is not used."""
    os.makedirs(project_dir, exist_ok=True)
    names = PROJECT_FILES[code]
    for role, name in names.items():
        src = (sources.get(role) or "").strip()
        dst = os.path.join(project_dir, name)
        if src:
            _copy_input(src, dst, progress)
            if role == "qe_projwfc":               # spin-down projections of an LSDA run sit next to the up file
                down = re.sub(r"up(\.gz)?$", r"down\1", src)
                if down != src and os.path.isfile(down):
                    _copy_input(down, os.path.join(project_dir, "qe_projwfc_down"), progress)
        elif role not in REQUIRED_FILES[code] and os.path.isfile(dst):
            _unlink_quiet(dst)
            if role == "qe_projwfc":
                _unlink_quiet(os.path.join(project_dir, "qe_projwfc_down"))
    src = project_sources(project_dir, code)
    cache = os.path.join(project_dir, "dft_cache.npz")
    if code == "espresso":
        stamp_paths, kpath_file = [src["qe_xml"], src["qe_scf"], src["qe_projwfc"]], src["qe_bands_in"]
    else:
        stamp_paths, kpath_file = [src["vasprun"]], src["kpoints"]
    dft = load_dft_cache(cache, stamp_paths, kpath_file, code)
    if dft is None:
        if code == "espresso":
            progress(-1, "Reading the Quantum ESPRESSO XML file (eigenvalues) ...")
            with captured(log):
                dft = load_dft_espresso(src["qe_xml"], src["qe_bands_in"], src["qe_scf"], src["qe_projwfc"], log=log)
        else:
            progress(-1, "Reading vasprun.xml (eigenvalues and orbital projections) ...")
            with captured(log):
                dft = load_dft(src["vasprun"], src["kpoints"], log=log)
        try:
            save_dft_cache(cache, dft)
        except OSError as exc:
            log(f"[skfitter] Could not write the DFT cache: {exc}")
    else:
        log("[skfitter] DFT data read from dft_cache.npz")
    return dft
