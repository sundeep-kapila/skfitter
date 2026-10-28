"""Starting values and bounds for an SKFitter fit from published tight-binding parameter sets.

SKFitter fits on-site energies ``e_orbit<eq>_<El>_<sub>``, two-centre integrals ``v_<type>_0`` (one value per orbital
pair, shared by all bonds, at the nearest-neighbour distance d0), overlaps ``s_<type>_0`` and decay exponents
``eta_<type>``. A good starting point saves many least-squares starts. This module reads four kinds of data sets and
turns them into starting values (and bounds around them) for the parameters of a model:

  kind        what it gives                                                   units in the source
  ----------  --------------------------------------------------------------  ---------------------------
  "dftb"      DFTB Slater-Koster tables (DFTB.org mio / 3ob / pbc / matsci,     Hartree, bohr (.skf files)
              or any .skf files): on-site energies of the free (confined)
              atom, H and S integrals as functions of distance
  "nist"      NIST Atomic Reference Data for Electronic Structure              Hartree
              Calculations: LDA / scalar-relativistic LDA orbital energies
              of the neutral atoms H-U (s, p, d and f): on-site only
  "harrison"  Harrison's universal matrix elements V = eta hbar^2/(m d^2)     eV, Angstrom
              for s and p (and, with the admin's r_d table, s-d, p-d, d-d)
              with the NIST on-site energies
  "nrl"       NRL tight-binding parameter files (Mehl & Papaconstantopoulos):   Rydberg, bohr (par files)
              environment-dependent on-site energies, H and S integrals

All energies returned are in eV and distances in Angstrom. Only numpy is needed (the web server, the workers and the
desktop program all use this module); the bundled data files are in ``skfitter_param_sets/``.

How the numbers are mapped onto a model (``starting_values``):
  * on-site: the set's value for the element and orbital family (s, p, d; NRL also t2g / eg); s* sits 6 eV above s;
  * integrals: the set's value for every nearest-neighbour bond type that carries the orbital pair, averaged
    (SKFitter has one value per orbital pair); with more than one neighbour shell the value is referred to d0 with
    the model's (d0/r)^eta law; eta itself from the slope of the set's integral;
  * the set's energies are atomic (or have their own zero): all on-site energies are shifted by one constant so that
    they match the DFT band centres of the orbitals (projection-weighted mean energies) as well as possible; without
    projections, so that the highest occupied atomic level sits 1 eV below the Fermi level;
  * bounds around each value; parameters the set does not cover keep the defaults.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

HARTREE_EV = 27.211386245988
RYDBERG_EV = HARTREE_EV / 2.0
BOHR_A = 0.529177210903
HBAR2_M = 7.62                     # hbar^2 / m_e in eV Angstrom^2 (Harrison)
FORMAT = 1
KINDS = ("dftb", "nist", "harrison", "nrl")
FAMILIES = ("s", "p", "d", "f")

# The ten two-centre integrals SKFitter uses (s, p, d), in the Slater-Koster convention (first orbital on the first
# atom, direction from the first atom to the second).
INTEGRALS = ("sss", "sps", "pps", "ppp", "sds", "pds", "pdp", "dds", "ddp", "ddd",
             "sfs", "pfs", "pfp", "dfs", "dfp", "dfd", "ffs", "ffp", "ffd", "fff")
_L = {"s": 0, "p": 1, "d": 2, "f": 3}

# .skf column order (Hamiltonian and overlap each): standard (s, p, d) and extended (with f) format
SKF_COLUMNS = ("dds", "ddp", "ddd", "pds", "pdp", "pps", "ppp", "sds", "sps", "sss")
SKF_COLUMNS_F = ("ffs", "ffp", "ffd", "fff", "dfs", "dfp", "dfd", "dds", "ddp", "ddd",
                 "pfs", "pfp", "pds", "pdp", "pps", "ppp", "sfs", "sds", "sps", "sss")
DFTB_GRID = (1.0, 0.05, 81)          # the bundled tables: r = 1.0 ... 5.0 Angstrom in steps of 0.05

# Harrison's universal coefficients (Harrison, Elementary Electronic Structure, 1999; as tabulated in L. Shi and
# D. A. Papaconstantopoulos, Phys. Rev. B 70, 205101 (2004), Table I). s-p: V = eta hbar^2/(m d^2); s-d, p-d:
# V = eta hbar^2 r_d^(3/2)/(m d^(7/2)); d-d: V = eta hbar^2 r_d^3/(m d^5).
HARRISON_ETA = {"sss": -1.32, "sps": 1.42, "pps": 2.22, "ppp": -0.63,
                "sds": -3.16, "pds": -2.95, "pdp": 1.36, "dds": -16.2, "ddp": 8.75, "ddd": -2.39}

# SKFitter key -> (integral, family on the first atom, family on the second)
_KEY_INTEGRAL = {
    "sss": "sss", "sps": "sps", "spzs": "sps", "spxs": "sps",
    "pps": "pps", "ppp": "ppp", "pxpxs": "pps", "pxpxp": "ppp", "pzpzs": "pps", "pzpzp": "ppp",
    "sds": "sds", "pds": "pds", "pdp": "pdp", "dds": "dds", "ddp": "ddp", "ddd": "ddd",
    "sfs": "sfs", "pfs": "pfs", "pfp": "pfp", "dfs": "dfs", "dfp": "dfp", "dfd": "dfd",
    "ffs": "ffs", "ffp": "ffp", "ffd": "ffd", "fff": "fff",
}
_SUB_FAMILY = {"s": "s", "s*": "s*", "px": "p", "py": "p", "pz": "p", "pxy": "p", "p": "p",
               "dxy": "d", "dyz": "d", "dzx": "d", "dx2y2": "d", "dz2": "d",
               "d_e1": "d", "d_e2": "d", "d_t2g": "d", "d_eg": "d", "d": "d",
               "fy3x2": "f", "fxyz": "f", "fyz2": "f", "fz3": "f", "fxz2": "f", "fzx2": "f", "fx3": "f",
               "f_m0": "f", "f_m1": "f", "f_m2": "f", "f_m3": "f", "f_a2u": "f", "f_t1u": "f", "f_t2u": "f", "f": "f"}
_T2G = {"dxy", "dyz", "dzx", "d_t2g"}
_EG = {"dx2y2", "dz2", "d_eg"}
_ONSITE_KEY = re.compile(r"^e_orbit(\d+)_([A-Za-z]+)_(.+)$")

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
           "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U").split()
Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}
_NAMES = {"hydrogen": "H", "helium": "He", "lithium": "Li", "beryllium": "Be", "boron": "B", "carbon": "C",
          "nitrogen": "N", "oxygen": "O", "fluorine": "F", "neon": "Ne", "sodium": "Na", "magnesium": "Mg",
          "aluminum": "Al", "aluminium": "Al", "silicon": "Si", "phosphorus": "P", "sulfur": "S", "chlorine": "Cl"}

BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "param_sets")


class ParamSetError(ValueError):
    """A parameter file or set that cannot be read (the message says why, for the admin page)."""


def usual_sign(integral: str) -> float:
    """The sign a two-centre integral V_{l1 l2 m} (l1 <= l2) usually has with SKFitter's orbitals: facing lobes of
    the two orbitals have the signs +1 and (-1)^(l2 + m), and H ~ -K S, so sign = (-1)^(l2 + m + 1)
    (ss-sigma -, sp-sigma +, pp-pi -, dd-delta -, ... - the signs of Harrison's coefficients)."""
    l2, m = _L[integral[1]], {"s": 0, "p": 1, "d": 2, "f": 3}[integral[2]]
    return -1.0 if (l2 + m + 1) % 2 else 1.0


def integral_key(key: str) -> Optional[str]:
    """'v_Mo_pz-S_d_pds_0' / 's_C_p-C_p_pps_0' / 'eta_C_p-C_p_pps' / 'eta_s_...' -> the integral it stands for
    ('pds', 'pps'; a group pair: the standard integral it splits); None for integrals no data set has (s*)."""
    p = integral_parts(key)
    return None if p is None else _KEY_INTEGRAL.get(p[2])


def integral_parts(key: str):
    """(element 1, element 2, label) of an integral, overlap or decay key in key order ('v_Ir_s-Mn_d_sds_1' ->
    ('Ir', 'Mn', 'sds')); None for other keys."""
    from skfitter import core as sk
    m = re.match(r"^(?:eta_s_|eta_)([A-Z].*)$", key)
    p = sk.parse_integral_key(f"v_{m.group(1)}" if m else key)
    if p is None:
        return None
    return p["el1"], p["el2"], p["label"]


def period(Z: int) -> int:
    for n, last in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if Z <= last:
            return n
    return 7


# =====================================================================================================================
#  Reading source files
# =====================================================================================================================
def _numbers(line: str) -> List[float]:
    """The numbers on one .skf line: separators are blanks and commas; 'n*x' means n copies of x."""
    out = []
    for tok in line.replace(",", " ").split():
        if "*" in tok:
            n, _, v = tok.partition("*")
            try:
                out.extend([float(v)] * int(n))
            except ValueError:
                return out
        else:
            try:
                out.append(float(tok))
            except ValueError:
                return out                     # e.g. the 'T 2 ...' flags at the end of some lines
    return out


def parse_skf(text: str, homonuclear: bool) -> dict:
    """One .skf file -> {'r': Angstrom, 'H': {integral: eV array}, 'S': {integral: array}, 'onsite': {family: eV}}.

    Format (DFTB+ "Slater-Koster files"): line 1 grid spacing and number of points (bohr); homonuclear files then have
    the on-site energies (Ed Ep Es SPE Ud Up Us fd fp fs; extended format with f: a leading '@' line and Ef first),
    then the mass / repulsive-polynomial line; then one line per grid point r = i * spacing with 10 (or 20) H and
    10 (or 20) S integrals; a 'Spline' section follows (ignored)."""
    lines = [ln for ln in text.splitlines()]
    extended = bool(lines) and lines[0].lstrip().startswith("@")
    if extended:
        lines = lines[1:]
    cols = SKF_COLUMNS_F if extended else SKF_COLUMNS
    try:
        head = _numbers(lines[0])
        dr, n = float(head[0]), int(head[1])
    except (IndexError, ValueError):
        raise ParamSetError("not a Slater-Koster (.skf) file: the first line must hold the grid spacing and size") from None
    if not (0 < dr < 1.0 and 10 <= n <= 100000):
        raise ParamSetError(f"implausible grid ({dr} bohr x {n} points)")
    onsite = {}
    start = 1
    if homonuclear:
        e = _numbers(lines[1])
        fams = ("f", "d", "p", "s") if extended else ("d", "p", "s")
        if len(e) < len(fams):
            raise ParamSetError("the on-site energies (line 2) are missing")
        for fam, v in zip(fams, e[:len(fams)]):
            if v != 0.0:
                onsite[fam] = v * HARTREE_EV
        start = 3
    else:
        start = 2
    rows = []
    for ln in lines[start:]:
        if ln.strip().lower().startswith("spline"):
            break
        vals = _numbers(ln)
        if len(vals) >= 2 * len(cols):
            rows.append(vals[:2 * len(cols)])
        if len(rows) == n:
            break
    if len(rows) < min(n, 20):
        raise ParamSetError(f"only {len(rows)} of {n} table lines could be read")
    tab = np.array(rows, dtype=float)
    r = (np.arange(len(rows)) + 1) * dr * BOHR_A
    H = {c: tab[:, i] * HARTREE_EV for i, c in enumerate(cols)}
    S = {c: tab[:, len(cols) + i] for i, c in enumerate(cols)}
    return {"r": r, "H": H, "S": S, "onsite": onsite}


def skf_pair_name(filename: str) -> Optional[Tuple[str, str]]:
    """'C-H.skf' -> ('C', 'H'); None for other files (and for variants such as 'H-N_short.skf')."""
    m = re.match(r"^([A-Z][a-z]?)-([A-Z][a-z]?)\.skf$", os.path.basename(filename))
    return (m.group(1), m.group(2)) if m else None


def dftb_set_from_files(files: Dict[str, str], meta: dict, grid=DFTB_GRID) -> dict:
    """A 'dftb' set from .skf file texts {filename: text}; the tables are resampled on ``grid`` (start, step, n in
    Angstrom) and only integrals that are not identically zero are kept."""
    r0, dr, n = grid
    rg = r0 + dr * np.arange(n)
    onsite, pairs, skipped = {}, {}, []
    flipped = 0
    for name, text in sorted(files.items()):
        ab = skf_pair_name(name)
        if ab is None:
            skipped.append(os.path.basename(name))
            continue
        a, b = ab
        for el in (a, b):
            if el not in Z_OF:
                raise ParamSetError(f"{os.path.basename(name)}: '{el}' is not an element symbol")
        try:
            t = parse_skf(text, homonuclear=(a == b))
        except ParamSetError as exc:
            raise ParamSetError(f"{os.path.basename(name)}: {exc}") from None
        if a == b and t["onsite"]:
            onsite[a] = {fam: round(v, 5) for fam, v in t["onsite"].items()}
        entry = {"H": {}, "S": {}}
        for integral in INTEGRALS:
            h, s = t["H"].get(integral), t["S"].get(integral)
            if h is None or not np.any(h != 0.0):
                continue
            # The sign of each atomic basis function is a free choice of the programme that made the table (3ob and
            # matsci flip some); bring every integral to the usual Slater-Koster sign: the overlap, where it is
            # largest, has the opposite sign of Harrison's coefficient (H ~ -K S), e.g. S_sp_sigma < 0, V_sp_sigma > 0.
            sign = 1.0
            invariant = a == b and integral[0] == integral[1]      # the same function on both ends: no choice
            if s is not None and np.any(s != 0.0) and not invariant:
                window = (t["r"] >= rg[0]) & (t["r"] <= rg[-1])
                sw = np.where(window, s, 0.0)
                if np.sign(sw[int(np.argmax(np.abs(sw)))]) == usual_sign(integral):
                    sign = -1.0
                    flipped += 1
            for part, col in (("H", h), ("S", s)):
                if col is None or not np.any(col != 0.0):
                    continue
                v = sign * np.interp(rg, t["r"], col, left=np.nan, right=0.0)
                entry[part][integral] = [None if not np.isfinite(x) else float(f"{x:.6g}") for x in v]
        pairs[f"{a}-{b}"] = entry
    if not pairs:
        raise ParamSetError("no Slater-Koster files (named like 'C-H.skf') were found")
    out = {**_meta_defaults(meta, "dftb"), "grid": [r0, dr, n], "onsite": onsite, "pairs": pairs}
    if flipped:
        out.setdefault("import_notes", []).append(f"{flipped} integral tables brought to the usual sign convention")
    if skipped:
        out.setdefault("import_notes", []).append(f"not used: {', '.join(skipped[:20])}"
                                                  + (" …" if len(skipped) > 20 else ""))
    return finalize(out)


def parse_nist_raw(text: str) -> dict:
    """The bundled NIST table ('Z|configuration|1s=LDA,ScRLDA;2s=...', Hartree) -> {El: {'conf', 'orbitals'}}."""
    atoms = {}
    for no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            z, conf, orbs = line.split("|")
            el = SYMBOLS[int(z) - 1]
            levels = {}
            for item in orbs.split(";"):
                name, vals = item.split("=")
                lda, sc = (float(x) for x in vals.split(","))
                levels[name.strip()] = [lda, sc]
        except (ValueError, IndexError):
            raise ParamSetError(f"line {no}: expected 'Z|configuration|1s=LDA,ScRLDA;...'") from None
        atoms[el] = {"conf": conf.strip(), "orbitals": levels}
    if not atoms:
        raise ParamSetError("no elements in the table")
    return atoms


def nist_valence(el: str, orbitals: dict, column: int = 1) -> dict:
    """The valence levels of an element (eV): s and p of the outer shell n (the period), d of shell n-1, f of n-2 -
    only those occupied in the neutral atom (the table has no empty levels). column 1: scalar-relativistic LDA."""
    Z = Z_OF[el]
    n = period(Z)
    want = {"s": n, "p": n, "d": n - 1, "f": n - 2}
    out = {}
    for fam, shell in want.items():
        v = orbitals.get(f"{shell}{fam}")
        if v is not None and shell >= 1:
            out[fam] = round(float(v[column]) * HARTREE_EV, 4)
    return out


def occupied_families(conf: str) -> set:
    """'[Ar] 3d^6 4s^2' -> {'d', 's'}: the families with electrons outside the noble-gas core."""
    return {m.group(1) for m in re.finditer(r"\d([spdf])\^?\d*", conf.split("]")[-1])}


def nist_set(raw_text: str, meta: dict) -> dict:
    atoms = parse_nist_raw(raw_text)
    onsite = {el: nist_valence(el, a["orbitals"]) for el, a in atoms.items()}
    return finalize({**_meta_defaults(meta, "nist"), "onsite": onsite, "atoms": atoms})


def harrison_set(nist: dict, meta: dict, rd: Optional[dict] = None, eta: Optional[dict] = None) -> dict:
    """Harrison's universal integrals with the on-site energies of a NIST set; ``rd`` (element -> r_d in Angstrom)
    enables the s-d, p-d and d-d integrals."""
    return finalize({**_meta_defaults(meta, "harrison"), "onsite": dict(nist.get("onsite") or {}),
                     "atoms": nist.get("atoms") or {},
                     "harrison": {"eta": dict(eta or HARRISON_ETA), "hbar2_m": HBAR2_M,
                                  "rd": {k: float(v) for k, v in (rd or {}).items()}}})


def parse_harrison_table(text: str) -> Tuple[dict, dict]:
    """Harrison's table as CSV (or blank-separated): ``element, r_d`` or, with a header line, any of the columns
    element, r_d (Angstrom), e_s, e_p, e_d, e_f (term values, eV). Returns ({El: r_d}, {El: {family: eV}})."""
    rows = [[p for p in re.split(r"[,;\t ]+", ln.strip()) if p] for ln in text.splitlines()]
    rows = [r for r in rows if r and not r[0].startswith("#")]
    if not rows:
        raise ParamSetError("the table is empty")
    cols = ["element", "r_d"]
    if rows[0][0] not in Z_OF:                              # a header line
        alias = {"el": "element", "symbol": "element", "rd": "r_d", "es": "e_s", "ep": "e_p", "ed": "e_d",
                 "ef": "e_f"}
        cols = [alias.get(c.lower().replace("(a)", "").replace("(ev)", "").strip("_ "), c.lower())
                for c in rows[0]]
        rows = rows[1:]
        unknown = [c for c in cols if c not in ("element", "r_d", "e_s", "e_p", "e_d", "e_f")]
        if unknown or "element" not in cols:
            raise ParamSetError(f"unknown column(s) {', '.join(unknown) or '(no element column)'}: use element, "
                                "r_d, e_s, e_p, e_d, e_f")
    rd, onsite = {}, {}
    for no, r in enumerate(rows, start=2):
        rec = dict(zip(cols, r))
        el = rec.get("element")
        if el not in Z_OF:
            raise ParamSetError(f"row {no - 1}: '{el}' is not an element symbol")
        for c, v in rec.items():
            if c == "element" or v in ("", "-", "NA", "na"):
                continue
            try:
                x = float(v)
            except ValueError:
                raise ParamSetError(f"row {no - 1} ({el}): {c} = '{v}' is not a number") from None
            if c == "r_d":
                if not 0.05 < x < 5.0:
                    raise ParamSetError(f"row {no - 1}: r_d = {x} Angstrom for {el} is implausible")
                rd[el] = x
            else:
                if not -100 < x < 20:
                    raise ParamSetError(f"row {no - 1}: {c} = {x} eV for {el} is implausible")
                onsite.setdefault(el, {})[c[2]] = x
    if not rd and not onsite:
        raise ParamSetError("no values found")
    return rd, onsite


# -- NRL tight-binding parameter files ---------------------------------------------------------------------------------
_NRL_H = re.compile(r"^(e|f|fbar|g)_\{(\w\w) (sigma|pi|delta)\}.*\((Hamiltonian|Overlap)\)")
_NRL_ON = re.compile(r"^([abcd])_(s|p|t2g|eg|d)\b")
_MSYM = {"sigma": "s", "pi": "p", "delta": "d"}


def parse_nrl(text: str) -> dict:
    """A single-element NRL-TB parameter file (e.g. si_par.spd) -> {element, rcut, screenl (bohr), lambda,
    onsite {fam: [a, b, c, d]}, H / S {integral: [e, f, fbar, g]}, overlap_style} (Rydberg / bohr).

    Layout: a header 'NN0000x' (x = 1: new overlap form for like atoms), a title naming the element, the number of
    atom types, 'RCUT SCREENL', orbitals, mass, valence occupancy, then one labelled parameter per line
    (lambda, a_s ... d_eg, e_{ss sigma} ... (Hamiltonian), ... (Overlap))."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 10 or not lines[0].strip().upper().startswith("NN"):
        raise ParamSetError("not an NRL tight-binding parameter file (the first line starts with 'NN000...')")
    style = lines[0].split()[0].upper()
    title = lines[1]
    m = re.search(r"\(([A-Z][a-z]?)\)", title)
    el = m.group(1) if m else next((s for name, s in _NAMES.items() if name in title.lower()), None)
    if el not in Z_OF:
        raise ParamSetError(f"cannot tell the element from the title line '{title.strip()}'")
    try:
        ntypes = int(lines[2].split()[0])
        rcut, screenl = (float(x) for x in lines[3].split()[:2])
    except (ValueError, IndexError):
        raise ParamSetError("lines 3-4 must hold the number of atom types and 'RCUT SCREENL'") from None
    if ntypes != 1:
        raise ParamSetError(f"files with {ntypes} atom types are not supported yet - import the single-element files")
    out = {"element": el, "title": title.strip(), "style": style, "rcut": rcut, "screenl": screenl,
           "lambda": None, "onsite": {}, "H": {}, "S": {}}
    for ln in lines[4:]:
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            val = float(parts[0].replace("D", "E").replace("d", "e"))
        except ValueError:
            continue
        label = parts[3].strip()
        if label.startswith("lambda"):
            out["lambda"] = val
            continue
        mo = _NRL_ON.match(label)
        if mo:
            fam = mo.group(2)
            out["onsite"].setdefault(fam, [0.0, 0.0, 0.0, 0.0])["abcd".index(mo.group(1))] = val
            continue
        mh = _NRL_H.match(label)
        if mh:
            coef, pair, sym, part = mh.groups()
            integral = pair + _MSYM[sym]
            if integral in INTEGRALS:
                tgt = out["H" if part == "Hamiltonian" else "S"].setdefault(integral, [0.0, 0.0, 0.0, 0.0])
                tgt[("e", "f", "fbar", "g").index(coef)] = val
    if out["lambda"] is None or not out["onsite"] or not out["H"]:
        raise ParamSetError("the lambda, on-site (a_s ...) or Hamiltonian (e_{ss sigma} ...) parameters are missing")
    return out


def nrl_set(entries: Iterable[dict], meta: dict) -> dict:
    return finalize({**_meta_defaults(meta, "nrl"), "nrl": {e["element"]: e for e in entries}})


def _texts(files: Dict[str, bytes]) -> Dict[str, str]:
    """Uploaded files -> {name: text}; zip archives are unpacked (their folders are ignored)."""
    import zipfile
    out = {}
    for name, blob in files.items():
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    for info in zf.infolist():
                        if info.is_dir() or info.file_size > 20_000_000:
                            continue
                        out[os.path.basename(info.filename)] = zf.read(info).decode("utf-8", errors="replace")
            except zipfile.BadZipFile:
                raise ParamSetError(f"{name}: not a valid zip archive") from None
        else:
            out[os.path.basename(name)] = blob.decode("utf-8", errors="replace")
    return out


def import_files(data: dict, files: Dict[str, bytes], replace: bool = False) -> dict:
    """A set ``data`` updated with uploaded source files (the kind decides what they are):
      dftb      .skf files (or a zip of them): pairs and on-site energies added / replaced
      nist      the NIST table in the bundled text format ('Z|configuration|1s=LDA,ScRLDA;...')
      harrison  a CSV of element, r_d (and optionally term values e_s, e_p, e_d, e_f in eV)
      nrl       NRL parameter files (single element; or a zip of them)
    ``replace``: start from an empty set of this kind instead of adding to it. Raises ParamSetError."""
    kind = data.get("kind")
    texts = _texts(files)
    if not texts:
        raise ParamSetError("no files")
    meta = {k: data.get(k) for k in ("id", "name", "description", "source", "url", "licence", "citation",
                                     "version", "notes", "enabled")}
    notes: List[str] = []
    if kind == "dftb":
        new = dftb_set_from_files(texts, meta)
        base = {"onsite": {}, "pairs": {}} if replace else data
        out = {**data, "grid": new["grid"], "onsite": {**(base.get("onsite") or {}), **new["onsite"]},
               "pairs": {**(base.get("pairs") or {}), **new["pairs"]}}
        notes.append(f"{len(new['pairs'])} pair tables and on-site energies of {len(new['onsite'])} elements read")
        notes += new.get("import_notes", [])
    elif kind == "nist":
        if len(texts) != 1:
            raise ParamSetError("upload one NIST table file")
        new = nist_set(next(iter(texts.values())), meta)
        base_on, base_at = ({}, {}) if replace else (data.get("onsite") or {}, data.get("atoms") or {})
        out = {**data, "onsite": {**base_on, **new["onsite"]}, "atoms": {**base_at, **new["atoms"]}}
        notes.append(f"{len(new['atoms'])} elements read")
    elif kind == "harrison":
        rd, onsite = {}, {}
        for name, text in texts.items():
            try:
                r, o = parse_harrison_table(text)
            except ParamSetError as exc:
                raise ParamSetError(f"{name}: {exc}") from None
            rd.update(r)
            for el, v in o.items():
                onsite.setdefault(el, {}).update(v)
        h = dict(data.get("harrison") or {"eta": dict(HARRISON_ETA), "hbar2_m": HBAR2_M})
        h["rd"] = {**({} if replace else (h.get("rd") or {})), **rd}
        base_on = {} if replace else {k: dict(v) for k, v in (data.get("onsite") or {}).items()}
        for el, v in onsite.items():
            base_on.setdefault(el, {}).update(v)
        out = {**data, "harrison": h, "onsite": base_on}
        notes.append(f"r_d of {len(rd)} and term values of {len(onsite)} elements read")
    elif kind == "nrl":
        entries, skipped = [], []
        for name, text in texts.items():
            if not text.lstrip().upper().startswith("NN"):
                skipped.append(name)
                continue
            try:
                entries.append(parse_nrl(text))
            except ParamSetError as exc:
                raise ParamSetError(f"{name}: {exc}") from None
        if not entries:
            raise ParamSetError("no NRL parameter files (they start with 'NN000...') were found")
        base = {} if replace else dict(data.get("nrl") or {})
        base.update({e["element"]: e for e in entries})
        out = {**data, "nrl": base}
        notes.append(f"parameters of {', '.join(e['element'] for e in entries)} read"
                     + (f"; not used: {', '.join(skipped)}" if skipped else ""))
    else:
        raise ParamSetError(f"unknown kind '{kind}'")
    out["import_notes"] = notes
    return finalize(out)


def edit_onsite(data: dict, changes: Dict[str, Dict[str, Optional[float]]]) -> dict:
    """On-site energies edited by hand: {El: {family: eV or None (remove)}} (not for NRL sets)."""
    if data.get("kind") == "nrl":
        raise ParamSetError("NRL on-site energies depend on the structure: import corrected parameter files instead")
    onsite = {k: dict(v) for k, v in (data.get("onsite") or {}).items()}
    for el, fams in (changes or {}).items():
        if el not in Z_OF:
            raise ParamSetError(f"'{el}' is not an element symbol")
        for fam, v in (fams or {}).items():
            if fam not in FAMILIES:
                raise ParamSetError(f"{el}: '{fam}' is not an orbital family (s, p, d, f)")
            if v is None:
                onsite.get(el, {}).pop(fam, None)
            else:
                v = float(v)
                if not (math.isfinite(v) and -200 < v < 50):
                    raise ParamSetError(f"{el} {fam}: {v} eV is implausible")
                onsite.setdefault(el, {})[fam] = round(v, 5)
        if el in onsite and not onsite[el]:
            del onsite[el]
    return finalize({**data, "onsite": onsite})


def empty_set(kind: str, meta: dict) -> dict:
    base = {"dftb": {"grid": list(DFTB_GRID), "onsite": {}, "pairs": {}}, "nist": {"onsite": {}, "atoms": {}},
            "harrison": {"onsite": {}, "harrison": {"eta": dict(HARRISON_ETA), "hbar2_m": HBAR2_M, "rd": {}}},
            "nrl": {"nrl": {}}}
    if kind not in base:
        raise ParamSetError(f"unknown kind '{kind}' (one of {', '.join(KINDS)})")
    return finalize({**_meta_defaults(meta, kind), **base[kind]})


def preview(data: dict, a: str, b: str, r: float) -> dict:
    """What the admin page shows for one pair: on-site energies of a and b, and every integral (H, S, eta) at r."""
    ps = ParamSet(data)
    env_rho = {}
    if ps.kind == "nrl":                            # a dimer-free guess: 12 neighbours at r (for the on-site shown)
        for el in {a, b}:
            env_rho[el] = ps.density(el, [(el, r, 12)])
    fams = ("s", "p", "d", "f")
    onsite = {el: {f: ps.onsite(el, f, env_rho.get(el)) for f in fams} for el in (a, b)}
    rows = []
    for name in INTEGRALS:
        h = ps.integral(a, b, name, r)
        s = ps.integral(a, b, name, r, overlap=True)
        e = ps.decay(a, b, name, r)
        if h is None and s is None:
            continue
        rows.append({"integral": name, "H": h, "S": s, "eta": e})
    return {"a": a, "b": b, "r": r, "onsite": onsite, "integrals": rows,
            "onsite_note": "NRL on-site energies shown for 12 neighbours at r" if ps.kind == "nrl" else ""}


# =====================================================================================================================
#  Set files: metadata, loading, saving
# =====================================================================================================================
def _meta_defaults(meta: dict, kind: str) -> dict:
    m = {"format": FORMAT, "kind": kind, "id": "", "name": "", "description": "", "source": "", "url": "",
         "licence": "", "citation": "", "version": "", "notes": ""}
    m.update({k: v for k, v in (meta or {}).items() if v is not None})
    m["kind"] = kind
    return m


def elements_of(data: dict) -> List[str]:
    kind = data.get("kind")
    els = set(data.get("onsite") or {})
    if kind == "dftb":
        for p in data.get("pairs") or {}:
            els.update(p.split("-"))
    if kind == "nrl":
        els.update(data.get("nrl") or {})
    return sorted(els, key=lambda e: Z_OF.get(e, 999))


def finalize(data: dict) -> dict:
    """Fill the derived fields (elements, what the set can give) and check the kind."""
    if data.get("kind") not in KINDS:
        raise ParamSetError(f"unknown kind '{data.get('kind')}' (one of {', '.join(KINDS)})")
    data["elements"] = elements_of(data)
    kind = data["kind"]
    data["provides"] = {
        "onsite": kind in ("dftb", "nist", "harrison", "nrl"),
        "hopping": kind in ("dftb", "harrison", "nrl"),
        "overlap": kind in ("dftb", "nrl"),
    }
    return data


def load_file(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") != FORMAT:
        raise ParamSetError(f"{os.path.basename(path)}: unsupported format {data.get('format')}")
    return finalize(data)


def dump_bytes(data: dict, compress: bool = True) -> bytes:
    raw = json.dumps(data, separators=(",", ":"), sort_keys=False).encode()
    if not compress:
        return raw
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:   # mtime=0: the same data gives the same bytes
        gz.write(raw)
    return buf.getvalue()


def save_file(data: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(dump_bytes(finalize(data), compress=path.endswith(".gz")))
    os.replace(tmp, path)


def bundled_sets(folder: str = BUNDLED_DIR) -> Dict[str, dict]:
    """The sets shipped with the program: {id: set} (read from ``*.json.gz`` in the data folder)."""
    out = {}
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json.gz") or name.endswith(".json"):
                d = load_file(os.path.join(folder, name))
                out[d["id"]] = d
    return out


def summary(data: dict) -> dict:
    """What a list of sets shows: no tables."""
    keep = ("id", "name", "kind", "description", "source", "url", "licence", "citation", "version", "notes",
            "elements", "provides", "enabled", "origin", "updated", "updated_by")
    s = {k: data.get(k) for k in keep if k in data}
    s.setdefault("enabled", True)
    return s


# =====================================================================================================================
#  Evaluating a set
# =====================================================================================================================
def cutoff_nrl(r_bohr, rcut, screenl):
    """The NRL smooth cutoff F(R) = 1 / (1 + exp((R - Rc)/l + 5)), zero beyond Rc."""
    r = np.asarray(r_bohr, float)
    with np.errstate(over="ignore"):
        f = 1.0 / (1.0 + np.exp((r - rcut) / screenl + 5.0))
    return np.where(r < rcut, f, 0.0)


class ParamSet:
    """Values of one set: ``onsite(el, sub, rho)``, ``integral(a, b, name, r)``, ``overlap(...)``, ``decay(...)``."""

    def __init__(self, data: dict):
        self.data = finalize(data)
        self.kind = data["kind"]
        self.id = data.get("id", "")
        if self.kind == "dftb":
            r0, dr, n = data["grid"]
            self._r = r0 + dr * np.arange(int(n))

    @property
    def elements(self) -> List[str]:
        return self.data["elements"]

    # -- on-site ---------------------------------------------------------------------------------------------------
    def density(self, el: str, neighbours) -> Optional[float]:
        """NRL: the atom's 'density' rho = sum_j exp(-lambda^2 R_j) F(R_j) over neighbours [(el_j, R Angstrom, n)]."""
        p = (self.data.get("nrl") or {}).get(el)
        if p is None:
            return None
        rho = 0.0
        for elj, r, n in neighbours:
            q = (self.data.get("nrl") or {}).get(elj, p)
            rb = r / BOHR_A
            rho += n * math.exp(-(q["lambda"] ** 2) * rb) * float(cutoff_nrl(rb, q["rcut"], q["screenl"]))
        return rho

    def onsite(self, el: str, sub: str, rho: Optional[float] = None) -> Optional[float]:
        """On-site energy (eV) of orbital ``sub`` (an SKFitter sub-shell: s, p, pz, d_t2g, ...) of element ``el``."""
        fam = _SUB_FAMILY.get(sub, sub)
        if fam == "s*":
            return None
        if self.kind == "nrl":
            p = (self.data.get("nrl") or {}).get(el)
            if p is None or rho is None:
                return None
            on = p["onsite"]

            def h(key):
                a, b, c, d = on[key]
                return (a + b * rho ** (2 / 3) + c * rho ** (4 / 3) + d * rho ** 2) * RYDBERG_EV
            if fam in ("s", "p"):
                return h(fam) if fam in on else None
            if fam == "d":
                t2g = h("t2g") if "t2g" in on else (h("d") if "d" in on else None)
                eg = h("eg") if "eg" in on else t2g
                if t2g is None:
                    return None
                if sub in _T2G:
                    return t2g
                if sub in _EG:
                    return eg
                return (3 * t2g + 2 * eg) / 5
            return None
        v = (self.data.get("onsite") or {}).get(el, {}).get(fam)
        return None if v is None else float(v)

    # -- two-centre integrals --------------------------------------------------------------------------------------
    def integral(self, a: str, b: str, name: str, r: float, overlap: bool = False) -> Optional[float]:
        """The integral ``name`` (e.g. 'sps': s on atom a, p on atom b) at distance r (Angstrom): eV, or the overlap."""
        if name not in INTEGRALS and (name[1] + name[0] + name[2:]) not in INTEGRALS:
            return None
        l1, l2 = _L[name[0]], _L[name[1]]
        if l1 > l2:          # stored with the lower l on the first atom: V_{l2 l1 m}(B->A) = (-1)^(l1+l2) V_{l1 l2 m}
            v = self.integral(b, a, name[1] + name[0] + name[2], r, overlap)
            return None if v is None else v * (-1) ** (l1 + l2)
        if self.kind == "dftb":
            entry = (self.data.get("pairs") or {}).get(f"{a}-{b}")
            col = None if entry is None else entry["S" if overlap else "H"].get(name)
            if col is None:
                return None
            arr = np.array([np.nan if x is None else x for x in col], float)
            if not (self._r[0] <= r <= self._r[-1]):
                return None
            v = float(np.interp(r, self._r, arr))
            return None if not np.isfinite(v) or v == 0.0 else v
        if self.kind == "harrison":
            if overlap:
                return None
            h = self.data["harrison"]
            eta, c, rd = h["eta"].get(name), h["hbar2_m"], h.get("rd") or {}
            if eta is None:
                return None
            if "d" not in name[:2]:
                return eta * c / r ** 2
            if name[0] == "d" and name[1] == "d":
                if a not in rd or b not in rd:
                    return None
                return eta * c * (rd[a] * rd[b]) ** 1.5 / r ** 5
            dat = b                                 # s-d / p-d: the d orbital is on atom b
            if dat not in rd:
                return None
            return eta * c * rd[dat] ** 1.5 / r ** 3.5
        if self.kind == "nrl":
            nrl = self.data.get("nrl") or {}
            if a != b:                              # unlike atoms: from the two like-atom sets (sign-preserving mean)
                va, vb = self.integral(a, a, name, r, overlap), self.integral(b, b, name, r, overlap)
                if va is None or vb is None or va * vb <= 0:
                    return None
                return math.copysign(math.sqrt(va * vb), va)
            p = nrl.get(a)
            if p is None:
                return None
            coef = p["S" if overlap else "H"].get(name)
            if coef is None:
                return None
            e, f, fb, g = coef
            rb = r / BOHR_A
            fc = float(cutoff_nrl(rb, p["rcut"], p["screenl"]))
            if overlap and p["style"].endswith("1"):         # new overlap form for like atoms
                poly = (1.0 if name[0] == name[1] else 0.0) + e * rb + f * rb ** 2 + fb * rb ** 3
            else:
                poly = e + f * rb + fb * rb ** 2
            v = poly * math.exp(-(g ** 2) * rb) * fc
            return v if overlap else v * RYDBERG_EV
        return None

    def decay(self, a: str, b: str, name: str, r: float, overlap: bool = False) -> Optional[float]:
        """eta = -d ln|V| / d ln r at r (the exponent of a local (d0/r)^eta law)."""
        v1, v2 = self.integral(a, b, name, r * 0.97, overlap), self.integral(a, b, name, r * 1.03, overlap)
        if v1 is None or v2 is None or v1 == 0 or v2 == 0 or v1 * v2 < 0:
            return None
        return -math.log(abs(v2) / abs(v1)) / math.log(1.03 / 0.97)


# =====================================================================================================================
#  The structure around each atom (for averaging over bonds and for the NRL densities)
# =====================================================================================================================
def neighbour_shells(lattice, symbols: List[str], frac, cutoff: float = 9.0, digits: int = 3) -> List[list]:
    """For each site: [[element, distance (Angstrom), count], ...] of all neighbours within ``cutoff``."""
    L = np.asarray(lattice, float)
    F = np.asarray(frac, float) % 1.0
    cart = F @ L
    inv = np.linalg.inv(L)
    # how many cells in each direction cover the cutoff: the distance between opposite faces
    heights = 1.0 / np.linalg.norm(inv, axis=0)
    nmax = [int(math.ceil(cutoff / h)) + 1 for h in heights]
    shifts = np.array([[i, j, k] for i in range(-nmax[0], nmax[0] + 1) for j in range(-nmax[1], nmax[1] + 1)
                       for k in range(-nmax[2], nmax[2] + 1)], float) @ L
    out = []
    for i, ci in enumerate(cart):
        acc: Dict[Tuple[str, float], int] = {}
        for j, cj in enumerate(cart):
            d = np.linalg.norm(cj + shifts - ci, axis=1)
            for x in d[(d > 1e-4) & (d <= cutoff)]:
                key = (symbols[j], round(float(x), digits))
                acc[key] = acc.get(key, 0) + 1
        out.append(sorted([[e, r, n] for (e, r), n in acc.items()], key=lambda t: (t[1], t[0])))
    return out


def model_environment(lattice, symbols, frac, eq_map, orbital_config, d0, n_neighbors, fit_eta, fixed_eta,
                      non_orthogonal, centres=None, hopping_mode="decay", shell_distances=None,
                      pair_shells=None) -> dict:
    """What ``starting_values`` needs to know about a built model (all JSON-friendly). ``pair_shells``: the neighbour
    shells of each element pair ({'Ir-Mn': [r0, r1, ...]}): the shell index of a per-shell integral and the
    reference length r0 of a pair's decay law."""
    return {"hopping_mode": str(hopping_mode or "decay"),
            "pair_shells": {str(k): [float(x) for x in v] for k, v in (pair_shells or {}).items()},
            "shell_distances": [float(x) for x in (shell_distances or [])],"lattice": np.asarray(lattice, float).tolist(), "symbols": list(symbols),
            "frac": np.asarray(frac, float).tolist(), "eq": [int(x) for x in eq_map],
            "orbitals": {k: list(v) for k, v in orbital_config.items()}, "d0": float(d0),
            "n_neighbors": int(n_neighbors), "fit_eta": bool(fit_eta), "fixed_eta": float(fixed_eta),
            "non_orthogonal": bool(non_orthogonal), "centres": dict(centres or {})}


# =====================================================================================================================
#  Starting values for a model
# =====================================================================================================================
def _families(subs: Iterable[str]) -> set:
    return {_SUB_FAMILY.get(s, s) for s in subs}


def _hop_bounds(v: float) -> Tuple[float, float]:
    w = max(1.0, 0.6 * abs(v))
    return v - w, v + w


def _overlap_bounds(v: float) -> Tuple[float, float]:
    w = max(0.1, 0.5 * abs(v))
    return max(-0.95, v - w), min(0.95, v + w)


def starting_values(data: dict, env: dict, keys: List[str], fallback_initial: dict, fallback_bounds: dict,
                    nist_atoms: Optional[dict] = None, below_ef: float = 1.0) -> dict:
    """Starting values and bounds for the parameters ``keys`` of a model described by ``env`` (see
    ``model_environment``) from the set ``data``. Parameters the set does not give keep ``fallback_*``.

    Returns {initial, bounds, source: {key: 'set' | 'default'}, shift (eV), alignment, coverage, notes}."""
    ps = ParamSet(data)
    name = data.get("name") or data.get("id")
    initial = {k: float(fallback_initial[k]) for k in keys if k in fallback_initial}
    bounds = {k: (float(fallback_bounds[k][0]), float(fallback_bounds[k][1])) for k in keys if k in fallback_bounds}
    source = {k: "default" for k in keys}
    notes: List[str] = []
    symbols, eq = env["symbols"], env["eq"]
    orbitals = env["orbitals"]
    d0 = float(env["d0"])
    scaled = int(env.get("n_neighbors", 1)) > 1
    fixed_eta = float(env.get("fixed_eta", 2.0))

    shells = None
    if ps.kind == "nrl":
        shells = neighbour_shells(env["lattice"], symbols, env["frac"], cutoff=9.0)

    # -- on-site energies (before the shift) ----------------------------------------------------------------------
    raw_onsite: Dict[str, float] = {}
    for k in keys:
        m = _ONSITE_KEY.match(k)
        if not m:
            continue
        eqid, el, sub = int(m.group(1)), m.group(2), m.group(3)
        if sub == "s*":
            continue
        rho = None
        if ps.kind == "nrl":
            sites = [i for i, s in enumerate(symbols) if s == el and eq[i] == eqid] or \
                    [i for i, s in enumerate(symbols) if s == el]
            dens = [ps.density(el, [(e, r, n) for e, r, n in shells[i]]) for i in sites]
            dens = [x for x in dens if x is not None]
            rho = float(np.mean(dens)) if dens else None
        v = ps.onsite(el, sub, rho)
        if v is not None:
            raw_onsite[k] = v

    # -- one constant shift: the set's energy zero -> the DFT one ----------------------------------------------------
    centres = {k: float(v) for k, v in (env.get("centres") or {}).items() if k in raw_onsite}
    shift, alignment = 0.0, "none"
    if centres:
        shift = float(np.mean([centres[k] - raw_onsite[k] for k in centres]))
        alignment = "dft"
        notes.append(f"On-site energies shifted by {shift:+.2f} eV to match the DFT band centres of "
                     f"{len(centres)} orbital{'s' if len(centres) != 1 else ''}.")
    elif raw_onsite:
        occ = []
        for k, v in raw_onsite.items():
            m = _ONSITE_KEY.match(k)
            el, fam = m.group(2), _SUB_FAMILY.get(m.group(3))
            conf = ((nist_atoms or data.get("atoms") or {}).get(el) or {}).get("conf", "")
            if not conf or fam in occupied_families(conf):
                occ.append(v)
        top = max(occ) if occ else max(raw_onsite.values())
        shift = -below_ef - top
        alignment = "fermi"
        notes.append(f"No orbital projections in the DFT data: on-site energies shifted by {shift:+.2f} eV so that "
                     f"the highest occupied atomic level is {below_ef:g} eV below the Fermi level.")
    s_values = {}
    for k, v in raw_onsite.items():
        val = v + shift
        initial[k] = val
        bounds[k] = (val - 4.0, val + 4.0)
        source[k] = "set"
        m = _ONSITE_KEY.match(k)
        if m.group(3) == "s":
            s_values[(m.group(1), m.group(2))] = val
    for k in keys:                                          # s*: 6 eV above s
        m = _ONSITE_KEY.match(k)
        if m and m.group(3) == "s*" and (m.group(1), m.group(2)) in s_values:
            val = s_values[(m.group(1), m.group(2))] + 6.0
            initial[k], bounds[k], source[k] = val, (val - 4.0, val + 6.0), "set"

    # -- integrals: average over the nearest-neighbour bonds that carry the orbital pair ------------------------------
    if ps.kind != "nist":
        from skfitter.core import element_pair
        fams = {el: _families(subs) for el, subs in orbitals.items()}
        pair_shells = env.get("pair_shells") or {}

        def pair_d0(a, b):
            sh = pair_shells.get(element_pair(a, b))
            return float(sh[0]) if sh else d0

        reach = max([v[0] for v in pair_shells.values()] or [d0])
        near = []                       # (el_i, el_j, r, count) of each element pair's nearest bonds (both directions)
        sh = neighbour_shells(env["lattice"], symbols, env["frac"], cutoff=reach * 1.1 + 0.05)
        for i, lst in enumerate(sh):
            for elj, r, n in lst:
                if r <= pair_d0(symbols[i], elj) * 1.08:
                    near.append((symbols[i], elj, r, n))

        shell_mode = env.get("hopping_mode") == "shell" and pair_shells
        by_shell = {}                   # shell mode: the bonds of shell n of each element pair, {(pair, n): [...]}
        if shell_mode:
            far = max(max(v) for v in pair_shells.values())
            sh_all = neighbour_shells(env["lattice"], symbols, env["frac"], cutoff=far + 0.05)
            for i, lst in enumerate(sh_all):
                for elj, r, n in lst:
                    pk = element_pair(symbols[i], elj)
                    dist = pair_shells.get(pk)
                    if not dist:
                        continue
                    k = int(np.argmin([abs(r - x) for x in dist]))
                    if abs(r - dist[k]) < 0.01:
                        by_shell.setdefault((pk, k), []).append((symbols[i], elj, r, n))

        def average(el1: str, el2: str, integral: str, overlap: bool, shell=None):
            """The set's integral between el1 (first family of ``integral``) and el2, averaged over that pair's
            nearest bonds (decay law: moved to the pair's d0 with the set's eta) or the bonds of the pair's shell."""
            fa, fb = integral[0], integral[1]
            vals, wts, etas = [], [], []
            bonds = near if shell is None else by_shell.get((element_pair(el1, el2), shell), [])
            for a, b, r, n in bonds:
                if a != el1 or b != el2 or fa not in fams.get(a, ()) or fb not in fams.get(b, ()):
                    continue
                v = ps.integral(a, b, integral, r, overlap)
                if v is None:
                    continue
                e = ps.decay(a, b, integral, r, overlap)
                if scaled and shell is None:                # decay law: the value at the pair's d0; per shell: at r_n
                    v = v * (r / pair_d0(a, b)) ** (e if (env.get("fit_eta") and e is not None) else fixed_eta)
                vals.append(v)
                wts.append(n)
                if e is not None:
                    etas.append(e)
            if not vals:
                return None, None
            return float(np.average(vals, weights=wts)), (float(np.mean(etas)) if etas else None)

        cache = {}
        for k in keys:
            integral = integral_key(k)
            if integral is None:
                continue
            el1, el2, _ = integral_parts(k)
            if k.startswith("v_") or k.startswith("s_"):
                ov = k.startswith("s_")
                shell = int(k.rsplit("_", 1)[1]) if shell_mode and re.search(r"_\d+$", k) else None
                ck = (el1, el2, integral, ov, shell)
                if ck not in cache:
                    cache[ck] = average(el1, el2, integral, ov, shell)
                v, _ = cache[ck]
                if v is None:
                    continue
                initial[k] = v
                bounds[k] = _overlap_bounds(v) if ov else _hop_bounds(v)
                source[k] = "set"
            elif k.startswith("eta_"):
                ov = k.startswith("eta_s_")
                ck = (el1, el2, integral, ov, None)
                if ck not in cache:
                    cache[ck] = average(el1, el2, integral, ov)
                _, e = cache[ck]
                if e is None or not np.isfinite(e):
                    continue
                e = float(np.clip(e, 0.5, 8.0))
                initial[k], bounds[k], source[k] = e, (max(0.5, e - 1.5), e + 1.5), "set"
        if not near:
            notes.append("No nearest-neighbour bonds found - the integrals keep their defaults.")

    for k in keys:                                      # every start inside its bounds
        if k in initial and k in bounds:
            lo, hi = bounds[k]
            if lo > hi:
                lo, hi = hi, lo
            initial[k] = float(np.clip(initial[k], lo, hi))
            bounds[k] = (round(lo, 6), round(hi, 6))
            initial[k] = round(initial[k], 6)

    groups = {"onsite": [k for k in keys if k.startswith("e_orbit")],
              "hopping": [k for k in keys if k.startswith("v_")],
              "overlap": [k for k in keys if k.startswith("s_")],
              "decay": [k for k in keys if k.startswith("eta_")]}
    coverage = {g: [sum(source[k] == "set" for k in ks), len(ks)] for g, ks in groups.items() if ks}
    missing = sorted({_ONSITE_KEY.match(k).group(2) for k in groups["onsite"] if source[k] != "set"})
    if missing:
        notes.append(f"{name} has no on-site energies for {', '.join(missing)} (or those orbitals) - "
                     "they start from the DFT band centres.")
    if ps.kind == "nist":
        notes.append("This set gives on-site energies only; the integrals keep their defaults.")
    if ps.kind == "harrison" and any(source[k] != "set" for k in groups["hopping"]):
        notes.append("Harrison's s-d, p-d and d-d integrals need an r_d table (Admin → Parameter sets); "
                     "without it they keep their defaults.")
    return {"set": data.get("id"), "name": name, "initial": initial,
            "bounds": {k: list(v) for k, v in bounds.items()}, "source": source, "shift": round(shift, 4),
            "alignment": alignment, "coverage": coverage, "notes": notes,
            "citation": data.get("citation") or "", "licence": data.get("licence") or ""}
