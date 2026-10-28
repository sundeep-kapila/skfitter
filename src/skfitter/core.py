"""
skfitter.core: the SKFitter class (Slater-Koster tight-binding fit to DFT bands from VASP or Quantum ESPRESSO).

    from skfitter.core import SKFitter

    fitter = SKFitter(vasprun_path="pt/vasprun.xml", kpoints_path="pt/KPOINTS",
                      orbital_config={"Pt": ["s", "p", "d"]}, n_bands_below=5, n_bands_above=1,
                      n_neighbors=2, include_soc=True)
    fitter.fit_or_load("pt_spd", "pt/", initial_params=..., custom_bounds=...)

Quantum ESPRESSO bands (pw.x >= 6.2) are read the same way; pass dft_code="espresso" and the files of the
calculation='bands' run instead of vasprun.xml / KPOINTS:

    fitter = SKFitter(dft_code="espresso", qe_xml_path="pt_qe/pt.xml",          # <outdir>/<prefix>.xml of the bands run
                      qe_bands_input="pt_qe/bands.in",                         # its pw.x input (K_POINTS crystal_b)
                      qe_scf_path="pt_qe/scf.out",                             # optional: E_F of the SCF run
                      orbital_config={"Pt": ["s", "p", "d"]}, ...)

The ``skfitter`` command (skfitter.cli) runs these steps from a TOML configuration.
"""
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import differential_evolution, least_squares
from scipy.linalg import eigh
import json
import os
import re
import itertools

def get_element_valency_new(elem):
    group = getattr(elem, 'group', None)
    if group is None: return 'N/A'
    if 1 <= group <= 2: return group
    elif 3 <= group <= 12: return group
    elif 13 <= group <= 18: return group - 10
    return 'N/A'

def get_valence_orbitals_from_Z(atomic_number):
    Z = atomic_number
    if Z <= 2 or Z in [3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88]:
        return 's'
    elif (21 <= Z <= 30) or (39 <= Z <= 48) or (72 <= Z <= 80):
        return 'd'
    elif (5 <= Z <= 10) or (13 <= Z <= 18) or (31 <= Z <= 36) or (49 <= Z <= 54) or (81 <= Z <= 86):
        return 'p'
    else:
        return 'd'

warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
warnings.filterwarnings("ignore", message="POTCAR data with symbol")


# ==============================================================================
# DFT sources: Quantum ESPRESSO readers and k-path segments (numpy + the standard library only, so the GUI can
# use them without pymatgen)
#
# A k-path is described by a list of straight segments, each a dict
#     {"k0", "k1": end points (fractional reciprocal coordinates), "l0", "l1": labels,
#      "i0", "i1": indices of the two end points in the list of DFT k-points (inclusive)}
# VASP line mode writes every segment with both end points (i0 = s*n, i1 = (s+1)*n - 1, the junction point appears
# twice); a Quantum ESPRESSO crystal_b / tpiba_b path shares the junction point between two segments (i1 of one
# segment = i0 of the next).  A jump (e.g. 'U|K') is simply a segment that does not start where the previous ended.
# ==============================================================================
HARTREE_EV = 27.211386245988          # eV
BOHR_ANG = 0.529177210903             # Angstrom

_DFT_CODE_ALIASES = {"vasp": "vasp", "espresso": "espresso", "qe": "espresso", "quantum espresso": "espresso",
                     "quantum-espresso": "espresso", "quantumespresso": "espresso", "pwscf": "espresso",
                     "pw.x": "espresso"}

_ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
    "Lv Ts Og").split()


def normalise_dft_code(code):
    """'vasp' or 'espresso' from the names people use ('VASP', 'QE', 'Quantum ESPRESSO', 'pwscf', ...)."""
    key = str(code or "vasp").strip().lower().replace("_", " ")
    if key not in _DFT_CODE_ALIASES:
        raise ValueError(f"Unknown DFT code {code!r}: use 'vasp' or 'espresso' (Quantum ESPRESSO).")
    return _DFT_CODE_ALIASES[key]


def normalise_kpoint_label(label):
    """'G', 'Gamma', '\\Gamma', 'GAMMA' -> 'Γ'; other labels are returned stripped."""
    s = str(label or "").strip().strip("!#").strip()
    t = s.lower().replace("$", "").replace("\\", "")
    if t in ("g", "gamma", "γ", "ɣ", "gm"):
        return "Γ"
    return s


def qe_element_symbol(name, pseudo_file=None):
    """Chemical symbol of a Quantum ESPRESSO species label ('Fe1', 'Fe_up', 'fe' -> 'Fe'); the pseudopotential
    file name ('Pt.rel-pbe-...UPF') is used when the label itself is not recognisable."""
    for text in (name, os.path.basename(pseudo_file or "")):
        m = re.match(r"\s*([A-Za-z]{1,2})", text or "")
        if not m:
            continue
        cand = m.group(1)
        for sym in (cand[:1].upper() + cand[1:].lower(), cand[:1].upper()):
            if sym in _ELEMENT_SYMBOLS:
                return sym
    raise ValueError(f"Cannot work out the element of the Quantum ESPRESSO species {name!r} "
                     f"(pseudopotential {pseudo_file!r}).")


def _xml_strip_namespaces(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _xml_floats(text):
    return [float(x.replace("D", "E").replace("d", "e")) for x in (text or "").split()]


def _xml_bool(el, default=None):
    if el is None or el.text is None:
        return default
    return el.text.strip().lower() in ("true", ".true.", "t", "1")


def _qe_xml_output(path):
    import xml.etree.ElementTree as ET
    try:
        root = _xml_strip_namespaces(ET.parse(path).getroot())
    except ET.ParseError as exc:
        raise ValueError(f"{path} is not a readable XML file ({exc}).") from exc
    out = root.find("output")
    if out is None and root.find("band_structure") is not None:
        out = root
    if out is None:
        raise ValueError(f"{path} is not a Quantum ESPRESSO XML data file: there is no <output> section. Use the "
                         "<outdir>/<prefix>.xml (or <prefix>.save/data-file-schema.xml) written by pw.x >= 6.2.")
    return root, out


def _qe_fermi_from_band_structure(bs):
    """E_F (eV) from a <band_structure> element: fermi_energy, else the mean of two_fermi_energies, else the HOMO."""
    for tag in ("fermi_energy", "two_fermi_energies", "highestOccupiedLevel"):
        el = bs.find(tag)
        if el is not None and el.text and el.text.strip():
            vals = _xml_floats(el.text)
            if vals:
                return float(np.mean(vals)) * HARTREE_EV, tag
    return None, None


def read_qe_xml(path):
    """
    Read a Quantum ESPRESSO XML data file (pw.x >= 6.2: <outdir>/<prefix>.xml or <prefix>.save/data-file-schema.xml),
    normally the one written by the calculation='bands' run along the k-path.

    Returns a dict with (Angstrom, 1/Angstrom, eV; k-points and bands in the order of the file):
        lattice (3, 3) rows a1..a3, alat, symbols / species (per atom), cart_coords (n_atoms, 3),
        kpoints_cart (nk, 3) incl. the 2*pi, kpoints_frac (nk, 3), k_weights (nk,),
        eig_up (nk, nb), eig_down (nk, nb) or None (LSDA only), efermi (or None), efermi_source,
        nelec, lsda, noncolin, spinorbit, calculation, prefix
    """
    root, out = _qe_xml_output(path)
    st = out.find("atomic_structure")
    if st is None:
        raise ValueError(f"{path}: no <atomic_structure> in the output section.")
    alat = float(st.get("alat")) * BOHR_ANG
    lattice = np.array([_xml_floats(st.find(f"cell/a{i}").text) for i in (1, 2, 3)]) * BOHR_ANG

    pseudo = {}
    for sp in out.findall("atomic_species/species"):
        pf = sp.find("pseudo_file")
        pseudo[sp.get("name")] = pf.text.strip() if pf is not None and pf.text else None
    atoms = st.findall("atomic_positions/atom")
    if atoms:
        cart = np.array([_xml_floats(a.text) for a in atoms]) * BOHR_ANG
    else:
        atoms = st.findall("crystal_positions/atom")
        if not atoms:
            raise ValueError(f"{path}: no atomic positions in <atomic_structure>.")
        cart = np.array([_xml_floats(a.text) for a in atoms]) @ lattice
    species = [a.get("name") for a in atoms]
    symbols = [qe_element_symbol(s, pseudo.get(s)) for s in species]

    bs = out.find("band_structure")
    if bs is None:
        raise ValueError(f"{path}: no <band_structure> in the output section.")
    mag = out.find("magnetization")

    def flag(tag):
        v = _xml_bool(bs.find(tag))
        if v is None and mag is not None:
            v = _xml_bool(mag.find(tag))
        return bool(v)

    lsda, noncolin, spinorbit = flag("lsda"), flag("noncolin"), flag("spinorbit")
    nelec = float(_xml_floats(bs.find("nelec").text)[0]) if bs.find("nelec") is not None else None

    ks = bs.findall("ks_energies")
    if not ks:
        raise ValueError(f"{path}: no <ks_energies> (eigenvalues) in <band_structure>.")
    k_tpiba = np.array([_xml_floats(k.find("k_point").text) for k in ks], dtype=float)
    k_w = np.array([float(k.find("k_point").get("weight", "0").replace("D", "E")) for k in ks])
    eig = np.array([_xml_floats(k.find("eigenvalues").text) for k in ks], dtype=float) * HARTREE_EV

    kcart = k_tpiba * (2.0 * np.pi / alat)
    kfrac = kcart @ lattice.T / (2.0 * np.pi)          # k = kfrac @ B with B = 2 pi inv(A)^T
    eig_down = None
    if lsda:
        el = bs.find("nbnd_up")
        n_up = int(el.text) if el is not None and el.text else eig.shape[1] // 2
        eig_up, eig_down = eig[:, :n_up], eig[:, n_up:]
    else:
        eig_up = eig

    efermi, source = _qe_fermi_from_band_structure(bs)
    calc = root.find("input/control_variables/calculation")
    prefix = root.find("input/control_variables/prefix")
    return {
        "lattice": lattice, "alat": alat, "symbols": symbols, "species": species, "cart_coords": cart,
        "kpoints_cart": kcart, "kpoints_frac": kfrac, "k_weights": k_w, "eig_up": eig_up, "eig_down": eig_down,
        "efermi": efermi, "efermi_source": source, "nelec": nelec, "lsda": lsda, "noncolin": noncolin,
        "spinorbit": spinorbit, "calculation": calc.text.strip() if calc is not None and calc.text else None,
        "prefix": prefix.text.strip() if prefix is not None and prefix.text else None,
    }


def read_vasprun_efermi(path, tail_bytes=8_000_000):
    """The last <i name="efermi"> of a vasprun.xml (for files whose DOS block pymatgen cannot read, e.g. trimmed
    ones that keep only E_F there). None if there is none."""
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


def read_qe_fermi(path):
    """
    Fermi energy (eV) from a Quantum ESPRESSO SCF run: either its XML data file or the pw.x standard output
    ('the Fermi energy is ...', 'the spin up/dw Fermi energies are ...', 'highest occupied(, lowest unoccupied)
    level ...'; the last occurrence wins). Returns None if nothing is found.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if text.lstrip().startswith("<"):
        try:
            _, out = _qe_xml_output(path)
            bs = out.find("band_structure")
            return None if bs is None else _qe_fermi_from_band_structure(bs)[0]
        except ValueError:
            pass
    num = r"([-+]?\d+\.\d*(?:[eEdD][-+]?\d+)?)"
    patterns = [
        (rf"the Fermi energy is\s+{num}\s*ev", 1),
        (rf"the spin up/dw Fermi energies are\s+{num}\s+{num}", 2),
        (rf"highest occupied, lowest unoccupied level \(ev\):\s+{num}\s+{num}", 1),
        (rf"highest occupied level \(ev\):\s+{num}", 1),
    ]
    best = None                                     # (position in the file, value)
    for pat, ngroups in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            vals = [float(m.group(i + 1).replace("D", "E").replace("d", "e")) for i in range(ngroups)]
            val = float(np.mean(vals))
            if best is None or m.start() > best[0]:
                best = (m.start(), val)
    return None if best is None else best[1]


def read_qe_site_moments(path, n_sites=None, vectors=False):
    """
    The magnetic moment of every site from a pw.x SCF output (standard output, not XML), the last ones printed:

      collinear (nspin=2)   'Magnetic moment per site' blocks: 'atom   1 (R=0.290)  charge= 13.6  magn=  4.58'
      non-collinear         'atom number   1 relative position : ...' then 'magnetization :  mx  my  mz'

    For a non-collinear run the moment of each site is its component along the axis of the largest moment (z for
    moments along z), so the sign says up or down - like the z component of a VASP MAGMOM. Returns a list of floats
    in site order, or None when the file has no per-site moments (or not one per site when n_sites is given).
    vectors=True: the moment vectors instead ([0, 0, m] for a collinear run).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    if text.lstrip().startswith("<"):
        return None
    num = r"([-+]?\d+\.\d*(?:[eEdD][-+]?\d+)?)"
    moments = {}
    blocks = list(re.finditer(r"Magnetic moment per site[^\n]*\n((?:[ \t]*atom[^\n]*\n)+)", text))
    if blocks:                                        # collinear: the last block
        for m in re.finditer(rf"atom\s+(\d+)[^\n]*?magn\s*=\s*{num}", blocks[-1].group(1)):
            moments[int(m.group(1))] = float(m.group(2).replace("D", "E").replace("d", "e"))
        if vectors:
            moments = {i: np.array([0.0, 0.0, v]) for i, v in moments.items()}
    else:                                             # non-collinear: the last vector printed for each atom
        vec = {}
        for m in re.finditer(rf"atom number\s+(\d+)\s+relative position[^\n]*\n(?:[^\n]*\n){{0,3}}?"
                             rf"\s*magnetization\s*:\s*{num}\s+{num}\s+{num}", text):
            vec[int(m.group(1))] = np.array([float(m.group(i).replace("D", "E").replace("d", "e")) for i in (2, 3, 4)])
        if vec and vectors:
            moments = vec
        elif vec:
            big = max(vec.values(), key=lambda v: float(np.linalg.norm(v)))
            axis = big / np.linalg.norm(big) if np.linalg.norm(big) > 1e-8 else np.array([0.0, 0.0, 1.0])
            if abs(axis[2]) > 0.999:                  # moments along +-z: the plain z component
                axis = np.array([0.0, 0.0, 1.0])
            moments = {i: float(np.dot(v, axis)) for i, v in vec.items()}
    if not moments:
        return None
    n = max(moments) if n_sites is None else int(n_sites)
    if sorted(moments) != list(range(1, n + 1)):
        return None
    return [moments[i] for i in range(1, n + 1)]


def read_qe_kpath(path):
    """
    The K_POINTS card of a pw.x input file (normally the one of the calculation='bands' run).

    Returns {'units': 'crystal_b' | 'tpiba_b' | 'crystal' | 'tpiba' | ..., 'band_path': bool,
             'nodes': [[x, y, z], ...], 'weights': [...], 'labels': [...]}.
    Labels are read from a comment after the weight ('0.5 0.0 0.5 20 ! X' or '# X') or from a fifth column.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    card = None
    for i, line in enumerate(lines):
        m = re.match(r"\s*K_POINTS\b\s*[\{\(]?\s*([A-Za-z_]*)", line, flags=re.IGNORECASE)
        if m:
            card = (i, (m.group(1) or "tpiba").lower())
            break
    if card is None:
        raise ValueError(f"{path}: no K_POINTS card found (expected a pw.x input file).")
    start, units = card
    if units in ("automatic", "gamma"):
        return {"units": units, "band_path": False, "nodes": [], "weights": [], "labels": []}
    body = [ln for ln in lines[start + 1:] if ln.strip() and not ln.strip().startswith(("!", "#"))]
    if not body:
        raise ValueError(f"{path}: the K_POINTS card is empty.")
    nks = int(body[0].split()[0])
    nodes, weights, labels = [], [], []
    for ln in body[1:1 + nks]:
        cm = re.search(r"[!#]", ln)
        data, comment = (ln[:cm.start()], ln[cm.end():]) if cm else (ln, "")
        parts = data.split()
        try:
            xyz = [float(p.replace("D", "E").replace("d", "e")) for p in parts[:3]]
            w = float(parts[3].replace("D", "E").replace("d", "e")) if len(parts) > 3 else 1.0
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}: cannot read the K_POINTS line {ln!r}") from exc
        lab = comment.split()[0] if comment.split() else (parts[4] if len(parts) > 4 else "")
        nodes.append(xyz)
        weights.append(w)
        labels.append(normalise_kpoint_label(lab))
    if len(nodes) != nks:
        raise ValueError(f"{path}: K_POINTS announces {nks} points but lists {len(nodes)}.")
    return {"units": units, "band_path": units.endswith("_b"), "nodes": nodes, "weights": weights, "labels": labels}


def qe_kpath_segments(nodes_frac, weights, labels, n_kpoints=None):
    """
    Segments (see the comment block above) of a pw.x crystal_b / tpiba_b path, following pw.x: node i contributes
    NINT(w_i) points starting at itself, and the last node is appended once.  A node with weight 1 is followed
    directly by the next node, i.e. a jump ('U|K'): no segment is drawn between the two.
    n_kpoints: number of k-points in the DFT file; a mismatch raises ValueError.
    """
    w = [int(round(x)) for x in weights]
    n = len(nodes_frac)
    if n < 2:
        return []
    starts, i = [], 0
    for j in range(n - 1):
        starts.append(i if w[j] >= 1 else None)
        i += max(w[j], 0)
    starts.append(i)
    total = i + 1
    if n_kpoints is not None and total != int(n_kpoints):
        raise ValueError(f"the K_POINTS path gives {total} k-points but the DFT data has {n_kpoints}: is it the "
                         "input of the same bands run?")
    segs = []
    for j in range(n - 1):
        if w[j] < 2 or starts[j] is None or starts[j + 1] is None:
            continue
        segs.append({"k0": [float(c) for c in nodes_frac[j]], "k1": [float(c) for c in nodes_frac[j + 1]],
                     "l0": labels[j], "l1": labels[j + 1], "i0": int(starts[j]), "i1": int(starts[j + 1])})
    return segs


def vasp_kpath_segments(n_per_segment, points):
    """Segments of a VASP line-mode KPOINTS path from SKFitter._read_kpath_file (both ends written per segment)."""
    n = int(n_per_segment)
    return [{"k0": list(points[2 * s][0]), "k1": list(points[2 * s + 1][0]),
             "l0": points[2 * s][1], "l1": points[2 * s + 1][1], "i0": s * n, "i1": (s + 1) * n - 1}
            for s in range(len(points) // 2)]


def detect_kpath_segments(kpoints_frac, lattice, angle_tol_deg=1.0, jump_factor=5.0):
    """
    Straight segments found in a list of k-points (used when no k-path file is given): a new segment starts where
    the direction changes, after a repeated point, or after a step much longer than the typical one (a jump).
    Labels are 'Γ' at the zone centre and '' elsewhere.
    """
    kf = np.asarray(kpoints_frac, float)
    nk = len(kf)
    if nk < 2:
        return []
    B = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, float)).T
    kc = kf @ B
    d = np.diff(kc, axis=0)
    L = np.linalg.norm(d, axis=1)
    pos = L[L > 1e-8]
    med = float(np.median(pos)) if len(pos) else 1.0
    cos_tol = np.cos(np.radians(angle_tol_deg))

    def label(i):
        return "Γ" if np.all(np.abs(kf[i] - np.round(kf[i])) < 1e-6) else ""

    bounds, a, u = [], 0, None
    for s in range(nk - 1):
        if L[s] < 1e-8 or L[s] > jump_factor * med:          # duplicate point or jump: close, restart after it
            if s > a:
                bounds.append((a, s))
            a, u = s + 1, None
            continue
        v = d[s] / L[s]
        if u is not None and float(u @ v) < cos_tol:          # corner: the node is shared by both segments
            bounds.append((a, s))
            a = s
        u = v
    if nk - 1 > a:
        bounds.append((a, nk - 1))
    return [{"k0": kf[i0].tolist(), "k1": kf[i1].tolist(), "l0": label(i0), "l1": label(i1), "i0": i0, "i1": i1}
            for i0, i1 in bounds]


def kpath_ticks(segments, n_kpoints=None):
    """Tick indices and labels of a band plot: the start of the path and the end of every segment; a jump shows
    as 'A|B' at the end of the segment before it."""
    if not segments:
        return [], []
    last = (int(n_kpoints) - 1) if n_kpoints else None
    clip = (lambda i: min(i, last)) if last is not None else (lambda i: i)
    indices, labels = [clip(segments[0]["i0"])], [segments[0]["l0"]]
    for s, sg in enumerate(segments):
        if s > 0 and labels[-1] != sg["l0"]:
            labels[-1] = f"{labels[-1]}|{sg['l0']}" if labels[-1] and sg["l0"] else (labels[-1] or sg["l0"])
        indices.append(clip(sg["i1"]))
        labels.append(sg["l1"])
    return indices, labels


def kpath_blocks(segments, n_kpoints):
    """Block number of every k-point: consecutive k-points belong to the same block when some segment contains
    both (so a block ends at a jump, and - for VASP - at the repeated junction point)."""
    nk = int(n_kpoints)
    blocks = np.zeros(nk, dtype=int)
    if not segments:
        return blocks
    joined = np.zeros(max(nk - 1, 0), dtype=bool)            # joined[i]: k-points i and i+1 on one segment
    for sg in segments:
        a, b = max(0, sg["i0"]), min(nk - 1, sg["i1"])
        if b > a:
            joined[a:b] = True
    blocks[1:] = np.cumsum(~joined)
    return blocks


def read_qe_projwfc(path):
    """
    Orbital projections written by projwfc.x through `filproj` (files <filproj>.projwfc_up / .projwfc_down).

    Returns {'states': [(atom index (0-based), l, m or None), ...], 'proj': (n_states, nk, nbnd) array of
    |<psi_nk|phi_state>|^2, 'noncolin', 'spinorbit', 'labels': [(atom index, species, shell label such as '4S'), ...]}.  m follows the real spherical harmonics of Quantum ESPRESSO
    (l=1: pz, px, py; l=2: dz2, dxz, dyz, dx2-y2, dxy) and is None for spin-orbit (j, mj) states.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    head = None
    for i in range(len(lines) - 1):
        t = lines[i].split()
        if len(t) == 3 and all(re.fullmatch(r"\d+", x) for x in t):
            u = lines[i + 1].replace(".", "").split()
            if len(u) == 2 and all(x.upper() in ("T", "F", "TRUE", "FALSE") for x in u):
                head = i
                break
    if head is None:
        raise ValueError(f"{path} does not look like a projwfc.x 'filproj' file (.projwfc_up / .projwfc_down).")
    nstates, nks, nbnd = (int(x) for x in lines[head].split())
    flags = [x.upper().startswith("T") for x in lines[head + 1].replace(".", "").split()]
    state_re = re.compile(r"^\s*(\d+)\s+(\d+)\s*([A-Za-z][A-Za-z0-9_\-]*)\s+(\S+)\s+(.*)$")
    states, labels, proj = [], [], np.empty((nstates, nks, nbnd))
    pos = head + 2
    block = nks * nbnd
    for s in range(nstates):
        m = state_re.match(lines[pos])
        if m is None:
            raise ValueError(f"{path}: cannot read the state header {lines[pos]!r}")
        nums = m.group(5).split()
        l = int(nums[1])
        mm = int(nums[2]) if len(nums) > 2 and re.fullmatch(r"[-+]?\d+", nums[2]) else None
        states.append((int(m.group(2)) - 1, l, mm))
        labels.append((int(m.group(2)) - 1, m.group(3), m.group(4).upper()))
        data = np.array(" ".join(lines[pos + 1:pos + 1 + block]).split(), dtype=float)
        if data.size != 3 * block:
            raise ValueError(f"{path}: state {s + 1} has {data.size // 3} values, expected {block}.")
        proj[s] = data[2::3].reshape(nks, nbnd)
        pos += 1 + block
    return {"states": states, "proj": proj, "noncolin": flags[0], "spinorbit": flags[1], "labels": labels}


def read_qe_projwfc_labels(path):
    """[(atom index (0-based), species, shell label such as '4S' or '3D'), ...] of the states of a projwfc.x
    'filproj' file, read from the state headers only (the projections are skipped)."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        prev, head = None, None
        for line in fh:
            t = line.split()
            if prev is not None and len(prev) == 3 and all(re.fullmatch(r"\d+", x) for x in prev) and len(t) == 2 \
                    and all(x.upper().replace(".", "") in ("T", "F", "TRUE", "FALSE") for x in t):
                head = [int(x) for x in prev]
                break
            prev = t
        if head is None:
            raise ValueError(f"{path} does not look like a projwfc.x 'filproj' file.")
        nstates, nks, nbnd = head
        state_re = re.compile(r"^\s*(\d+)\s+(\d+)\s*([A-Za-z][A-Za-z0-9_\-]*)\s+(\S+)\s+(.*)$")
        for _ in range(nstates):
            m = state_re.match(fh.readline())
            if m is None:
                break
            out.append((int(m.group(2)) - 1, m.group(3), m.group(4).upper()))
            for _ in range(nks * nbnd):
                fh.readline()
    return out


_TRANSITION_Z = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81)) | set(range(89, 113))
_F_Z = set(range(57, 72)) | set(range(89, 104))


def _cart_direction(d):
    """A direction for the conductivity routines: 'x' / 'y' / 'z' or three Cartesian components (normalised), e.g.
    [1, 1, 1] for the cubic [111] axis."""
    if isinstance(d, str):
        key = d.strip().lower()
        if key in ("x", "y", "z"):
            return np.eye(3)["xyz".index(key)]
        raise ValueError(f"direction {d!r}: use 'x', 'y', 'z' or three Cartesian components such as [1, 1, 1].")
    v = np.asarray(d, dtype=float).reshape(-1)
    if v.shape != (3,) or not np.linalg.norm(v) > 0:
        raise ValueError(f"direction {d!r}: use 'x', 'y', 'z' or three Cartesian components such as [1, 1, 1].")
    return v / np.linalg.norm(v)


def direction_label(d):
    """'x' for 'x', '[1-10]' style for a vector (integers when possible)."""
    if isinstance(d, str):
        return d.strip().lower()
    v = np.asarray(d, dtype=float).reshape(-1)
    if np.allclose(v, np.round(v)):
        return "[" + "".join(f"{int(round(x))}" if x >= 0 else f"-{int(round(-x))}" for x in v) + "]"
    return "[" + ",".join(f"{x:g}" for x in v) + "]"


def unprojected_valence_shells(projected_shells):
    """{element: ['4p', ...]}: valence shells that have no projection state, from {element: ['3D', '4S', ...]} (the
    state labels of projwfc.x). The valence shells are counted from the outermost s shell n_s: p at n_s, d at n_s - 1
    for transition metals and lanthanides / actinides, f at n_s - 2 for lanthanides and actinides. A pseudopotential
    without such a wavefunction gives no projection on it, so its character shows up as weight on no atomic orbital.
    A label with n = 0 (a projector without a principal quantum number) counts as present."""
    from pymatgen.core.periodic_table import Element
    out = {}
    for el, labs in (projected_shells or {}).items():
        have = {}
        for lab in labs:
            m = re.fullmatch(r"(\d+)([SPDF])", str(lab).upper())
            if m:
                have.setdefault(m.group(2).lower(), set()).add(int(m.group(1)))
        ns = max((n for n in have.get("s", ()) if n > 0), default=0)
        if not ns:
            continue
        try:
            z = Element(el).Z
        except Exception:  # noqa: BLE001
            continue
        want = {"p": ns}
        if z in _TRANSITION_Z:
            want["d"] = ns - 1
        if z in _F_Z:
            want["f"] = ns - 2
        miss = [f"{n}{fam}" for fam, n in want.items() if n > _L_OF_FAMILY[fam]
                and not (have.get(fam, set()) & {n, 0})]
        if miss:
            out[el] = miss
    return out



def find_espresso_files(directory="."):
    """
    Quantum ESPRESSO files in a folder, by the usual names (first match wins; None if nothing fits):
        xml          qe_bands.xml, bands.xml, *bands*.xml, <prefix>.xml, <prefix>.save/data-file-schema.xml
        bands_input  qe_bands.in, bands.in, *bands*.in, *bands*.pwi
        scf          qe_scf.*, scf.out, *scf*.out, *scf*.log, scf.xml, *scf*.xml
        projwfc      qe_projwfc_up, *.projwfc_up   (projwfc.x filproj output, optional; used by the GUI)
    """
    import glob

    def first(patterns, exclude=()):
        for pat in patterns:
            hits = sorted(h for h in glob.glob(os.path.join(glob.escape(directory), pat))
                          if os.path.isfile(h) and os.path.basename(h) not in exclude)
            if hits:
                return hits[0]
        return None

    scf = first(["qe_scf.*", "scf.out", "*scf*.out", "*scf*.log", "scf.xml", "*scf*.xml"])
    skip = ("vasprun.xml",) + ((os.path.basename(scf),) if scf else ())
    return {
        "xml": first(["qe_bands.xml", "bands.xml", "*bands*.xml", "*.xml", "*.save/data-file-schema.xml"], skip),
        "bands_input": first(["qe_bands.in", "bands.in", "*bands*.in", "*bands*.pwi"]),
        "scf": scf,
        "projwfc": first(["qe_projwfc_up", "*.projwfc_up"]),
    }


# ==============================================================================
# Slater-Koster blocks and angular-momentum matrices for any l (used for the f shell; the s, p, d blocks keep their
# closed forms below and the tests check that this general construction reproduces them).
#
# Orbitals are real spherical harmonics written as polynomials in x, y, z, in SKFitter's order:
#   l=0  s
#   l=1  px, py, pz
#   l=2  dxy, dyz, dzx, dx2-y2, dz2
#   l=3  fy(3x2-y2), fxyz, fyz2, fz3, fxz2, fz(x2-y2), fx(x2-3y2)      (VASP's LORBIT=11 order, m = -3 ... 3)
# each with its |m| and cos / sin type. For a bond along z, orbital (|m|, type) on one atom couples only to the same
# (|m|, type) on the other, with the two-centre integral V_{l1 l2 |m|} (sigma, pi, delta, phi). For another bond
# direction n the orbitals are expanded in those of a frame whose z axis is n:
#   H(n) = sum_|m| V_|m| * sum_type C1[:, (|m|, type)] C2[:, (|m|, type)]^T,
# with C_l[i, j] = < orbital i | orbital j of the rotated frame > (an orthogonal matrix).
# ==============================================================================
def _poly(*terms):
    """{(a, b, c): coefficient} for sum coefficient * x^a y^b z^c; terms are (coefficient, a, b, c)."""
    out = {}
    for c, a, b, d in terms:
        out[(a, b, d)] = out.get((a, b, d), 0.0) + c
    return out


_REAL_HARMONICS = {   # l -> [(name, |m|, 'c' | 's', polynomial)] in SKFitter's orbital order
    0: [("s", 0, "c", _poly((1, 0, 0, 0)))],
    1: [("px", 1, "c", _poly((1, 1, 0, 0))), ("py", 1, "s", _poly((1, 0, 1, 0))), ("pz", 0, "c", _poly((1, 0, 0, 1)))],
    2: [("dxy", 2, "s", _poly((1, 1, 1, 0))), ("dyz", 1, "s", _poly((1, 0, 1, 1))), ("dzx", 1, "c", _poly((1, 1, 0, 1))),
        ("dx2y2", 2, "c", _poly((1, 2, 0, 0), (-1, 0, 2, 0))),
        ("dz2", 0, "c", _poly((2, 0, 0, 2), (-1, 2, 0, 0), (-1, 0, 2, 0)))],
    3: [("fy3x2", 3, "s", _poly((3, 2, 1, 0), (-1, 0, 3, 0))),                        # y(3x^2 - y^2)
        ("fxyz", 2, "s", _poly((1, 1, 1, 1))),                                        # xyz
        ("fyz2", 1, "s", _poly((4, 0, 1, 2), (-1, 2, 1, 0), (-1, 0, 3, 0))),          # y(5z^2 - r^2)
        ("fz3", 0, "c", _poly((2, 0, 0, 3), (-3, 2, 0, 1), (-3, 0, 2, 1))),           # z(5z^2 - 3r^2)
        ("fxz2", 1, "c", _poly((4, 1, 0, 2), (-1, 3, 0, 0), (-1, 1, 2, 0))),          # x(5z^2 - r^2)
        ("fzx2", 2, "c", _poly((1, 2, 0, 1), (-1, 0, 2, 1))),                         # z(x^2 - y^2)
        ("fx3", 3, "c", _poly((1, 3, 0, 0), (-3, 1, 2, 0)))],                         # x(x^2 - 3y^2)
}
_L_OF_FAMILY = {"s": 0, "s*": 0, "p": 1, "d": 2, "f": 3}
_M_NAMES = {0: "s_sigma", 1: "p_pi", 2: "d_delta", 3: "f_phi"}


def _sphere_integral(a, b, c):
    """Integral of x^a y^b z^c over the unit sphere."""
    from math import gamma
    if a % 2 or b % 2 or c % 2:
        return 0.0
    return 2.0 * gamma((a + 1) / 2) * gamma((b + 1) / 2) * gamma((c + 1) / 2) / gamma((a + b + c + 3) / 2)


def _poly_mul(p, q):
    out = {}
    for (a1, b1, c1), v1 in p.items():
        for (a2, b2, c2), v2 in q.items():
            k = (a1 + a2, b1 + b2, c1 + c2)
            out[k] = out.get(k, 0.0) + v1 * v2
    return out


def _poly_inner(p, q):
    return sum(v * _sphere_integral(*k) for k, v in _poly_mul(p, q).items())


def _poly_d(p, axis):
    out = {}
    for k, v in p.items():
        if k[axis]:
            kk = list(k)
            kk[axis] -= 1
            out[tuple(kk)] = out.get(tuple(kk), 0.0) + v * k[axis]
    return out


_NORMED = {}


def _harmonics(l):
    """[(name, |m|, type, normalised polynomial)] for shell l."""
    if l not in _NORMED:
        _NORMED[l] = [(n, m, t, {k: v / np.sqrt(_poly_inner(p, p)) for k, v in p.items()})
                      for n, m, t, p in _REAL_HARMONICS[l]]
    return _NORMED[l]


def _eval_poly(p, pts):
    return sum(v * pts[:, 0] ** a * pts[:, 1] ** b * pts[:, 2] ** c for (a, b, c), v in p.items())


_SAMPLE = None


def _sample_points():
    """Fixed, well-spread points on the unit sphere (enough to determine any polynomial of degree <= 3)."""
    global _SAMPLE
    if _SAMPLE is None:
        n = 64
        i = np.arange(n) + 0.5
        th = np.arccos(1 - 2 * i / n)
        ph = np.pi * (1 + 5 ** 0.5) * i
        _SAMPLE = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    return _SAMPLE


def _rotation_to(n):
    """A rotation matrix (columns x', y', z') whose z' axis is the unit vector n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = helper - np.dot(helper, n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n], axis=1)


def _expansion(l, R):
    """C[i, j] = < orbital i | orbital j of the frame R > for shell l (orbital j of the frame: f_j(R^T r))."""
    H = _harmonics(l)
    P = _sample_points()
    A = np.stack([_eval_poly(p, P @ R) for _, _, _, p in H], axis=1)       # rotated orbitals at the points
    B = np.stack([_eval_poly(p, P) for _, _, _, p in H], axis=1)           # orbitals at the points
    C_T, *_ = np.linalg.lstsq(A, B, rcond=None)                            # B = A C^T
    return C_T.T


_SK_UNIT_CACHE = {}


def sk_unit_blocks(l1, l2, direction):
    """[U_0, U_1, ...]: the (2 l1 + 1) x (2 l2 + 1) Slater-Koster block for a bond along ``direction`` is
    sum_m V_{l1 l2 m} U_m (m = 0: sigma, 1: pi, 2: delta, 3: phi). Cached per direction."""
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    key = (l1, l2, n.tobytes())                 # exact direction: bonds repeat their own vectors
    hit = _SK_UNIT_CACHE.get(key)
    if hit is not None:
        return hit
    R = _rotation_to(n)
    C1, C2 = _expansion(l1, R), _expansion(l2, R)
    H1, H2 = _harmonics(l1), _harmonics(l2)
    units = []
    for m in range(min(l1, l2) + 1):
        U = np.zeros((2 * l1 + 1, 2 * l2 + 1))
        for t in ("c", "s"):
            i1 = [j for j, h in enumerate(H1) if h[1] == m and h[2] == t]
            i2 = [j for j, h in enumerate(H2) if h[1] == m and h[2] == t]
            if i1 and i2:
                U += np.outer(C1[:, i1[0]], C2[:, i2[0]])
        units.append(U)
    if len(_SK_UNIT_CACHE) > 20000:
        _SK_UNIT_CACHE.clear()
    _SK_UNIT_CACHE[key] = units
    return units


def angular_momentum_matrices(l):
    """(Lx, Ly, Lz) in the real orbitals of shell l: L_a[i, j] = < orbital i | L_a | orbital j >, L = -i r x grad."""
    H = _harmonics(l)
    ops = {   # r x grad components: (y d/dz - z d/dy), (z d/dx - x d/dz), (x d/dy - y d/dx)
        "x": ((1, 2, 1.0), (2, 1, -1.0)), "y": ((2, 0, 1.0), (0, 2, -1.0)), "z": ((0, 1, 1.0), (1, 0, -1.0))}
    out = []
    for a in ("x", "y", "z"):
        K = np.zeros((len(H), len(H)))
        for j, (_, _, _, pj) in enumerate(H):
            Dp = {}
            for coord, deriv, sign in ops[a]:
                term = _poly_mul(_poly(*[(1, *(1 if q == coord else 0 for q in range(3)))]), _poly_d(pj, deriv))
                for k, v in term.items():
                    Dp[k] = Dp.get(k, 0.0) + sign * v
            for i, (_, _, _, pi) in enumerate(H):
                K[i, j] = _poly_inner(pi, Dp)
        out.append(-1j * K)
    return tuple(out)


# ---------------------------------------------------------------------------------------------------------------
# Neighbour shells and bonds (shared by SKFitter and the structure / neighbour views of the GUIs)
# ---------------------------------------------------------------------------------------------------------------
NEIGHBOUR_SEARCH_RADIUS = 15.0      # Å: the neighbour shells are the distinct distances found within this radius
SHELL_DECIMALS = 3                  # distances equal to 0.001 Å form one shell
SHELL_MARGIN = 0.1                  # Å: the cutoff sits this far beyond the last included shell


def neighbour_shells(structure, search_radius=NEIGHBOUR_SEARCH_RADIUS):
    """The distinct atom-atom distances (Å, rounded to 0.001 Å, ascending) within ``search_radius``, over all sites
    of the cell: shell n of the model is the n-th of these (the same list for every site)."""
    all_neighbors = structure.get_all_neighbors(search_radius)
    distances = [nb.nn_distance for site_nb in all_neighbors for nb in site_nb if nb.nn_distance > 1e-4]
    if not distances:
        raise ValueError(f"No neighbors found within {search_radius} Å search radius.")
    return np.sort(np.unique(np.round(distances, decimals=SHELL_DECIMALS)))


def cutoff_from_shells(structure, n_shells):
    """The bond cutoff (Å) of a model with ``n_shells`` neighbour shells: shell n_shells + 0.1 Å."""
    unique_shells = neighbour_shells(structure)
    if n_shells > len(unique_shells):
        n_shells = len(unique_shells)
    return unique_shells[n_shells - 1] + SHELL_MARGIN


def neighbour_pairs(structure, cutoff):
    """The bonds of the model: every (i, j, R) with 0 < |r_j + R - r_i| < cutoff, both directions, as dicts
    {i, j, dist, d_vec, dir_cosines}; and d0 = the shortest bond (the reference length of the distance decay)."""
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)
    pairs = []
    min_dist = float('inf')
    for site_i_idx, neighbors in enumerate(all_neighbors):
        site_i = structure[site_i_idx]
        for nb in neighbors:
            site_j_idx = nb.index
            d_vec = nb.coords - site_i.coords
            dist = nb.nn_distance
            if dist > 1e-4:
                if dist < min_dist:
                    min_dist = dist
                pairs.append({
                    'i': site_i_idx,
                    'j': site_j_idx,
                    'dist': dist,
                    'd_vec': d_vec,
                    'dir_cosines': d_vec / dist
                })
    return pairs, min_dist


SHELL_TOLERANCE = 2e-3      # Å: bond lengths closer than this belong to the same neighbour shell
HOPPING_MODES = ("decay", "shell")


def bond_shell_distances(pairs, tol=SHELL_TOLERANCE):
    """The distinct bond lengths (Å, ascending) of the bonds ``pairs`` (from neighbour_pairs): the neighbour shells of
    the model. Lengths within ``tol`` of the previous one join its shell; a shell's distance is their mean."""
    shells = []
    for d in sorted(float(p['dist']) for p in pairs):
        if shells and d - shells[-1][-1] <= tol:
            shells[-1].append(d)
        else:
            shells.append([d])
    return [float(np.mean(x)) for x in shells]


def pair_shell_distances(pairs, site_elements, tol=SHELL_TOLERANCE):
    """{element pair ('Ir-Mn'): its bond lengths (Å, ascending)}: the neighbour shells of each element pair among the
    bonds ``pairs`` (from neighbour_pairs). Shell 0 of a pair is its nearest bond, the reference length d0 of its
    distance decay."""
    by = {}
    for p in pairs:
        by.setdefault(element_pair(site_elements[p['i']], site_elements[p['j']]), []).append(p)
    return {k: bond_shell_distances(v, tol) for k, v in sorted(by.items())}


def shell_index(shells, r):
    """(n, r_n): the shell of the list ``shells`` (Å) nearest to the bond length r."""
    dist = np.asarray(shells, dtype=float)
    n = int(np.argmin(np.abs(dist - r)))
    return n, float(dist[n])


def bond_shell_keys(pairs, site_orbitals, site_elements, pair_shells):
    """Sorted [(n, key), ...]: the hopping integrals (keys 'v_...') the bonds of shell n of their element pair use."""
    out = set()
    for pair in pairs:
        ei, ej = site_elements[pair['i']], site_elements[pair['j']]
        n, _ = shell_index(pair_shells[element_pair(ei, ej)], pair['dist'])
        for key in pair_hopping_keys(ei, site_orbitals[pair['i']], ej, site_orbitals[pair['j']], pair['d_vec']):
            out.add((n, key))
    return sorted(out)


def normalise_hopping_mode(mode):
    """'decay' (one integral per bond type, V(r) = V0 (d0/r)^eta) or 'shell' (independent integrals per neighbour
    shell, V_n); None -> 'decay'."""
    mode = str(mode or "decay").strip().lower()
    aliases = {"decay": "decay", "decay_law": "decay", "law": "decay", "shell": "shell", "per_shell": "shell",
               "shells": "shell", "per-shell": "shell"}
    if mode not in aliases:
        raise ValueError(f"hopping_mode must be one of {HOPPING_MODES}, not {mode!r}")
    return aliases[mode]


_S_TYPES = ('s', 's*')
_P_TYPES = ('p', 'px', 'py', 'pxy', 'pz')
_D_TYPES = ('d', 'dxy', 'dyz', 'dzx', 'dx2y2', 'dz2', 'd_e1', 'd_e2', 'd_t2g', 'd_eg')
_F_TYPES = ('f', 'fy3x2', 'fxyz', 'fyz2', 'fz3', 'fxz2', 'fzx2', 'fx3', 'f_m0', 'f_m1', 'f_m2', 'f_m3',
            'f_a2u', 'f_t1u', 'f_t2u')
_FAMILIES = ('s', 'p', 'd', 'f')


def _family_of(sub):
    if sub in _S_TYPES: return 's'
    if sub in _P_TYPES: return 'p'
    if sub in _D_TYPES: return 'd'
    if sub in _F_TYPES: return 'f'
    return sub


def exchange_key(elem, sub, magnetic_orbitals):
    """The exchange-splitting parameter of the subshell ``sub`` (e.g. 'd_t2g', 'pz', 'd') of element ``elem``, or None.

    ``magnetic_orbitals[elem]`` lists orbital families ('s', 'p', 'd', 'f') and/or symmetry subshells of the element's
    orbital set ('pz', 'pxy', 'dz2', 'd_e1', 'd_e2', 'd_t2g', 'd_eg', 'dxy', ...):
      * a subshell listed by name has its own splitting, delta_spin_<El>_<sub> (e.g. delta_spin_Fe_d_t2g);
      * the other subshells of a listed family share delta_spin_<El>_<family> (the behaviour without symmetry
        subshells: one delta per family).
    So ['s', 'd'] with d split into t2g / e_g gives one delta for all d; ['s', 'd_t2g', 'd_eg'] gives two."""
    mags = (magnetic_orbitals or {}).get(elem) or ()
    fam = _family_of(sub)
    if sub != fam and sub in mags:
        return f"delta_spin_{elem}_{sub}"
    if fam in mags:
        return f"delta_spin_{elem}_{fam}"
    return None


def magnetic_orbitals_from_params(params):
    """The magnetic_orbitals that a parameter set's delta_spin_<El>_<x> keys imply (for models saved without it)."""
    out = {}
    for k in params or {}:
        if k.startswith("delta_spin_"):
            elem, _, x = k[len("delta_spin_"):].partition("_")
            if elem and x:
                out.setdefault(elem, [])
                if x not in out[elem]:
                    out[elem].append(x)
    return out


def check_soc_orbitals(soc_orbitals, orbital_config, family_map):
    """Errors in ``soc_orbitals`` ({element: ['p', 'd', 'f', ...]}, which orbital families carry on-site spin-orbit
    coupling) against ``orbital_config``: an entry must be a p, d or f family the element has in the model."""
    errs = []
    for elem, fams in (soc_orbitals or {}).items():
        if elem not in orbital_config:
            errs.append(f"{elem} is not in the model")
            continue
        have = {family_map[s] for s in orbital_config[elem]}
        for f in fams:
            if f not in ("p", "d", "f"):
                errs.append(f"{elem} {f!r}: only p, d and f families carry spin-orbit coupling")
            elif f not in have:
                errs.append(f"{elem} has no {f} orbitals in the model")
    return errs


def check_magnetic_orbitals(magnetic_orbitals, orbital_config):
    """Errors in ``magnetic_orbitals`` against ``orbital_config`` ({element: [subshells]}): an entry must be a family the
    element has, or one of its subshells (a symmetry subshell such as 'd_t2g' needs the d shell split that way)."""
    errs = []
    for elem, entries in (magnetic_orbitals or {}).items():
        if orbital_config is not None and elem not in orbital_config:
            continue                                   # an element of another structure: nothing to check
        subs = list((orbital_config or {}).get(elem) or [])
        fams = {_family_of(x) for x in subs}
        for e in entries or ():
            if e in _FAMILIES:
                if subs and e not in fams:
                    errs.append(f"{elem}: no {e} orbitals in the orbital set ({', '.join(subs)})")
            elif _family_of(e) == e:
                errs.append(f"{elem}: unknown orbital {e!r} (a family s, p, d, f or a subshell such as d_t2g, pz)")
            elif subs and e not in subs:
                errs.append(f"{elem}: {e} is not in the orbital set ({', '.join(subs)}) - split the "
                            f"{_family_of(e)} shell into {e} and its partners to give it its own exchange splitting")
    return errs


# ---------------------------------------------------------------------------------------------------------------
# Two-centre integrals per pair of (element, orbital group)
#
# Every orbital choice of an element ('s', 'p', 'pz', 'pxy', 'd', 'dz2', 'd_e1', 'd_t2g', 'f_m1', ...) is a group of
# real orbitals of one shell l. The block between group A (shell l1) on element E1 and group B (shell l2) on element
# E2 on a bond along n is
#     sum_m V^{AB}_m U_m[A, B],   m = sigma, pi, delta, phi (0 ... min(l1, l2)),
# with U_m the Slater-Koster angular blocks of the whole shells (sk_unit_blocks) restricted to the two groups. Each
# unordered pair (E1 A, E2 B) has its own integrals, named
#     v_<E1>_<A>-<E2>_<B>_<l1 l2 m>        e.g. v_Mn_d-Mn_d_dds, v_Ir_s-Mn_d_sds, v_Bi_pz-Mn_dz2_pds
# with the pair in key order (lower l first, then element, then group) and the integral label of the two families
# and m (sss, sps, pds, ddd, ss*s, ...). A bond's integral carries a suffix: _<n> = the n-th neighbour shell of that
# element pair (per-shell model; _0 = the pair's nearest bond), or _0 = V0 at the pair's nearest bond d0 (decay law,
# V(r) = V0 (d0/r)^eta with eta_<E1>_<A>-<E2>_<B>_<label>). Overlaps are s_<same>. With all the integrals of the
# groups of a shell pair equal, the block is exactly the standard Slater-Koster block. A split must respect the
# crystal symmetry (check_orbital_symmetry).
# ---------------------------------------------------------------------------------------------------------------
_M_LETTER = "spdf"                  # sigma, pi, delta, phi
_INTEGRAL_RE = re.compile(r"^(v|s)_([A-Z][a-z]?)_([^_\-]+)-([A-Z][a-z]?)_([^_\-]+)_([spdf*]+)(?:_(\d+))?$")
_BTYPE_RE = re.compile(r"^([A-Z][a-z]?)_([^_\-]+)-([A-Z][a-z]?)_([^_\-]+)_([spdf*]+)$")


def group_tag(sub):
    """The name of an orbital group inside integral keys (no underscores: 'd_t2g' -> 'dt2g')."""
    return sub.replace("_", "")


def _l_of(sub):
    return _L_OF_FAMILY[_family_of(sub)]


def element_pair(el1, el2):
    """The label of an unordered element pair ('Mn', 'Ir' -> 'Ir-Mn'): the key of its neighbour shells."""
    return "-".join(sorted((str(el1), str(el2))))


def _canonical_pair(sub1, el1, sub2, el2):
    """(A, EA, B, EB, swapped): the two ends in key order - lower l first, then element, then group name."""
    k1, k2 = (_l_of(sub1), el1, group_tag(sub1)), (_l_of(sub2), el2, group_tag(sub2))
    return (sub1, el1, sub2, el2, False) if k1 <= k2 else (sub2, el2, sub1, el1, True)


def _key_family(sub):
    """The family a group belongs to in integral labels ('s*' is a family of its own: ss*s, s*ps, ...)."""
    return sub if sub in ("s", "s*") else _family_of(sub)


def integral_keys(sub1, sub2, el1, el2):
    """[key for m = 0 ... min(l1, l2)]: the hopping integrals between group sub1 of element el1 and group sub2 of
    element el2 (either order), e.g. integral_keys('d', 'd', 'Mn', 'Ir') = ['v_Ir_d-Mn_d_dds', ..._ddp, ..._ddd]."""
    a, ea, b, eb, _ = _canonical_pair(sub1, el1, sub2, el2)
    fa, fb = _key_family(a), _key_family(b)
    nm = min(_l_of(a), _l_of(b)) + 1
    return [f"v_{ea}_{group_tag(a)}-{eb}_{group_tag(b)}_{fa}{fb}{_M_LETTER[m]}" for m in range(nm)]


def parse_integral_key(key):
    """'v_Ir_d-Mn_d_dds_1' -> {pre: 'v', el1: 'Ir', tag1: 'd', el2: 'Mn', tag2: 'd', label: 'dds', shell: 1,
    btype: 'Ir_d-Mn_d_dds', pair: 'Ir-Mn'}; None for anything else (shell None without a suffix)."""
    m = _INTEGRAL_RE.match(str(key))
    if not m:
        return None
    pre, e1, t1, e2, t2, lab, n = m.groups()
    return {"pre": pre, "el1": e1, "tag1": t1, "el2": e2, "tag2": t2, "label": lab,
            "shell": None if n is None else int(n), "btype": f"{e1}_{t1}-{e2}_{t2}_{lab}",
            "pair": element_pair(e1, e2)}


def bond_type(key):
    """The integral of a key without prefix and shell ('v_Ir_d-Mn_d_dds_0', 's_..._2' -> 'Ir_d-Mn_d_dds'): the
    name of its decay exponent eta_<bond type> (overlaps: eta_s_<bond type>). A bond type itself is returned as it is."""
    p = parse_integral_key(key)
    if p is not None:
        return p["btype"]
    if _BTYPE_RE.match(str(key)):
        return str(key)
    raise ValueError(f"{key!r} is not a two-centre integral (v_<El>_<orbital>-<El>_<orbital>_<integral>[_n])")


def integral_pair(key):
    """The element pair ('Ir-Mn') of an integral key or bond type."""
    m = _BTYPE_RE.match(bond_type(key))
    return element_pair(m.group(1), m.group(3))


def integral_name(key):
    """The integral of a key ('v_Bi_pz-Mn_dz2_pds_0' -> 'pds'); None for other keys."""
    p = parse_integral_key(key)
    return None if p is None else p["label"]


_SUB_INDICES = None
_GROUP_UNIT_CACHE = {}


def _sub_indices(sub):
    global _SUB_INDICES
    if _SUB_INDICES is None:
        _SUB_INDICES = {k: v["indices"] for k, v in SKFitter.SUB_ORBITALS.items()}
    return _SUB_INDICES[sub]


_SUB_BASIS = {}


def sub_basis(sub):
    """B (dim x 2l+1): the orbitals of group ``sub`` as rows of coefficients in the real orbitals of its shell. A group of
    real orbitals (pz, d_t2g, f_m1, ...) is a set of rows of the unit matrix; a cubic f group (f_a2u, f_t1u, f_t2u) is
    a set of cubic harmonics, linear combinations of the real f orbitals (orthonormal rows)."""
    hit = _SUB_BASIS.get(sub)
    if hit is None:
        info = SKFitter.SUB_ORBITALS[sub]
        if info.get("basis") is not None:
            hit = np.asarray(info["basis"], float)
        else:
            hit = np.eye(2 * _l_of(sub) + 1)[list(info["indices"])]
        _SUB_BASIS[sub] = hit
    return hit


def groups_overlap(sub1, sub2, tol=1e-8):
    """True when two groups of one shell share an orbital (their spans are not orthogonal)."""
    if _family_of(sub1) != _family_of(sub2):
        return False
    return bool(np.abs(sub_basis(sub1) @ sub_basis(sub2).T).max() > tol)


def group_unit_blocks(sub1, sub2, direction):
    """[U_m restricted to (sub1 rows, sub2 columns)] for a bond along ``direction`` from the sub1 site to the sub2 site:
    the block is sum_m V_m U_m with V_m = integral_keys(sub1, sub2, el1, el2)[m]."""
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    ck = (sub1, sub2, n.tobytes())
    hit = _GROUP_UNIT_CACHE.get(ck)
    if hit is not None:
        return hit
    l1, l2 = _l_of(sub1), _l_of(sub2)
    b1, b2 = sub_basis(sub1), sub_basis(sub2)
    if l1 <= l2:
        out = [b1 @ U @ b2.T for U in sk_unit_blocks(l1, l2, n)]
    else:                                        # V_{l2 l1} = (-1)^(l1+l2) V_{l1 l2}: the transposed block
        out = [(-1) ** (l1 + l2) * (b2 @ U @ b1.T).T for U in sk_unit_blocks(l2, l1, n)]
    if len(_GROUP_UNIT_CACHE) > 50000:
        _GROUP_UNIT_CACHE.clear()
    _GROUP_UNIT_CACHE[ck] = out
    return out


def pair_hopping_keys(el_i, orbs_i, el_j, orbs_j, direction=None, tol=1e-10):
    """The hopping integrals (keys 'v_...') a bond between a site of element ``el_i`` with the orbital groups
    ``orbs_i`` and a site of ``el_j`` with ``orbs_j`` uses. With ``direction`` (the bond vector) only those that
    contribute on that bond (a split integral can vanish on some bonds by symmetry)."""
    keys = set()
    for a in orbs_i:
        for b in orbs_j:
            ks = integral_keys(a, b, el_i, el_j)
            if direction is None:
                keys.update(ks)
                continue
            for key, U in zip(ks, group_unit_blocks(a, b, direction)):
                if np.abs(U).max() > tol:
                    keys.add(key)
    return keys


_GROUP_OF_TAG = None


def _group_of_tag(tag):
    global _GROUP_OF_TAG
    if _GROUP_OF_TAG is None:
        _GROUP_OF_TAG = {group_tag(sub): sub for sub in SKFitter.SUB_ORBITALS}
    return _GROUP_OF_TAG.get(tag)


def is_group_integral(key):
    """True for an integral between orbital groups that are not whole families ('v_Bi_pz-Mn_dz2_pds_0')."""
    p = parse_integral_key(key) or parse_integral_key(f"v_{key}")
    if p is None:
        return False
    return any(_group_of_tag(t) is not None and _key_family(_group_of_tag(t)) != _group_of_tag(t)
               for t in (p["tag1"], p["tag2"]))


def _rebuild(p, a, b, suffix=True):
    """The key of parsed integral ``p`` with its groups replaced by a, b (same elements, same m)."""
    ks = integral_keys(a, b, p["el1"], p["el2"])
    k = f"{p['pre']}_{ks[_M_LETTER.index(p['label'][-1])][2:]}"
    return k + (f"_{p['shell']}" if suffix and p["shell"] is not None else "")


def family_integral_key(key):
    """The standard integral a group-pair integral splits ('v_Bi_pz-Mn_dz2_pds_3' -> 'v_Bi_p-Mn_d_pds_3'); a key
    between whole families (or anything else) is returned unchanged."""
    p = parse_integral_key(key)
    if p is None:
        return key
    a, b = _group_of_tag(p["tag1"]), _group_of_tag(p["tag2"])
    if a is None or b is None:
        return key
    return _rebuild(p, _key_family(a), _key_family(b))


def coarser_integral_keys(key):
    """Keys of the same integral with one or both groups merged back into their family, finest first
    ('v_Ce_deg-Ce_ft1u_dfd_0' -> ['v_Ce_d-Ce_ft1u_dfd_0', 'v_Ce_deg-Ce_f_dfd_0', 'v_Ce_d-Ce_f_dfd_0']): starting
    values when only one side was split."""
    p = parse_integral_key(key)
    if p is None:
        return []
    a, b = _group_of_tag(p["tag1"]), _group_of_tag(p["tag2"])
    if a is None or b is None:
        return []
    fa, fb = _key_family(a), _key_family(b)
    out = []
    for x, y in ((fa, b), (a, fb), (fa, fb)):
        if (x, y) == (a, b):
            continue
        k = _rebuild(p, x, y)
        if k != key and k not in out:
            out.append(k)
    return out


# The split p integrals before the group-pair scheme (v_pzpzs, v_pxpxp, v_spzs, ...) and their names now. The old
# pxy-pz cross term used the px-px integrals, so they also start the new pxy-pz ones.
_LEGACY_P = {"pzpzs": ["pz.pz.s"], "pzpzp": ["pz.pz.p"], "pxpxs": ["pxy.pxy.s", "pxy.pz.s"],
             "pxpxp": ["pxy.pxy.p", "pxy.pz.p"], "spzs": ["s.pz.s"], "spxs": ["s.pxy.s"],
             "s*pzs": ["s*.pz.s"], "s*pxs": ["s*.pxy.s"]}


def migrate_legacy_keys(params):
    """``params`` with the split-p integral names of older fits and models renamed to the group-pair names
    (v_pzpzs_0 -> v_pz.pz.s_0, eta_pxpxp -> eta_pxy.pxy.p and eta_pxy.pz.p, ...). Other keys are kept."""
    out = {}
    for k, v in (params or {}).items():
        m = re.match(r"^(v_|s_|eta_s_|eta_)(s\*pzs|s\*pxs|pzpzs|pzpzp|pxpxs|pxpxp|spzs|spxs)((?:_\d+)?)$", k)
        if not m:
            out[k] = v
            continue
        for new in _LEGACY_P[m.group(2)]:
            out.setdefault(f"{m.group(1)}{new}{m.group(3)}", v)
    return out


# Integral names before the element-pair scheme (rc4 and older): one integral per orbital pair for all element pairs,
# 'v_dds_0', 'v_pz.dz2.s_1', 'eta_dds', with the shells counted over all bonds of the cell and the decay referred
# to the shortest bond of the cell. convert_legacy_hoppings maps such a parameter set onto a model's element-pair
# integrals so that the converted set gives exactly the same bands.
_LEGACY_INTEGRAL_RE = re.compile(r"^(v|s)_([a-z*][^_]*)_(\d+)$")


def has_legacy_hoppings(params):
    """True when ``params`` uses the integral names before the element-pair scheme ('v_dds_0', 'v_pz.dz2.s_1')."""
    return any(_LEGACY_INTEGRAL_RE.match(k) for k in (params or {}))


def _legacy_integral_keys(sub1, sub2):
    """The old (element-independent) names of the integrals between groups sub1 and sub2 ('v_pds', 'v_pz.dz2.s')."""
    k1, k2 = (_l_of(sub1), group_tag(sub1)), (_l_of(sub2), group_tag(sub2))
    a, b = (sub1, sub2) if k1 <= k2 else (sub2, sub1)
    fa, fb = _key_family(a), _key_family(b)
    nm = min(_l_of(a), _l_of(b)) + 1
    if a == fa and b == fb:
        return [f"v_{fa}{fb}{_M_LETTER[m]}" for m in range(nm)]
    return [f"v_{group_tag(a)}.{group_tag(b)}.{_M_LETTER[m]}" for m in range(nm)]


def convert_legacy_hoppings(params, neighbor_list, site_elements, site_orbitals, n_neighbors, hopping_mode=None,
                            pair_shells=None):
    """``params`` with old integral names (has_legacy_hoppings) renamed to the element-pair integrals of the model with
    these bonds, so that the bands do not change: every element pair gets the old integral's value; per shell, the old
    shell index (counted over the whole cell) becomes the pair's own shell index; with the decay law, V0 is moved
    from the cell's shortest bond d0 to the pair's nearest bond d0_pair (V0 (d0/d0_pair)^eta) and eta is kept.
    ``hopping_mode``: the mode of the old fit (None: 'shell' if any shell index above 0 appears, else 'decay').
    Other parameters are kept; parameters without old names are returned unchanged."""
    params = migrate_legacy_keys(params)
    if not has_legacy_hoppings(params):
        return dict(params)
    if hopping_mode is None:
        hopping_mode = "shell" if any((m := _LEGACY_INTEGRAL_RE.match(k)) and int(m.group(3)) > 0
                                      for k in params) else "decay"
    hopping_mode = normalise_hopping_mode(hopping_mode)
    out = {k: v for k, v in params.items()
           if not (_LEGACY_INTEGRAL_RE.match(k) or re.match(r"^eta_(s_)?[a-z*][^_]*$", k))}
    glob = bond_shell_distances(neighbor_list)
    d0 = min((float(p['dist']) for p in neighbor_list), default=0.0)       # the old reference: the shortest bond
    pair_shells = pair_shells or pair_shell_distances(neighbor_list, site_elements)
    for bond in neighbor_list:
        ei, ej = site_elements[bond['i']], site_elements[bond['j']]
        shells = pair_shells[element_pair(ei, ej)]
        n, _ = shell_index(shells, bond['dist'])
        g, _ = shell_index(glob, bond['dist'])
        for a in site_orbitals[bond['i']]:
            for b in site_orbitals[bond['j']]:
                for new, old in zip(integral_keys(a, b, ei, ej), _legacy_integral_keys(a, b)):
                    obt, nbt = old[2:], bond_type(new)
                    for pre, eta_pre in (("v", "eta_"), ("s", "eta_s_")):
                        if hopping_mode == "shell":
                            if f"{pre}_{obt}_{g}" in params:
                                out[f"{pre}_{nbt}_{n}"] = float(params[f"{pre}_{obt}_{g}"])
                            if f"{eta_pre}{obt}" in params:
                                out[f"{eta_pre}{nbt}"] = float(params[f"{eta_pre}{obt}"])
                            continue
                        if f"{pre}_{obt}_0" not in params:
                            continue
                        eta = float(params.get(f"{eta_pre}{obt}", 2.5))
                        scale = (d0 / shells[0]) ** eta if n_neighbors > 1 and shells[0] > 1e-4 else 1.0
                        out[f"{pre}_{nbt}_0"] = float(params[f"{pre}_{obt}_0"]) * scale
                        if f"{eta_pre}{obt}" in params:
                            out[f"{eta_pre}{nbt}"] = eta
    return out


def check_orbital_symmetry(structure, site_orbitals, symprec=1e-3, tol=1e-3):
    """Errors when an orbital split breaks the crystal symmetry: every group of every site must be mapped onto itself
    by every symmetry operation of the crystal (in the Cartesian axes of the structure). E.g. pz / pxy and dz2 / d_e1 /
    d_e2 need z along a 3-, 4- or 6-fold axis; d_t2g / d_eg need cubic axes along x, y, z."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    split = {}
    for i, subs in enumerate(site_orbitals):
        for sub in subs:
            fam = _key_family(sub)
            if sub != fam:
                split.setdefault(structure[i].specie.symbol, set()).add(sub)
    if not split:
        return []
    try:
        ops = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations(cartesian=True)
    except Exception:  # noqa: BLE001 - no symmetry found: nothing to check against
        return []
    errs, done = [], set()
    for elem, subs in split.items():
        for sub in sorted(subs):
            l = _l_of(sub)
            B = sub_basis(sub)
            Q = np.eye(2 * l + 1) - B.T @ B                 # projector onto the rest of the shell
            for op in ops:
                W = np.asarray(op.rotation_matrix, float)
                C = _expansion(l, W)
                if np.abs(Q @ C @ B.T).max() > tol:          # the operation takes the group out of itself
                    if (elem, sub) not in done:
                        done.add((elem, sub))
                        errs.append(f"{elem} {sub}: a symmetry operation of the crystal mixes it with other "
                                    f"{_family_of(sub)} orbitals, so a separate {sub} breaks the symmetry - use the "
                                    f"whole {_family_of(sub)} shell or groups adapted to this crystal and its axes")
                    break
    return errs


def _site_representation(site_orbitals, W):
    """D(W) on a site's orbital basis (its groups in order): D[block a, block b] = B_a C_l(W) B_b^T within a family,
    zero between families. An orbital |b> of the site, moved by the operation, is sum_a D[a, b] |a> on the image."""
    rows = []
    for sub in site_orbitals:
        rows.append((SKFitter.FAMILY_MAP[sub], _l_of(sub), sub_basis(sub)))
    dims = [B.shape[0] for _, _, B in rows]
    off = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    D = np.zeros((off[-1], off[-1]))
    cache = {}
    for a, (fa, la, Ba) in enumerate(rows):
        for b, (fb, lb, Bb) in enumerate(rows):
            if fa != fb:
                continue
            if la not in cache:
                cache[la] = _expansion(la, W)
            D[off[a]:off[a + 1], off[b]:off[b + 1]] = Ba @ cache[la] @ Bb.T
    return D, off


def onsite_mixing_terms(structure, site_orbitals, eq_site_map, elements=None, symprec=1e-3, tol=1e-3):
    """The on-site terms beyond one energy per orbital group that the site symmetry allows: real symmetric matrices
    M on a site's orbitals with D M D^T = M for every operation that leaves the site in place, with the identity of
    every group taken out. Between two groups of the same symmetry type (E_1 / E_2 of d on a trigonal site, two e_u
    sets of f, ...) they mix the groups; between families on a site without inversion they mix e.g. s with d_z2 or
    p_z with d_z2; within a group that is not irreducible (a whole d shell on a trigonal site) they are its crystal
    field. One parameter per matrix and class of equivalent sites, e_mix<eq>_<El>_<n>; the matrices of the other
    sites of a class are the representative's moved by an operation that maps it there. ``elements``: the elements
    that get these terms (None: all). Returns (per site: [(key, M)], [keys in order])."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    n = len(structure)
    out, keys = [[] for _ in range(n)], []
    want = [i for i in range(n) if site_orbitals[i] and (elements is None or structure[i].specie.symbol in elements)]
    if not want:
        return out, keys
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    ops_c = sga.get_symmetry_operations(cartesian=True)
    ops_f = sga.get_symmetry_operations(cartesian=False)
    frac = np.asarray(structure.frac_coords)
    tol_f = max(10 * symprec, 1e-4)

    def maps(op, i, j):
        d = op.operate(frac[i]) - frac[j]
        return np.abs(d - np.round(d)).max() < tol_f

    done = {}
    for i in want:
        eq = int(eq_site_map[i])
        elem = structure[i].specie.symbol
        if eq not in done:
            rep = next(r for r in want if int(eq_site_map[r]) == eq)
            stab = [np.asarray(oc.rotation_matrix, float) for oc, of in zip(ops_c, ops_f) if maps(of, rep, rep)]
            Ds = [_site_representation(site_orbitals[rep], W)[0] for W in stab]
            _, off = _site_representation(site_orbitals[rep], np.eye(3))
            m = off[-1]
            basis = []
            for p in range(m):
                for q in range(p, m):
                    X = np.zeros((m, m)); X[p, q] = X[q, p] = 1.0
                    basis.append(sum(D @ X @ D.T for D in Ds) / len(Ds))
            V = np.array([b.ravel() for b in basis])
            ids = []
            for a in range(len(off) - 1):
                Ia = np.zeros((m, m)); Ia[off[a]:off[a + 1], off[a]:off[a + 1]] = np.eye(off[a + 1] - off[a])
                ids.append(Ia.ravel() / np.linalg.norm(Ia))
            Qi, _ = np.linalg.qr(np.array(ids).T)
            V = V - (V @ Qi) @ Qi.T                          # the group identities are the on-site energies
            _, sv, vt = np.linalg.svd(V, full_matrices=False)
            # (tol: the operations of a cell that is symmetric only to symprec leave small residues)
            mats = []
            for r in range(int((sv > tol * max(1.0, sv[0] if len(sv) else 1.0)).sum())):
                M = vt[r].reshape(m, m); M = 0.5 * (M + M.T)
                M = np.where(np.abs(M) < 1e-10, 0.0, M)
                M /= np.abs(np.linalg.eigvalsh(M)).max()        # unit coupling: the levels move by +- the parameter
                nz = M.ravel()[np.abs(M.ravel()) > 1e-8]
                if nz.size and nz[0] < 0:
                    M = -M
                mats.append(M)
            terms = [(f"e_mix{eq}_{elem}_{r}", M) for r, M in enumerate(mats)]
            keys += [k for k, _ in terms]
            done[eq] = (rep, terms)
        rep, terms = done[eq]
        if i == rep:
            out[i] = [(k, M.copy()) for k, M in terms]
            continue
        g = next(((oc, of) for oc, of in zip(ops_c, ops_f) if maps(of, rep, i)), None)
        if g is None:
            raise ValueError(f"on-site mixing: no symmetry operation takes site {rep} to site {i}")
        D, _ = _site_representation(site_orbitals[rep], np.asarray(g[0].rotation_matrix, float))
        out[i] = [(k, D @ M @ D.T) for k, M in terms]
    return out, keys


# Partitions of a shell into orbital groups, tried finest first by symmetry_groups.
GROUP_SETS = {
    "p": [["px", "py", "pz"], ["pz", "pxy"]],
    "d": [["dxy", "dyz", "dzx", "dx2y2", "dz2"], ["dz2", "dx2y2", "dxy", "d_e1"], ["dz2", "d_e1", "d_e2"],
          ["d_t2g", "d_eg"]],
    "f": [["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"], ["f_m0", "f_m1", "f_m2", "f_m3"],
          ["f_a2u", "f_t1u", "f_t2u"]],
}


def site_symmetry_symbols(structure, symprec=1e-3):
    """The site point group of every site (Hermann-Mauguin, e.g. '4/mmm', 'm-3m'), or '?' per site."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        ds = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_dataset()
        sy = getattr(ds, "site_symmetry_symbols", None)
        if sy is None and isinstance(ds, dict):
            sy = ds.get("site_symmetry_symbols")
        return [str(x).replace(".", "") or "1" for x in sy]
    except Exception:  # noqa: BLE001
        return ["?"] * len(structure)


def symmetry_groups(structure, element, family, symprec=1e-3):
    """The finest split of an element's p, d or f shell into the orbital groups SKFitter knows that the crystal
    symmetry allows on all its sites (check_orbital_symmetry), and the on-site terms the site symmetry still allows
    within the shell (onsite_mixing_terms: a crystal field inside a group, e.g. t2g -> b2g + eg on a tetragonal
    site with its own axis, or mixing between groups of the same type). Groups are in the Cartesian axes of the
    structure; a site whose symmetry axis is not along x, y or z keeps a coarser split plus its crystal field.
    Returns {groups: [...] (the whole shell [family] when nothing finer is allowed), mixing: number of e_mix terms
    per class of sites, site_symmetry: point groups of the element's sites}."""
    sites = [i for i, site in enumerate(structure) if site.specie.symbol == element]
    syms = site_symmetry_symbols(structure, symprec)
    out = {"groups": [family], "mixing": 0, "site_symmetry": sorted({syms[i] for i in sites})}
    if family not in GROUP_SETS or not sites:
        return out
    for groups in GROUP_SETS[family]:
        orbs = [list(groups) if i in sites else [] for i in range(len(structure))]
        if not check_orbital_symmetry(structure, orbs, symprec):
            out["groups"] = list(groups)
            break
    orbs = [list(out["groups"]) if i in sites else [] for i in range(len(structure))]
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        eq = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_dataset()
        eq = getattr(eq, "equivalent_atoms", None) if not isinstance(eq, dict) else eq["equivalent_atoms"]
        _, keys = onsite_mixing_terms(structure, orbs, [int(v) for v in eq], elements=[element], symprec=symprec)
        classes = {int(eq[i]) for i in sites}
        out["mixing"] = len(keys) // max(1, len(classes))
    except Exception:  # noqa: BLE001 - the split alone is still useful
        pass
    return out


def _cubic_f_basis():
    """Rows of the cubic f harmonics in SKFitter's real f orbitals (fy3x2, fxyz, fyz2, fz3, fxz2, fzx2, fx3)."""
    targets = {
        "f_a2u": [{(1, 1, 1): 1.0}],
        "f_t1u": [{(3, 0, 0): 2.0, (1, 2, 0): -3.0, (1, 0, 2): -3.0},      # x(5x2 - 3r2)
                  {(0, 3, 0): 2.0, (2, 1, 0): -3.0, (0, 1, 2): -3.0},      # y(5y2 - 3r2)
                  {(0, 0, 3): 2.0, (2, 0, 1): -3.0, (0, 2, 1): -3.0}],     # z(5z2 - 3r2)
        "f_t2u": [{(1, 2, 0): 1.0, (1, 0, 2): -1.0},                        # x(y2 - z2)
                  {(0, 1, 2): 1.0, (2, 1, 0): -1.0},                        # y(z2 - x2)
                  {(2, 0, 1): 1.0, (0, 2, 1): -1.0}],                       # z(x2 - y2)
    }
    H = _harmonics(3)
    out = {}
    for name, polys in targets.items():
        rows = []
        for q in polys:
            c = np.array([_poly_inner(h[3], q) for h in H])
            rows.append(c / np.linalg.norm(c))
        out[name] = np.array(rows)
    return out


_CUBIC_F = _cubic_f_basis()


class SKFitter:
    SUB_ORBITALS = {
        's':      {'dim': 1, 'indices': [0]},
        's*':     {'dim': 1, 'indices': [0]},
        'px':     {'dim': 1, 'indices': [0]},
        'py':     {'dim': 1, 'indices': [1]},
        'pz':     {'dim': 1, 'indices': [2]},
        'dxy':    {'dim': 1, 'indices': [0]},
        'dyz':    {'dim': 1, 'indices': [1]},
        'dzx':    {'dim': 1, 'indices': [2]},
        'dx2y2':  {'dim': 1, 'indices': [3]},
        'dz2':    {'dim': 1, 'indices': [4]},

        'pxy':    {'dim': 2, 'indices': [0, 1]},
        'p':      {'dim': 3, 'indices': [0, 1, 2]},
        'd_e1':   {'dim': 2, 'indices': [1, 2]},
        'd_e2':   {'dim': 2, 'indices': [0, 3]},
        'd_t2g':  {'dim': 3, 'indices': [0, 1, 2]},
        'd_eg':   {'dim': 2, 'indices': [3, 4]},
        'd':      {'dim': 5, 'indices': [0, 1, 2, 3, 4]},

        # f: fy3x2, fxyz, fyz2, fz3, fxz2, fzx2 (= z(x2-y2)), fx3 (VASP's order, m = -3 ... 3); sets by |m|
        'fy3x2':  {'dim': 1, 'indices': [0]},
        'fxyz':   {'dim': 1, 'indices': [1]},
        'fyz2':   {'dim': 1, 'indices': [2]},
        'fz3':    {'dim': 1, 'indices': [3]},
        'fxz2':   {'dim': 1, 'indices': [4]},
        'fzx2':   {'dim': 1, 'indices': [5]},
        'fx3':    {'dim': 1, 'indices': [6]},
        'f_m0':   {'dim': 1, 'indices': [3]},
        'f_m1':   {'dim': 2, 'indices': [2, 4]},
        'f_m2':   {'dim': 2, 'indices': [1, 5]},
        'f_m3':   {'dim': 2, 'indices': [0, 6]},
        # cubic f groups (O_h / T_d sites, cubic axes along x, y, z): a2u = xyz; t1u = x(5x2-3r2), y(5y2-3r2),
        # z(5z2-3r2); t2u = x(y2-z2), y(z2-x2), z(x2-y2). 'indices' = the real f orbitals they are built from.
        'f_a2u':  {'dim': 1, 'indices': [1], 'basis': _CUBIC_F['f_a2u'], 'labels': ['fxyz']},
        'f_t1u':  {'dim': 3, 'indices': [0, 2, 3, 4, 6], 'basis': _CUBIC_F['f_t1u'], 'labels': ['fx3c', 'fy3c', 'fz3c']},
        'f_t2u':  {'dim': 3, 'indices': [0, 2, 4, 5, 6], 'basis': _CUBIC_F['f_t2u'], 'labels': ['fx(y2-z2)', 'fy(z2-x2)', 'fz(x2-y2)']},
        'f':      {'dim': 7, 'indices': [0, 1, 2, 3, 4, 5, 6]}
    }

    overlap_floor = 0.05         # non-orthogonal fits: smallest eigenvalue S(k) may take at the fitted k-points
    k_workers = 1          # threads over k-point blocks in the Brillouin-zone sums (skfitter berry --workers)
    k_progress = None      # callable(level, k-points done, k-points of the level): progress of those sums
    refine_window = 0.0    # eV; > 0: only bands within this of the scanned Fermi energies trigger adaptive refinement

    FAMILY_MAP = {
        's': 's', 's*': 's*',
        'px': 'p', 'py': 'p', 'pz': 'p', 'pxy': 'p', 'p': 'p',
        'dxy': 'd', 'dyz': 'd', 'dzx': 'd', 'dx2y2': 'd', 'dz2': 'd',
        'd_e1': 'd', 'd_e2': 'd', 'd_t2g': 'd', 'd_eg': 'd', 'd': 'd',
        'fy3x2': 'f', 'fxyz': 'f', 'fyz2': 'f', 'fz3': 'f', 'fxz2': 'f', 'fzx2': 'f', 'fx3': 'f',
        'f_m0': 'f', 'f_m1': 'f', 'f_m2': 'f', 'f_m3': 'f', 'f_a2u': 'f', 'f_t1u': 'f', 'f_t2u': 'f', 'f': 'f'
    }

    def __init__(self, vasprun_path=None, kpoints_path=None, orbital_config=None, magnetic_orbitals=None,
                 dft_nelect=None, tb_nelect=None, n_bands_below=6, n_bands_above=5, n_neighbors=2,
                 symprec=1e-3, fit_eta=True, fixed_eta=2.5, include_soc=None, non_orthogonal=False,
                 use_fast_model=True, dft_code="vasp", qe_xml_path=None, qe_bands_input=None, qe_scf_path=None,
                 moment_signs=None, hopping_mode="decay", onsite_mixing=None, moment_directions=None,
                 soc_orbitals=None):
        """
        magnetic_orbitals: {element: [...]} - what carries the exchange splitting: orbital families ('s', 'p', 'd', 'f';
                      one delta_spin_<El>_<family> for the whole shell) and/or symmetry subshells of the element's
                      orbital set ('d_t2g', 'd_eg', 'dz2', 'd_e1', 'd_e2', 'pz', 'pxy', ...; their own
                      delta_spin_<El>_<subshell>). E.g. {'Fe': ['s', 'd_t2g', 'd_eg']} with d split into t2g / e_g
                      fits separate t2g and e_g splittings (see exchange_key).
        soc_orbitals: {element: ['p', 'd', 'f', ...]} - the orbital families that carry on-site spin-orbit coupling
                      lambda_<El>_<family> L.S when include_soc is on (None, the default: every p, d and f family of
                      the model). E.g. {'Bi': ['p'], 'Eu': ['d']} leaves out Eu 4f, which a scalar-relativistic
                      pseudopotential treats without spin-orbit coupling.
        onsite_mixing: False / None (default), True (every element) or a list of elements: add the on-site terms
                      beyond one energy per group that the site symmetry allows (onsite_mixing_terms): the
                      mixing of two groups of the same symmetry type (d E_1 / E_2 on a trigonal site), of
                      families on a site without inversion (s - d_z2, p_z - d_z2), and the crystal field of a group
                      that the site splits further. Parameters e_mix<eq>_<El>_<n> (eV).
        moment_directions: the direction of every site's moment (3-vectors, normalised; None: from the DFT run - the
                      moment vectors of a non-collinear run, or +-z from moment_signs). The exchange term is
                      -(delta/2) n_i . sigma on the site's magnetic orbitals.
        hopping_mode: how the two-centre integrals depend on the bond length.
                      'decay' (default): one integral V0 per bond type at the nearest-neighbour distance d0, scaled
                      to longer bonds by (d0/r)^eta (eta fitted, or fixed_eta);
                      'shell': an independent integral per neighbour shell, V_n (keys v_<type>_<n>, n = 0 for the
                      nearest shell); no eta is fitted. Off the fitted bond lengths (strain) V_n is scaled by
                      (r_n/r)^eta with the fixed eta (fixed_eta).
        moment_signs: the direction (+1 / -1) of the magnetic moment of every site, for the exchange splitting
                      δ(site) = sign × delta_spin_<element>_<family> (or _<subshell>, see magnetic_orbitals). None:
                      from the signs of the VASP MAGMOM, or of the site moments in the pw.x SCF output (Quantum ESPRESSO)
                      (antiferromagnets, altermagnets), else +1 on every site (ferromagnet).
                      With sign × δ > 0 the site's spin-up orbitals lie δ/2 below and its spin-down orbitals δ/2
                      above the on-site energy: spin up is the majority, so the model's moments point like the DFT's.
        DFT source (dft_code):
          'vasp'      vasprun_path = vasprun.xml of the band run, kpoints_path = its line-mode KPOINTS file.
          'espresso'  (Quantum ESPRESSO; also 'qe') qe_xml_path = <outdir>/<prefix>.xml (or <prefix>.save/
                      data-file-schema.xml) written by the calculation='bands' run (pw.x >= 6.2);
                      qe_bands_input = the pw.x input of that run (its K_POINTS crystal_b / tpiba_b card gives the
                      high-symmetry points and, from '! X' comments, their labels; optional - without it the
                      segments are found from the k-points and only Γ is labelled);
                      qe_scf_path = optional SCF output (pw.x stdout or XML) to take E_F from when the bands XML
                      does not hold the SCF Fermi energy.
        """
        self.dft_code = normalise_dft_code(dft_code)
        self.vasprun_path = vasprun_path
        self.qe_xml_path = qe_xml_path
        if self.dft_code == "espresso" and not qe_bands_input:
            qe_bands_input = kpoints_path                  # kpoints_path may name the pw.x bands input as well
        self.qe_bands_input = qe_bands_input
        self.qe_scf_path = qe_scf_path
        # kpoints_path = the file the k-path is read from (KPOINTS for VASP, the pw.x bands input for QE)
        self.kpoints_path = qe_bands_input if self.dft_code == "espresso" else kpoints_path
        if self.dft_code == "vasp" and (not vasprun_path or not kpoints_path):
            raise ValueError("dft_code='vasp' needs vasprun_path and kpoints_path.")
        if self.dft_code == "espresso" and not qe_xml_path:
            raise ValueError("dft_code='espresso' needs qe_xml_path (the XML data file of the pw.x bands run).")
        self.orbital_config = orbital_config
        self.magnetic_orbitals = magnetic_orbitals or {}
        self.soc_orbitals = {k: list(v) for k, v in soc_orbitals.items()} if soc_orbitals else None
        self.tb_nelect = tb_nelect 
        self.dft_nelect = dft_nelect 
        self.n_bands_below = n_bands_below
        self.n_bands_above = n_bands_above
        self.n_neighbors = n_neighbors
        self.symprec = symprec
        
        self.fit_eta = fit_eta
        self.fixed_eta = fixed_eta
        self.hopping_mode = normalise_hopping_mode(hopping_mode)
        self.non_orthogonal = non_orthogonal
        self.use_fast_model = use_fast_model   # vectorised band evaluation (see _prepare_fast_model)

        self.efermi = 0.0
        self.structure = None
        self.kpoints = []
        
        self.target_indices = []
        self.target_dft_bands = None 
        self._sqrt_weights = None              # optional per-(k, band) fit weights, see fit(add_weight_at_*)
        
        self.is_magnetic = False
        self._magmom = None                    # the initial moments of the DFT run (VASP MAGMOM), if known
        self._magmom_vec = None                # their vectors (non-collinear runs), if known

        self._setup_structure_and_symmetry(include_soc)
        self._setup_orbital_basis(orbital_config)
        errs = check_magnetic_orbitals(self.magnetic_orbitals,
                                       {self.structure[i].specie.symbol: list(self.site_orbitals[i])
                                        for i in range(len(self.structure))})
        if errs:
            raise ValueError("magnetic_orbitals: " + "; ".join(errs))
        errs = check_soc_orbitals(self.soc_orbitals,
                                  {self.structure[i].specie.symbol: list(self.site_orbitals[i])
                                   for i in range(len(self.structure))}, self.FAMILY_MAP)
        if errs:
            raise ValueError("soc_orbitals: " + "; ".join(errs))
        errs = check_orbital_symmetry(self.structure, self.site_orbitals, symprec=self.symprec)
        if errs:
            raise ValueError("orbital groups: " + "; ".join(errs))
        self.onsite_mixing = onsite_mixing
        self.onsite_mix_terms = [[] for _ in range(len(self.structure))]
        if onsite_mixing:
            els = None if onsite_mixing is True else list(onsite_mixing)
            self.onsite_mix_terms, mix_keys = onsite_mixing_terms(self.structure, self.site_orbitals,
                                                                  self.eq_site_map, els, symprec=self.symprec)
            print(f"[INFO] On-site mixing: {len(mix_keys)} parameters ({', '.join(mix_keys) or 'none allowed'})")

        if self.dft_code == "espresso":
            self._load_espresso_data()
        else:
            self._load_vasp_data()
        if not self.is_magnetic and self.noncollinear_dft and self.magnetic_orbitals:
            # a non-collinear (spin-orbit) run has one list of spinor bands, so magnetism cannot be seen as distinct
            # up / down bands: the model is magnetic when magnetic orbitals are chosen (moments along z, their
            # directions from MAGMOM)
            self.is_magnetic = True
            print("[INFO] Non-collinear DFT data with magnetic orbitals: magnetic mode ENABLED (moments along z).")
        self.site_moment_signs = self._moment_signs(moment_signs)
        self.site_moment_dirs = self._moment_dirs(moment_directions)

        self.total_dim = 2 * self.base_dim if (self.soc or self.is_magnetic) else self.base_dim
        print(f"[INFO] Spatial Basis (N) = {self.base_dim} | Total Hamiltonian Dimension = {self.total_dim}")
        print(f"[INFO] Fitting Basis Mode: {'NON-ORTHOGONAL' if self.non_orthogonal else 'ORTHOGONAL'}")
        
        self.target_tb_band_indices = [] 
        self.tb_bands = None                      
        self._calculate_tb_filling()
        
        self.cutoff_radius = self._get_cutoff_from_neighbors(self.n_neighbors)
        self._precompute_neighbor_pairs()

    def _moment_signs(self, given=None):
        """+1 / -1 per site: the direction of each site's moment in the exchange splitting (see __init__)."""
        n = len(self.structure)
        if given is not None:
            signs = [(-1 if float(v) < 0 else 1) for v in given]
            if len(signs) != n:
                raise ValueError(f"moment_signs needs one sign per site ({n}), got {len(signs)}")
        elif self._magmom is not None:
            signs = [(-1 if m < 0 else 1) for m in self._magmom]
        else:
            signs = [1] * n
        if any(v < 0 for v in signs):
            desc = ", ".join(f"{self.structure[i].specie.symbol}{i + 1} {'+' if v > 0 else '-'}" for i, v in enumerate(signs)
                             if self.structure[i].specie.symbol in self.magnetic_orbitals)
            print(f"[INFO] Antiparallel moments (exchange splitting per site): {desc or 'no magnetic sites'}")
        return signs

    @property
    def site_moment_dirs(self):
        """Unit vector n_i of every site's moment: the directions set explicitly (non-collinear moments, or
        moment_directions), else s_i z from site_moment_signs (so changing the signs changes the model)."""
        explicit = getattr(self, "_moment_dirs_explicit", None)
        if explicit is not None:
            return explicit
        signs = getattr(self, "site_moment_signs", None)
        if signs is None:
            return None
        return [np.array([0.0, 0.0, float(s)]) for s in signs]

    @site_moment_dirs.setter
    def site_moment_dirs(self, dirs):
        """Set the directions (None: back to s_i z from site_moment_signs). Directions along +-z are stored as
        signs, so the collinear case keeps one source of truth."""
        if dirs is None:
            self._moment_dirs_explicit = None
            return
        dirs = [np.asarray(d, float).reshape(3) / max(np.linalg.norm(d), 1e-12) for d in dirs]
        if all(abs(d[2]) > 1 - 1e-9 for d in dirs):
            self.site_moment_signs = [1 if d[2] > 0 else -1 for d in dirs]
            self._moment_dirs_explicit = None
        else:
            self._moment_dirs_explicit = dirs

    def _moment_dirs(self, given=None):
        """Unit vector n_i of every site's moment for the exchange term -(delta/2) n_i . sigma: given, or the moment
        vectors of a non-collinear DFT run (sites without a moment: the sign along z), else s_i z (collinear)."""
        n = len(self.structure)
        signs = self.site_moment_signs
        dirs = [np.array([0.0, 0.0, float(s)]) for s in signs]
        src = given if given is not None else (self._magmom_vec if getattr(self, "noncollinear_dft", False) else None)
        if src is not None:
            if len(src) != n:
                raise ValueError(f"moment_directions needs one vector per site ({n}), got {len(src)}")
            for i, v in enumerate(src):
                v = np.asarray(v, float).reshape(3)
                if np.linalg.norm(v) > 1e-6:
                    dirs[i] = v / np.linalg.norm(v)
        if any(abs(d[2]) < 1 - 1e-6 for d in dirs) and self.is_magnetic:
            desc = ", ".join(f"{self.structure[i].specie.symbol}{i + 1} ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})"
                             for i, d in enumerate(dirs) if self.structure[i].specie.symbol in self.magnetic_orbitals)
            print(f"[INFO] Non-collinear moment directions: {desc}")
        return dirs

    def _vasp_site_moments(self, vrun, noncollinear, vectors=False):
        """The initial moment of every site from MAGMOM (the z component - along SAXIS - of a non-collinear run): the
        value the run used (vasprun <parameters>), else the INCAR's; None if neither has one per site."""
        n = len(self.structure)
        for source in (getattr(vrun, "parameters", None), getattr(vrun, "incar", None)):
            try:
                magmom = [float(m) for m in np.ravel((source or {}).get("MAGMOM") or [])]
            except (TypeError, ValueError, AttributeError):
                continue
            if noncollinear and len(magmom) == 3 * n:
                saxis = np.ravel((getattr(vrun, "parameters", {}) or {}).get("SAXIS", [0, 0, 1])).astype(float)
                if not np.allclose(saxis / np.linalg.norm(saxis), [0, 0, 1]):
                    print(f"[WARN] SAXIS = {list(saxis)}: MAGMOM is read in the Cartesian frame (moments along z "
                          "of SAXIS are not rotated).")
                if vectors:
                    return [np.array(magmom[3 * i:3 * i + 3]) for i in range(n)]
                return magmom[2::3]
            if not noncollinear and len(magmom) == n:
                return magmom
        return None

    def _pairs_of_bands(self):
        """True when bands are counted in Kramers pairs (n_bands_below / above): spin-orbit coupling, except for
        the merged up / down channels of a collinear magnetic run (counted one by one)."""
        collinear_magnetic = self.is_magnetic and not getattr(self, "noncollinear_dft", False)
        return bool(self.soc and not collinear_magnetic)

    def _setup_structure_and_symmetry(self, include_soc=None):
        """Structure, E_F, NELECT and the SOC flag from the DFT source, then the space group of the structure."""
        if self.dft_code == "espresso":
            dft_soc = self._read_espresso_structure()
            code = "Quantum ESPRESSO"
        else:
            vrun = Vasprun(self.vasprun_path, parse_projected_eigen=False)
            self.efermi = vrun.efermi if vrun.efermi is not None else read_vasprun_efermi(self.vasprun_path)
            if self.efermi is None:
                raise ValueError(f"{self.vasprun_path} holds no Fermi energy (<i name=\"efermi\">).")
            self.structure = vrun.final_structure
            self.dft_nelect = vrun.parameters.get("NELECT")
            params = getattr(vrun, "parameters", {})
            dft_soc = bool(params.get("LSORBIT", False) or params.get("LNONCOLLINEAR", False))
            code = "VASP"
            self._magmom = self._vasp_site_moments(vrun, dft_soc)
            self._magmom_vec = self._vasp_site_moments(vrun, dft_soc, vectors=True) if dft_soc else None

        self.lattice_vectors = self.structure.lattice.matrix

        if include_soc is not None:
            self.soc = include_soc
        else:
            self.soc = dft_soc

        self.noncollinear_dft = bool(dft_soc)     # spinor bands (one list), not up / down channels
        print(f"[INFO] SOC mode: {'ENABLED' if self.soc else 'DISABLED'} ({code} auto-detected: {dft_soc})")

        sga = SpacegroupAnalyzer(self.structure, symprec=self.symprec)
        sym_dataset = sga.get_symmetry_dataset()
        
        self.eq_site_map = getattr(sym_dataset, "equivalent_atoms", sym_dataset["equivalent_atoms"] if isinstance(sym_dataset, dict) else None)
        self.spacegroup = getattr(sym_dataset, "international", sym_dataset["international"] if isinstance(sym_dataset, dict) else "Unknown")
        spacegroup_num = getattr(sym_dataset, "number", sym_dataset["number"] if isinstance(sym_dataset, dict) else "N/A")

        print(f"[INFO] Crystal Spacegroup: {self.spacegroup} (No. {spacegroup_num})")

    def _setup_orbital_basis(self, orbital_config):
        self.site_orbitals = []
        self.site_dims = []
        self.site_offsets = []
        self.site_indices = []
        current_offset = 0

        for idx, site in enumerate(self.structure):
            elem = site.specie.symbol
            Z = site.specie.Z

            if orbital_config is None:
                assigned = [get_valence_orbitals_from_Z(Z)]
            elif isinstance(orbital_config, dict):
                assigned = orbital_config.get(idx, orbital_config.get(elem, get_valence_orbitals_from_Z(Z)))
            elif isinstance(orbital_config, (list, tuple)):
                assigned = orbital_config[idx]
            else:
                assigned = [orbital_config]

            if isinstance(assigned, str):
                assigned = [assigned]

            self.site_orbitals.append(assigned)
            
            site_indices_list = []
            site_dim = 0
            for sub in assigned:
                indices = self.SUB_ORBITALS[sub]['indices']
                site_indices_list.extend(indices)
                site_dim += self.SUB_ORBITALS[sub]['dim']

            self.site_indices.append(site_indices_list)
            self.site_dims.append(site_dim)
            self.site_offsets.append(current_offset)
            current_offset += site_dim

        self.base_dim = current_offset

    @staticmethod
    def _normalise_kpoint_label(label):
        if label.strip().lower() in ('g', 'gamma', '\\gamma', 'γ'):
            return 'Γ'
        return label

    @staticmethod
    def _read_kpath_file(path):
        """
        Read a line-mode KPOINTS file. Returns (points_per_segment, [(coords, label), ...]).
        Copes with an optional weight column ("kx ky kz 1 LABEL"), "! LABEL" / "# LABEL"
        comment-style labels, and UTF-8 labels (e.g. Γ) regardless of the platform default
        encoding.
        """
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.strip() for line in f if line.strip()]

        n_per_segment = int(lines[1].split()[0])
        points = []
        for line in lines[4:]:
            parts = line.replace('!', ' ').replace('#', ' ').split()
            if len(parts) < 3:
                continue
            try:
                coords = [float(x) for x in parts[:3]]
            except ValueError:
                continue
            rest = parts[3:]
            if rest:
                try:                      # numeric 4th column = weight, label follows
                    float(rest[0])
                    rest = rest[1:]
                except ValueError:
                    pass
            label = SKFitter._normalise_kpoint_label(rest[0]) if rest else ""
            points.append((coords, label))
        return n_per_segment, points

    # ---- the k-path: a list of straight segments with their DFT index ranges (see vasp_kpath_segments) ----
    def _read_vasp_kpath_segments(self):
        """Segments of the VASP line-mode KPOINTS file (empty list if there is none)."""
        if not self.kpoints_path or not os.path.exists(self.kpoints_path):
            return []
        try:
            n, pts = self._read_kpath_file(self.kpoints_path)
            return vasp_kpath_segments(n, pts)
        except Exception as e:
            print(f"[WARN] Could not read the k-path from {self.kpoints_path}: {e}")
            return []

    def _get_kpath_segments(self):
        segs = getattr(self, "kpath_segments", None)
        if segs is None:
            segs = self._read_vasp_kpath_segments() if getattr(self, "dft_code", "vasp") == "vasp" else []
            self.kpath_segments = segs
        return segs

    def _parse_kpoint_labels(self):
        """
        Tick positions/labels for band plots from the k-path segments: the first k-point and the last point of
        every segment (VASP line mode writes the junction point twice, Quantum ESPRESSO once); a jump between
        segments gets a combined label such as 'H|P'.
        """
        try:
            self.high_sym_indices, self.high_sym_labels = kpath_ticks(self._get_kpath_segments(), len(self.kpoints))
        except Exception as e:
            self.high_sym_indices, self.high_sym_labels = [], []
            print(f"[WARN] Could not work out the high-symmetry labels from {self.kpoints_path}: {e}")
    
    def _calculate_dft_filling(self):
        if self.dft_nelect is not None:
            nelect = self.dft_nelect
        else:
            raise ValueError("Cannot calculate DFT filling: 'dft_nelect' is missing.")

        is_spin_polarized = getattr(self, 'soc', False) or getattr(self, 'is_magnetic', False)
    
        if is_spin_polarized:
            num_occupied_bands = int(np.round(nelect))
        else:
            num_occupied_bands = int(np.round(nelect / 2.0))

        self.dft_vbm_index = max(0, num_occupied_bands - 1)
        return self.dft_vbm_index

    def _load_vasp_data(self):
        vrun = Vasprun(self.vasprun_path, parse_projected_eigen=False)
        eigen_dict = vrun.eigenvalues
        
        if Spin.up in eigen_dict and Spin.down in eigen_dict:
            up_test = eigen_dict[Spin.up][:, :, 0]
            down_test = eigen_dict[Spin.down][:, :, 0]
            if not np.allclose(up_test, down_test, atol=1e-4):
                self.is_magnetic = True
                print("[INFO] VASP Data shows distinct spin bands. Magnetic mode ENABLED.")
        
        if Spin.up in eigen_dict:
            up_energies = eigen_dict[Spin.up][:, :, 0] - self.efermi
            if Spin.down in eigen_dict:
                down_energies = eigen_dict[Spin.down][:, :, 0] - self.efermi
                if self.is_magnetic:
                    all_dft_energies = np.sort(np.hstack([up_energies, down_energies]), axis=1)
                else:
                    all_dft_energies = up_energies
            else:
                all_dft_energies = up_energies
        else:
            all_dft_energies = list(eigen_dict.values())[0][:, :, 0] - self.efermi

        recip_lattice = self.structure.lattice.reciprocal_lattice
        raw_kpoints = recip_lattice.get_cartesian_coords(vrun.actual_kpoints)

        weights = getattr(vrun, "kpoints_weights", None)
        if weights is not None and len(weights) == len(raw_kpoints):
            weights = np.array(weights)
            if np.any(weights == 0) and np.any(weights > 0):
                mask = (weights == 0)
                raw_kpoints = raw_kpoints[mask]
                all_dft_energies = all_dft_energies[mask, :]

        valid_mask = [True]
        for idx in range(1, len(raw_kpoints)):
            dist = np.linalg.norm(raw_kpoints[idx] - raw_kpoints[idx-1])
            if dist < 1e-5 and idx > len(raw_kpoints) - 5:
                valid_mask.append(False)
            else:
                valid_mask.append(True)

        valid_mask = np.array(valid_mask)
        self.kpoints = raw_kpoints[valid_mask]
        all_dft_energies = all_dft_energies[valid_mask, :]

        self._select_target_bands(all_dft_energies)
        self.kpath_segments = self._read_vasp_kpath_segments()
        self._parse_kpoint_labels()

    def _select_target_bands(self, all_dft_energies):
        """The fitted DFT window (n_bands_below / n_bands_above around the VBM) of the (nk, nbands) array."""
        num_kpoints, num_bands = all_dft_energies.shape

        if not hasattr(self, 'dft_vbm_index') or self.dft_vbm_index is None:
            self._calculate_dft_filling()

        i_vmax = self.dft_vbm_index

        spin_factor = 2 if self._pairs_of_bands() else 1
        bands_below = self.n_bands_below * spin_factor
        bands_above = self.n_bands_above * spin_factor

        start_idx = max(0, i_vmax - bands_below + 1)
        end_idx = min(num_bands, i_vmax + bands_above + 1)

        self.target_indices = list(range(start_idx, end_idx))
        self.target_dft_bands = all_dft_energies[:, self.target_indices]
        self._all_dft_energies = np.asarray(all_dft_energies, float)
        self.state_filter = None
        
        print(f"[INFO] Dynamic Band Alignment (SOC={self.soc}, MAG={self.is_magnetic}) - VBM Index: {i_vmax}")
        print(f"[INFO] Window: [{start_idx}:{end_idx}] | Target DFT bands shape: {self.target_dft_bands.shape}")

    def apply_scissor(self, delta):
        """Shift the DFT conduction bands (every band above the last occupied one, dft_vbm_index) by ``delta`` eV
        before fitting: the fit then reproduces a corrected gap (e.g. from experiment or a hybrid functional)
        while keeping its physical parameters and distance scaling. Needs a gap in the DFT bands; call after the
        band window (and any state filter) is set. Repeated calls add up (``self.scissor`` holds the total)."""
        delta = float(delta)
        if delta == 0.0:
            return
        E = getattr(self, "_all_dft_energies", None)
        if E is None or self.target_dft_bands is None:
            raise ValueError("No DFT bands loaded.")
        vbm = int(self.dft_vbm_index)
        if vbm + 1 >= E.shape[1]:
            raise ValueError("Scissor: there is no conduction band in the DFT data.")
        gap = float(np.nanmin(E[:, vbm + 1]) - np.nanmax(E[:, vbm]))
        if gap <= 0:
            raise ValueError(f"Scissor: the DFT bands have no gap (the conduction band minimum is {-gap:.3f} eV below "
                             "the valence band maximum), so 'conduction bands' are not defined.")
        E = np.array(E, float)
        E[:, vbm + 1:] += delta
        self._all_dft_energies = E
        n_below = sum(1 for i in self.target_indices if i <= vbm)
        T = np.array(self.target_dft_bands, float)
        T[:, n_below:] += delta
        self.target_dft_bands = T
        self.scissor = float(getattr(self, "scissor", 0.0)) + delta
        print(f"[INFO] Scissor: DFT conduction bands shifted by {delta:+.4f} eV (DFT gap {gap:.4f} -> "
              f"{gap + delta:.4f} eV)")

    # ---- Quantum ESPRESSO ----
    def _read_espresso_structure(self):
        """Structure, E_F and NELECT from the pw.x XML file (see read_qe_xml). Returns the SOC/noncollinear flag."""
        from pymatgen.core import Lattice, Structure
        data = read_qe_xml(self.qe_xml_path)
        self._qe_data = data
        self.structure = Structure(Lattice(data["lattice"]), data["symbols"], data["cart_coords"],
                                   coords_are_cartesian=True)
        efermi, source = None, None
        if self.qe_scf_path:
            efermi = read_qe_fermi(self.qe_scf_path)
            if efermi is None:
                print(f"[WARN] No Fermi energy found in {self.qe_scf_path}; using the one of {self.qe_xml_path}.")
            else:
                source = os.path.basename(self.qe_scf_path)
        if efermi is None and data["efermi"] is not None:
            efermi, source = data["efermi"], f"{os.path.basename(self.qe_xml_path)} <{data['efermi_source']}>"
        if efermi is None:
            raise ValueError(f"{self.qe_xml_path} holds no Fermi energy: pass qe_scf_path=<pw.x SCF output or XML>.")
        self.efermi = float(efermi)
        if self.qe_scf_path:                     # the moment directions (up / down per site), as VASP's MAGMOM
            self._magmom = read_qe_site_moments(self.qe_scf_path, len(self.structure))
            self._magmom_vec = read_qe_site_moments(self.qe_scf_path, len(self.structure), vectors=True)
        if self.dft_nelect is None:
            self.dft_nelect = data["nelec"]
        calc = data.get("calculation")
        if calc and calc.lower() != "bands":
            print(f"[WARN] {os.path.basename(self.qe_xml_path)} comes from a calculation='{calc}' run; the fit expects "
                  "the bands along a k-path (calculation='bands').")
        print(f"[INFO] Quantum ESPRESSO: {len(data['kpoints_cart'])} k-points, {data['eig_up'].shape[1]} bands"
              f"{' per spin' if data['lsda'] else ''} | NELEC = {data['nelec']} | E_F = {self.efermi:.4f} eV "
              f"(from {source}) | lsda={data['lsda']} noncolin={data['noncolin']} spinorbit={data['spinorbit']}")
        return bool(data["spinorbit"] or data["noncolin"])

    def _load_espresso_data(self):
        """Eigenvalues (relative to E_F) and k-points of the pw.x bands run, the fitted window and the k-path."""
        data = getattr(self, "_qe_data", None) or read_qe_xml(self.qe_xml_path)
        up = np.asarray(data["eig_up"], float) - self.efermi
        down = data["eig_down"]
        if down is not None:
            down = np.asarray(down, float) - self.efermi
            if down.shape != up.shape or not np.allclose(up, down, atol=1e-4):
                self.is_magnetic = True
                print("[INFO] Quantum ESPRESSO data shows distinct spin bands (lsda). Magnetic mode ENABLED.")
        all_dft_energies = np.sort(np.hstack([up, down]), axis=1) if self.is_magnetic else up

        self.kpoints = np.asarray(data["kpoints_cart"], float)
        self._select_target_bands(all_dft_energies)
        self.kpath_segments = self._read_espresso_kpath_segments(data)
        self._parse_kpoint_labels()

    def _read_espresso_kpath_segments(self, data):
        """Segments of the pw.x K_POINTS crystal_b / tpiba_b path (qe_bands_input), checked against the k-points of
        the XML file; without a usable input file they are found from the k-points themselves."""
        kf = np.asarray(data["kpoints_frac"], float)
        path = self.qe_bands_input
        if path:
            try:
                kp = read_qe_kpath(path)
                if not kp["band_path"]:
                    raise ValueError(f"K_POINTS {kp['units']} is not a band path (use crystal_b or tpiba_b)")
                nodes = np.array(kp["nodes"], float)
                if kp["units"].startswith("tpiba"):          # Cartesian, units 2 pi / alat -> fractional
                    nodes = nodes @ np.asarray(data["lattice"]).T / data["alat"]
                segs = qe_kpath_segments(nodes, kp["weights"], kp["labels"], len(kf))
                for sg in segs:
                    if not (np.allclose(kf[sg["i0"]], sg["k0"], atol=1e-4)
                            and np.allclose(kf[sg["i1"]], sg["k1"], atol=1e-4)):
                        raise ValueError("its high-symmetry points do not coincide with the k-points of "
                                         f"{os.path.basename(self.qe_xml_path)}")
                print(f"[INFO] k-path from {os.path.basename(path)}: {len(segs)} segment(s), "
                      f"nodes {' '.join(kpath_ticks(segs)[1])}")
                return segs
            except (OSError, ValueError, IndexError) as exc:
                print(f"[WARN] Could not use the k-path of {path} ({exc}); finding the segments from the k-points.")
        segs = detect_kpath_segments(kf, data["lattice"])
        print(f"[INFO] k-path found from the k-points: {len(segs)} straight segment(s) (only Γ is labelled; pass the "
              "pw.x bands input for the labels)")
        return segs

    def _calculate_tb_filling(self):
        if self.tb_nelect is not None:
            nelect = self.tb_nelect
        else:
            nelect = 0.0
            for i, site in enumerate(self.structure):
                elem = site.specie.element if hasattr(site.specie, 'element') else site.specie
                orbs = self.site_orbitals[i]
                group = getattr(elem, 'group', 0)
                
                val = get_element_valency_new(elem)
                if val == 'N/A': val = 0
                
                fam_dims = {'s': 0, 's*': 0, 'p': 0, 'd': 0, 'f': 0}
                for orb in orbs:
                    fam = self.FAMILY_MAP.get(orb, 's')
                    fam_dims[fam] += self.SUB_ORBITALS[orb]['dim']
                
                e_count = 0.0
                if getattr(elem, 'is_lanthanoid', False) or getattr(elem, 'is_actinoid', False):
                    # rare earths: the outer s, d and f electrons of the free atom (e.g. Ce 4f1 5d1 6s2)
                    occ = {'s': 0.0, 'd': 0.0, 'f': 0.0}
                    n_outer = 6 if getattr(elem, 'Z', 0) <= 71 else 7
                    for n_shell, l_name, n_el in getattr(elem, 'full_electronic_structure', []):
                        if (l_name == 's' and n_shell == n_outer) or (l_name == 'd' and n_shell == n_outer - 1) \
                                or (l_name == 'f' and n_shell == n_outer - 2):
                            occ[l_name] += n_el
                    e_count += fam_dims['s'] * occ['s']
                    e_count += (fam_dims['d'] / 5.0) * occ['d']
                    e_count += (fam_dims['f'] / 7.0) * occ['f']
                elif 13 <= group <= 15 and fam_dims['s'] == 0 and fam_dims['p'] == 1:
                    # a single p orbital (pz of a planar sp2 layer): the pi electrons, B 0, C 1, N 2 (as in hBN)
                    e_count = float(group - 13)
                elif 13 <= group <= 18:
                    s_e = min(2, val)
                    p_e = max(0, val - s_e)
                    e_count += (fam_dims['s'] / 1.0) * s_e
                    e_count += (fam_dims['p'] / 3.0) * p_e
                elif 3 <= group <= 12:
                    s_e = min(2, val)
                    d_e = max(0, val - s_e)
                    e_count += (fam_dims['s'] / 1.0) * s_e
                    e_count += (fam_dims['d'] / 5.0) * d_e
                elif 1 <= group <= 2:
                    s_e = val
                    e_count += (fam_dims['s'] / 1.0) * s_e
                    
                nelect += e_count

            nelect = round(nelect)
            self.tb_nelect = nelect
        
        is_spin_polarized = self.soc or self.is_magnetic
        self.tb_vbm_index = int(nelect) - 1 if is_spin_polarized else int(nelect // 2) - 1
        self.tb_vbm_index = max(0, self.tb_vbm_index)
        
        spin_factor = 2 if self._pairs_of_bands() else 1
        bands_below = self.n_bands_below * spin_factor
        bands_above = self.n_bands_above * spin_factor

        start_idx = max(0, self.tb_vbm_index - bands_below + 1)
        end_idx = min(self.total_dim, self.tb_vbm_index + bands_above + 1)
        
        n_dft_bands = len(self.target_indices)
        if (end_idx - start_idx) != n_dft_bands:
            end_idx = min(self.total_dim, start_idx + n_dft_bands)
            if (end_idx - start_idx) < n_dft_bands:
                start_idx = max(0, end_idx - n_dft_bands)

        self.target_tb_band_indices = list(range(start_idx, end_idx))

        print(f"[INFO] TB Filling - Electrons: {self.tb_nelect} | VBM Index: {self.tb_vbm_index}")
        print(f"[INFO] Target TB Band Indices: {self.target_tb_band_indices} (Total Dim: {self.total_dim})")

    def restrict_to_states(self, keep, label="states of the model's orbitals"):
        """
        Fit only the DFT states marked in `keep` (bool, (n_k, n_bands) in the order of all DFT bands at each k-point,
        i.e. the array the fitted window is cut from). At every k-point the window is taken from the kept states:
        the last n_below kept states up to the VBM and the first n_above kept states above it (n_below / n_above as
        in the plain window). E.g. with the nearly-free-electron vacuum states of a 2D layer left out, the fitted pi*
        band continues with pi* instead of jumping to the vacuum state near Gamma; with only the pz-like states kept,
        a pz model is fitted to the pi bands even where sigma bands lie in between. Where a k-point has too few kept
        states (e.g. the pi* band above the highest computed band), the missing entries are NaN and left out of
        the fit and of the errors. Returns the number of k-points whose window changed.
        """
        E = getattr(self, "_all_dft_energies", None)
        if E is None:
            raise ValueError("No DFT bands loaded.")
        keep = np.asarray(keep, bool)
        if keep.shape != E.shape:
            raise ValueError(f"keep has shape {keep.shape}, the DFT bands {E.shape}.")
        vbm = int(self.dft_vbm_index)
        n_below = sum(1 for i in self.target_indices if i <= vbm)
        n_above = len(self.target_indices) - n_below
        target = np.full((E.shape[0], len(self.target_indices)), np.nan)
        changed = short = 0
        for ik in range(E.shape[0]):
            idx = np.flatnonzero(keep[ik])
            occ, emp = idx[idx <= vbm][-n_below:] if n_below else idx[:0], idx[idx > vbm][:n_above]
            if len(occ) < n_below or len(emp) < n_above:
                short += 1                                 # the missing states stay NaN: left out of the fit
            target[ik, n_below - len(occ):n_below] = E[ik, occ]
            target[ik, n_below:n_below + len(emp)] = E[ik, emp]
            sel = np.concatenate([occ, emp]).astype(int)
            if not (len(sel) == len(self.target_indices) and np.array_equal(sel, self.target_indices)):
                changed += 1
        self.target_dft_bands = target
        self.state_filter = {"label": label, "kpoints_changed": changed, "kpoints_short": short,
                             "n_removed": int(np.size(keep) - keep.sum())}
        print(f"[INFO] Fitting {label} only: the window changed at {changed} of {E.shape[0]} k-points"
              + (f"; at {short} k-points some fitted bands have no such state and are left out" if short else ""))
        return changed

    def _key_cache_signature(self):
        """Everything the key lists depend on; cached lists are rebuilt when it changes."""
        return (self.soc, self.is_magnetic, self.non_orthogonal, self.n_neighbors, self.fit_eta,
                getattr(self, 'hopping_mode', 'decay'),
                id(getattr(self, 'neighbor_list', None)),
                tuple(tuple(o) for o in self.site_orbitals), tuple(self.eq_site_map),
                tuple(sorted((k, tuple(v)) for k, v in self.magnetic_orbitals.items())), self._mix_signature(),
                tuple(sorted((k, tuple(v)) for k, v in (getattr(self, "soc_orbitals", None) or {}).items())))

    def _mix_signature(self):
        return tuple(tuple(k for k, _ in site) for site in (getattr(self, "onsite_mix_terms", None) or []))

    def _cached_keys(self, name, compute):
        cache = self.__dict__.setdefault('_key_cache', {})
        sig = self._key_cache_signature()
        entry = cache.get(name)
        if entry is None or entry[0] != sig:
            entry = (sig, compute())
            cache[name] = entry
        return list(entry[1])

    def _get_unique_onsite_keys(self):
        return self._cached_keys('onsite', self._compute_unique_onsite_keys)

    def _get_required_hopping_keys(self):
        return self._cached_keys('hopping', self._compute_required_hopping_keys)

    def _get_param_keys(self):
        return self._cached_keys('params', self._compute_param_keys)

    def _compute_unique_onsite_keys(self):
        unique_keys = []
        for site_idx, subshells in enumerate(self.site_orbitals):
            eq_rep = self.eq_site_map[site_idx]
            elem = self.structure[site_idx].specie.symbol
            for sub in subshells:
                key = f"e_orbit{eq_rep}_{elem}_{sub}"
                if key not in unique_keys:
                    unique_keys.append(key)
                    
            if self.soc:
                families = set([self.FAMILY_MAP[sub] for sub in subshells])
                for fam in families:
                    if fam in ['p', 'd', 'f'] and self._has_soc(elem, fam):
                        lam_key = f"lambda_{elem}_{fam}"
                        if lam_key not in unique_keys:
                            unique_keys.append(lam_key)

            for key, _ in (getattr(self, "onsite_mix_terms", None) or [[]] * (site_idx + 1))[site_idx]:
                if key not in unique_keys:
                    unique_keys.append(key)

            if self.is_magnetic and elem in self.magnetic_orbitals:
                for sub in subshells:              # one delta per family, or per symmetry subshell (exchange_key)
                    delta_key = exchange_key(elem, sub, self.magnetic_orbitals)
                    if delta_key and delta_key not in unique_keys:
                        unique_keys.append(delta_key)
                            
        return unique_keys

    def _has_soc(self, elem, fam):
        """Does family ``fam`` of ``elem`` carry on-site spin-orbit coupling (soc_orbitals; None: every p, d, f)?"""
        so = getattr(self, "soc_orbitals", None)
        return True if not so else fam in (so.get(elem) or ())

    def _shell_mode(self):
        return getattr(self, 'hopping_mode', 'decay') == 'shell'

    def _site_elements(self):
        return [site.specie.symbol for site in self.structure]

    def _pair_shells(self):
        """{element pair: its shell distances (Å)} of the model's bonds (pair_shell_distances)."""
        ps = getattr(self, 'pair_shells', None)
        if ps is None:
            ps = pair_shell_distances(getattr(self, 'neighbor_list', None) or [], self._site_elements())
            self.pair_shells = ps
        return ps

    def _pair_d0(self, pair):
        """The reference length of the distance decay of an element pair: its nearest bond (Å)."""
        sh = self._pair_shells().get(pair)
        return float(sh[0]) if sh else float(getattr(self, 'd0', 0.0))

    def _shell_of(self, r, pair):
        """(index, distance) of the neighbour shell of element pair ``pair`` a bond of length ``r`` belongs to."""
        dist = self._pair_shells().get(pair) or [getattr(self, 'd0', r)]
        return shell_index(dist, r)

    def _get_shell_hopping_keys(self):
        return self._cached_keys('shell_hopping', self._compute_shell_hopping_keys)

    def _compute_shell_hopping_keys(self):
        """Per neighbour shell n: [(n, key), ...] of the hopping integrals some bond of that shell needs."""
        return bond_shell_keys(self.neighbor_list, self.site_orbitals, self._site_elements(), self._pair_shells())

    def shell_start_from_decay(self, params):
        """Starting values of this model's per-shell integrals from a decay-law parameter set ``params``
        (v_<type>_0, eta_<type>, ...): V_n = V0 (d0/r_n)^eta, the decay-law integral at each shell distance r_n of
        its element pair (d0 = the pair's nearest bond). Returns {param key: value} for the keys it can fill (s_*
        overlaps likewise)."""
        out = {}
        shells = self._pair_shells()
        for n, key in self._get_shell_hopping_keys():
            btype = bond_type(key)
            dist = shells[integral_pair(key)]
            for pre, eta_pre in (("v_", "eta_"), ("s_", "eta_s_")):
                name = f"{pre}{btype}"
                if f"{name}_0" not in params:
                    continue
                eta = float(params.get(f"{eta_pre}{btype}", 2.5))
                out[f"{name}_{n}"] = float(params[f"{name}_0"]) * (dist[0] / dist[n]) ** eta
        return out

    def convert_legacy_params(self, params, settings=None):
        """A parameter set saved before the element-pair integral names (rc4 and older) renamed for this model, with the
        same bands (convert_legacy_hoppings); ``settings``: the saved fit's settings record (its hopping_mode)."""
        if not has_legacy_hoppings(migrate_legacy_keys(params)):
            return migrate_legacy_keys(params)
        mode = (settings or {}).get("hopping_mode") if settings else None
        return convert_legacy_hoppings(params, self.neighbor_list, self._site_elements(), self.site_orbitals,
                                       self.n_neighbors, mode, self._pair_shells())

    def split_start(self, params):
        """Starting values of this model's parameters from a fit with coarser orbital groups (``params``): every
        group-pair integral starts at the integral it splits (v_pz.dz2.s_n <- v_pds_n, also v_dz2.dz2.s <- v_dds),
        every group's on-site energy and exchange at its family's. The first fit of the split model then starts from
        exactly the coarser model's bands. Returns {param key: value} for the keys it can fill."""
        out = {}
        for k in self._get_param_keys():
            if k in params:
                continue
            fk = next((c for c in coarser_integral_keys(k) if c in params), None)
            if fk is not None:
                out[k] = float(params[fk])
                continue
            m = re.match(r"^(e_orbit\d+_[A-Z][a-z]?_|delta_spin_[A-Z][a-z]?_)(.+)$", k)
            if m:
                fam = _family_of(m.group(2))
                if fam != m.group(2) and f"{m.group(1)}{fam}" in params:
                    out[k] = float(params[f"{m.group(1)}{fam}"])
                    continue
            m = re.match(r"^(eta_(?:s_)?)(.+)$", k)
            if m:
                fk = next((c[2:] for c in coarser_integral_keys(f"v_{m.group(2)}")
                           if f"{m.group(1)}{c[2:]}" in params), None)
                if fk is not None:
                    out[k] = float(params[f"{m.group(1)}{fk}"])
        return out

    def _fixed_eta_of(self, btype, overlap=False):
        if isinstance(self.fixed_eta, dict):
            return float(self.fixed_eta.get(f"s_{btype}" if overlap else btype, 2.5))
        return float(self.fixed_eta)

    def _compute_param_keys(self):
        param_keys = []
        param_keys.extend(self._get_unique_onsite_keys())
        if self._shell_mode():                  # independent integrals per shell: v_<type>_<n> (+ s_<type>_<n>)
            pairs = self._get_shell_hopping_keys()
            param_keys.extend(f"{key}_{n}" for n, key in pairs)
            if self.non_orthogonal:
                param_keys.extend(f"s_{key[2:]}_{n}" for n, key in pairs)
            return param_keys

        hopping_keys = self._get_required_hopping_keys()
        for key in hopping_keys:
            param_keys.append(f"{key}_0")
            
        if self.non_orthogonal:
            for key in hopping_keys:
                param_keys.append(f"s_{key[2:]}_0")

        if self.n_neighbors > 1 and self.fit_eta:
            unique_bond_types = sorted(set(bond_type(k) for k in hopping_keys))
            for btype in unique_bond_types:
                param_keys.append(f"eta_{btype}")
                if self.non_orthogonal:
                    param_keys.append(f"eta_s_{btype}")
                
        return param_keys

    def _param_vec_to_dict(self, param_vec):
        sk_dict = {}
        idx = 0

        unique_onsite_keys = self._get_unique_onsite_keys()
        for key in unique_onsite_keys:
            sk_dict[key] = param_vec[idx]
            idx += 1

        for site_idx, subshells in enumerate(self.site_orbitals):
            eq_rep = self.eq_site_map[site_idx]
            elem = self.structure[site_idx].specie.symbol
            for sub in subshells:
                orbit_key = f"e_orbit{eq_rep}_{elem}_{sub}"
                site_key = f"e_site{site_idx}_{elem}_{sub}"
                sk_dict[site_key] = sk_dict[orbit_key]

        hopping_keys = self._get_required_hopping_keys()
        if self._shell_mode():
            shells = self._pair_shells()
            for key in hopping_keys:                           # integrals no bond of a shell of the pair uses: 0
                for n in range(len(shells.get(integral_pair(key), [0]))):
                    sk_dict[f'{key}_{n}'] = 0.0
                    if self.non_orthogonal:
                        sk_dict[f's_{key[2:]}_{n}'] = 0.0
            pairs = self._get_shell_hopping_keys()
            for n, key in pairs:
                sk_dict[f'{key}_{n}'] = param_vec[idx]
                idx += 1
            if self.non_orthogonal:
                for n, key in pairs:
                    sk_dict[f's_{key[2:]}_{n}'] = param_vec[idx]
                    idx += 1
            for btype in sorted(set(bond_type(k) for k in hopping_keys)):   # local decay off the fitted lengths
                sk_dict[f'eta_{btype}'] = self._fixed_eta_of(btype)
                if self.non_orthogonal:
                    sk_dict[f'eta_s_{btype}'] = self._fixed_eta_of(btype, overlap=True)
            return sk_dict

        for key in hopping_keys:
            sk_dict[f'{key}_0'] = param_vec[idx]
            idx += 1

        if self.non_orthogonal:
            for key in hopping_keys:
                sk_dict[f's_{key[2:]}_0'] = param_vec[idx]
                idx += 1

        if self.n_neighbors > 1:
            unique_bond_types = sorted(set(bond_type(k) for k in hopping_keys))
            for btype in unique_bond_types:
                if self.fit_eta:
                    sk_dict[f'eta_{btype}'] = param_vec[idx]
                    idx += 1
                    if self.non_orthogonal:
                        sk_dict[f'eta_s_{btype}'] = param_vec[idx]
                        idx += 1
                else:
                    if isinstance(self.fixed_eta, dict):
                        sk_dict[f'eta_{btype}'] = self.fixed_eta.get(btype, 2.5)
                        if self.non_orthogonal:
                            sk_dict[f'eta_s_{btype}'] = self.fixed_eta.get(f's_{btype}', 2.5)
                    else:
                        sk_dict[f'eta_{btype}'] = float(self.fixed_eta)
                        if self.non_orthogonal:
                            sk_dict[f'eta_s_{btype}'] = float(self.fixed_eta)
        else:
            unique_bond_types = sorted(set(bond_type(k) for k in hopping_keys))
            for btype in unique_bond_types:
                sk_dict[f'eta_{btype}'] = 2.5
                if self.non_orthogonal:
                    sk_dict[f'eta_s_{btype}'] = 2.5

        return sk_dict

    def _get_cutoff_from_neighbors(self, n_shells):
        return cutoff_from_shells(self.structure, n_shells)

    def _precompute_neighbor_pairs(self):
        self.neighbor_list, self.d0 = neighbour_pairs(self.structure, self.cutoff_radius)
        self.shell_distances = bond_shell_distances(self.neighbor_list)
        self.pair_shells = pair_shell_distances(self.neighbor_list, self._site_elements())

    _S_TYPES, _P_TYPES, _D_TYPES, _F_TYPES = _S_TYPES, _P_TYPES, _D_TYPES, _F_TYPES

    def _compute_required_hopping_keys(self):
        """
        Hopping integrals that are actually needed, decided bond by bond: an integral is
        required only if the two orbital types sit on the two ends of some neighbour pair
        (i, j) (see pair_hopping_keys).
        """
        keys = set()
        el = self._site_elements()
        for pair in self.neighbor_list:
            keys |= pair_hopping_keys(el[pair['i']], self.site_orbitals[pair['i']], el[pair['j']],
                                      self.site_orbitals[pair['j']], pair['d_vec'])
        return sorted(keys)
    
    @staticmethod
    def _sk_hopping_s_s(l, m, n, v_sss):
        return np.array([[v_sss]])

    @staticmethod
    def _sk_hopping_s_p(l, m, n, v_sps):
        return np.array([[l * v_sps, m * v_sps, n * v_sps]])

    @staticmethod
    def _sk_hopping_s_d(l, m, n, v_sds):
        val_dz2 = (n**2 - 0.5 * (l**2 + m**2)) * v_sds
        val_dx2y2 = (np.sqrt(3) / 2.0) * (l**2 - m**2) * v_sds
        val_dxy = np.sqrt(3) * l * m * v_sds
        val_dyz = np.sqrt(3) * m * n * v_sds
        val_dzx = np.sqrt(3) * l * n * v_sds
        return np.array([[val_dxy, val_dyz, val_dzx, val_dx2y2, val_dz2]])

    @staticmethod
    def _sk_hopping_p_p(l, m, n, v_pps, v_ppp):
        H_pp = np.zeros((3, 3))
        dir_cos = [l, m, n]
        for a in range(3):
            for b in range(3):
                if a == b:
                    H_pp[a, b] = (dir_cos[a] ** 2) * v_pps + (1.0 - dir_cos[a] ** 2) * v_ppp
                else:
                    H_pp[a, b] = dir_cos[a] * dir_cos[b] * (v_pps - v_ppp)
        return H_pp

    @staticmethod
    def _sk_hopping_p_d(l, m, n, v_pds, v_pdp):
        root3 = np.sqrt(3)
        H_pd = np.zeros((3, 5))
        H_pd[0, 0] = root3 * (l**2) * m * v_pds + m * (1.0 - 2.0 * l**2) * v_pdp       
        H_pd[0, 1] = root3 * l * m * n * v_pds - 2.0 * l * m * n * v_pdp               
        H_pd[0, 2] = root3 * (l**2) * n * v_pds + n * (1.0 - 2.0 * l**2) * v_pdp       
        H_pd[0, 3] = (root3 / 2.0) * l * (l**2 - m**2) * v_pds + l * (1.0 - l**2 + m**2) * v_pdp  
        H_pd[0, 4] = l * (n**2 - 0.5 * (l**2 + m**2)) * v_pds - root3 * l * n**2 * v_pdp         

        H_pd[1, 0] = root3 * l * (m**2) * v_pds + l * (1.0 - 2.0 * m**2) * v_pdp       
        H_pd[1, 1] = root3 * (m**2) * n * v_pds + n * (1.0 - 2.0 * m**2) * v_pdp       
        H_pd[1, 2] = root3 * l * m * n * v_pds - 2.0 * l * m * n * v_pdp               
        H_pd[1, 3] = (root3 / 2.0) * m * (l**2 - m**2) * v_pds - m * (1.0 + l**2 - m**2) * v_pdp  
        H_pd[1, 4] = m * (n**2 - 0.5 * (l**2 + m**2)) * v_pds - root3 * m * n**2 * v_pdp         

        H_pd[2, 0] = root3 * l * m * n * v_pds - 2.0 * l * m * n * v_pdp               
        H_pd[2, 1] = root3 * m * (n**2) * v_pds + m * (1.0 - 2.0 * n**2) * v_pdp       
        H_pd[2, 2] = root3 * l * (n**2) * v_pds + l * (1.0 - 2.0 * n**2) * v_pdp       
        H_pd[2, 3] = (root3 / 2.0) * n * (l**2 - m**2) * v_pds - n * (l**2 - m**2) * v_pdp       
        H_pd[2, 4] = n * (n**2 - 0.5 * (l**2 + m**2)) * v_pds + root3 * n * (l**2 + m**2) * v_pdp 
        return H_pd

    @staticmethod
    def _sk_hopping_d_d(l, m, n, v_dds, v_ddp, v_ddd):
        root3 = np.sqrt(3)
        H_dd = np.zeros((5, 5))
        H_dd[0, 0] = 3.0 * (l**2) * (m**2) * v_dds + (l**2 + m**2 - 4.0 * l**2 * m**2) * v_ddp + (n**2 + l**2 * m**2) * v_ddd
        H_dd[0, 1] = 3.0 * l * (m**2) * n * v_dds + l * n * (1.0 - 4.0 * m**2) * v_ddp + l * n * (m**2 - 1.0) * v_ddd
        H_dd[0, 2] = 3.0 * (l**2) * m * n * v_dds + m * n * (1.0 - 4.0 * l**2) * v_ddp + m * n * (l**2 - 1.0) * v_ddd
        H_dd[0, 3] = 1.5 * l * m * (l**2 - m**2) * v_dds + 2.0 * l * m * (m**2 - l**2) * v_ddp + 0.5 * l * m * (l**2 - m**2) * v_ddd
        H_dd[0, 4] = root3 * l * m * (n**2 - 0.5 * (l**2 + m**2)) * v_dds - 2.0 * root3 * l * m * n**2 * v_ddp + 0.5 * root3 * l * m * (1.0 + n**2) * v_ddd

        H_dd[1, 0] = H_dd[0, 1]
        H_dd[1, 1] = 3.0 * (m**2) * (n**2) * v_dds + (m**2 + n**2 - 4.0 * m**2 * n**2) * v_ddp + (l**2 + m**2 * n**2) * v_ddd
        H_dd[1, 2] = 3.0 * l * m * (n**2) * v_dds + l * m * (1.0 - 4.0 * n**2) * v_ddp + l * m * (n**2 - 1.0) * v_ddd
        H_dd[1, 3] = 1.5 * m * n * (l**2 - m**2) * v_dds - m * n * (1.0 + 2.0 * (l**2 - m**2)) * v_ddp + m * n * (1.0 + 0.5 * (l**2 - m**2)) * v_ddd
        H_dd[1, 4] = root3 * m * n * (n**2 - 0.5 * (l**2 + m**2)) * v_dds + root3 * m * n * (l**2 + m**2 - n**2) * v_ddp - 0.5 * root3 * m * n * (l**2 + m**2) * v_ddd

        H_dd[2, 0] = H_dd[0, 2]
        H_dd[2, 1] = H_dd[1, 2]
        H_dd[2, 2] = 3.0 * (l**2) * (n**2) * v_dds + (l**2 + n**2 - 4.0 * l**2 * n**2) * v_ddp + (m**2 + l**2 * n**2) * v_ddd
        H_dd[2, 3] = 1.5 * l * n * (l**2 - m**2) * v_dds + l * n * (1.0 - 2.0 * (l**2 - m**2)) * v_ddp - l * n * (1.0 - 0.5 * (l**2 - m**2)) * v_ddd
        H_dd[2, 4] = root3 * l * n * (n**2 - 0.5 * (l**2 + m**2)) * v_dds + root3 * l * n * (l**2 + m**2 - n**2) * v_ddp - 0.5 * root3 * l * n * (l**2 + m**2) * v_ddd

        H_dd[3, 0] = H_dd[0, 3]
        H_dd[3, 1] = H_dd[1, 3]
        H_dd[3, 2] = H_dd[2, 3]
        H_dd[3, 3] = 0.75 * ((l**2 - m**2)**2) * v_dds + (l**2 + m**2 - (l**2 - m**2)**2) * v_ddp + (n**2 + 0.25 * ((l**2 - m**2)**2)) * v_ddd
        H_dd[3, 4] = (root3 / 2.0) * (l**2 - m**2) * (n**2 - 0.5 * (l**2 + m**2)) * v_dds + root3 * n**2 * (m**2 - l**2) * v_ddp + (root3 / 4.0) * (1.0 + n**2) * (l**2 - m**2) * v_ddd

        H_dd[4, 0] = H_dd[0, 4]
        H_dd[4, 1] = H_dd[1, 4]
        H_dd[4, 2] = H_dd[2, 4]
        H_dd[4, 3] = H_dd[3, 4]
        H_dd[4, 4] = ((n**2 - 0.5 * (l**2 + m**2))**2) * v_dds + 3.0 * n**2 * (l**2 + m**2) * v_ddp + 0.75 * ((l**2 + m**2)**2) * v_ddd
        return H_dd
   
    def _integral_value(self, key_name, btype, r, sk_params, is_overlap=False):
        """The two-centre integral ``key_name`` ('v_Mn_d-Mn_d_dds', 's_...') of a bond of length r (Å).
        decay: V0 (d0/r)^eta with d0 the nearest bond of the element pair (V0 = <key>_0; no scaling with one shell);
        shell: V_n of the bond's shell n of its element pair, scaled by (r_n/r)^eta (= 1 on the fitted bond lengths)."""
        pair = integral_pair(btype)
        if self._shell_mode():
            n, rn = self._shell_of(r, pair)
            if f"{key_name}_{n}" not in sk_params and is_group_integral(key_name):
                key_name, btype = self._family_fallback(key_name, sk_params, n)
            eta = sk_params.get(f"eta_s_{btype}" if is_overlap else f"eta_{btype}", 2.5)
            v = sk_params.get(f"{key_name}_{n}", 0.0)
            return v * (rn / r) ** eta if abs(r - rn) > 1e-5 and r > 1e-6 else v
        if f"{key_name}_0" not in sk_params and key_name not in sk_params and is_group_integral(key_name):
            key_name, btype = self._family_fallback(key_name, sk_params, 0)
        eta = sk_params.get(f"eta_s_{btype}" if is_overlap else f"eta_{btype}", 2.5)
        v0 = sk_params.get(f"{key_name}_0", sk_params.get(key_name, 0.0))
        d0 = self._pair_d0(pair)
        if self.n_neighbors > 1 and d0 > 1e-4:
            return v0 * ((d0 / r) ** eta)
        return v0

    @staticmethod
    def _family_fallback(key_name, sk_params, n):
        """A model saved before its groups had their own integrals: the integral the group pair splits."""
        fam = family_integral_key(key_name)
        if fam != key_name and (f"{fam}_{n}" in sk_params or fam in sk_params):
            return fam, bond_type(fam)
        return key_name, bond_type(key_name)

    def _get_submatrix_hopping(self, elem1, sub1, elem2, sub2, R_vec, sk_params, is_overlap=False):
        """The hopping (or overlap) block between orbital group sub1 of element elem1 on one site and sub2 of elem2 on
        a site at R_vec from it: sum_m V_m U_m with the integrals V_m = integral_keys(sub1, sub2, elem1, elem2)
        (see group_unit_blocks)."""
        info1 = self.SUB_ORBITALS[sub1]
        info2 = self.SUB_ORBITALS[sub2]
        r = np.linalg.norm(R_vec)
        if r < 1e-6:
            return np.zeros((info1['dim'], info2['dim']))
        block = np.zeros((info1['dim'], info2['dim']))
        for key, U in zip(integral_keys(sub1, sub2, elem1, elem2),
                          group_unit_blocks(sub1, sub2, np.asarray(R_vec, float) / r)):
            if np.abs(U).max() <= 1e-10:            # this integral does not act on this bond (pair_hopping_keys)
                continue
            name = f"s_{key[2:]}" if is_overlap else key
            v = self._integral_value(name, bond_type(key), r, sk_params, is_overlap)
            if v:
                block = block + v * U
        return block

    @staticmethod
    def _get_angular_momentum_matrices(family):
        if family == 'p':
            Lx = 1j * np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
            Ly = 1j * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
            Lz = 1j * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
            return Lx, Ly, Lz
        elif family == 'd':
            r3 = np.sqrt(3)
            Lz = 1j * np.array([
                [ 0,  0,  0,  2,  0],
                [ 0,  0, 1,  0,  0],
                [ 0,  -1,  0,  0,  0],
                [-2,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0]
            ])
            Lx = 1j * np.array([
                [ 0,  0, -1,  0,  0],
                [ 0,  0,  0, -1, -r3],
                [ 1,  0,  0,  0,  0],
                [ 0,  1,  0,  0,  0],
                [ 0, r3,  0,  0,  0]
            ])
            Ly = 1j * np.array([
                [ 0,  1,  0,  0,  0],
                [-1,  0,  0,  0,  0],
                [ 0,  0,  0, -1,  r3],
                [ 0,  0,  1,  0,  0],
                [ 0,  0, -r3, 0,  0]
            ])
            return Lx, Ly, Lz
        elif family == 'f':
            return angular_momentum_matrices(3)
        return None, None, None

    def build_overlap_matrix(self, k_vec, sk_params):
        N = self.base_dim
        S0 = np.eye(N, dtype=complex)

        for nb in self.neighbor_list:
            site_i = nb['i']
            site_j = nb['j']
            R_vec = nb['d_vec']   
        
            elem_i = self.structure[site_i].specie.symbol
            elem_j = self.structure[site_j].specie.symbol
        
            subs_i = self.site_orbitals[site_i]
            subs_j = self.site_orbitals[site_j]
        
            offset_i = self.site_offsets[site_i]
            offset_j = self.site_offsets[site_j]

            phase = np.exp(1j * np.dot(k_vec, R_vec))

            loc_i = offset_i
            for sub1 in subs_i:
                dim1 = self.SUB_ORBITALS[sub1]['dim']
            
                loc_j = offset_j
                for sub2 in subs_j:
                    dim2 = self.SUB_ORBITALS[sub2]['dim']

                    S_sub = self._get_submatrix_hopping(
                        elem1=elem_i, 
                        sub1=sub1, 
                        elem2=elem_j, 
                        sub2=sub2, 
                        R_vec=R_vec, 
                        sk_params=sk_params,
                        is_overlap=True
                    )

                    S0[loc_i : loc_i + dim1, loc_j : loc_j + dim2] += S_sub * phase
                    loc_j += dim2
                loc_i += dim1

        S0 = 0.5 * (S0 + S0.conj().T)

        if not self.soc and not self.is_magnetic:
            return S0

        S_full = np.zeros((2 * N, 2 * N), dtype=complex)
        S_full[0:N, 0:N] = S0
        S_full[N:2*N, N:2*N] = S0

        return 0.5 * (S_full + S_full.conj().T)

    def build_hamiltonian(self, k_vec, sk_params):
        N = self.base_dim
        H0 = np.zeros((N, N), dtype=complex)

        for site_idx in range(len(self.structure)):
            elem = self.structure[site_idx].specie.symbol
            eq_rep = self.eq_site_map[site_idx]
            subshells = self.site_orbitals[site_idx]
            site_offset = self.site_offsets[site_idx]

            curr_offset = site_offset
            for sub in subshells:
                sub_dim = self.SUB_ORBITALS[sub]['dim']
                key = f"e_orbit{eq_rep}_{elem}_{sub}"
                e_val = sk_params.get(key, 0.0)
            
                for d in range(sub_dim):
                    H0[curr_offset + d, curr_offset + d] = e_val
                curr_offset += sub_dim
            mix = getattr(self, "onsite_mix_terms", None)
            if mix:
                for key, M in mix[site_idx]:
                    v = sk_params.get(key, 0.0)
                    if v:
                        m = M.shape[0]
                        H0[site_offset:site_offset + m, site_offset:site_offset + m] += v * M

        for nb in self.neighbor_list:
            site_i = nb['i']
            site_j = nb['j']
            R_vec = nb['d_vec']   
        
            elem_i = self.structure[site_i].specie.symbol
            elem_j = self.structure[site_j].specie.symbol
        
            subs_i = self.site_orbitals[site_i]
            subs_j = self.site_orbitals[site_j]
        
            offset_i = self.site_offsets[site_i]
            offset_j = self.site_offsets[site_j]

            phase = np.exp(1j * np.dot(k_vec, R_vec))

            loc_i = offset_i
            for sub1 in subs_i:
                dim1 = self.SUB_ORBITALS[sub1]['dim']
            
                loc_j = offset_j
                for sub2 in subs_j:
                    dim2 = self.SUB_ORBITALS[sub2]['dim']

                    H_sub = self._get_submatrix_hopping(
                        elem1=elem_i, 
                        sub1=sub1, 
                        elem2=elem_j, 
                        sub2=sub2, 
                        R_vec=R_vec, 
                        sk_params=sk_params,
                        is_overlap=False
                    )

                    H0[loc_i : loc_i + dim1, loc_j : loc_j + dim2] += H_sub * phase
                    loc_j += dim2
                loc_i += dim1

        H0 = 0.5 * (H0 + H0.conj().T)

        if not self.soc and not self.is_magnetic:
            return H0

        H_full = np.zeros((2 * N, 2 * N), dtype=complex)
        H_full[0:N, 0:N] = H0
        H_full[N:2*N, N:2*N] = H0

        if self.soc:
            for site_idx in range(len(self.structure)):
                elem = self.structure[site_idx].specie.symbol
                subshells = self.site_orbitals[site_idx]
                offset_i = self.site_offsets[site_idx]

                loc_a = offset_i
                for sub_a in subshells:
                    fam_a = self.FAMILY_MAP[sub_a]
                    dim_a = self.SUB_ORBITALS[sub_a]['dim']
                    B_a = sub_basis(sub_a)

                    loc_b = offset_i
                    for sub_b in subshells:
                        fam_b = self.FAMILY_MAP[sub_b]
                        dim_b = self.SUB_ORBITALS[sub_b]['dim']
                        B_b = sub_basis(sub_b)

                        if fam_a == fam_b and fam_a in ['p', 'd', 'f'] and self._has_soc(elem, fam_a):
                            lam = sk_params.get(f"lambda_{elem}_{fam_a}", 0.0)
                            Lx, Ly, Lz = self._get_angular_momentum_matrices(fam_a)

                            if Lx is not None:
                                Lx_sub = B_a @ Lx @ B_b.T     # L between the two groups (real or cubic orbitals)
                                Ly_sub = B_a @ Ly @ B_b.T
                                Lz_sub = B_a @ Lz @ B_b.T

                                H_full[loc_a:loc_a+dim_a, loc_b:loc_b+dim_b] += 0.5 * lam * Lz_sub
                                H_full[N+loc_a:N+loc_a+dim_a, N+loc_b:N+loc_b+dim_b] -= 0.5 * lam * Lz_sub
                                H_full[loc_a:loc_a+dim_a, N+loc_b:N+loc_b+dim_b] += 0.5 * lam * (Lx_sub - 1j * Ly_sub)
                                H_full[N+loc_a:N+loc_a+dim_a, loc_b:loc_b+dim_b] += 0.5 * lam * (Lx_sub + 1j * Ly_sub)

                        loc_b += dim_b
                    loc_a += dim_a

        # Exchange splitting is independent of SOC: apply it whenever the model is magnetic
        if self.is_magnetic:
            signs = getattr(self, "site_moment_signs", None) or [1] * len(self.structure)
            dirs = getattr(self, "site_moment_dirs", None) or [np.array([0.0, 0.0, float(s)]) for s in signs]
            mags = getattr(self, "magnetic_orbitals", None)
            if mags is None:                      # a model rebuilt from saved parameters: their delta keys say it
                mags = magnetic_orbitals_from_params(sk_params)
            for site_idx in range(len(self.structure)):
                elem = self.structure[site_idx].specie.symbol
                subshells = self.site_orbitals[site_idx]
                offset_i = self.site_offsets[site_idx]

                loc_a = offset_i
                for sub_a in subshells:
                    fam_a = self.FAMILY_MAP[sub_a]
                    dim_a = self.SUB_ORBITALS[sub_a]['dim']
                    
                    delta_key = exchange_key(elem, sub_a, mags)
                    delta_val = sk_params.get(delta_key, 0.0) if delta_key else 0.0
                    nx, ny, nz = (float(c) for c in dirs[site_idx])           # moment direction n_i

                    if delta_val != 0.0:             # -(delta/2) n . sigma: with n = +z (delta > 0) spin up is the
                        for d in range(dim_a):       # majority, lowered by delta/2, as in DFT (more up electrons)
                            a = loc_a + d
                            H_full[a, a] -= 0.5 * delta_val * nz
                            H_full[N + a, N + a] += 0.5 * delta_val * nz
                            H_full[a, N + a] -= 0.5 * delta_val * (nx - 1j * ny)
                            H_full[N + a, a] -= 0.5 * delta_val * (nx + 1j * ny)
                            
                    loc_a += dim_a

        return 0.5 * (H_full + H_full.conj().T)

    # ------------------------------------------------------------------
    # Fast (vectorised) band evaluation
    #
    # For a fixed structure/basis, every hopping block is linear in the Slater-Koster
    # integrals, the on-site / SOC / exchange terms are linear in their parameters, and
    # the Bloch phases exp(i k.d) do not depend on the parameters at all. So we precompute
    #   A[key, bond]   angular coefficient of each integral on each bond (obtained by calling
    #                  the reference _get_submatrix_hopping with that integral set to 1),
    #   L[key]         constant matrices of the on-site / lambda / delta_spin terms (obtained
    #                  from the reference build_hamiltonian),
    #   phase[k, bond] the Bloch phases,
    # and each evaluation reduces to a few matrix products plus one batched eigvalsh.
    # ------------------------------------------------------------------
    def _fast_signature(self):
        kp = getattr(self, 'kpoints', None)
        return (id(self.neighbor_list), len(self.neighbor_list), id(kp), 0 if kp is None else len(kp),
                self.soc, self.is_magnetic, self.non_orthogonal, self.n_neighbors,
                getattr(self, 'hopping_mode', 'decay'),
                tuple((k, tuple(v)) for k, v in sorted(self._pair_shells().items())),
                self.base_dim, self.total_dim, tuple(tuple(o) for o in self.site_orbitals),
                tuple(getattr(self, "site_moment_signs", None) or ()), self._mix_signature(),
                tuple(tuple(np.round(d, 8)) for d in (getattr(self, "site_moment_dirs", None) or ())))

    def _prepare_fast_model(self):
        sig = self._fast_signature()
        fm = getattr(self, "_fast", None)
        if fm is not None and fm["sig"] == sig:
            return fm

        N, D = self.base_dim, self.total_dim
        bonds = self.neighbor_list
        nb = len(bonds)
        keys = self._get_required_hopping_keys()
        bond_types = [bond_type(k) for k in keys]
        site_el = self._site_elements()
        bond_pair = [element_pair(site_el[b['i']], site_el[b['j']]) for b in bonds]

        unit_eta = {}
        for b in set(bond_types):
            unit_eta[f"eta_{b}"] = 0.0       # eta = 0  ->  distance scaling factor (d0/r)^0 = 1
            unit_eta[f"eta_s_{b}"] = 0.0

        n_sh = max([len(v) for v in self._pair_shells().values()] or [1]) if self._shell_mode() else 1
        shells = list(range(n_sh))
        A = np.zeros((len(keys), nb, N, N))
        for ik, key in enumerate(keys):
            unit = {f"{k}_{n}": 0.0 for k in keys for n in range(len(shells))}
            unit.update(unit_eta)
            for n in range(len(shells)):        # every shell of this integral at 1 (decay: only <key>_0 is read)
                unit[f"{key}_{n}"] = 1.0
            for ib, bond in enumerate(bonds):
                i, j = bond['i'], bond['j']
                elem_i = self.structure[i].specie.symbol
                elem_j = self.structure[j].specie.symbol
                loc_i = self.site_offsets[i]
                for sub1 in self.site_orbitals[i]:
                    d1 = self.SUB_ORBITALS[sub1]['dim']
                    loc_j = self.site_offsets[j]
                    for sub2 in self.site_orbitals[j]:
                        d2 = self.SUB_ORBITALS[sub2]['dim']
                        A[ik, ib, loc_i:loc_i + d1, loc_j:loc_j + d2] += self._get_submatrix_hopping(
                            elem_i, sub1, elem_j, sub2, bond['d_vec'], unit)
                        loc_j += d2
                    loc_i += d1

        static_keys = self._get_unique_onsite_keys()      # on-site energies, lambda_*, delta_spin_*
        L = np.zeros((len(static_keys), D, D), dtype=complex)
        for ik, key in enumerate(static_keys):
            L[ik] = self.build_hamiltonian(np.zeros(3), {key: 1.0})

        d_vec = np.array([b['d_vec'] for b in bonds], dtype=float).reshape(nb, 3)
        r = np.linalg.norm(d_vec, axis=1)
        d0 = np.array([self._pair_d0(pp) for pp in bond_pair], dtype=float)
        kpts = np.asarray(getattr(self, 'kpoints', None) if getattr(self, 'kpoints', None) is not None else [],
                          dtype=float).reshape(-1, 3)
        phase_full = np.exp(1j * (kpts @ d_vec.T))

        self._fast = {
            "sig": sig, "N": N, "D": D, "nb": nb, "keys": keys,
            "A2": A.reshape(len(keys), nb, N * N),
            "eta_names": [f"eta_{b}" for b in bond_types],
            "eta_s_names": [f"eta_s_{b}" for b in bond_types],
            "static_keys": static_keys, "L": L,
            "d_vec": d_vec, "ratio": (d0 / r) if nb else r,
            "use_eta": bool(self.n_neighbors > 1 and nb and d0.min() > 1e-4),
            "phase_full": phase_full,
            "shell_mode": self._shell_mode(),
            "bond_shell": np.array([self._shell_of(x, pp)[0] for x, pp in zip(r, bond_pair)], dtype=int),
            "bond_rn": np.array([self._shell_of(x, pp)[1] for x, pp in zip(r, bond_pair)], dtype=float),
            "n_shells": len(shells),
        }
        self._active = None
        return self._fast

    def _fast_weights(self, params, overlap=False):
        """v0 * (d0/r)^eta for every (integral, bond): shape (n_keys, n_bonds)."""
        fm = self._fast
        if fm.get("shell_mode"):                 # V_n of each bond's shell (bonds sit on the shell distances)
            pre = (lambda k: f"s_{k[2:]}") if overlap else (lambda k: k)
            V = np.array([[params.get(f"{pre(k)}_{n}", 0.0) for n in range(fm["n_shells"])] for k in fm["keys"]])
            W = V[:, fm["bond_shell"]]
            # bonds a little off their shell distance (a cell symmetric only to symprec, or strain): V_n (r_n/r)^eta,
            # as in _integral_value
            rn, rr = fm["bond_rn"], np.linalg.norm(fm["d_vec"], axis=1)
            off = (np.abs(rr - rn) > 1e-5) & (rr > 1e-6)
            if off.any():
                names = fm["eta_s_names"] if overlap else fm["eta_names"]
                eta = np.array([params.get(nm, 2.5) for nm in names])
                W = W.copy()
                W[:, off] *= (rn[off] / rr[off])[None, :] ** eta[:, None]
            return W
        if overlap:
            v0 = np.array([params[f"s_{k[2:]}_0"] for k in fm["keys"]])
            eta_names = fm["eta_s_names"]
        else:
            v0 = np.array([params[f"{k}_0"] for k in fm["keys"]])
            eta_names = fm["eta_names"]
        if fm["use_eta"]:
            eta = np.array([params.get(n, 2.5) for n in eta_names])
            return v0[:, None] * fm["ratio"][None, :] ** eta[:, None]
        return np.repeat(v0[:, None], fm["nb"], axis=1)

    def _fast_H_S(self, params, phase, need_S=False):
        """Batched H(k) (and S(k)) for the Bloch phases `phase` of shape (n_k, n_bonds)."""
        fm = self._fast
        N, D = fm["N"], fm["D"]

        T = np.einsum('kb,kbm->bm', self._fast_weights(params), fm["A2"])
        H0 = (phase @ T).reshape(-1, N, N)
        if D == N:
            H = H0
        else:                                             # spin-doubled: same hopping in both blocks
            H = np.zeros((H0.shape[0], D, D), dtype=complex)
            H[:, :N, :N] = H0
            H[:, N:, N:] = H0
        if fm["L"].shape[0]:
            pv = np.array([params.get(k, 0.0) for k in fm["static_keys"]])
            H = H + np.tensordot(pv, fm["L"], axes=1)[None]
        H = 0.5 * (H + H.conj().transpose(0, 2, 1))

        S = None
        if need_S:
            Ts = np.einsum('kb,kbm->bm', self._fast_weights(params, overlap=True), fm["A2"])
            S0 = (phase @ Ts).reshape(-1, N, N) + np.eye(N)
            S0 = 0.5 * (S0 + S0.conj().transpose(0, 2, 1))
            if D == N:
                S = S0
            else:
                S = np.zeros((S0.shape[0], D, D), dtype=complex)
                S[:, :N, :N] = S0
                S[:, N:, N:] = S0
        return H, S

    OVERLAP_PENALTY = 10.0      # eV of residual per unit of S(k) eigenvalue below overlap_floor

    def _overlap_min_eig(self, params, phase):
        """Smallest eigenvalue of the overlap matrix S(k) (spin-free part) at every k of `phase`."""
        fm = self._fast
        Ts = np.einsum('kb,kbm->bm', self._fast_weights(params, overlap=True), fm["A2"])
        S0 = (phase @ Ts).reshape(-1, fm["N"], fm["N"]) + np.eye(fm["N"])
        return np.linalg.eigvalsh(0.5 * (S0 + S0.conj().transpose(0, 2, 1)))[:, 0]

    def overlap_conditioning(self, params=None, n_random=256, seed=0):
        """Smallest eigenvalue of S(k) for a non-orthogonal model: {'path': on the fitted k-points, 'bz': on
        n_random random k-points of the Brillouin zone} (None for an orthogonal model)."""
        if not self.non_orthogonal:
            return None
        params = self.opt_params if params is None else params
        fm = self._prepare_fast_model()
        rec = self.structure.lattice.reciprocal_lattice.matrix
        q = np.random.default_rng(seed).random((int(n_random), 3)) @ rec
        out = {"path": float(self._overlap_min_eig(params, fm["phase_full"]).min())}
        d_vec = np.array([b['d_vec'] for b in self.neighbor_list], dtype=float).reshape(-1, 3)
        if d_vec.shape[0] == fm["A2"].shape[1]:           # bonds of the fast model: vectorised over the random k
            out["bz"] = float(self._overlap_min_eig(params, np.exp(1j * (q @ d_vec.T))).min())
        else:
            out["bz"] = float(min(np.linalg.eigvalsh(self.build_overlap_matrix(k, params)).min() for k in q))
        return out

    def _overlap_penalty(self, params):
        """Residuals that keep S(k) well conditioned in a non-orthogonal fit: OVERLAP_PENALTY * (overlap_floor -
        lambda_min(S(k))) at every active k where the smallest eigenvalue of S drops below overlap_floor. Without
        it a fit can drive S(k) towards singular at a few k-points (Mn3Ir: Gamma), which sends states to
        +-infinity there, shuffles which states fall in the fitted window, and breaks the model elsewhere."""
        floor = float(getattr(self, "overlap_floor", 0.0) or 0.0)
        if not self.non_orthogonal or floor <= 0:
            return np.zeros(0)
        lam = self._overlap_min_eig(params, self._get_active()["phase"])
        return self.OVERLAP_PENALTY * np.maximum(0.0, floor - lam)

    def _fast_bands(self, params, phase):
        """All D eigenvalues (ascending) at every k of `phase`."""
        H, S = self._fast_H_S(params, phase, self.non_orthogonal)
        if S is not None:                                  # generalised problem via batched Cholesky
            L = np.linalg.cholesky(S)                      # raises LinAlgError if S is not positive definite
            Y = np.linalg.solve(L, H)
            H = np.linalg.solve(L, Y.conj().transpose(0, 2, 1))
            H = 0.5 * (H + H.conj().transpose(0, 2, 1))
        return np.linalg.eigvalsh(H)

    def _reference_bands(self, params, idx):
        """Original one-k-at-a-time evaluation (slow, but the ground truth)."""
        out = []
        for i in idx:
            k = self.kpoints[i]
            H_k = self.build_hamiltonian(k, params)
            if self.non_orthogonal:
                S_k = self.build_overlap_matrix(k, params)
                try:
                    out.append(eigh(H_k, S_k, eigvals_only=True))
                except np.linalg.LinAlgError as exc:
                    raise np.linalg.LinAlgError(
                        f"The overlap matrix S(k) is not positive definite at k-point {i} (smallest eigenvalue "
                        f"{np.linalg.eigvalsh(S_k).min():.2e}). The overlap parameters of this fit make S nearly "
                        "singular; refit with [fit] overlap_floor (default 0.05) to keep S(k) well conditioned."
                    ) from exc
            else:
                out.append(np.sort(np.linalg.eigvalsh(H_k)))
        return np.array(out)

    def check_fast_model(self, param_vec=None, n_k=8, tol=1e-7, verbose=True):
        """Compare the vectorised bands with the reference implementation. True if they agree."""
        if param_vec is None:
            param_vec = np.zeros(len(self._get_param_keys()))
        params = self._param_vec_to_dict(np.asarray(param_vec, dtype=float))
        fm = self._prepare_fast_model()
        nk = len(self.kpoints)
        idx = np.unique(np.linspace(0, nk - 1, min(n_k, nk)).astype(int))
        try:
            ref = self._reference_bands(params, idx)
            fast = self._fast_bands(params, fm["phase_full"][idx])
        except np.linalg.LinAlgError as exc:
            if verbose:
                print(f"[WARN] Could not verify the fast model at these parameters ({exc}).")
            return True
        # compare the fitted bands, relative to their size: with a nearly singular overlap matrix a few states sit
        # at +-1e8 eV, where the two eigensolvers legitimately differ by eV (Mn3Ir non-orthogonal fit at Gamma)
        win = np.asarray(getattr(self, "target_tb_band_indices", None) if getattr(self, "target_tb_band_indices", None)
                         is not None else np.arange(ref.shape[1]), dtype=int)
        ref, fast = ref[:, win], fast[:, win]
        err = float(np.max(np.abs(ref - fast) / np.maximum(1.0, np.abs(ref))))
        if verbose:
            print(f"[INFO] Fast model check: max |E_fast - E_reference| / max(1, |E|) = {err:.2e} on {len(idx)} k-points")
        return err < tol

    # ---- active k-point set (lets the global search use a subset, refinement the full set) ----
    def _select_k_indices(self, stride):
        nk = len(self.kpoints)
        if stride is None or int(stride) <= 1:
            return None
        idx = set(range(0, nk, int(stride)))
        idx.add(nk - 1)
        idx.update(i for i in getattr(self, 'high_sym_indices', []) if 0 <= i < nk)
        return np.array(sorted(idx))

    def _set_active_k(self, k_idx=None):
        self._active_idx = None if k_idx is None else np.asarray(k_idx, dtype=int)
        self._active = None

    def _get_active(self):
        fm = self._prepare_fast_model()
        act = getattr(self, "_active", None)
        if act is None or act["sig"] != fm["sig"]:
            idx = getattr(self, "_active_idx", None)
            if idx is None:
                idx = np.arange(len(self.kpoints))
            act = {"sig": fm["sig"], "idx": idx, "phase": fm["phase_full"][idx],
                   "dft": np.asarray(self.target_dft_bands)[idx]}
            self._active = act
        return act

    def _selected_bands(self, params):
        """(TB bands, DFT bands) restricted to the target window and the active k-points."""
        if getattr(self, "use_fast_model", True):
            act = self._get_active()
            E = self._fast_bands(params, act["phase"])[:, self.target_tb_band_indices]
            return E, act["dft"]
        idx = getattr(self, "_active_idx", None)
        if idx is None:
            idx = np.arange(len(self.kpoints))
        E = self._reference_bands(params, idx)[:, self.target_tb_band_indices]
        return E, np.asarray(self.target_dft_bands)[idx]

    # ---- optional emphasis (weights) on the parts of the band structure that matter for an application ----
    def _symmetry_k_indices(self):
        """
        Indices of the DFT k-points at the two ends of every line segment of the k-path (so both sides of a jump
        such as 'H|P' are included) together with the block number of every k-point (see kpath_blocks; None if
        there is no k-path). Weights around a symmetry point never spread into another block.
        """
        nk = len(self.kpoints)
        idx = set(int(i) for i in getattr(self, "high_sym_indices", []))
        blocks = None
        try:
            segs = self._get_kpath_segments()
            for sg in segs:
                idx.add(min(sg["i0"], nk - 1))
                idx.add(min(sg["i1"], nk - 1))
            if segs:
                blocks = kpath_blocks(segs, nk)
        except Exception:
            pass
        return sorted(i for i in idx if 0 <= i < nk), blocks

    def _build_fit_weights(self, add_weight_at_symmetry_points, add_weight_at_fermi_bands, symmetry_weight,
                           symmetry_width, fermi_weight, fermi_window):
        """
        Weight W[k, band] (>= 1 where emphasised) multiplying the squared band error of the fit. Two independent
        factors that multiply where they overlap:
          symmetry points: k-points at the ends of the k-path segments (all fitted bands) get weight
              symmetry_weight, decaying linearly to 1 over `symmetry_width` neighbouring k-points on each side;
          Fermi bands: an entry (k, band) is weighted by 1 + (fermi_weight - 1) exp(-(E_DFT/fermi_window)^2), i.e.
              full weight for bands within ~fermi_window (eV) of the Fermi level, smoothly falling off.
        """
        nk, nb = np.asarray(self.target_dft_bands).shape
        W = np.ones((nk, nb))
        if add_weight_at_symmetry_points:
            if symmetry_weight <= 0 or int(symmetry_width) < 0:
                raise ValueError("symmetry_weight must be > 0 and symmetry_width >= 0.")
            sym, blocks = self._symmetry_k_indices()
            if not sym:
                raise ValueError("add_weight_at_symmetry_points needs a k-path (VASP line-mode KPOINTS file or pw.x "
                                 "crystal_b / tpiba_b bands input) to locate the high-symmetry points "
                                 f"(read: {self.kpoints_path}).")
            width = int(symmetry_width)
            profile = np.zeros(nk)
            for i in sym:
                for d in range(-width, width + 1):
                    j = i + d
                    if 0 <= j < nk and (blocks is None or blocks[j] == blocks[i] or d == 0):
                        profile[j] = max(profile[j], 1.0 - abs(d) / (width + 1.0))
            W *= (1.0 + (symmetry_weight - 1.0) * profile)[:, None]
            self._sym_rows = np.array(sym)
        if add_weight_at_fermi_bands:
            if fermi_weight <= 0 or fermi_window <= 0:
                raise ValueError("fermi_weight and fermi_window must be > 0.")
            W *= 1.0 + (fermi_weight - 1.0) * np.exp(-(np.nan_to_num(np.asarray(self.target_dft_bands), nan=1e3)
                                                       / fermi_window) ** 2)
        return W

    def _active_sqrt_weights(self):
        w = getattr(self, "_sqrt_weights", None)
        if w is None:
            return None
        idx = getattr(self, "_active_idx", None)
        return w if idx is None else w[idx]

    def fit_error_report(self, params=None, fermi_window=0.5):
        """
        Unweighted band errors (meV) of `params` (default: the fitted ones) on every k-point: over all fitted
        bands, at the high-symmetry k-points, and for the entries with |E_DFT - E_F| < fermi_window (eV). Returns a
        dict with 'all', 'symmetry_points', 'near_fermi' (RMSE in meV; NaN if there are no such entries).
        """
        params = self.opt_params if params is None else params
        keep_idx, self._active_idx = getattr(self, "_active_idx", None), None
        self._active = None
        try:
            E, dft = self._selected_bands(params)
        finally:
            self._active_idx, self._active = keep_idx, None
        dft = np.asarray(dft)
        err2 = (np.asarray(E) - dft) ** 2                   # NaN where a DFT state was left out (restrict_to_states)
        sym, _ = self._symmetry_k_indices()
        near = np.abs(np.nan_to_num(dft, nan=np.inf)) < fermi_window
        rms = lambda a: 1000.0 * float(np.sqrt(np.nanmean(a))) if np.isfinite(a).any() else float("nan")
        return {"all": rms(err2), "symmetry_points": rms(err2[sym]) if sym else float("nan"),
                "near_fermi": rms(err2[near])}

    def objective_function(self, param_vec):
        try:
            E, dft = self._selected_bands(self._param_vec_to_dict(param_vec))
            if not np.all(np.isfinite(E)):
                return 1e9
            d = np.where(np.isnan(dft), 0.0, E - dft)       # NaN: DFT state left out (restrict_to_states)
            w = self._active_sqrt_weights()
            if w is not None:
                d = d * w
            total_loss = float(np.sum(d ** 2)) + float(np.sum(self._overlap_penalty(self._param_vec_to_dict(param_vec)) ** 2))
            return total_loss if np.isfinite(total_loss) else 1e9
        except Exception:
            return 1e9

    def _residuals(self, param_vec):
        try:
            E, dft = self._selected_bands(self._param_vec_to_dict(param_vec))
            w = self._active_sqrt_weights()
            d = np.where(np.isnan(dft), 0.0, E - dft)       # NaN: DFT state left out (restrict_to_states)
            r = (d if w is None else d * w).ravel()
            if self.non_orthogonal and float(getattr(self, "overlap_floor", 0.0) or 0.0) > 0:
                r = np.concatenate([r, self._overlap_penalty(self._param_vec_to_dict(param_vec))])
            if not np.all(np.isfinite(r)):
                raise FloatingPointError
            return r
        except Exception:
            nk = len(getattr(self, "_active_idx", None) if getattr(self, "_active_idx", None) is not None
                     else self.kpoints)
            n = nk * len(self.target_tb_band_indices)
            if self.non_orthogonal and float(getattr(self, "overlap_floor", 0.0) or 0.0) > 0:
                n += nk
            return np.full(n, 1e3)

    def _parallel_jacobian(self, lo, hi, workers):
        """Forward-difference Jacobian of the residuals with one column per task on ``workers`` threads (the
        diagonalisations release the GIL). Steps as scipy's '2-point' scheme, taken backwards at an upper bound."""
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(int(workers))
        eps = np.finfo(float).eps ** 0.5

        def jac(x):
            x = np.asarray(x, dtype=float)
            r0 = np.asarray(self._residuals(x))
            h = eps * np.maximum(1.0, np.abs(x)) * np.where(x >= 0, 1.0, -1.0)
            h = np.where((x + h > hi) | (x + h < lo), -h, h)

            def column(i):
                xi = x.copy()
                xi[i] += h[i]
                return (np.asarray(self._residuals(xi)) - r0) / h[i]
            return np.column_stack(list(pool.map(column, range(len(x)))))
        return jac, pool

    def _least_squares_refine(self, x0, bounds, max_nfev=300):
        """Bounded Levenberg-Marquardt-type refinement of the summed squared band error. With
        ``self.fit_workers`` > 1 the Jacobian is evaluated on that many threads."""
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        workers = int(getattr(self, "fit_workers", 1) or 1)
        tol = float(getattr(self, "fit_tolerance", 1e-10) or 1e-10)     # relative change at which a start stops
        jac, pool = self._parallel_jacobian(lo, hi, workers) if workers > 1 else ("2-point", None)
        try:
            res = least_squares(self._residuals, np.clip(x0, lo, hi), jac=jac, bounds=(lo, hi), method='trf',
                                x_scale='jac', max_nfev=max_nfev, xtol=tol, ftol=tol, gtol=tol)
        finally:
            if pool is not None:
                pool.shutdown(wait=False)
        return res.x, float(2.0 * res.cost)

    def fit(self, maxiter=100, popsize=15, initial_params=None, custom_bounds=None, fit_eta=None, fixed_eta=None,
            k_stride=1, refine=True, refine_max_nfev=300, method="ls", n_starts=8, seed=None,
            add_weight_at_symmetry_points=False, add_weight_at_fermi_bands=False,
            symmetry_weight=10.0, symmetry_width=2, fermi_weight=10.0, fermi_window=0.5, workers=None,
            tolerance=None, overlap_floor=None):
        """
        Fit the Slater-Koster parameters to the DFT bands.

        method="ls"   (default) multi-start bounded least squares: the first start is your initial
                      guess, the others are random perturbations of it; maxiter/popsize are unused.
                      Fast and accurate when the initial estimates are reasonable.
        method="de"   differential evolution (global) -> optional least-squares refinement.
                      Slower, but does not rely on good initial estimates: use it for a new
                      material or basis where you have no starting values.
        k_stride      global stage uses every k_stride-th k-point (plus high-symmetry ones);
                      the refinement and all reported errors always use every k-point
        refine        finish with a bounded least-squares polish on the full k-set
        seed          makes the stochastic parts reproducible

        Optional weights (off by default, since not every fit needs them) multiply the squared error of a band
        at a k-point, so the fit trades accuracy elsewhere for accuracy where it matters, e.g. for quantities
        such as Berry curvature or the spin Hall conductivity that are controlled by the bands near the Fermi
        level and by the degeneracies at high-symmetry points:
        add_weight_at_symmetry_points  emphasise all fitted bands at the high-symmetry k-points (the ends of the
                      line segments of the KPOINTS path): weight symmetry_weight there, decaying linearly to 1 over
                      symmetry_width neighbouring k-points on each side
        add_weight_at_fermi_bands      emphasise the bands close to the Fermi level: an entry (k, band) is
                      weighted by 1 + (fermi_weight - 1) exp(-(E_DFT/fermi_window)^2) with E_DFT relative to E_F
                      in eV, i.e. up to fermi_weight for bands within ~fermi_window eV of E_F
        If both are on the weights multiply, so the near-E_F bands at the high-symmetry points count the most.
        The reported RMSE values are always the plain (unweighted) errors; fit_error_report() gives them
        separately for all bands, at the symmetry points and near E_F. Note that fit_or_load() re-uses saved
        parameters unless force_refit=True.
        """
        if workers is not None:
            self.fit_workers = max(1, int(workers))
        if tolerance is not None:
            self.fit_tolerance = float(tolerance)
        if overlap_floor is not None:
            self.overlap_floor = float(overlap_floor)
        if method not in ("de", "ls"):
            raise ValueError("method must be 'de' or 'ls'")
        if fit_eta is not None: self.fit_eta = fit_eta
        if fixed_eta is not None: self.fixed_eta = fixed_eta

        param_keys = self._get_param_keys()
        bounds = []
        x0_vec = []

        for key in param_keys:
            if custom_bounds and key in custom_bounds:
                bnd = custom_bounds[key]
            elif key.startswith("lambda_"):
                bnd = (0.0, 1.0)
            elif key.startswith("delta_spin_"):
                bnd = (-5.0, 5.0)
            elif key.startswith("e_orbit"):
                bnd = (-10.0, 15.0)
            elif key.startswith("s_"):
                bnd = (-0.8, 0.8)
            elif key.startswith("v_"):
                if "pds" in key or "dds" in key or "sss" in key or "s*s*s" in key or "ss*s" in key or "sds" in key or "s*ds" in key:
                    bnd = (-10.0, 2.0)
                elif "pdp" in key or "ddp" in key or "pps" in key:
                    bnd = (-2.0, 8.0)
                elif "ddd" in key:
                    bnd = (-2.0, 2.0)
                else:
                    bnd = (-8.0, 8.0)
            elif key.startswith("eta_"):
                bnd = (1.0, 5.0)
            else:
                bnd = (-8.0, 8.0)
            
            bounds.append(bnd)

            val = None
            if initial_params and key in initial_params:
                val = initial_params[key]
            else:
                val = 0.1 if key.startswith("s_") else 1.0 if key.startswith("delta_spin_") else -2.0 if ("pds" in key or "dds" in key or "sds" in key) else 1.0 if ("pdp" in key or "ddp" in key) else 0.0
            
            x0_vec.append(np.clip(val, bnd[0], bnd[1]))
        
        for label, given in (("initial_params", initial_params), ("custom_bounds", custom_bounds)):
            unused = sorted(set(given or {}) - set(param_keys))
            if unused:
                print(f"[WARN] {label} has keys that are not free parameters of this model "
                      f"and are IGNORED: {unused}")

        n_params = len(param_keys)
        x0 = np.array(x0_vec, dtype=float)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(seed)

        # Make sure the vectorised evaluator reproduces the reference one before relying on it
        if getattr(self, "use_fast_model", True):
            self._prepare_fast_model()
            if not self.check_fast_model(x0):
                msg = ("[WARN] Fast model disagrees with the reference implementation - "
                       "falling back to the slow reference path (much slower).")
                print(msg)
                try:
                    sys.__stderr__.write(msg + "\n")
                    sys.__stderr__.flush()
                except Exception:  # noqa: BLE001
                    pass
                self.use_fast_model = False

        n_points = len(self.kpoints) * len(self.target_tb_band_indices)
        def rmse_meV(loss):
            return 1000.0 * np.sqrt(loss / n_points)

        weighted = bool(add_weight_at_symmetry_points or add_weight_at_fermi_bands)
        if weighted:
            W = self._build_fit_weights(add_weight_at_symmetry_points, add_weight_at_fermi_bands, symmetry_weight,
                                        symmetry_width, fermi_weight, fermi_window)
            self._sqrt_weights = np.sqrt(W)
            what = [n for n, on in (("high-symmetry points", add_weight_at_symmetry_points),
                                    ("bands near E_F", add_weight_at_fermi_bands)) if on]
            print(f"[INFO] Weighted fit ({' and '.join(what)}): weights {W.min():.1f}-{W.max():.1f}, "
                  f"{100.0 * np.mean(W > 1.0):.0f}% of the (k, band) entries emphasised. Reported RMSE values "
                  f"are unweighted.")
        else:
            self._sqrt_weights = None
        self._active = None

        def plain_rmse(x):                       # unweighted RMSE (meV) on every k-point; inf if H(k) cannot be solved
            try:
                E, dft = self._selected_bands(self._param_vec_to_dict(x))
            except np.linalg.LinAlgError:
                return float("inf")
            return 1000.0 * float(np.sqrt(np.nanmean((E - dft) ** 2)))

        self._set_active_k(self._select_k_indices(k_stride))
        if k_stride and int(k_stride) > 1:
            print(f"[INFO] Global stage uses {len(self._active_idx)} of {len(self.kpoints)} k-points.")

        if method == "de":
            pop_count = popsize * n_params
            init_pop = np.zeros((pop_count, n_params))
            for i, (bnd, val) in enumerate(zip(bounds, x0)):
                init_pop[0, i] = val
                span = bnd[1] - bnd[0]
                rand_vals = val + rng.normal(0, 0.1 * span, pop_count - 1)
                init_pop[1:, i] = np.clip(rand_vals, bnd[0], bnd[1])

            print(f"[INFO] Running Differential Evolution with {n_params} parameters...")
            result = differential_evolution(
                self.objective_function,
                bounds=bounds,
                init=init_pop,
                maxiter=maxiter,
                popsize=popsize,
                polish=False,
                disp=True,
                seed=None if seed is None else int(seed)
            )
            x_best = np.array(result.x, dtype=float)
            stage = "Differential evolution"
        else:
            print(f"[INFO] Running multi-start least squares ({n_starts} starts, {n_params} parameters)...")
            x_best, best_loss = None, np.inf
            for s in range(max(1, int(n_starts))):
                start = x0 if s == 0 else np.clip(x0 + rng.normal(0, 0.1 * (hi - lo)), lo, hi)
                try:
                    x_s, loss_s = self._least_squares_refine(start, bounds, refine_max_nfev)
                except Exception as exc:
                    print(f"  start {s}: failed ({exc})")
                    continue
                print(f"  start {s}: loss = {loss_s:.5g}")
                if loss_s < best_loss:
                    x_best, best_loss = x_s, loss_s
            if x_best is None:
                raise RuntimeError("All least-squares starts failed.")
            stage = "Multi-start least squares"

        # From here on, everything is evaluated on the full k-point set
        self._set_active_k(None)
        best_loss = self.objective_function(x_best)
        if not np.isfinite(plain_rmse(x_best)):
            raise ValueError(
                "No usable fit: for every start the overlap matrix S(k) of the non-orthogonal model stopped being "
                "positive definite. Narrow the bounds of the overlap parameters s_* (e.g. to +-0.3), start them "
                "closer to 0, or fit an orthogonal model first." if self.non_orthogonal else
                "No usable fit: the tight-binding bands could not be computed for the best parameters.")
        print(f"[INFO] {stage}: RMSE = {plain_rmse(x_best):.2f} meV (all k-points, unweighted)")

        # a multi-start least-squares run on all k-points is already fully refined
        if refine and not (method == "ls" and getattr(self, "_active_idx", None) is None and not (k_stride and int(k_stride) > 1)):
            try:
                x_ref, loss_ref = self._least_squares_refine(x_best, bounds, refine_max_nfev)
                loss_ref = self.objective_function(x_ref)
                if loss_ref < best_loss:
                    x_best, best_loss = x_ref, loss_ref
                    print(f"[INFO] Least-squares refinement: RMSE = {plain_rmse(x_best):.2f} meV")
                else:
                    print("[INFO] Least-squares refinement did not improve the fit; keeping previous result.")
            except Exception as exc:
                print(f"[WARN] Least-squares refinement skipped: {exc}")

        self.opt_params = self._param_vec_to_dict(x_best)
        if weighted:
            rep = self.fit_error_report(self.opt_params, fermi_window)
            print(f"[INFO] Unweighted RMSE: all {rep['all']:.1f} meV | high-symmetry points "
                  f"{rep['symmetry_points']:.1f} meV | bands within {fermi_window:g} eV of E_F {rep['near_fermi']:.1f} meV")
        print("\n=== Optimized Slater-Koster Parameters ===")
        for k, v in self.opt_params.items():
            print(f"  {k:25s}: {v:8.4f}")
            
        self._compute_tb_bands(self.opt_params)

        return self.opt_params

    # ------------------------------------------------------------------
    # Persistence: re-use previously optimised parameters (looked up by seed)
    # ------------------------------------------------------------------
    PARAM_FILE_SUFFIXES = ("_fit.json", "_fit.txt", "_params.json", "_params.txt")

    def _compute_tb_bands(self, params):
        """Diagonalise H(k) [and S(k)] on self.kpoints; stores and returns self.tb_bands."""
        if getattr(self, "use_fast_model", True):
            try:
                fm = self._prepare_fast_model()
                self.tb_bands = self._fast_bands(params, fm["phase_full"])
                return self.tb_bands
            except np.linalg.LinAlgError:
                pass                                   # let the reference path report/raise as before
        self.tb_bands = self._reference_bands(params, range(len(self.kpoints)))
        return self.tb_bands

    def _settings_snapshot(self):
        """Model settings that change the meaning of the fitted numbers."""
        return {
            "n_neighbors": int(self.n_neighbors),
            "soc": bool(self.soc),
            "is_magnetic": bool(self.is_magnetic),
            "non_orthogonal": bool(self.non_orthogonal),
            "hopping_mode": getattr(self, "hopping_mode", "decay"),
            "orbitals": [list(o) for o in self.site_orbitals],
            **({"moment_signs": list(self.site_moment_signs)} if any(v < 0 for v in getattr(self, "site_moment_signs", None) or ()) else {}),
            **({"moment_directions": [[round(float(c), 6) for c in d] for d in self.site_moment_dirs]}
               if any(abs(d[2]) < 1 - 1e-6 for d in (getattr(self, "site_moment_dirs", None) or ())) else {}),
            **({"onsite_mixing": sorted({k for site in self.onsite_mix_terms for k, _ in site})}
               if any(getattr(self, "onsite_mix_terms", None) or []) else {}),
        }

    @classmethod
    def find_param_file(cls, seedname, directory="."):
        """Return the most recently modified saved-parameter file for `seedname`, or None."""
        existing = [
            os.path.join(directory, f"{seedname}{suffix}")
            for suffix in cls.PARAM_FILE_SUFFIXES
            if os.path.isfile(os.path.join(directory, f"{seedname}{suffix}"))
        ]
        return max(existing, key=os.path.getmtime) if existing else None

    @staticmethod
    def read_param_file(path):
        """
        Parse a parameter file written by test_and_plot_fit (.json or .txt).
        Returns (params: dict[str, float], metrics: dict, settings: dict | None).
        """
        params, metrics, settings = {}, {}, None

        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            params = {k: float(v) for k, v in data.get("parameters", {}).items()}
            metrics = data.get("metrics", {})
            settings = data.get("settings")
            return params, metrics, settings

        num = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        row_re = re.compile(rf"^\s*(\S+)\s*\|\s*({num})\s*$")
        rmse_re = re.compile(rf"RMSE:\s*({num})\s*eV")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("SETTINGS:"):
                    try:
                        settings = json.loads(line.split(":", 1)[1])
                    except ValueError:
                        settings = None
                    continue
                m = row_re.match(line)
                if m:
                    params[m.group(1)] = float(m.group(2))
                    continue
                m = rmse_re.search(line)
                if m:
                    metrics["rmse_eV"] = float(m.group(1))
        return params, metrics, settings

    def load_optimised_params(self, seedname=None, directory=".", param_file=None,
                              strict=True, verbose=True):
        """
        Initialise this fitter with previously optimised parameters, if they exist.

        The file is located from `seedname` + `directory` (see PARAM_FILE_SUFFIXES),
        or given explicitly through `param_file`. On success `self.opt_params` and
        `self.tb_bands` are populated, so test_and_plot_fit() and
        the property routines work without running fit().

        strict=True  -> refuse (return False) if the saved model settings differ from
                        this fitter (neighbours, SOC, magnetism, orthogonality, orbitals).
        strict=False -> only warn about such differences.
        Returns True if parameters were loaded, False otherwise (never raises for a
        missing/incompatible file, so it can be used as an "if exists" check).
        """
        path = param_file or (self.find_param_file(seedname, directory) if seedname else None)
        if path is None or not os.path.isfile(path):
            if verbose:
                print(f"[INFO] No saved parameters for seed '{seedname}' in '{directory}'.")
            return False

        try:
            saved, metrics, settings = self.read_param_file(path)
        except (OSError, ValueError) as exc:
            print(f"[WARN] Could not read parameter file {path}: {exc}")
            return False
        if not saved:
            print(f"[WARN] No parameters found in {path}.")
            return False
        saved = self.convert_legacy_params(saved, settings)

        # 1. Model-setting consistency (only verifiable if the file recorded them)
        if settings:
            current = self._settings_snapshot()
            diffs = [f"{k}: saved={settings[k]!r} vs current={current[k]!r}"
                     for k in current if k in settings and settings[k] != current[k]]
            if diffs:
                msg = "Saved fit was made with different settings:\n    " + "\n    ".join(diffs)
                if strict:
                    print(f"[WARN] {msg}\n[WARN] Not loading {path}.")
                    return False
                print(f"[WARN] {msg}\n[WARN] Loading anyway (strict=False).")
        elif verbose:
            print("[WARN] Saved file has no settings record; cannot verify it matches this model.")

        # 2. Key consistency: every free parameter of the *current* model must be present
        keys = self._get_param_keys()
        missing = [k for k in keys if k not in saved]
        if missing:
            print(f"[WARN] {path} lacks parameters required by this model: {missing}. Not loading.")
            return False

        vec = np.array([saved[k] for k in keys], dtype=float)
        if not np.all(np.isfinite(vec)):
            print(f"[WARN] {path} contains non-finite values. Not loading.")
            return False

        # Rebuild the full dict (derived e_site*/eta_* entries) from the free parameters
        params = self._param_vec_to_dict(vec)
        unused = sorted(set(saved) - set(params))
        if unused and verbose:
            print(f"[WARN] Ignoring parameters in file not used by this model: {unused}")

        self.opt_params = params
        self._compute_tb_bands(params)
        self.loaded_param_file = path

        if verbose:
            print(f"[INFO] Loaded {len(keys)} optimised parameters from {path}")
            if "rmse_eV" in metrics:
                print(f"[INFO] Stored RMSE: {metrics['rmse_eV'] * 1000:.2f} meV")
        return True

    def fit_or_load(self, seedname, directory=".", param_file=None, force_refit=False,
                    warm_start=False, strict=True, fit_overrides=None, **fit_kwargs):
        """
        Use saved parameters for `seedname` if available, otherwise run fit(**fit_kwargs).

        force_refit=True  -> always run the optimiser.
        warm_start=True   -> (with force_refit) start DE from the saved parameters
                             instead of the initial_params you pass in.
        """
        fit_kwargs.update(fit_overrides or {})      # e.g. {"method": "de", "maxiter": 300}
        if not force_refit and self.load_optimised_params(seedname, directory, param_file, strict):
            return self.opt_params

        if warm_start:
            path = param_file or self.find_param_file(seedname, directory)
            if path and os.path.isfile(path):
                try:
                    saved, _, _ = self.read_param_file(path)
                    free = set(self._get_param_keys())
                    init = dict(fit_kwargs.get("initial_params") or {})
                    init.update({k: v for k, v in saved.items() if k in free})  # saved values win
                    fit_kwargs["initial_params"] = init
                    print(f"[INFO] Warm-starting optimiser from {path}")
                except (OSError, ValueError):
                    pass
        return self.fit(**fit_kwargs)

    @classmethod
    def from_saved(cls, seedname, directory=".", param_file=None, strict=True, **init_kwargs):
        """
        Build an SKFitter and load saved parameters in one call. For VASP, vasprun/KPOINTS default to
        <directory>/vasprun.xml and <directory>/KPOINTS; for dft_code='espresso' the pw.x files are looked up in
        <directory> with find_espresso_files. `fitter.params_loaded` tells you whether saved parameters were
        found and applied.
        """
        if normalise_dft_code(init_kwargs.get("dft_code", "vasp")) == "espresso":
            found = find_espresso_files(directory)
            init_kwargs.setdefault("qe_xml_path", found["xml"])
            init_kwargs.setdefault("qe_bands_input", found["bands_input"])
            init_kwargs.setdefault("qe_scf_path", found["scf"])
        else:
            init_kwargs.setdefault("vasprun_path", os.path.join(directory, "vasprun.xml"))
            init_kwargs.setdefault("kpoints_path", os.path.join(directory, "KPOINTS"))
        fitter = cls(**init_kwargs)
        fitter.params_loaded = fitter.load_optimised_params(seedname, directory, param_file, strict)
        return fitter

    def test_and_plot_fit(self, save_fig=None, save_param=None):
        best_params = getattr(self, 'opt_params', None)
        if best_params is None:
            raise ValueError("No fitted parameters found. Run fitter.fit() first.")

        E_tb = self.tb_bands[:, self.target_tb_band_indices]
        E_dft = self.target_dft_bands

        diff = E_tb - E_dft
        rmse = np.sqrt(np.nanmean(diff ** 2))
    
        print("=" * 45)
        print(f"  FIT QUALITY METRIC")
        print(f"  Final RMSE: {rmse:.6f} eV ({rmse * 1000:.2f} meV)")
        print("=" * 45)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(E_dft, color='black', linestyle='-')
        ax.plot(E_tb, color='red', linestyle='--')
        
        if hasattr(self, 'high_sym_indices') and getattr(self, 'high_sym_labels', None):
            valid_indices = [idx for idx in self.high_sym_indices if idx < len(self.kpoints)]
            valid_labels = self.high_sym_labels[:len(valid_indices)]
            
            for k_idx in valid_indices:
                ax.axvline(x=k_idx, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            
            ax.set_xticks(valid_indices)
            ax.set_xticklabels(valid_labels, fontsize=11)
            ax.set_xlim(0, len(self.kpoints) - 1)
            
        ax.text(
            0.03, 0.95, f"RMSE = {rmse*1000:.1f} meV", 
            transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
    
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"Tight-Binding vs DFT Band Structure ({'Non-Orthogonal' if self.non_orthogonal else 'Orthogonal'})")
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='-', label='DFT'),
            Line2D([0], [0], color='red', linestyle='--', label='TB Fit')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, linestyle=':', alpha=0.6)

        if save_fig:
            dirname = os.path.dirname(save_fig)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_fig}")
        
        plt.show()

        if save_param:
            dirname = os.path.dirname(save_param)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            if save_param.endswith('.json'):
                output_data = {
                    "metrics": {
                    "rmse_eV": float(rmse),
                    "rmse_meV": float(rmse * 1000)
                },
                "settings": self._settings_snapshot(),
                "parameters": best_params
                }
                with open(save_param, 'w', encoding="utf-8") as f:
                    json.dump(output_data, f, indent=4)
                print(f"Parameters and metrics saved to JSON: {save_param}")
            
            else:
                with open(save_param, 'w', encoding="utf-8") as f:
                    f.write("====================================================\n")
                    f.write("      TIGHT-BINDING FITTED PARAMETERS REPORT        \n")
                    f.write("====================================================\n\n")
                    f.write(f"FIT QUALITY METRICS:\n")
                    f.write(f"  RMSE: {rmse:.6f} eV ({rmse * 1000:.2f} meV)\n")
                    f.write("-" * 52 + "\n\n")
                    f.write(f"SETTINGS: {json.dumps(self._settings_snapshot())}\n\n")
                
                    f.write(f"{'Parameter Name':<25} | {'Optimized Value':<15}\n")
                    f.write("-" * 52 + "\n")
                
                    for key, val in sorted(best_params.items()):
                        f.write(f"{key:<25} | {val:<15.6f}\n")
                
                    f.write("\n====================================================\n")
                
                print(f"Parameters report saved to text file: {save_param}")
            
        return rmse
    
    def build_gauge_transformation(self, k_vec):
        N = self.base_dim
        U_diag = np.zeros(N, dtype=complex)
        
        for site_idx in range(len(self.structure)):
            tau = self.structure[site_idx].coords  
            phase = np.exp(1j * np.dot(k_vec, tau))
            offset = self.site_offsets[site_idx]
            dim = self.site_dims[site_idx]
            U_diag[offset : offset + dim] = phase

        if self.soc or self.is_magnetic:
            U_diag = np.concatenate([U_diag, U_diag])

        return np.diag(U_diag)
    



    def _kpath_pieces(self, n_kpoints=None):
        """
        Continuous pieces of the k-path, as [first DFT index, last DFT index, nodes (reduced coordinates), node
        labels], built from the segments (self.kpath_segments: VASP line-mode KPOINTS or the pw.x crystal_b /
        tpiba_b path); a new piece starts wherever a segment does not begin where the previous one ended (a jump,
        e.g. 'H|P'). n_kpoints (number of DFT k-points) only trims the index ranges; nodes do not depend on it.
        """
        try:
            segments = self._get_kpath_segments()
        except (OSError, ValueError, IndexError):
            segments = []
        last = (n_kpoints - 1) if n_kpoints else 10 ** 12
        pieces = []
        for sg in segments:
            c0, lab0, c1, lab1 = sg["k0"], sg["l0"], sg["k1"], sg["l1"]
            first, end = sg["i0"], min(sg["i1"], last)
            if first >= end:
                continue
            if pieces and np.allclose(pieces[-1][2][-1], c0):
                pieces[-1][1] = end
                pieces[-1][2].append(c1)
                pieces[-1][3].append(lab1 or f"K_{len(pieces[-1][2])}")
            else:
                pieces.append([first, end, [c0, c1], [lab0 or "K_1", lab1 or "K_2"]])
        return pieces

    # ------------------------------------------------------------------
    # Berry curvature
    # ------------------------------------------------------------------
    _BC_AXES = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}       # Omega_x = Omega_yz, Omega_y = Omega_zx, Omega_z = Omega_xy

    @staticmethod
    def _berry_curvature_kubo(H, dHa, dHb, bands, degen_thresh=1e-4, return_energies=False,
                              average_degenerate=False):
        """
        Band-resolved Berry curvature Omega_n = -2 Im sum_{m != n} <n|dH/dk_a|m><m|dH/dk_b|n> / (E_n - E_m)^2
        (A_n = i<u_n|grad u_n>, Omega = curl A; units Angstrom^2 for H in eV and k in 1/Angstrom) for the
        bands in `bands` (indices into the ascending spectrum); the sum over m runs over ALL bands.
        H, dHa, dHb: (nk, D, D) Hermitian arrays. Pairs closer than degen_thresh (eV) are left out, as in
        the Kubo formula; inside an exactly degenerate multiplet only the sum over the multiplet is meaningful
        (the split of a multiplet's curvature among its members depends on the arbitrary basis the
        diagonalisation returns).
        average_degenerate=True: every band gets the MEAN over its degenerate group (bands closer than
        degen_thresh, chained), which is gauge invariant.
        The sum over the bands of a group, and hence every total, is unchanged.
        dHa may be any Hermitian operator (e.g. a spin current); return_energies=True also returns E (nk, D).
        """
        E, U = np.linalg.eigh(H)
        Ud = U.conj().transpose(0, 2, 1)
        Va = Ud @ dHa @ U                                    # Va[k, n, m] = <n|dH/dk_a|m>
        Vb = Ud @ dHb @ U
        b = list(bands)
        rows = list(range(E.shape[1])) if average_degenerate else b     # partners of a group may lie outside `bands`
        dE = E[:, rows, None] - E[:, None, :]                # (nk, nrows, D)
        w = np.zeros_like(dE)
        ok = np.abs(dE) > degen_thresh
        w[ok] = 1.0 / dE[ok] ** 2
        term = Va[:, rows, :] * Vb[:, :, rows].transpose(0, 2, 1)  # Va[n, m] * Vb[m, n]
        omega = -2.0 * np.imag(np.sum(term * w, axis=2))
        if average_degenerate:
            gid = np.concatenate([np.zeros((E.shape[0], 1), dtype=int),
                                  np.cumsum(np.diff(E, axis=1) > degen_thresh, axis=1)], axis=1)
            same = gid[:, :, None] == gid[:, None, :]                  # (nk, D, D): same degenerate group
            omega = (np.einsum("kij,kj->ki", same.astype(float), omega) / same.sum(axis=2))[:, b]
        return (omega, E) if return_energies else omega

    def _tb_orthogonal_hamiltonians(self, k_cart, chunk=512):
        """
        H(k) for Cartesian k (n_k, 3) in the phase convention of build_hamiltonian (exp(i k.d), orbitals at their
        atoms; this is what makes the Berry curvature physical), Loewdin-orthogonalised for non-orthogonal
        models. Shape (n_k, D, D).
        """
        fm = self._prepare_fast_model()
        k_cart = np.asarray(k_cart, dtype=float).reshape(-1, 3)
        out = np.empty((len(k_cart), fm["D"], fm["D"]), dtype=complex)
        for s in range(0, len(k_cart), chunk):
            kc = k_cart[s:s + chunk]
            H, S = self._fast_H_S(self.opt_params, np.exp(1j * (kc @ fm["d_vec"].T)), self.non_orthogonal)
            if S is not None:
                w, V = np.linalg.eigh(S)
                w = np.maximum(w, 1e-10)
                inv_sqrt_S = (V / np.sqrt(w)[:, None, :]) @ V.conj().transpose(0, 2, 1)
                H = inv_sqrt_S @ H @ inv_sqrt_S
            out[s:s + chunk] = 0.5 * (H + H.conj().transpose(0, 2, 1))
        return out

    def _berry_curvature_sk(self, k_cart, bands, component="z", delta=1e-4, degen_thresh=1e-4, chunk=256,
                            average_degenerate=True):
        """Omega_component(k) of the fitted SK model for the given bands at Cartesian k (1/Angstrom): (n_k, len(bands))."""
        a, b = self._BC_AXES[component]
        ea, eb = np.eye(3)[a] * delta, np.eye(3)[b] * delta       # dH/dk by central differences
        k_cart = np.asarray(k_cart, dtype=float).reshape(-1, 3)
        out = np.empty((len(k_cart), len(bands)))
        for s in range(0, len(k_cart), chunk):
            kc = k_cart[s:s + chunk]
            ham = self._tb_orthogonal_hamiltonians
            dHa = (ham(kc + ea) - ham(kc - ea)) / (2.0 * delta)
            dHb = (ham(kc + eb) - ham(kc - eb)) / (2.0 * delta)
            out[s:s + chunk] = self._berry_curvature_kubo(ham(kc), dHa, dHb, bands, degen_thresh,
                                                          average_degenerate=average_degenerate)
        return out

    def _dense_kpath(self, dk):
        """
        Dense k-path along the KPOINTS nodes with spacing ~dk (1/Angstrom). A jump between continuous pieces takes
        no width on the axis. Returns a dict: frac/cart (n, 3), x (n,), per-piece 'pieces' (slice, x0, x1, nodes,
        labels) and axis 'ticks' (positions, labels; 'H|P' at a jump).
        """
        raw = self._kpath_pieces()
        if not raw:
            raise ValueError("Need a k-path (VASP line-mode KPOINTS or pw.x crystal_b / tpiba_b bands input) to build the "
                             f"k-path (read: {self.kpoints_path}).")
        B = self.structure.lattice.reciprocal_lattice.matrix
        frac, xs, pieces, tick_x, tick_l = [], [], [], [], []
        x0, n_done = 0.0, 0
        for _, _, nodes, labels in raw:
            nodes = np.array(nodes, dtype=float)
            pf, px, x = [nodes[0]], [x0], x0
            if tick_l:                                            # junction with the previous piece
                if labels[0] != tick_l[-1]:
                    tick_l[-1] = f"{tick_l[-1]}|{labels[0]}"
            else:
                tick_x.append(x0); tick_l.append(labels[0])
            for a, b, lab in zip(nodes[:-1], nodes[1:], labels[1:]):
                length = np.linalg.norm((b - a) @ B)
                n = max(1, int(round(length / dk)))
                t = np.arange(1, n + 1) / n
                pf.extend(a + (b - a) * ti for ti in t)
                px.extend(x + length * t)
                x += length
                tick_x.append(x); tick_l.append(lab)
            frac.extend(pf); xs.extend(px)
            pieces.append((slice(n_done, n_done + len(pf)), x0, x, nodes.tolist(), list(labels)))
            n_done += len(pf)
            x0 = x
        frac = np.array(frac)
        return {"frac": frac, "cart": frac @ B, "x": np.array(xs), "pieces": pieces, "ticks": (tick_x, tick_l)}

    def _fold_to_bz(self, k_cart):
        """Translate Cartesian k (n, 3) by reciprocal lattice vectors b1, b2 into the first Brillouin zone (in-plane)."""
        B = self.structure.lattice.reciprocal_lattice.matrix
        G = np.array([i * B[0] + j * B[1] for i in range(-2, 3) for j in range(-2, 3)])
        cand = k_cart[:, None, :] + G[None, :, :]
        best = np.argmin(np.linalg.norm(cand, axis=2), axis=1)
        return cand[np.arange(len(k_cart)), best]

    def _bz_outline(self):
        """Closed polygon (m, 2) of the first Brillouin zone in the kx-ky plane (Voronoi cell of Gamma)."""
        from scipy.spatial import Voronoi
        B = self.structure.lattice.reciprocal_lattice.matrix
        G = np.array([i * B[0, :2] + j * B[1, :2] for i in range(-2, 3) for j in range(-2, 3)])
        vor = Voronoi(G)
        verts = vor.vertices[vor.regions[vor.point_region[int(np.argmin(np.linalg.norm(G, axis=1)))]]]
        verts = verts[np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))]
        return np.vstack([verts, verts[:1]])

    def _bc_map_grid(self, nk, k3):
        """nk x nk grid of reduced (k1, k2) in [-1/2, 1/2) at fixed reduced k3; needs b1, b2 in the xy plane."""
        B = self.structure.lattice.reciprocal_lattice.matrix
        if np.abs(B[:2, 2]).max() > 1e-6:
            raise ValueError("mode='map' needs the first two lattice vectors in the xy plane (a 2D-type cell); "
                             "use mode='path' for this structure.")
        u = np.arange(nk) / nk - 0.5
        U, V = np.meshgrid(u, u, indexing="ij")
        return np.stack([U.ravel(), V.ravel(), np.full(nk * nk, float(k3))], axis=1)

    def _bc_result(self, mode, omega, idx, **extra):
        """Bundle band-wise values with the two totals: all fitted bands and the occupied ones (index <= tb_vbm_index)."""
        occ_cols = [c for c, i in enumerate(idx) if i <= self.tb_vbm_index]
        res = {"mode": mode, "bands": list(idx), "omega": omega, "total": omega.sum(axis=1),
               "occupied_bands": [idx[c] for c in occ_cols],
               "total_occupied": omega[:, occ_cols].sum(axis=1) if occ_cols else np.zeros(len(omega))}
        res.update(extra)
        return res

    @staticmethod
    def _robust_limit(a, q=99.0):
        v = np.nanpercentile(np.abs(a), q)
        return float(v) if v > 1e-12 else 1.0

    def _plot_berry_curvature(self, res, source, component, ylim=None, save_fig=None, save_data=None, show=True,
                              band_offset=None):
        idx, om = res["bands"], res["omega"]
        nb = len(idx)
        comp = rf"$\Omega_{component}$"
        unit = r"(Å$^2$)"
        title = f"Berry curvature {comp} from {source} ({nb} fitted bands, idx {idx[0]}-{idx[-1]})"
        occ = res["occupied_bands"]
        occ_txt = ("occupied fitted bands (none)" if not occ else
                   f"occupied fitted bands (idx {occ[0]})" if len(occ) == 1 else
                   f"occupied fitted bands (idx {occ[0]}-{occ[-1]})")

        if res["mode"] == "path":
            x = res["x"]
            colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, max(nb, 2)))[:nb]
            # Bands are stacked: band j is drawn on its own baseline j*band_offset (lowest fitted band at the
            # bottom), so they no longer overlap. Each curve is clipped to +-0.48*band_offset around its baseline.
            vanishing = bool(np.nanmax(np.abs(om)) < 1e-3)   # nothing but numerical noise (e.g. inversion + time reversal)
            if band_offset is None:                        # auto: no clipping unless a few huge spikes would flatten everything
                peak, robust = float(np.nanmax(np.abs(om))), self._robust_limit(om, 99.0)
                band_offset = 2.2 * (peak if peak <= 3.0 * robust else robust)
                if band_offset < 1e-9 or vanishing:
                    band_offset = 1.0
            band_offset = float(band_offset)
            half = 0.48 * band_offset
            clipped = bool(np.any(np.abs(om) > half))
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, min(14.0, 4.5 + 0.55 * nb)),
                                    constrained_layout=True, gridspec_kw={"height_ratios": [max(2.0, 0.45 * nb), 1.2]})
            for j in range(nb):
                axs[0].axhline(j * band_offset, color=colours[j], linewidth=0.8, alpha=0.55, zorder=1)
            for sl in res["slices"]:                       # each continuous piece on its own: nothing across a jump
                for j in range(nb):
                    axs[0].plot(x[sl], np.clip(om[sl, j], -half, half) + j * band_offset, color=colours[j],
                                linewidth=1.3, zorder=2)
                axs[1].plot(x[sl], res["total"][sl], color="black", linewidth=1.8)
                axs[1].plot(x[sl], res["total_occupied"][sl], color="#eb6834", linestyle="--", linewidth=1.8)
            axs[0].set_ylim(-0.5 * band_offset, (nb - 0.5) * band_offset)
            ystep = max(1, int(np.ceil(nb / 24)))
            axs[0].set_yticks([j * band_offset for j in range(0, nb, ystep)])
            axs[0].set_yticklabels([f"{idx[j]}" for j in range(0, nb, ystep)])
            # totals: own robust scale, but never zoomed into numerical noise when the bands cancel
            tlim = 1.15 * max(self._robust_limit(np.concatenate([res["total"], res["total_occupied"]]), 99.5),
                              0.05 * self._robust_limit(om))
            if vanishing:
                tlim = 1.15 * 0.05
            axs[1].set_ylim(*(ylim if ylim is not None else (-tlim, tlim)))
            for t in res["ticks"][0]:
                for ax in axs:
                    ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            axs[1].axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
            axs[0].grid(False)
            axs[1].grid(True, linestyle=":", alpha=0.6)
            note = (f"baselines {band_offset:.3g} {unit} apart" + (f"; curves clipped at \u00b1{half:.3g}" if clipped else "")
                    + ("; degenerate bands show the mean of their group" if res.get("averaged") else ""))
            if vanishing:
                note = ("|\u03a9| < 1e-3 \u00c5$^2$ for every band: the Berry curvature vanishes (e.g. inversion and "
                        "time-reversal symmetry with degenerate bands); what is left is numerical noise")
            axs[1].set_xticks(res["ticks"][0])
            axs[1].set_xticklabels(res["ticks"][1], fontsize=11)
            axs[1].set_xlim(0.0, x[-1])
            axs[0].set_ylabel("band index")
            axs[0].text(0.5, 1.008, note, transform=axs[0].transAxes, ha="center", va="bottom", fontsize=8, color="dimgray")
            axs[1].set_ylabel(f"total {comp} {unit}")
            axs[0].set_title(title, pad=16)
            axs[1].legend(handles=[plt.Line2D([0], [0], color="black", linewidth=1.8, label=f"sum, all {nb} fitted bands"),
                                   plt.Line2D([0], [0], color="#eb6834", linestyle="--", linewidth=1.8,
                                              label=f"sum, {occ_txt}")], loc="upper center",
                          bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9, frameon=False)
            axs = list(axs)
        else:
            from matplotlib.tri import Triangulation
            panels = [(f"band {i}", om[:, j]) for j, i in enumerate(idx)]
            panels += [(f"sum, all {nb} fitted", res["total"]), ("sum, occupied fitted", res["total_occupied"])]
            ncols = min(4, len(panels))
            nrows = int(np.ceil(len(panels) / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), constrained_layout=True,
                                    squeeze=False)
            tri = Triangulation(res["kx"], res["ky"])
            outline = self._bz_outline()
            # one colour scale for every panel (so cancellation in the sums shows as white, not as amplified noise)
            v = (max(abs(ylim[0]), abs(ylim[1])) if ylim is not None
                 else max(self._robust_limit(vals) for _, vals in panels))
            for ax, (name, vals) in zip(axs.ravel(), panels):
                im = ax.tripcolor(tri, vals, cmap="RdBu_r", vmin=-v, vmax=v, shading="gouraud")
                ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=0.8)
                ax.set_aspect("equal")
                ax.set_title(name, fontsize=10)
                ax.set_xlabel(r"$k_x$ (Å$^{-1}$)", fontsize=9)
                ax.set_ylabel(r"$k_y$ (Å$^{-1}$)", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            for ax in axs.ravel()[len(panels):]:
                ax.set_visible(False)
            fig.suptitle(f"{title}; colour scale in {unit}", fontsize=11)
            axs = list(axs.ravel())

        if save_fig:
            dirname = os.path.dirname(save_fig)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_fig, dpi=300, bbox_inches="tight")
            print(f"[INFO] Berry curvature plot saved to {save_fig}")
        if save_data:
            dirname = os.path.dirname(save_data)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            if res["mode"] == "path":
                cols, head = [res["x"]], "k_distance(1/A)"
            else:
                cols, head = [res["kx"], res["ky"]], "kx(1/A) ky(1/A)"
            cols += [res["total"], res["total_occupied"]] + [om[:, j] for j in range(nb)]
            head += " total_fitted total_occupied " + " ".join(f"band{i}" for i in idx)
            np.savetxt(save_data, np.column_stack(cols), header=f"{source}; Omega_{component} in A^2; {head}", fmt="%.8e", encoding="utf-8")
            print(f"[INFO] Berry curvature data saved to {save_data}")
        if show:
            plt.show()
        return fig, axs

    def _bc_setup(self, component):
        if getattr(self, "opt_params", None) is None:
            raise ValueError("No fitted parameters found. Run fit() or load_optimised_params() first.")
        component = str(component).lower()
        if component not in self._BC_AXES:
            raise ValueError("component must be 'x', 'y' or 'z'.")
        return component, list(self.target_tb_band_indices)

    def plot_berry_curvature_sk(self, mode="path", component="z", dk=0.01, nk=40, k3=0.0, ylim=None,
                                save_fig=None, save_data=None, show=True, delta=1e-4, degen_thresh=1e-4,
                                band_offset=None, average_degenerate=True):
        """
        Compute and plot the Berry curvature Omega_component(k) of the fitted SK Hamiltonian: band by band for the
        bands used in the fit, and their sum. The Kubo sum over intermediate states runs over all model bands, so
        each Omega_n is the exact curvature of band n of the model; only the bands plotted/summed are restricted to
        the fitted ones. Two totals are drawn: all fitted bands, and the occupied ones (index <= tb_vbm_index).
        For non-orthogonal models the Loewdin-orthogonalised Hamiltonian is used.

        mode='path': along the KPOINTS path (jumps such as 'H|P' are honoured), points spaced by dk (1/Angstrom).
        mode='map' : nk x nk map over the first Brillouin zone in the plane of fixed reduced k3 (structures whose
                     first two lattice vectors lie in the xy plane, i.e. 2D-type cells).
        band_offset: (path mode) the fixed spacing, in Angstrom^2, between the baselines of successive fitted bands.
        The bands are stacked (lowest fitted band at the bottom, its index on the y axis) instead of overlapping;
        each curve is clipped to +-0.48*band_offset around its own baseline. Default: 2.2x the largest |Omega|, or
        2.2x its 99th percentile when a few huge spikes would otherwise flatten everything, i.e. it adapts to the data; give a number to fix it (e.g. band_offset=50) or to see more/less of the
        spikes near band (near-)degeneracies.
        ylim: (min, max) of the totals axis (path) or of the colour scale (map); default is robust to the spikes.
        average_degenerate: bands closer than degen_thresh (eV) form a degenerate group (e.g. the spin- or
        Kramers-degenerate pairs of a material with inversion and time-reversal symmetry, where the curvature of
        the group sums to zero). The curvature of a single member of such a group is gauge dependent, i.e. it
        depends on the arbitrary basis the diagonalisation returns and shows up as large, rapidly oscillating
        values. With True (default) every band shows the mean over its group, which is gauge invariant; the totals
        do not change. False shows the raw per-state values.
        Values in Angstrom^2. Returns (fig, axes, data dict); data['omega'] holds the raw, un-offset values.
        """
        component, idx = self._bc_setup(component)
        if mode == "path":
            kp = self._dense_kpath(dk)
            omega = self._berry_curvature_sk(kp["cart"], idx, component, delta, degen_thresh,
                                             average_degenerate=average_degenerate)
            res = self._bc_result("path", omega, idx, x=kp["x"], ticks=kp["ticks"],
                                  slices=[p[0] for p in kp["pieces"]], averaged=bool(average_degenerate))
        elif mode == "map":
            frac = self._bc_map_grid(nk, k3)
            k_cart = self._fold_to_bz(frac @ self.structure.lattice.reciprocal_lattice.matrix)
            omega = self._berry_curvature_sk(k_cart, idx, component, delta, degen_thresh,
                                             average_degenerate=average_degenerate)
            res = self._bc_result("map", omega, idx, kx=k_cart[:, 0], ky=k_cart[:, 1],
                                  averaged=bool(average_degenerate))
        else:
            raise ValueError("mode must be 'path' or 'map'.")
        print(f"[INFO] SK Berry curvature ({mode}): {len(omega)} k-points, fitted bands {idx[0]}-{idx[-1]} "
              f"of {self.total_dim}; occupied: {res['occupied_bands']}")
        fig, axs = self._plot_berry_curvature(res, "the SK Hamiltonian", component, ylim, save_fig, save_data, show,
                                              band_offset)
        return fig, axs, res

    # ------------------------------------------------------------------
    # Valley topology of 2D hexagonal cells: Wilson loops around K and K', Chern and valley Chern numbers
    # ------------------------------------------------------------------
    _VALLEY_COLOURS = {"K": "#c0392b", "K'": "#1f6fb2"}

    def _basis_positions(self):
        """Cartesian centre (Angstrom) of every basis state of H, in the order of H (the spin blocks repeat the sites)."""
        pos = np.repeat(np.asarray(self.structure.cart_coords, dtype=float), self.site_dims, axis=0)
        if self.total_dim == 2 * self.base_dim:
            pos = np.concatenate([pos, pos], axis=0)
        return pos

    def _valley_points(self, k3=0.0):
        """
        The two inequivalent corners K and K' of a hexagonal 2D Brillouin zone in the plane of fixed reduced k3.
        Corners that differ by a reciprocal lattice vector are the same valley. K is the corner labelled 'K' on the
        KPOINTS path when there is one, otherwise the corner class holding the corner of smallest polar angle.
        Returns {'K': (3,), "K'": (3,), 'corners': (m, 2), 'corner_valley': ['K'|"K'", ...], 'd_KM': |K - M| (1/A)}.
        """
        B = self.structure.lattice.reciprocal_lattice.matrix
        if np.abs(B[:2, 2]).max() > 1e-6:
            raise ValueError("The valley analysis needs the first two lattice vectors in the xy plane (a 2D-type cell).")
        verts = self._bz_outline()[:-1]
        red = verts @ np.linalg.inv(B[:2, :2])                   # verts = red @ B[:2, :2]
        cls = -np.ones(len(verts), dtype=int)
        n_cls = 0
        for i in range(len(verts)):
            if cls[i] >= 0:
                continue
            for j in range(i, len(verts)):
                d = red[j] - red[i]
                if cls[j] < 0 and np.allclose(d, np.round(d), atol=1e-4):
                    cls[j] = n_cls
            n_cls += 1
        if n_cls != 2:
            raise ValueError("Wilson loops around K and K' and valley Chern numbers need a hexagonal 2D Brillouin zone "
                             f"(two inequivalent corners K and K'); this cell's zone has {n_cls} kind(s) of corner.")
        ang = np.mod(np.arctan2(verts[:, 1], verts[:, 0]), 2 * np.pi)
        k_cls = int(cls[int(np.argmin(ang + 10.0 * (ang > 2 * np.pi - 1e-6)))])
        try:                                                     # the KPOINTS label decides which corner is K
            for _, _, nodes, labels in self._kpath_pieces():
                for node, lab in zip(nodes, labels):
                    if str(lab).strip().upper() == "K":
                        d = red - np.asarray(node, dtype=float)[None, :2]
                        hit = np.where(np.all(np.abs(d - np.round(d)) < 1e-3, axis=1))[0]
                        if len(hit):
                            k_cls = int(cls[hit[0]])
                            raise StopIteration
        except StopIteration:
            pass
        except Exception:  # noqa: BLE001  (no k-path: keep the geometric choice)
            pass
        kz = float(k3) * B[2]
        rep = {}
        for name, c in (("K", k_cls), ("K'", 1 - k_cls)):
            members = np.where(cls == c)[0]
            first = members[int(np.argmin(ang[members]))]
            rep[name] = np.array([verts[first, 0], verts[first, 1], 0.0]) + kz
        edge = float(np.min(np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1)))
        return {"K": rep["K"], "K'": rep["K'"], "corners": verts,
                "corner_valley": ["K" if c == k_cls else "K'" for c in cls], "d_KM": 0.5 * edge}

    def _gap_at(self, k_cart):
        """E_{vbm+1} - E_vbm (eV) of the model at one Cartesian k; nan when the model has no band above the VBM."""
        e = np.linalg.eigvalsh(self._tb_orthogonal_hamiltonians(np.asarray(k_cart, dtype=float)[None, :])[0])
        i = int(self.tb_vbm_index)
        return float(e[i + 1] - e[i]) if 0 <= i < len(e) - 1 else float("nan")

    def _loop_phases(self, centre, radius, n_pts, idx, occ_cols):
        """
        Berry phases on the counter-clockwise circle |k - centre| = radius in the kx-ky plane, discretised with n_pts
        points: gamma = -Im ln prod_i <u(k_i)|u(k_i+1)> (= the flux of Omega_z through the disk, modulo 2 pi).
        Returns (per fitted band (len(idx),), occupied fitted bands together (det of the overlap matrices),
        smallest |overlap| per band). A band whose overlaps collapse (degenerate with another band somewhere on the
        loop) has no gauge-invariant phase of its own: its entry is nan.
        """
        th = 2.0 * np.pi * np.arange(n_pts) / n_pts
        k = np.asarray(centre, dtype=float)[None, :] + radius * np.stack([np.cos(th), np.sin(th), 0.0 * th], axis=1)
        _, U = np.linalg.eigh(self._tb_orthogonal_hamiltonians(k))
        Ub = U[:, :, list(idx)]
        M = np.einsum("kai,kaj->kij", Ub.conj(), np.roll(Ub, -1, axis=0))      # (n, nb, nb) overlaps
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

    def compute_wilson_loops_sk(self, k3=0.0, radius=None, n_loop=120, n_radii=30, save_fig=None, save_data=None,
                                show=True):
        """
        Wilson loops (Berry phases) of the fitted SK Hamiltonian on circles around the valleys K and K' of a
        hexagonal 2D Brillouin zone, as a function of the circle radius r = r_max/n_radii ... r_max (1/Angstrom):
        gamma(r) = flux of Omega_z through the disk of radius r (modulo 2 pi), for every fitted band and for the
        occupied fitted bands together (a multi-band, U(N) gauge-invariant Wilson loop). The phases are unwrapped
        along r starting from the smallest circle, so the curves are continuous.

        - Gapless Dirac point (graphene): gamma = pi for every r, although Omega(k) = 0 everywhere else (P*T
          symmetry): the whole Berry flux of a valley is a delta function sitting exactly at K. Being defined modulo
          2 pi, +pi and -pi are the same number here; a gap gives the valleys their sign.
        - Gapped valley (hBN, graphene on hBN): gamma grows from 0 and saturates with opposite signs at K and K'
          (valley Hall physics); gamma/2pi at large r approaches the valley Chern number.

        radius: the largest circle (default/None or <= 0: 0.4 x the K-M distance; clipped to 0.95 x |K-M| so that the
        disks of K and K' never overlap). Returns (fig, axes, data dict).
        """
        if getattr(self, "opt_params", None) is None:
            raise ValueError("No fitted parameters found. Run fit() or load_optimised_params() first.")
        vp = self._valley_points(k3)
        idx = list(self.target_tb_band_indices)
        occ_cols = [c for c, i in enumerate(idx) if i <= self.tb_vbm_index]
        notes = []
        r_max = float(radius) if radius is not None and float(radius) > 0 else 0.4 * vp["d_KM"]
        if r_max > 0.95 * vp["d_KM"]:
            notes.append(f"loop radius clipped to {0.95 * vp['d_KM']:.3g} 1/Å (0.95 × |K−M|) so the K and K' disks do not overlap")
            r_max = 0.95 * vp["d_KM"]
        n_loop, n_radii = max(8, int(n_loop)), max(1, int(n_radii))
        radii = r_max * np.arange(1, n_radii + 1) / n_radii
        out = {}
        for name in ("K", "K'"):
            gb = np.empty((n_radii, len(idx)))
            go = np.empty(n_radii)
            mags = np.empty((n_radii, len(idx)))
            for j, r in enumerate(radii):
                gb[j], go[j], mags[j] = self._loop_phases(vp[name], r, n_loop, idx, occ_cols)
            go = np.unwrap(go)
            for c in range(len(idx)):
                if np.all(np.isfinite(gb[:, c])):
                    gb[:, c] = np.unwrap(gb[:, c])
            out[name] = {"centre": vp[name], "band": gb, "occupied": go, "min_overlap": mags.min(axis=0),
                         "gap": self._gap_at(vp[name])}
        gapless = [n for n in ("K", "K'") if np.isfinite(out[n]["gap"]) and out[n]["gap"] < 1e-3]
        res = {"mode": "wilson", "bands": idx, "occupied_bands": [idx[c] for c in occ_cols], "radii": radii,
               "r_max": r_max, "n_loop": n_loop, "valleys": out, "valley_points": vp, "gapless": gapless,
               "notes": notes}
        g_k, g_kp = out["K"]["occupied"][-1] / np.pi, out["K'"]["occupied"][-1] / np.pi
        print(f"[INFO] Wilson loops around K and K': {n_radii} circles up to r = {r_max:.4g} 1/A, {n_loop} points each; "
              f"gamma_occ(K) = {g_k:+.4f} pi, gamma_occ(K') = {g_kp:+.4f} pi")
        fig, axs = self._plot_wilson_loops(res)
        if save_fig:
            if os.path.dirname(save_fig):
                os.makedirs(os.path.dirname(save_fig), exist_ok=True)
            fig.savefig(save_fig, dpi=300, bbox_inches="tight")
            print(f"[INFO] Wilson-loop plot saved to {save_fig}")
        if save_data:
            if os.path.dirname(save_data):
                os.makedirs(os.path.dirname(save_data), exist_ok=True)
            cols, head = [radii], ["r(1/A)"]
            for name, tagn in (("K", "K"), ("K'", "Kp")):
                cols.append(out[name]["occupied"] / np.pi); head.append(f"gamma_occ_{tagn}/pi")
            for name, tagn in (("K", "K"), ("K'", "Kp")):
                for c, i in enumerate(idx):
                    cols.append(out[name]["band"][:, c] / np.pi); head.append(f"gamma_band{i}_{tagn}/pi")
            np.savetxt(save_data, np.column_stack(cols), fmt="%.8e",
                       header="SK Hamiltonian; Berry phase of circles around the valleys, in units of pi; " + " ".join(head), encoding="utf-8")
            print(f"[INFO] Wilson-loop data saved to {save_data}")
        if show:
            plt.show()
        return fig, axs, res

    def _plot_wilson_loops(self, res):
        vp, out, idx = res["valley_points"], res["valleys"], res["bands"]
        radii = res["radii"]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True,
                                       gridspec_kw={"width_ratios": [1.0, 1.45]})
        outline = self._bz_outline()
        ax0.plot(outline[:, 0], outline[:, 1], color="black", linewidth=0.9)
        ax0.plot([0], [0], marker="o", color="dimgray", ms=4)
        ax0.annotate("Γ", (0, 0), textcoords="offset points", xytext=(5, 4), fontsize=10)
        th = np.linspace(0, 2 * np.pi, 200)
        for (x, y), v in zip(vp["corners"], vp["corner_valley"]):
            c = self._VALLEY_COLOURS[v]
            ax0.plot([x], [y], marker="o" if v == "K" else "s", ms=6, color=c, mfc=c if v == "K" else "white", zorder=3)
        for name in ("K", "K'"):
            c = self._VALLEY_COLOURS[name]
            kx, ky = out[name]["centre"][:2]
            for frac, ls in ((1.0, "-"), (0.5, ":"), (0.2, ":")):
                r = res["r_max"] * frac
                ax0.plot(kx + r * np.cos(th), ky + r * np.sin(th), color=c, linestyle=ls, linewidth=1.3 if ls == "-" else 0.8)
            ax0.annotate("", xy=(kx + res["r_max"] * np.cos(0.35), ky + res["r_max"] * np.sin(0.35)),
                         xytext=(kx + res["r_max"] * np.cos(0.0), ky + res["r_max"] * np.sin(0.0)),
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
            up = ky >= -1e-9
            ax0.text(kx, ky + (1.12 if up else -1.12) * res["r_max"], f"{name}: γ = {out[name]['occupied'][-1] / np.pi:+.3f}π",
                     ha="center", va="bottom" if up else "top", fontsize=9, color=c,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax0.set_aspect("equal")
        pad = 0.08 * np.ptp(outline[:, 0])
        ax0.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
        ax0.set_ylim(outline[:, 1].min() - 2.5 * pad, outline[:, 1].max() + 2.5 * pad)
        ax0.set_xlabel(r"$k_x$ (Å$^{-1}$)")
        ax0.set_ylabel(r"$k_y$ (Å$^{-1}$)")
        ax0.set_title("loops around the valleys (counter-clockwise)", fontsize=10)

        show_bands = len(idx) <= 6
        dashes = [(0, (4, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), (0, (2, 3)), (0, (8, 3)), (0, (3, 1, 1, 1))]
        for name in ("K", "K'"):
            c = self._VALLEY_COLOURS[name]
            gap = out[name]["gap"]
            gtxt = f"gap {gap:.3g} eV" if np.isfinite(gap) else ""
            ax1.plot(radii, out[name]["occupied"] / np.pi, color=c, linewidth=2.4,
                     label=f"{name}: occupied fitted bands ({gtxt})")
            if show_bands:
                for col, i in enumerate(idx):
                    g = out[name]["band"][:, col]
                    if np.all(np.isnan(g)):
                        continue
                    ax1.plot(radii, g / np.pi, color=c, linewidth=1.0, alpha=0.75, linestyle=dashes[col % len(dashes)],
                             label=f"{name}: band {i}")
        ax1.axhline(0.0, color="gray", linewidth=0.8)
        for v in (-1, 1):
            ax1.axhline(v, color="gray", linewidth=0.6, linestyle=":")
        ax1.set_xlim(0.0, radii[-1])
        ax1.set_xlabel(r"loop radius $r$ (Å$^{-1}$)")
        ax1.set_ylabel(r"Berry phase $\gamma$ / π")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(fontsize=8, loc="best", ncol=2 if show_bands and len(idx) > 1 else 1)
        if res["gapless"]:
            note = ("Gap closes at " + " and ".join(res["gapless"]) + ": γ = π on every loop although Ω = 0 away from the "
                    "Dirac point - the Berry flux is a δ-function at the valley (±π are the same phase mod 2π)")
        else:
            note = "γ(r) = flux of Ω_z through the disk (mod 2π); opposite signs at K and K' = valley Hall"
        ax1.set_title(note, fontsize=8.5, color="dimgray", wrap=True)
        fig.suptitle(f"Wilson loops around K and K' - SK Hamiltonian ({len(idx)} fitted bands, idx {idx[0]}-{idx[-1]}; "
                     f"{res['n_loop']} points per loop)", fontsize=11)
        return fig, [ax0, ax1]

    def compute_chern_numbers_sk(self, nk=60, k3=0.0, save_fig=None, save_data=None, show=True):
        """
        Chern numbers and valley Chern numbers of the fitted SK Hamiltonian on an nk x nk mesh of the 2D Brillouin
        zone (plane of fixed reduced k3) with the lattice method of Fukui, Hatsugai and Suzuki (J. Phys. Soc. Jpn. 74,
        1674 (2005)): every plaquette carries the gauge-invariant Berry flux F = arg of the product of the four link
        overlaps, so sum(F)/2pi is an integer for an isolated band (set) even on a coarse mesh, and narrow curvature
        peaks are never missed. The mesh is offset by half a step so that K and K' fall inside plaquettes.

        Valley Chern numbers: the plaquettes are split between the valleys by their nearest zone corner (for the
        hexagonal zone: lines from Gamma to the M points), C_K = sum_{K half} F / 2pi and the same for K'.
        C_K + C_K' = C. For a massive Dirac valley C_K -> +-1/2 when the gap is small against the bandwidth; a large
        gap (hBN) spreads the curvature over the zone and |C_K| is smaller.

        Per fitted band and for the occupied fitted bands together (multi-band link = det of the overlap matrix).
        A band that touches another one on the mesh (e.g. gapless graphene at K, or Kramers pairs) has no Chern number
        of its own: this is reported in 'touching'. Returns (fig, axes, data dict).
        """
        if getattr(self, "opt_params", None) is None:
            raise ValueError("No fitted parameters found. Run fit() or load_optimised_params() first.")
        vp = self._valley_points(k3)
        nk = max(6, int(nk))
        B = self.structure.lattice.reciprocal_lattice.matrix
        idx = list(self.target_tb_band_indices)
        occ_cols = [c for c, i in enumerate(idx) if i <= self.tb_vbm_index]
        u = (np.arange(nk) + 0.5) / nk
        U1, U2 = np.meshgrid(u, u, indexing="ij")
        frac = np.stack([U1.ravel(), U2.ravel(), np.full(nk * nk, float(k3))], axis=1)
        k_cart = frac @ B
        E, V = np.linalg.eigh(self._tb_orthogonal_hamiltonians(k_cart))
        D = E.shape[1]
        E = E.reshape(nk, nk, D)
        V = V.reshape(nk, nk, D, D)[..., idx]                    # (nk, nk, D, nb)
        tau = self._basis_positions()
        # periodic continuation in the convention of H (orbitals at their atoms): u(k + G) = diag(exp(-i G.tau)) u(k)
        nxt = []
        for ax, G in ((0, B[0]), (1, B[1])):
            Vn = np.roll(V, -1, axis=ax)
            ph = np.exp(-1j * (tau @ G))[:, None]
            if ax == 0:
                Vn[-1] = ph * V[0]
            else:
                Vn[:, -1] = ph * V[:, 0]
            nxt.append(Vn)
        M = [np.einsum("ijan,ijam->ijnm", V.conj(), Vn) for Vn in nxt]        # link matrices along b1, b2
        orient = 1.0 if np.cross(B[0], B[1])[2] > 0 else -1.0

        def plaquette_flux(L1, L2):
            L1 = L1 / np.maximum(np.abs(L1), 1e-300)
            L2 = L2 / np.maximum(np.abs(L2), 1e-300)
            # Berry phase of the plaquette, -arg of the product of the links (A = i<u|grad u>, Omega = curl A)
            return -orient * np.angle(L1 * np.roll(L2, -1, axis=0) * np.conj(np.roll(L1, -1, axis=1)) * np.conj(L2))

        F_band = np.stack([plaquette_flux(M[0][..., c, c], M[1][..., c, c]) for c in range(len(idx))], axis=-1)
        if occ_cols:
            F_occ = plaquette_flux(np.linalg.det(M[0][..., occ_cols, :][..., :, occ_cols]),
                                   np.linalg.det(M[1][..., occ_cols, :][..., :, occ_cols]))
        else:
            F_occ = np.zeros((nk, nk))
        # plaquette centres, folded into the first zone, and their valley (nearest corner)
        cfrac = np.stack([U1.ravel() + 0.5 / nk, U2.ravel() + 0.5 / nk, np.full(nk * nk, float(k3))], axis=1)
        cen = self._fold_to_bz(cfrac @ B)
        dist = np.linalg.norm(cen[:, None, :2] - vp["corners"][None, :, :], axis=2)
        in_K = np.array([vp["corner_valley"][j] == "K" for j in np.argmin(dist, axis=1)]).reshape(nk, nk)

        def numbers(F):
            return {"C": float(F.sum() / (2 * np.pi)), "C_K": float(F[in_K].sum() / (2 * np.pi)),
                    "C_Kp": float(F[~in_K].sum() / (2 * np.pi))}

        # smallest gap of every fitted band to its neighbours (all model bands) on the mesh and at K, K' themselves
        # (the offset mesh never contains K: a Dirac point there would otherwise go unnoticed), and above the
        # occupied set. A plaquette flux near +-pi also gives a touching point (or a too coarse mesh) away.
        Ev = np.linalg.eigvalsh(self._tb_orthogonal_hamiltonians(np.array([vp["K"], vp["K'"]])))
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
        vbm = int(self.tb_vbm_index)
        occ_gap = float(np.min(Eall[:, vbm + 1] - Eall[:, vbm])) if 0 <= vbm < D - 1 else float("nan")
        occ_touch = bool(occ_cols) and np.isfinite(occ_gap) and occ_gap < 1e-3
        res = {"mode": "chern", "bands": idx, "occupied_bands": [idx[c] for c in occ_cols], "nk": nk,
               "per_band": {i: numbers(F_band[..., c]) for c, i in enumerate(idx)},
               "occupied": numbers(F_occ), "min_gap": min_gap, "occupied_gap": occ_gap, "touching": touching,
               "occupied_touching": occ_touch, "coarse": singular or
               (["occupied"] if not occ_touch and np.max(np.abs(F_occ)) > 0.75 * np.pi else []),
               "gap_K": self._gap_at(vp["K"]), "gap_Kp": self._gap_at(vp["K'"]),
               "flux_occ": F_occ.ravel(), "centres": cen, "in_K": in_K.ravel(),
               "plaquette_area": abs(np.cross(B[0], B[1])[2]) / nk ** 2, "valley_points": vp}
        o = res["occupied"]
        print(f"[INFO] Chern numbers on a {nk}x{nk} mesh (occupied fitted bands): C = {o['C']:+.4f}, "
              f"C_K = {o['C_K']:+.4f}, C_K' = {o['C_Kp']:+.4f}; smallest gap above them {occ_gap:.4g} eV")
        fig, axs = self._plot_chern_numbers(res)
        if save_fig:
            if os.path.dirname(save_fig):
                os.makedirs(os.path.dirname(save_fig), exist_ok=True)
            fig.savefig(save_fig, dpi=300, bbox_inches="tight")
            print(f"[INFO] Chern-number plot saved to {save_fig}")
        if save_data:
            if os.path.dirname(save_data):
                os.makedirs(os.path.dirname(save_data), exist_ok=True)
            rows = [[i, d["C"], d["C_K"], d["C_Kp"], d["C_K"] - d["C_Kp"], min_gap[i]] for i, d in res["per_band"].items()]
            rows.append([-1, o["C"], o["C_K"], o["C_Kp"], o["C_K"] - o["C_Kp"], occ_gap])
            np.savetxt(save_data, np.array(rows, dtype=float), fmt="%.8e",
                       header=f"SK Hamiltonian; Fukui-Hatsugai-Suzuki on a {nk}x{nk} mesh; band -1 = occupied; "
                              "band C C_K C_Kp C_K-C_Kp min_gap(eV)", encoding="utf-8")
            print(f"[INFO] Chern-number data saved to {save_data}")
        if show:
            plt.show()
        return fig, axs, res

    def _plot_chern_numbers(self, res):
        from matplotlib.colors import SymLogNorm, Normalize
        from matplotlib.tri import Triangulation
        vp = res["valley_points"]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True,
                                       gridspec_kw={"width_ratios": [1.0, 1.35]})
        omega = res["flux_occ"] / res["plaquette_area"]              # mean Omega_z of each plaquette (A^2)
        peak = float(np.max(np.abs(omega))) if omega.size else 0.0
        robust = self._robust_limit(omega, 90.0)
        v = max(peak, 1e-9)
        norm = (SymLogNorm(linthresh=max(robust, v * 1e-3), vmin=-v, vmax=v, base=10) if peak > 30.0 * robust
                else Normalize(vmin=-v, vmax=v))
        tri = Triangulation(res["centres"][:, 0], res["centres"][:, 1])
        im = ax0.tripcolor(tri, omega, cmap="RdBu_r", norm=norm, shading="gouraud")
        outline = self._bz_outline()
        ax0.plot(outline[:, 0], outline[:, 1], color="black", linewidth=0.9)
        corners = vp["corners"]
        mids = 0.5 * (corners + np.roll(corners, -1, axis=0))
        for m in mids:                                                # the valley halves: Gamma -> M lines
            ax0.plot([0, m[0]], [0, m[1]], color="black", linewidth=0.6, linestyle=":")
        for (x, y), name in zip(corners, vp["corner_valley"]):
            c = self._VALLEY_COLOURS[name]
            ax0.plot([x], [y], marker="o" if name == "K" else "s", ms=6, color=c, mfc=c if name == "K" else "white")
        for name in ("K", "K'"):
            kx, ky = vp[name][:2]
            ax0.annotate(name, (kx, ky), textcoords="offset points", xytext=(8, 6), color=self._VALLEY_COLOURS[name],
                         fontsize=11, fontweight="bold")
        ax0.set_aspect("equal")
        ax0.set_xlabel(r"$k_x$ (Å$^{-1}$)")
        ax0.set_ylabel(r"$k_y$ (Å$^{-1}$)")
        ax0.set_title(r"$\Omega_z$ of the occupied fitted bands (plaquette flux / area, Å$^2$)" +
                      ("\nsymmetric log colour scale" if isinstance(norm, SymLogNorm) else ""), fontsize=9.5)
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
        if res["occupied_touching"]:
            ax0.text(0.5, 0.04, "bands touch at the valleys: the flux sits in the\nplaquettes holding K and K' (a δ-function)",
                     transform=ax0.transAxes, ha="center", fontsize=8, color="#b03a2e",
                     bbox=dict(boxstyle="round", fc="white", ec="#b03a2e", alpha=0.85))

        groups = [(f"band {i}", d, i in res["touching"]) for i, d in res["per_band"].items()]
        groups.append(("occupied\n(together)", res["occupied"], res["occupied_touching"]))
        x = np.arange(len(groups))
        w = 0.26
        series = [("C_K", "$C_K$", self._VALLEY_COLOURS["K"]), ("C_Kp", "$C_{K'}$", self._VALLEY_COLOURS["K'"]),
                  ("C", "$C = C_K + C_{K'}$", "#444444")]
        for s, (key, lab, col) in enumerate(series):
            vals = np.array([g[1][key] for g in groups])
            bars = ax1.bar(x + (s - 1) * w, vals, w * 0.92, color=col, label=lab,
                           hatch=None if key != "C" else "//", edgecolor="white" if key != "C" else "#222222")
            for b_, g in zip(bars, groups):
                if g[2]:                                  # not defined: keep the number visible but faded
                    b_.set_alpha(0.25)
            for b_, val, g in zip(bars, vals, groups):
                ax1.annotate(f"{val:+.3f}", (b_.get_x() + b_.get_width() / 2, val), textcoords="offset points",
                             xytext=(0, 3 if val >= 0 else -10), ha="center", fontsize=7.5,
                             color="gray" if g[2] else "black")
        for k_, g in enumerate(groups):
            if g[2]:
                ax1.text(k_, 0.0, "bands touch:\nnot defined", ha="center", va="center", fontsize=7.5, color="#b03a2e",
                         bbox=dict(boxstyle="round", fc="white", ec="#b03a2e", alpha=0.85))
        for yv in (-1, -0.5, 0.5, 1):
            ax1.axhline(yv, color="gray", linewidth=0.5, linestyle=":")
        ax1.axhline(0, color="gray", linewidth=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([g[0] for g in groups])
        ax1.set_xlim(-0.6, len(groups) - 0.4)
        lim = max(1.1, 1.25 * max(abs(g[1][k]) for g in groups for k in ("C", "C_K", "C_Kp")))
        ax1.set_ylim(-lim, lim)
        ax1.set_ylabel("Chern number")
        from matplotlib.patches import Patch
        ax1.legend(handles=[Patch(facecolor=col, edgecolor="#222222" if key == "C" else "white",
                                  hatch="//" if key == "C" else None, label=lab) for key, lab, col in series],
                   fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
        og = res["occupied_gap"]
        ax1.set_title(f"Fukui-Hatsugai-Suzuki on a {res['nk']}×{res['nk']} mesh; gap at K {res['gap_K']:.3g} eV, "
                      f"at K' {res['gap_Kp']:.3g} eV; smallest gap above the occupied bands {og:.3g} eV", fontsize=8.5)
        idx = res["bands"]
        fig.suptitle(f"Chern and valley Chern numbers - SK Hamiltonian ({len(idx)} fitted bands, idx {idx[0]}-{idx[-1]})",
                     fontsize=11)
        return fig, [ax0, ax1]


    # ------------------------------------------------------------------
    # Spin Berry curvature along the k-path (the integrand of the spin Hall conductivity)
    # ------------------------------------------------------------------
    _BOHR_ANGSTROM = 0.529177210903

    def _spin_berry_curvature_sk(self, k_cart, spin="z", alpha="x", beta="y", delta=1e-4, degen_thresh=1e-4,
                                 chunk=256):
        """
        Spin Berry curvature of EVERY band of the fitted SK model at Cartesian k (n_k, 3) in 1/Angstrom:

            Omega^spin_{n,alpha beta}(k) = -2 Im sum_{m != n} <n|J_alpha|m> <m|dH/dk_beta|n> / (E_n - E_m)^2,
            J_alpha = (1/2) {sigma_spin, dH/dk_alpha}

        (sigma = Pauli matrix; the same quantity that compute_spin_hall_conductivity sums over the Fermi sea).
        Returns (omega, E), both (n_k, total_dim): omega in Angstrom^2 (Pauli convention), E in eV relative to the
        DFT Fermi level.
        """
        axis = {"x": 0, "y": 1, "z": 2}
        sop = self._spin_operators()[spin]
        ea, eb = np.eye(3)[axis[alpha]] * delta, np.eye(3)[axis[beta]] * delta
        k_cart = np.asarray(k_cart, dtype=float).reshape(-1, 3)
        D = self.total_dim
        bands = list(range(D))
        omega, energies = np.empty((len(k_cart), D)), np.empty((len(k_cart), D))
        for s in range(0, len(k_cart), chunk):
            kc = k_cart[s:s + chunk]
            ham = self._tb_orthogonal_hamiltonians
            dHa = (ham(kc + ea) - ham(kc - ea)) / (2.0 * delta)
            dHb = dHa if alpha == beta else (ham(kc + eb) - ham(kc - eb)) / (2.0 * delta)
            J = 0.5 * (sop @ dHa + dHa @ sop)
            omega[s:s + chunk], energies[s:s + chunk] = self._berry_curvature_kubo(
                ham(kc), J, dHb, bands, degen_thresh, return_energies=True)
        return omega, energies

    @staticmethod
    def _fermi_occupation(E, efermi, kT):
        """Fermi-Dirac occupation f((E - efermi)/kT); kT <= 0 gives the step function."""
        if kT > 0:
            return 0.5 * (1.0 - np.tanh(0.5 * (np.asarray(E) - efermi) / kT))
        return (np.asarray(E) < efermi).astype(float)

    def plot_spin_berry_curvature_sk(self, spin="z", alpha="x", beta="y", dk=0.01, efermi=0.0, smearing=0.0,
                                     pair_degenerate="auto", units="au", band_offset=None, ylim=None,
                                     save_fig=None, save_data=None, show=True, delta=1e-4, degen_thresh=1e-4):
        """
        Compute and plot the spin Berry curvature Omega^spin_{alpha beta}(k) of the fitted SK Hamiltonian along the
        KPOINTS path: band by band for the fitted bands (stacked, like plot_berry_curvature_sk) and, in the top
        panel, the Fermi-sea sum that is integrated over the Brillouin zone to give the spin Hall conductivity
        (compare Fig. 2 of Guo et al., PRL 100, 096401, for Pt). Needs a spin-doubled model (SOC and/or magnetic).

            Omega^spin_{n,alpha beta}(k) = -2 Im sum_{m != n} <n|J_alpha|m> <m|dH/dk_beta|n> / (E_n - E_m)^2
            J_alpha = (1/2) {sigma_spin, dH/dk_alpha}          (spin current along alpha, field along beta)

        spin, alpha, beta: 'x' | 'y' | 'z'; the default is Omega^z_xy (spin z, current x, field y).
        dk: spacing of the path points (1/Angstrom); jumps in the KPOINTS path ('H|P') are honoured.
        Top panel: sum_n f(E_n(k) - E_F) Omega_n(k) over ALL model bands n (the exact SHC integrand, so it is what
        compute_spin_hall_conductivity averages over the BZ), and, dashed, the same restricted to the fitted bands.
        efermi (eV, relative to the DFT Fermi level) and smearing (Fermi-Dirac kT in eV; 0 = step function) set f.
        Lower panel: the fitted bands, the band with the lowest energy at the bottom, each on its own baseline
        band_offset apart (default: adapts to the data, 2.2x the 99th percentile of |Omega|; curves are clipped to
        +-0.48*band_offset so that spikes at (near-)degeneracies do not swamp the other bands).

        pair_degenerate: in a material with inversion and time-reversal symmetry (e.g. Pt) every band is doubly
        degenerate (Kramers). The curvature of an individual member of a degenerate pair is gauge dependent - only
        the sum over the pair is meaningful - so the two are added and shown as one band ('0-1', '2-3', ...).
        'auto' (default) does this when the fitted window consists of complete pairs (even number of bands starting
        at an even index) that are degenerate (< degen_thresh) at every path point; True forces it (error if the
        window cannot be paired); False shows single states (right for magnetic or non-centrosymmetric materials,
        where nothing is degenerate along the path).

        units: 'au' (default) = atomic units with the spin current in units of hbar/2, i.e. Omega_au =
        (1/2) Omega_A2 / a0^2, so that sigma = (e^2/hbar) (1/V) sum_k f Omega_au with V in bohr^3 (checked numerically
        against compute_spin_hall_conductivity); this is the natural reading of the 'atomic units' of Fig. 2 in the
        literature, but that paper does not spell out its normalisation, so compare shapes and orders of magnitude
        rather than the last factor. 'A2' = Angstrom^2 with Pauli matrices, the convention of
        compute_spin_hall_conductivity.

        Returns (fig, axes, result). result: 'x', 'ticks', 'slices', 'omega' (n_k, n_plotted: raw, without offsets,
        in `units`), 'groups' (band indices behind each column), 'labels', 'total' (all bands), 'total_fitted',
        'energies' (fitted bands, eV), 'paired', 'units', 'spin', 'alpha', 'beta', 'efermi', 'smearing'.
        """
        if getattr(self, "opt_params", None) is None:
            raise ValueError("No fitted parameters found. Run fit() or load_optimised_params() first.")
        spin, alpha, beta = str(spin).lower(), str(alpha).lower(), str(beta).lower()
        for name, val in (("spin", spin), ("alpha", alpha), ("beta", beta)):
            if val not in ("x", "y", "z"):
                raise ValueError(f"{name} must be 'x', 'y' or 'z'.")
        units = str(units).lower()
        if units not in ("au", "a2"):
            raise ValueError("units must be 'au' or 'A2'.")
        self._spin_operators()                                   # raises for a model without spin
        idx = [int(i) for i in self.target_tb_band_indices]
        kp = self._dense_kpath(dk)
        omega_all, E_all = self._spin_berry_curvature_sk(kp["cart"], spin, alpha, beta, delta, degen_thresh)
        conv = 0.5 / self._BOHR_ANGSTROM ** 2 if units == "au" else 1.0
        occ = self._fermi_occupation(E_all, float(efermi), float(smearing))
        total = conv * np.sum(occ * omega_all, axis=1)
        total_fit = conv * np.sum(occ[:, idx] * omega_all[:, idx], axis=1)
        om_fit, E_fit = conv * omega_all[:, idx], E_all[:, idx]

        can_pair = len(idx) % 2 == 0 and idx[0] % 2 == 0 and idx == list(range(idx[0], idx[-1] + 1))
        gap = float(np.abs(E_fit[:, 0::2] - E_fit[:, 1::2]).max()) if can_pair else float("inf")
        if isinstance(pair_degenerate, str):
            if pair_degenerate.lower() != "auto":
                raise ValueError("pair_degenerate must be 'auto', True or False.")
            paired = can_pair and gap < degen_thresh
        elif pair_degenerate:
            if not can_pair:
                raise ValueError("pair_degenerate=True needs an even number of consecutive fitted bands starting at an "
                                 f"even band index; the fitted bands are {idx}.")
            if gap >= degen_thresh:
                print(f"[WARN] The fitted bands are not degenerate in pairs along the path (largest splitting "
                      f"{gap:.2g} eV); the pair sums are still shown.")
            paired = True
        else:
            paired = False
        if paired:
            omega = om_fit[:, 0::2] + om_fit[:, 1::2]
            groups = [(idx[2 * j], idx[2 * j + 1]) for j in range(len(idx) // 2)]
            labels = [f"{a}-{b}" for a, b in groups]
        else:
            omega = om_fit
            groups = [(i,) for i in idx]
            labels = [str(i) for i in idx]
        res = {"mode": "path", "x": kp["x"], "ticks": kp["ticks"], "slices": [p[0] for p in kp["pieces"]],
               "omega": omega, "groups": groups, "labels": labels, "total": total, "total_fitted": total_fit,
               "energies": E_fit, "paired": paired, "units": "a.u." if units == "au" else "A^2", "spin": spin,
               "alpha": alpha, "beta": beta, "efermi": float(efermi), "smearing": float(smearing), "bands": idx}
        print(f"[INFO] SK spin Berry curvature Omega^{spin}_{alpha}{beta}: {len(omega)} path points, fitted bands "
              f"{idx[0]}-{idx[-1]} of {self.total_dim}, "
              + ("Kramers-degenerate pairs summed" if paired else "single states")
              + f"; E_F = {efermi:g} eV, kT = {smearing:g} eV; total in [{total.min():.4g}, {total.max():.4g}] {res['units']}")
        fig, axs = self._plot_spin_berry_curvature(res, band_offset, ylim, save_fig, save_data, show)
        return fig, axs, res

    def _plot_spin_berry_curvature(self, res, band_offset, ylim, save_fig, save_data, show):
        x, om = res["x"], res["omega"]
        nb = om.shape[1]
        spin, alpha, beta = res["spin"], res["alpha"], res["beta"]
        comp = rf"$\Omega^{{{spin}}}_{{{alpha}{beta}}}$"
        unit = "a.u." if res["units"] == "a.u." else r"Å$^2$"
        colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, max(nb, 2)))[:nb]
        noise = 1e-3 * (0.5 / self._BOHR_ANGSTROM ** 2 if res["units"] == "a.u." else 1.0)   # finite-difference noise level
        negligible = bool(self._robust_limit(om, 98.0) < noise)
        res["negligible"] = negligible
        if band_offset is None:                            # adapt to the data; a few huge spikes must not flatten the rest
            peak, robust = max(float(np.nanmax(np.abs(om))), noise), max(self._robust_limit(om, 98.0), noise)
            band_offset = 2.2 * (peak if peak <= 1.5 * robust else robust)
        band_offset = float(band_offset)
        half = 0.48 * band_offset
        clipped = bool(np.any(np.abs(om) > half))
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, min(14.0, 5.5 + 0.5 * nb)), constrained_layout=True,
                                gridspec_kw={"height_ratios": [1.0, max(1.6, 0.32 * nb)]})
        same = bool(np.allclose(res["total"], res["total_fitted"], rtol=1e-6, atol=1e-9 * max(1.0, np.abs(res["total"]).max())))
        fermi_txt = (f"E$_F$ = {res['efermi']:g} eV, " + (f"kT = {res['smearing']:g} eV" if res["smearing"] > 0 else "T = 0"))
        for sl in res["slices"]:                           # each continuous piece on its own: nothing across a jump
            axs[0].plot(x[sl], res["total"][sl], color="black", linewidth=1.8)
            if not same:
                axs[0].plot(x[sl], res["total_fitted"][sl], color="#eb6834", linestyle="--", linewidth=1.5)
            for j in range(nb):
                axs[1].plot(x[sl], np.clip(om[sl, j], -half, half) + j * band_offset, color=colours[j], linewidth=1.3,
                            zorder=2)
        for j in range(nb):
            axs[1].axhline(j * band_offset, color=colours[j], linewidth=0.8, alpha=0.55, zorder=1)
        axs[1].set_ylim(-0.5 * band_offset, (nb - 0.5) * band_offset)
        ystep = max(1, int(np.ceil(nb / 24)))
        axs[1].set_yticks([j * band_offset for j in range(0, nb, ystep)])
        axs[1].set_yticklabels([res["labels"][j] for j in range(0, nb, ystep)])
        if ylim is None:                                   # robust to isolated spikes, asymmetric so that nothing is wasted
            both = np.concatenate([res["total"], res["total_fitted"]])
            lo, hi = min(0.0, float(np.nanpercentile(both, 0.25))), max(0.0, float(np.nanpercentile(both, 99.75)))
            floor = max(0.05 * self._robust_limit(om), noise)   # never zoom into numerical noise when the bands cancel
            hi, lo = max(hi, floor), min(lo, -floor)
            pad = 0.08 * (hi - lo)
            ylim = (lo - pad, hi + pad)
        axs[0].set_ylim(*ylim)
        for t in res["ticks"][0]:
            for ax in axs:
                ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        axs[0].axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        axs[0].grid(True, linestyle=":", alpha=0.6)
        axs[1].grid(False)
        axs[1].set_xticks(res["ticks"][0])
        axs[1].set_xticklabels(res["ticks"][1], fontsize=11)
        axs[1].set_xlim(0.0, x[-1])
        axs[0].set_ylabel(f"{comp}(k) ({unit})")
        axs[1].set_ylabel(("band pair" if res["paired"] else "band") + " (model index)")
        handles = [plt.Line2D([0], [0], color="black", linewidth=1.8, label=f"sum f$\\cdot$Ω, all model bands ({fermi_txt})")]
        if not same:
            handles.append(plt.Line2D([0], [0], color="#eb6834", linestyle="--", linewidth=1.5, label="sum, fitted bands only"))
        axs[0].legend(handles=handles, loc="best", fontsize=9)
        note = (f"baselines {band_offset:.3g} {unit} apart" + (f"; curves clipped at ±{half:.3g}" if clipped else "")
                + ("; Kramers-degenerate pairs summed" if res["paired"] else ""))
        if negligible:
            note = (f"|Ω| < {noise:.1g} {unit} for (almost) all k: the spin Berry curvature vanishes here "
                    f"(no spin-orbit coupling, or symmetry); what is left is numerical noise")
            print(f"[INFO] The spin Berry curvature of the fitted bands is below {noise:.1g} {res['units']}: it vanishes "
                  f"within numerical noise (a model without spin-orbit coupling, or a symmetry-forbidden component).")
        axs[1].text(0.5, 1.01, note, transform=axs[1].transAxes, ha="center", va="bottom", fontsize=8, color="dimgray")
        axs[0].set_title(f"Spin Berry curvature {comp} from the SK Hamiltonian", pad=8)

        if save_fig:
            dirname = os.path.dirname(save_fig)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_fig, dpi=300, bbox_inches="tight")
            print(f"[INFO] Spin Berry curvature plot saved to {save_fig}")
        if save_data:
            dirname = os.path.dirname(save_data)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            cols = [x, res["total"], res["total_fitted"]] + [om[:, j] for j in range(nb)]
            head = ("k_distance(1/A) total_all_bands total_fitted_bands "
                    + " ".join(f"band{lab}" for lab in res["labels"]))
            unit_txt = "a.u." if res["units"] == "a.u." else "A^2"
            np.savetxt(save_data, np.column_stack(cols),
                       header=(f"SK_Hamiltonian; spin_Berry_curvature_Omega^{spin}_{alpha}{beta}_in_{unit_txt}; "
                               f"E_F={res['efermi']:g}_eV_kT={res['smearing']:g}_eV; {head}"), fmt="%.8e", encoding="utf-8")
            print(f"[INFO] Spin Berry curvature data saved to {save_data}")
        if show:
            plt.show()
        return fig, list(axs)

    # ------------------------------------------------------------------
    # Fermi-sea Hall responses: anomalous and spin Hall conductivity
    # ------------------------------------------------------------------
    _E2_OVER_HBAR = 2.434134807e-4          # e^2 / hbar in siemens

    @staticmethod
    def _fermi_sea_sum(ham, recip, nk, alpha, beta, efermi, smearing, spin_op=None, chunk=256, delta=1e-4,
                       degen_thresh=1e-4, refine=True, refine_threshold=1.0, refine_levels=3, refine_factor=4,
                       bands=None, shift=None):
        """
        acc(E_F) = sum_k w_k sum_n f(E_n - E_F) Omega_n(k) on a Gamma-centred nk1 x nk2 x nk3 mesh (w_k = 1), with
        Omega_n = -2 Im sum_{m != n} <n|A|m> <m|dH/dk_beta|n> / (E_n - E_m)^2 over ALL bands m, where A = dH/dk_alpha
        (Berry curvature) or A = 1/2 {spin_op, dH/dk_alpha} (spin Berry curvature). Fermi-Dirac occupation with kT =
        smearing (0: step function). ham(k_cart) -> (n, D, D) Hamiltonians, recip = reciprocal lattice vectors (rows).
        smearing may also be an array with one kT per entry of `efermi`: column j of the result is then evaluated at
        (efermi[j], smearing[j]), all from the same k-points (this is how a temperature scan costs one pass).
        bands: indices (into the ascending spectrum) of the bands n that enter the sum; None = all bands. The sum
        over the intermediate states m always runs over all bands. shift: offset of the whole mesh in units of one
        mesh cell along each axis (default 0 = Gamma-centred); a shifted mesh is an equally valid uniform sample.

        Adaptive refinement: near a band (near-)crossing Omega ~ 1/L^2 varies on the k-length
        L, so a mesh point whose cell (linear size h, 1/Angstrom) has  max_n |Omega_n| h^2 > refine_threshold  is
        replaced by a refine_factor^d sub-mesh over its own cell (d = number of axes with nk > 1), the weights split
        equally, and so on for up to refine_levels levels. This resolves the sharp hot spots that a uniform mesh
        misses, at a small fraction of the cost of a uniformly dense mesh. Returns (acc over `efermi`, sum of
        weights = n_k, list of the number of points refined per level).
        """
        grid = np.stack(np.meshgrid(*[np.arange(n) / n for n in nk], indexing="ij"), axis=-1).reshape(-1, 3)
        if shift is not None:
            grid = grid + np.asarray(shift, dtype=float)[None, :] / np.array(nk, dtype=float)[None, :]
        n_k = len(grid)
        efermi = np.asarray(efermi, dtype=float)
        kT = np.asarray(smearing, dtype=float)
        kT = np.full(len(efermi), float(kT)) if kT.ndim == 0 else kT
        if kT.shape != efermi.shape:
            raise ValueError("smearing must be a scalar or have one entry per Fermi energy.")
        thermal = kT > 0.0
        ea, eb = _cart_direction(alpha) * delta, _cart_direction(beta) * delta
        same_axis = bool(np.allclose(ea, eb))
        active = np.array([n > 1 for n in nk])
        r = int(refine_factor)
        sub = [((np.arange(r) + 0.5) / r - 0.5) if act else np.zeros(1) for act in active]
        sub_off = np.stack(np.meshgrid(*sub, indexing="ij"), axis=-1).reshape(-1, 3)      # child offsets in units of the parent cell
        n_child = len(sub_off)
        acc = np.zeros(len(efermi))
        info = []
        sel = None if bands is None else [int(b) for b in bands]
        rwin = float(getattr(SKFitter, "refine_window", 0.0) or 0.0)
        kfrac, weight = grid, np.ones(n_k)
        cell = 1.0 / np.array(nk, dtype=float)                                            # reduced-coordinate cell size per axis
        next_report = 0.25
        for level in range(int(refine_levels) + 1):
            flagged_k, flagged_w = [], []
            can_refine = bool(refine) and level < int(refine_levels) and n_child > 1
            d_act = int(active.sum())
            if d_act:                                                                    # linear cell size h in 1/Angstrom
                rows = recip[active] * cell[active][:, None]
                h2 = np.sqrt(np.linalg.det(rows @ rows.T)) ** (2.0 / d_act)
            else:
                can_refine, h2 = False, 0.0
            m = len(kfrac)

            def block(s, kfrac=kfrac, weight=weight, h2=h2, can_refine=can_refine):
                """The contribution of the k-points s ... s+chunk: (acc part, flagged k, their weights)."""
                kf, w = kfrac[s:s + chunk], weight[s:s + chunk]
                kc = kf @ recip
                H0 = ham(kc)
                dHa = (ham(kc + ea) - ham(kc - ea)) / (2.0 * delta)
                dHb = dHa if same_axis else (ham(kc + eb) - ham(kc - eb)) / (2.0 * delta)
                A = dHa if spin_op is None else 0.5 * (spin_op @ dHa + dHa @ spin_op)
                bsel = sel if sel is not None else list(range(H0.shape[1]))
                omega, E = SKFitter._berry_curvature_kubo(H0, A, dHb, bsel, degen_thresh, return_energies=True)
                E = E[:, bsel]
                if can_refine:
                    om = np.abs(omega)
                    if rwin > 0:                           # only bands near the scanned Fermi energies decide
                        om = np.where((E >= efermi.min() - rwin) & (E <= efermi.max() + rwin), om, 0.0)
                    flag = om.max(axis=1) * h2 > refine_threshold
                else:
                    flag = np.zeros(len(kf), bool)
                keep = ~flag
                part = np.zeros(len(efermi))
                if keep.any():
                    x = E[keep].reshape(-1, 1) - efermi[None, :]
                    occ = (x < 0.0).astype(float)
                    if thermal.any():
                        occ[:, thermal] = 0.5 * (1.0 - np.tanh(x[:, thermal] / (2.0 * kT[thermal])[None, :]))
                    part = (omega[keep] * w[keep, None]).reshape(-1) @ occ
                return part, kf[flag], w[flag]

            starts = list(range(0, m, chunk))
            workers = max(1, int(getattr(SKFitter, "k_workers", 1) or 1))
            report = getattr(SKFitter, "k_progress", None)
            if callable(report):
                report(level, 0, m)
            pool = None
            if workers > 1 and len(starts) > 1:            # numpy releases the GIL in eigh and the matrix products
                from concurrent.futures import ThreadPoolExecutor
                first = block(starts[0])                   # prepares the fast model once, before the threads
                pool = ThreadPoolExecutor(workers)
                results = itertools.chain([first], pool.map(block, starts[1:]))     # in order, as they finish
            else:
                results = (block(s0) for s0 in starts)
            for i, (part, fk_, fw_) in enumerate(results):
                acc += part
                if len(fk_):
                    flagged_k.append(fk_); flagged_w.append(fw_)
                done_frac = min(1.0, (starts[i] + chunk) / m)
                if callable(report):
                    report(level, min(m, starts[i] + chunk), m)
                if level == 0 and done_frac >= next_report and done_frac < 1.0:
                    print(f"[INFO]   {int(round(100 * done_frac))}% of the base mesh done")
                    next_report += 0.25
            if pool is not None:
                pool.shutdown()
            if not flagged_k:
                break
            fk, fw = np.concatenate(flagged_k), np.concatenate(flagged_w)
            info.append(len(fk))
            print(f"[INFO]   level {level}: refining {len(fk)} of {m} points ({100.0 * len(fk) / m:.1f}%)")
            kfrac = (fk[:, None, :] + sub_off[None, :, :] * cell[None, None, :]).reshape(-1, 3)
            weight = np.repeat(fw / n_child, n_child)
            cell = cell / np.where(active, r, 1)
        return acc, n_k, info

    def _plot_conductivity(self, efermi, sigma, sigma_ef, label, ylabel, title, color, save_fig, save_data,
                           data_header, show):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(efermi, sigma, color=color, lw=2, label=label)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label=r"$E_F$ (DFT)")
        ax.plot([0.0], [sigma_ef], "o", color="black", zorder=5, label=f"{label}($E_F$) = {sigma_ef:.0f}")
        ax.set_xlabel(r"Fermi energy $E_F$ (eV)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(efermi[0], efermi[-1])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        if save_fig:
            dirname = os.path.dirname(save_fig)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_fig, dpi=300, bbox_inches="tight")
            print(f"[INFO] Plot saved to {save_fig}")
        if save_data:
            dirname = os.path.dirname(save_data)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            np.savetxt(save_data, np.column_stack([efermi, sigma]), fmt="%.8e", header=data_header, encoding="utf-8")
            print(f"[INFO] Data saved to {save_data}")
        if show:
            plt.show()
        return fig, ax

    def _hall_setup(self, nk, efermi_range, num_efermi):
        if getattr(self, "opt_params", None) is None:
            raise ValueError("No fitted parameters found. Run fit() or load_optimised_params() first.")
        nk = (int(nk),) * 3 if np.isscalar(nk) else tuple(int(n) for n in nk)
        if len(nk) != 3 or min(nk) < 1:
            raise ValueError("nk must be an int or three positive ints.")
        efermi = np.union1d(np.linspace(efermi_range[0], efermi_range[1], int(num_efermi)), [0.0])
        return nk, efermi

    def _spin_operators(self):
        """Pauli matrices sigma_x, sigma_y, sigma_z in the model's basis: the first N orbitals are spin up, the
        next N spin down (the layout of build_hamiltonian), so sigma = pauli (x) 1_orbital."""
        N, D = self.base_dim, self.total_dim
        if D != 2 * N:
            raise ValueError("The spin Hall conductivity and the spin Berry curvature need a spin-doubled model (include_soc=True or magnetic "
                             "orbitals); this model has no spin degree of freedom.")
        I, Z = np.eye(N), np.zeros((N, N))
        return {"x": np.block([[Z, I], [I, Z]]).astype(complex),
                "y": np.block([[Z, -1j * I], [1j * I, Z]]),
                "z": np.block([[I, Z], [Z, -I]]).astype(complex)}

    def _resolve_bands(self, bands):
        """None -> None (all model bands); 'fitted' -> the bands used in the fit; else a list of band indices."""
        if bands is None:
            return None
        if isinstance(bands, str):
            if bands.lower() != "fitted":
                raise ValueError("bands must be None (all bands), 'fitted' or a list of band indices.")
            return [int(i) for i in self.target_tb_band_indices]
        idx = sorted({int(i) for i in np.atleast_1d(bands)})
        if not idx or idx[0] < 0 or idx[-1] >= self.total_dim:
            raise ValueError(f"bands needs at least one band index, all within 0..{self.total_dim - 1}.")
        return idx

    def _anomalous_hall(self, ham, source, nk, component, efermi_range, num_efermi, smearing, chunk, delta,
                        degen_thresh, refine, refine_threshold, refine_levels, bands, save_fig, save_data, show,
                        color):
        """Shared driver of the AHC: ham(k_cart) -> (n, D, D) is the Hamiltonian in the tau gauge."""
        axes_of = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}
        component = str(component).lower()
        if component not in axes_of:
            raise ValueError("component must be 'x', 'y' or 'z'.")
        alpha, beta = axes_of[component]
        nk, efermi = self._hall_setup(nk, efermi_range, num_efermi)
        sel = self._resolve_bands(bands)
        volume = abs(np.linalg.det(self.structure.lattice.matrix))
        name = {"x": "yz", "y": "zx", "z": "xy"}[component]
        which = "all bands" if sel is None else f"bands {sel[0]}-{sel[-1]} of {self.total_dim}" \
            if sel == list(range(sel[0], sel[-1] + 1)) else f"bands {sel} of {self.total_dim}"
        print(f"[INFO] Anomalous Hall conductivity sigma_{name} from {source}: {nk[0]}x{nk[1]}x{nk[2]} k-mesh "
              f"({nk[0] * nk[1] * nk[2]} points), {which}, kT = {smearing} eV")
        acc, n_k, _ = self._fermi_sea_sum(ham, self.structure.lattice.reciprocal_lattice.matrix,
                                          nk, alpha, beta, efermi, smearing, None, chunk, delta, degen_thresh,
                                          refine, refine_threshold, refine_levels, bands=sel)
        # -(e^2/hbar)[S] * 1e8 [1/Angstrom -> 1/cm] / V[Angstrom^3] * (1/N_k) sum f Omega[Angstrom^2]
        sigma = -self._E2_OVER_HBAR * 1e8 / volume * acc / n_k
        sigma_ef = float(sigma[np.argmin(np.abs(efermi))])
        print(f"[INFO] sigma_{name}(E_F = 0) = {sigma_ef:.1f} S/cm  ({source}, {which})")
        res = {"efermi": efermi, "sigma": sigma, "sigma_ef": sigma_ef, "component": component, "nk": nk,
               "smearing": smearing, "units": "S/cm", "bands": sel, "source": source}
        label = rf"$\sigma_{{{name}}}$"
        sub = "" if sel is None else ", fitted bands only" if sel == [int(i) for i in self.target_tb_band_indices] \
            else ", selected bands"
        fig, ax = self._plot_conductivity(
            efermi, sigma, sigma_ef, label, rf"Anomalous Hall conductivity {label} (S/cm)",
            f"Anomalous Hall conductivity from {source} ({nk[0]}x{nk[1]}x{nk[2]} k-mesh, kT = {smearing} eV{sub})",
            color, save_fig, save_data,
            f"E_Fermi(eV)  sigma_{name} (S/cm); nk={nk}, kT={smearing} eV, {source}, {which}", show)
        return fig, ax, res

    def compute_anomalous_hall_conductivity(self, nk=(40, 40, 40), component="z", efermi_range=(-2.0, 2.0),
                                            num_efermi=200, smearing=0.02, chunk=256, delta=1e-4,
                                            degen_thresh=1e-4, refine=True, refine_threshold=1.0, refine_levels=3,
                                            save_fig=None, save_data=None, show=True, bands=None):
        """
        Intrinsic anomalous Hall conductivity of the fitted SK Hamiltonian versus Fermi energy (Fermi-sea Kubo
        formula):

            O_delta(E_F) = -(e^2/hbar) (1/V N_k) sum_k sum_n f(E_n - E_F) Omega^delta_n(k)
            sigma_{alpha beta} = epsilon_{alpha beta delta} O_delta

        component 'z' gives sigma_xy (Omega_z = Omega_xy), 'x' gives sigma_yz, 'y' gives sigma_zx. Omega_n is the
        band-resolved Berry curvature of plot_berry_curvature_sk (sum over intermediate states over all model bands).
        The result is in S/cm (multiply by 100 for S/m).
        Energies are relative to the DFT Fermi level. A non-zero result needs broken time-reversal symmetry
        (magnetic model) together with SOC (include_soc=True); without SOC it vanishes. V is the cell volume, so a 2D
        material (vacuum in the cell) needs multiplying by the cell height to get a sheet conductivity.

        bands: which bands n enter the sum over occupied states: None (default) = every band of the model, 'fitted'
        = only the bands used in the fit (self.target_tb_band_indices; bands below the fitted window, which the fit
        never saw, are then left out), or an explicit list of band indices. The intermediate states always run
        over all model bands.

        The k-sum runs over a Gamma-centred nk1 x nk2 x nk3 mesh of the whole BZ (no symmetry used) with Fermi-Dirac
        smearing kT = `smearing` (eV; 0 = zero temperature). The Berry curvature is sharply peaked at band
        (near-)crossings, so converge nk (e.g. 40 -> 80 -> 120): the AHC of a metal such as Fe needs a dense mesh.
        Non-orthogonal models use the Loewdin-orthogonalised H.

        Returns (fig, ax, result) with result['efermi'], result['sigma'] (arrays over the E_F grid, which always
        contains E_F = 0), result['sigma_ef'] (value at E_F = 0) and the settings.
        """
        return self._anomalous_hall(self._tb_orthogonal_hamiltonians, "the SK Hamiltonian", nk, component,
                                    efermi_range, num_efermi, smearing, chunk, delta, degen_thresh, refine,
                                    refine_threshold, refine_levels, bands, save_fig, save_data, show, "tab:red")




    def compute_spin_hall_conductivity(self, nk=(40, 40, 40), spin="z", alpha="x", beta="y",
                                       efermi_range=(-2.0, 2.0), num_efermi=200, smearing=0.02,
                                       chunk=256, delta=1e-4, degen_thresh=1e-4, refine=True,
                                       refine_threshold=1.0, refine_levels=3,
                                       save_fig=None, save_data=None, show=True):
        """
        Intrinsic spin Hall conductivity sigma^spin_{alpha beta}(E_F) of the fitted SK Hamiltonian from the Kubo
        formula (Guo et al., PRL 100, 096401; Qiao et al., PRB 98, 214402):

            sigma = +(e^2/hbar) (1/2) (1/V N_k) sum_k sum_n f(E_n - E_F) Omega^spin_n(k)
            Omega^spin_n = -2 Im sum_{m != n} <n| J_alpha |m> <m| dH/dk_beta |n> / (E_n - E_m)^2,
            J_alpha = (1/2) {sigma_spin, dH/dk_alpha}     (sigma = Pauli matrix; spin current (hbar/2) J)

        (The sign is opposite to the anomalous Hall formula because the spin current carries no charge -e. As a
        check, a Rashba 2D electron gas gives the universal -e/8pi of Sinova et al.)

        i.e. spin current flowing along `alpha`, polarised along `spin`, in response to a field along `beta`; the
        default is sigma^z_xy. Each of spin, alpha and beta is 'x', 'y', 'z' or a Cartesian vector (e.g. [1, 1, 1]
        and [1, -1, 0] for the frame of a (111) kagome plane). The result is in (hbar/e)*S/cm, the usual unit (as in the
        literature; Pt is ~2000). Multiply by 2 for units of hbar/2e. The energy axis is relative to the DFT Fermi
        level, like all TB energies here. V is the cell volume, so for a 2D material (vacuum in the cell) multiply by
        the cell height to get a sheet conductivity.

        The k-sum runs over a Gamma-centred nk1 x nk2 x nk3 mesh of the whole BZ (no symmetry used) with Fermi-Dirac
        smearing kT = `smearing` (eV; 0 = zero temperature). The sum over intermediate states m runs over all model
        bands. Non-orthogonal models use the Loewdin-orthogonalised H (sigma commutes
        with S^-1/2, so the spin operator is unchanged). Needs a spin-doubled model (SOC and/or magnetic).

        The integrand is sharply peaked at band (near-)crossings, so converge nk (e.g. 40 -> 80 -> 120) and treat the
        smearing as a numerical parameter, especially for metals such as Pt.

        Returns (fig, ax, result) with result['efermi'], result['sigma'] (arrays over the E_F grid, which always
        contains E_F = 0), result['sigma_ef'] (value at E_F = 0) and the settings.
        """
        sv = _cart_direction(spin)
        nk, efermi = self._hall_setup(nk, efermi_range, num_efermi)
        ops = self._spin_operators()
        sigma_op = sv[0] * ops["x"] + sv[1] * ops["y"] + sv[2] * ops["z"]
        alpha_v, beta_v = _cart_direction(alpha), _cart_direction(beta)
        spin, alpha, beta = direction_label(spin), direction_label(alpha), direction_label(beta)
        volume = abs(np.linalg.det(self.structure.lattice.matrix))
        print(f"[INFO] Spin Hall conductivity sigma^{spin}_{alpha}{beta}: {nk[0]}x{nk[1]}x{nk[2]} k-mesh "
              f"({nk[0] * nk[1] * nk[2]} points), {self.total_dim} bands, kT = {smearing} eV")
        acc, n_k, _ = self._fermi_sea_sum(self._tb_orthogonal_hamiltonians, self.structure.lattice.reciprocal_lattice.matrix,
                                       nk, alpha_v, beta_v, efermi, smearing, sigma_op, chunk, delta, degen_thresh,
                                       refine, refine_threshold, refine_levels)
        # (e^2/hbar)[S] * (1/2 for hbar/e units) * 1e8 [1/Angstrom -> 1/cm] / V[Angstrom^3]
        sigma = 0.5 * self._E2_OVER_HBAR * 1e8 / volume * acc / n_k
        sigma_ef = float(sigma[np.argmin(np.abs(efermi))])
        print(f"[INFO] sigma^{spin}_{alpha}{beta}(E_F = 0) = {sigma_ef:.1f} (hbar/e) S/cm")
        res = {"efermi": efermi, "sigma": sigma, "sigma_ef": sigma_ef, "spin": spin, "alpha": alpha, "beta": beta,
               "nk": nk, "smearing": smearing, "units": "(hbar/e) S/cm"}
        label = rf"$\sigma^{{{spin}}}_{{{alpha}{beta}}}$"
        fig, ax = self._plot_conductivity(
            efermi, sigma, sigma_ef, label, rf"Spin Hall conductivity {label} ($\hbar/e \cdot$ S/cm)",
            f"Spin Hall conductivity from the SK Hamiltonian ({nk[0]}x{nk[1]}x{nk[2]} k-mesh, kT = {smearing} eV)",
            "tab:blue", save_fig, save_data,
            f"E_Fermi(eV)  sigma^{spin}_{alpha}{beta} ((hbar/e) S/cm); nk={nk}, kT={smearing} eV, SK Hamiltonian", show)
        return fig, ax, res

    _KB_EV = 8.617333262e-5                 # Boltzmann constant in eV/K

    def compute_spin_hall_conductivity_vs_temperature(self, temperatures=None, nk=(40, 40, 40), spin="z",
                                                      alpha="x", beta="y", efermi=0.0, n_shifts=4, seed=0,
                                                      chunk=256, delta=1e-4, degen_thresh=1e-4, refine=True,
                                                      refine_threshold=1.0, refine_levels=3,
                                                      save_fig=None, save_data=None, show=True):
        """
        Intrinsic spin Hall conductivity sigma^spin_{alpha beta} at a fixed Fermi energy `efermi` (default 0 = the
        DFT Fermi level) as a function of temperature T, from the same Kubo formula as
        compute_spin_hall_conductivity (see there for the formula, sign and units, (hbar/e) S/cm) but with the
        Fermi-Dirac occupation f((E_n - efermi)/k_B T) of each temperature. `temperatures` is a list in kelvin;
        the default runs from very low temperature to room temperature (5, 10, 20, 40, 60, 80, 100, 150, 200, 250
        and 300 K). T = 0 gives the step function. All temperatures share the same k-points, so the cost is that of
        one conductivity at a single smearing, and the chemical potential is held at `efermi` (for a metal its shift
        with T is a few meV at 300 K, which changes sigma by much less than the numerical uncertainty).

        Resolving the temperature dependence needs care: k_B T (26 meV at 300 K, 0.4 meV at 5 K) is far smaller
        than the energy spacing between the neighbouring points of any practical k-mesh, so on a single mesh the
        change of sigma with T is buried in the mesh error (a T = 0 sum over a mesh is a noisy estimate of the
        integral). To make this visible the calculation is repeated on `n_shifts` meshes: the Gamma-centred mesh
        plus n_shifts - 1 randomly shifted ones (reproducible via `seed`), each with the adaptive refinement of
        compute_spin_hall_conductivity. The curve shows the mean over the meshes, the error bars the standard error
        of that mean (the individual meshes are drawn as faint dots): a temperature dependence smaller than the
        error bars is not resolved at this nk; raise nk and/or n_shifts to shrink them. The mesh error is larger at
        low T (where the step function is sharpest), so the lowest-T points are the least reliable.

        Returns (fig, ax, result): result['temperatures'] (K), result['sigma'] (mean over the meshes),
        result['sigma_err'] (standard error; NaN for n_shifts = 1), result['sigma_meshes'] (n_shifts, n_T),
        result['kT'] (eV), result['efermi'] and the settings.
        """
        axes = {"x": 0, "y": 1, "z": 2}
        spin, alpha, beta = str(spin).lower(), str(alpha).lower(), str(beta).lower()
        for name, val in (("spin", spin), ("alpha", alpha), ("beta", beta)):
            if val not in axes:
                raise ValueError(f"{name} must be 'x', 'y' or 'z'.")
        if temperatures is None:
            temperatures = (5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300)
        T = np.unique(np.atleast_1d(np.asarray(temperatures, dtype=float)))
        if T.size == 0 or T[0] < 0.0 or not np.all(np.isfinite(T)):
            raise ValueError("temperatures must be a non-empty list of temperatures in kelvin (>= 0).")
        n_shifts = int(n_shifts)
        if n_shifts < 1:
            raise ValueError("n_shifts must be at least 1.")
        efermi = float(efermi)
        nk, _ = self._hall_setup(nk, (0.0, 0.0), 1)
        sigma_op = self._spin_operators()[spin]
        volume = abs(np.linalg.det(self.structure.lattice.matrix))
        kT = self._KB_EV * T
        rng = np.random.RandomState(int(seed))
        active = np.array([n > 1 for n in nk], dtype=float)
        print(f"[INFO] Spin Hall conductivity sigma^{spin}_{alpha}{beta} vs temperature at E_F = {efermi:g} eV: "
              f"{T.size} temperatures ({T[0]:g}-{T[-1]:g} K), {nk[0]}x{nk[1]}x{nk[2]} k-mesh, {n_shifts} mesh(es), "
              f"{self.total_dim} bands")
        pref = 0.5 * self._E2_OVER_HBAR * 1e8 / volume
        recip = self.structure.lattice.reciprocal_lattice.matrix
        per_mesh = np.empty((n_shifts, T.size))
        for i in range(n_shifts):
            shift = np.zeros(3) if i == 0 else rng.uniform(0.0, 1.0, 3) * active
            print(f"[INFO] mesh {i + 1}/{n_shifts}" + ("" if i == 0 else f" (shifted by {np.round(shift, 3)} cells)"))
            acc, n_k, _ = self._fermi_sea_sum(self._tb_orthogonal_hamiltonians, recip, nk, alpha, beta,
                                              np.full(T.size, efermi), kT, sigma_op, chunk, delta, degen_thresh,
                                              refine, refine_threshold, refine_levels, shift=shift)
            per_mesh[i] = pref * acc / n_k
        sigma = per_mesh.mean(axis=0)
        err = per_mesh.std(axis=0, ddof=1) / np.sqrt(n_shifts) if n_shifts > 1 else np.full(T.size, np.nan)
        for j in sorted({0, T.size - 1}):
            e = "" if n_shifts == 1 else f" +- {err[j]:.1f}"
            print(f"[INFO] sigma^{spin}_{alpha}{beta}(E_F = {efermi:g} eV, T = {T[j]:g} K) = {sigma[j]:.1f}{e} (hbar/e) S/cm")
        res = {"temperatures": T, "kT": kT, "sigma": sigma, "sigma_err": err, "sigma_meshes": per_mesh,
               "efermi": efermi, "spin": spin, "alpha": alpha, "beta": beta, "nk": nk, "n_shifts": n_shifts,
               "units": "(hbar/e) S/cm"}

        label = rf"$\sigma^{{{spin}}}_{{{alpha}{beta}}}$"
        fig, ax = plt.subplots(figsize=(8, 5))
        if n_shifts > 1:
            ax.plot(np.tile(T, (n_shifts, 1)).ravel(), per_mesh.ravel(), ".", color="tab:blue", alpha=0.3, ms=5,
                    label=f"individual k-meshes ({n_shifts})")
            ax.errorbar(T, sigma, yerr=err, fmt="o-", color="tab:blue", lw=2, capsize=3, ms=5,
                        label=f"mean $\\pm$ standard error")
        else:
            ax.plot(T, sigma, "o-", color="tab:blue", lw=2, ms=5, label=label)
        lo, hi = float(np.nanmin(per_mesh)), float(np.nanmax(per_mesh))
        pad = max(0.0, 0.05 * max(abs(lo), abs(hi)) - 0.5 * (hi - lo)) + 0.05 * (hi - lo)   # never zoom in on noise
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.0, 1.03 * T[-1])
        ax.set_xlabel("Temperature $T$ (K)", fontsize=12)
        ax.set_ylabel(rf"Spin Hall conductivity {label} ($\hbar/e \cdot$ S/cm)", fontsize=12)
        ax.set_title(f"Spin Hall conductivity vs temperature at $E_F$ = {efermi:g} eV (SK Hamiltonian, "
                     f"{nk[0]}x{nk[1]}x{nk[2]} k-mesh)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        if save_fig:
            dirname = os.path.dirname(save_fig)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_fig, dpi=300, bbox_inches="tight")
            print(f"[INFO] Plot saved to {save_fig}")
        if save_data:
            dirname = os.path.dirname(save_data)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            np.savetxt(save_data, np.column_stack([T, kT, sigma, err, per_mesh.T]), fmt="%.8e",
                       header=f"T(K) kT(eV) sigma_mean sigma_stderr sigma_mesh1..{n_shifts}; sigma^{spin}_{alpha}{beta} "
                              f"((hbar/e) S/cm) at E_F = {efermi:g} eV; nk={nk}, SK Hamiltonian", encoding="utf-8")
            print(f"[INFO] Data saved to {save_data}")
        if show:
            plt.show()
        return fig, ax, res
