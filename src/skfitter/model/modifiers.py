"""Model-level modifiers (phase M2a): fields and gap changes applied on top of an unchanged base model.

    from skfitter.model import apply
    m = apply(model, [{"kind": "zeeman", "B": [0, 0, 10]}])                       # 10 T out of plane
    m = apply(model, [{"kind": "onsite_shift", "shift": 0.1, "sites": [0]},
                      {"kind": "onsite_shift", "shift": -0.1, "sites": [1]}])      # a sublattice mass (graphene)
    m = apply(model, [{"kind": "scissor", "delta": 0.5}])                          # conduction bands up by 0.5 eV
    m.save("graphene_B10_model.json")                                               # base + modifiers, re-applied on load

Modifiers, applied in order (each is a dict with ``kind``):

``spinful``        doubles the basis for spin (spin-up orbitals of a site first, then spin-down); needed by the
                   spin Zeeman term and added automatically before it.
``onsite_shift``   ``shift`` (eV) added to the on-site energy of the selected orbitals. Selection (all optional, all
                   must match): ``element``, ``sites`` (site indices of the base model's cell), ``orbitals``
                   (labels such as "pz", "dz2"), ``l`` (0-3), ``spin`` ("up"/"down"). ``sublattice_mass()`` writes
                   the pair of shifts that open a gap of ``delta`` between two sublattices.
``zeeman``         a magnetic field ``B`` (tesla, Cartesian [Bx, By, Bz]) on every site:
                   spin  (g mu_B / 2) B.sigma (``g`` default 2.0023, ``spin`` true/false), and
                   orbit mu_B L.B for orbitals with an atomic label (``orbital`` true/false; SK models; with the same
                   L matrices as SKFitter's spin-orbit term). Wannier functions have no L: the orbital part is left
                   out with a warning. Spinor Wannier functions use the model's spin matrices sigma (from the run's
                   .spn) when it has them - on each site and, between sites, as hoppings
                   (g mu_B / 2) B.sigma(i, j, R), exact on the run's mesh - else they are taken as pure spin up/down
                   (with a warning).
                   (The orbital effect of B on the motion between atoms - Peierls phases, Landau levels - is a
                   device-level effect and is not a modifier.)
``exchange``       an exchange field ``energy`` (eV) along ``direction`` (default [0, 0, 1]): + energy n.sigma on the
                   selected sites / orbitals (``element``, ``sites``, ``orbitals``; default all) - the energy scale of a
                   magnetic proximity effect or a magnetic adatom, far above what a Zeeman field in tesla gives
                   (10 T is ~1 meV). Makes the model spinful first. Same sign as ``zeeman`` with B along n.
``atomic_soc``     on-site (atomic) spin-orbit coupling lambda L.S = (lambda / 2) L.sigma on each site's p, d or f
                   shell - the term SKFitter fits as lambda_<El>_<family>, with the same L matrices - for a model made
                   without spin-orbit coupling (a fit or Wannier run of a calculation without SOC, e.g. a meta-GGA
                   that the DFT code cannot run non-collinear). ``lam`` (eV): {"Te_p": 0.49, "Bi": 1.2} - a key
                   "<El>_<family>" for one shell, or "<El>" for every p, d and f shell of the element. Wannier
                   functions carry no atomic label: give ``labels`` {"Te": ["pz", "px", "py"]} (the spatial functions
                   of each site of that element, in the model's order - Wannier90's order for a p projection) when
                   they are atom-centred atomic-like orbitals with the global axes (projections, little or no
                   maximal localisation); the result is then only as good as that assumption. Makes the model spinful
                   first; refused on a model that has spin-orbit coupling already.
``rashba``         a Rashba spin-orbit coupling ``lam`` (eV) on the nearest-neighbour bonds between the ``orbitals``
                   (default ["pz"]) of the ``element`` (optional): i lam (sigma x d)_z on each bond i -> j (d the unit
                   vector from i to j) - Kane and Mele's term for a perpendicular field or substrate (graphene with an
                   exchange field: a quantum anomalous Hall insulator, C = ±2; Qiao et al., PRB 82, 161414 (2010)).
                   Makes the model spinful first; breaks the conservation of S_z.
``efield``         a uniform electric field ``E`` (V/Å, Cartesian) across a direction without hoppings (a layer's
                   vacuum, across a wire): + E.(r - origin) (``origin`` default: the mean orbital centre) - on the
                   orbital centres and, for Wannier models with position matrices r(R) (the run's _tb.dat, W4d),
                   between the functions too (the interband Stark terms, as hoppings E.r(i, j, R)). A fixed field:
                   no screening (it overestimates the response). Along a periodic direction it is refused.
``scissor``        conduction states (above mid-gap; the valence bands are the ``n_valence`` lowest, by default
                   those below E_F = 0, or give ``midgap`` in eV) shifted by ``delta`` (eV): H'(k) = H(k) + delta S P_c(k) S on a
                   uniform ``mesh`` (default 12 per periodic direction - it holds the points 1/2, 1/3, 1/4 and 1/6, e.g.
                   H of a hexagonal cell; 18 for a 2D material), transformed back to real space with minimal-image
                   hoppings. Exact on the mesh; the largest error between mesh points is measured and recorded (a
                   warning above 5 meV: raise the mesh). Needs a gap at every k. With ``n_valence`` the conduction
                   states are the bands above the n-th at each k: only a direct gap is needed, so a DFT semimetal
                   (bands overlapping across the zone, e.g. PBE Te) is opened too. The result is a model of numbers (an
                   SK model loses its distance scaling).

The modified model offers the common interface, so the properties use it as any model. Its
file (``prayog-modified-model`` version 1) holds the base model and the modifiers; reading it applies them again.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from .model import TBModel, _vector_matrices_of
from .orbitals import L_matrices, strip_spin

FORMAT = "prayog-modified-model"
VERSION = 1
MU_B = 5.7883818060e-5          # Bohr magneton, eV/T (CODATA 2018)
G_ELECTRON = 2.00231930436      # free-electron g-factor
KINDS = ("spinful", "onsite_shift", "zeeman", "exchange", "rashba", "scissor", "efield", "atomic_soc")
SOC_FAMILIES = ("p", "d", "f")
SCISSOR_WARN_MEV = 5.0
SCISSOR_MAX_BYTES = 1.5e9       # memory of H(k) on the mesh and of the aliased H(R) together
E_F_TOL = 1e-3                  # eV: a state up to this far above E_F = 0 counts as occupied (E_F at the VBM)

_SIGMA = (np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]],
                                                                                             dtype=complex))


# == checks =======================================================================================================
def check_modifier(mod: dict) -> dict:
    """A modifier with its defaults filled in; ValueError for anything malformed."""
    if not isinstance(mod, dict) or mod.get("kind") not in KINDS:
        raise ValueError(f"A modifier is a dict with kind one of {', '.join(KINDS)}; got {mod!r}.")
    m = copy.deepcopy(mod)
    k = m["kind"]
    allowed = {"spinful": {"auto"},
               "onsite_shift": {"shift", "element", "sites", "orbitals", "l", "spin"},
               "zeeman": {"B", "g", "spin", "orbital"},
               "exchange": {"energy", "direction", "element", "sites", "orbitals"},
               "rashba": {"lam", "orbitals", "element"},
               "scissor": {"delta", "mesh", "midgap", "n_valence"},
               "efield": {"E", "origin"},
               "atomic_soc": {"lam", "labels"}}[k] | {"kind", "label"}
    extra = set(m) - allowed
    if extra:
        raise ValueError(f"{k}: unknown field(s) {', '.join(sorted(extra))}.")
    if k == "onsite_shift":
        m["shift"] = float(m.get("shift", 0.0))
        if m.get("sites") is not None:
            m["sites"] = [int(x) for x in m["sites"]]
        if m.get("orbitals") is not None:
            m["orbitals"] = [str(x) for x in m["orbitals"]]
        if m.get("l") is not None:
            m["l"] = int(m["l"])
        if m.get("spin") not in (None, "up", "down"):
            raise ValueError("onsite_shift: spin is 'up', 'down' or absent.")
    elif k == "zeeman":
        B = np.asarray(m.get("B", [0, 0, 0]), dtype=float).reshape(-1)
        if B.shape != (3,) or not np.all(np.isfinite(B)):
            raise ValueError("zeeman: B is [Bx, By, Bz] in tesla.")
        m["B"] = B.tolist()
        m["g"] = float(m.get("g", G_ELECTRON))
        m["spin"] = bool(m.get("spin", True))
        m["orbital"] = bool(m.get("orbital", True))
    elif k == "exchange":
        m["energy"] = float(m.get("energy", 0.0))
        n = np.asarray(m.get("direction", [0, 0, 1]), dtype=float).reshape(-1)
        if n.shape != (3,) or not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-12:
            raise ValueError("exchange: direction is a non-zero [nx, ny, nz].")
        m["direction"] = (n / np.linalg.norm(n)).tolist()
        if m.get("sites") is not None:
            m["sites"] = [int(x) for x in m["sites"]]
        if m.get("orbitals") is not None:
            m["orbitals"] = [str(x) for x in m["orbitals"]]
    elif k == "rashba":
        m["lam"] = float(m.get("lam", 0.0))
        m["orbitals"] = [str(x) for x in (m.get("orbitals") or ["pz"])]
    elif k == "efield":
        E = np.asarray(m.get("E", [0, 0, 0]), dtype=float).reshape(-1)
        if E.shape != (3,) or not np.all(np.isfinite(E)):
            raise ValueError("efield: E is [Ex, Ey, Ez] in V/Å.")
        m["E"] = E.tolist()
        if m.get("origin") is not None:
            o = np.asarray(m["origin"], dtype=float).reshape(-1)
            if o.shape != (3,):
                raise ValueError("efield: origin is [x, y, z] in Å.")
            m["origin"] = o.tolist()
    elif k == "atomic_soc":
        lam = m.get("lam")
        if not isinstance(lam, dict) or not lam:
            raise ValueError('atomic_soc: lam is {"<El>_<family>" or "<El>": lambda in eV}, e.g. {"Te_p": 0.49}.')
        out = {}
        for key, v in lam.items():
            el, _, fam = str(key).partition("_")
            if not el or (fam and fam not in SOC_FAMILIES):
                raise ValueError(f"atomic_soc: {key!r} is not '<El>' or '<El>_p' / '_d' / '_f'.")
            try:
                out[str(key)] = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"atomic_soc: lambda of {key} must be a number (eV).") from None
            if not np.isfinite(out[str(key)]):
                raise ValueError(f"atomic_soc: lambda of {key} must be finite.")
        m["lam"] = out
        if m.get("labels") is not None:
            if not isinstance(m["labels"], dict):
                raise ValueError('atomic_soc: labels is {"<El>": ["pz", "px", "py"], ...}.')
            m["labels"] = {str(el): [str(x) for x in labs] for el, labs in m["labels"].items()}
    elif k == "scissor":
        m["delta"] = float(m.get("delta", 0.0))
        if m.get("mesh") is not None:
            mesh = [int(x) for x in m["mesh"]]
            if len(mesh) != 3 or min(mesh) < 1:
                raise ValueError("scissor: mesh is three positive integers.")
            m["mesh"] = mesh
        if m.get("midgap") is not None:
            m["midgap"] = float(m["midgap"])
        if m.get("n_valence") is not None:
            m["n_valence"] = int(m["n_valence"])
    return m


def sublattice_mass(delta: float, plus_sites, minus_sites) -> list:
    """The two on-site shifts (+delta/2 on ``plus_sites``, -delta/2 on ``minus_sites``) of a sublattice mass:
    they open a gap of ``delta`` at a Dirac point (graphene: sites [0] and [1])."""
    return [{"kind": "onsite_shift", "shift": 0.5 * float(delta), "sites": list(plus_sites), "label": "mass +"},
            {"kind": "onsite_shift", "shift": -0.5 * float(delta), "sites": list(minus_sites), "label": "mass -"}]


# == generic helpers ==============================================================================================
def bloch_of(m, k_cart, overlap: bool = False) -> np.ndarray:
    """H(k) (or S(k)) of any model from its blocks, phases exp(i k . d) (as every model's own bloch)."""
    k = np.asarray(k_cart, dtype=float)
    dims = [m.norbs(i) for i in range(m.n_sites)]
    offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    if overlap and not m.non_orthogonal:
        return np.eye(offs[-1], dtype=complex)
    H = np.eye(offs[-1], dtype=complex) if overlap else np.zeros((offs[-1], offs[-1]), dtype=complex)
    if not overlap:
        for i in range(m.n_sites):
            H[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] += m.onsite(i)
    for b in m.bonds:
        i, j = b["i"], b["j"]
        H[offs[i]:offs[i + 1], offs[j]:offs[j + 1]] += m.hopping(i, j, b["d_vec"], overlap=overlap) * np.exp(
            1j * k @ np.asarray(b["d_vec"], float))
    return 0.5 * (H + H.conj().T)


def parent_sites(base, sc) -> list:
    """For each site of ``sc`` (a supercell of ``base``), the base site it is a translate of."""
    inv = np.linalg.inv(base.lattice)
    out = []
    for r, sp in zip(sc.cart_coords, sc.species):
        found = None
        for s in range(base.n_sites):
            f = (r - base.cart_coords[s]) @ inv
            if base.species[s] == sp and np.abs(f - np.rint(f)).max() < 1e-5:
                found = s
                break
        if found is None:
            raise RuntimeError("A supercell site is not a translate of any site of the model's cell.")
        out.append(found)
    return out


def periodic_axes(m) -> list:
    """Lattice directions along which the model has hoppings (a 2D material: two; its vacuum direction has none)."""
    return [a for a in range(3) if m.reach[a] > 0]


# == static delegation (typing.Protocol checks attributes statically, so __getattr__ is not enough) ============
def _delegated(name):
    return property(lambda self: getattr(self._target(), name))


class _Delegate:
    """Properties that read the target model's attributes (the common interface's data members)."""

    def _target(self):
        raise NotImplementedError

    @property
    def n_hoppings(self):
        return getattr(self._target(), "n_hoppings", None)

    @property
    def spin_matrices(self):
        return getattr(self._target(), "spin_matrices", None)

    @property
    def position_matrices(self):
        return getattr(self._target(), "position_matrices", None)


for _n in ("kind", "lattice", "species", "frac_coords", "efermi", "n_neighbors", "non_orthogonal", "soc",
           "is_magnetic", "n_sites", "cart_coords", "reach", "bonds"):
    setattr(_Delegate, _n, _delegated(_n))
del _n


# == the layer: a model plus spin doubling or on-site terms =======================================================
class _Layer(_Delegate):
    """``inner`` with its basis doubled for spin (``double``) and/or extra on-site blocks ``terms`` {site: block}.
    Offers the common interface; hoppings are the inner model's (so an SK model keeps its distance scaling)."""

    def __init__(self, inner, *, double=False, terms=None, step=None, conserves_sz=None, rashba=None, soc=None):
        self.inner = inner
        self._soc = soc
        self.double = bool(double)
        self.rashba = dict(rashba) if rashba else None       # {lam, orbitals, element, nn}: on the NN bonds
        if self.double and inner.spinful:
            raise ValueError("The model is spinful already.")
        self.terms = {int(i): np.asarray(t, dtype=complex) for i, t in (terms or {}).items()}
        self.step = dict(step or {})
        self._conserves = conserves_sz
        for i, t in self.terms.items():
            if t.shape != (self.norbs(i), self.norbs(i)):
                raise ValueError(f"on-site term of site {i} has shape {t.shape}")

    def _target(self):
        return self.inner

    @property
    def source(self) -> dict:
        return self.inner.source

    @property
    def spinful(self) -> bool:
        return self.double or self.inner.spinful

    @property
    def soc(self) -> bool:
        return bool(self._soc) or bool(self.inner.soc)

    @property
    def conserves_sz(self) -> bool:
        if self._conserves is not None:
            return self._conserves
        return (not self.inner.soc) if self.double else getattr(self.inner, "conserves_sz",
                                                               self.inner.spinful and not self.inner.soc)

    @property
    def site_orbitals(self) -> list:
        if self.double:
            return [self.orbital_labels(i) for i in range(self.n_sites)]
        return self.inner.site_orbitals

    @property
    def provenance(self) -> list:
        return list(self.inner.provenance) + ([self.step] if self.step else [])

    def norbs(self, i: int) -> int:
        return self.inner.norbs(i) * (2 if self.double else 1)

    def orbital_labels(self, i: int) -> list:
        lab = self.inner.orbital_labels(i)
        return [x + "↑" for x in lab] + [x + "↓" for x in lab] if self.double else lab

    def orbital_info(self, i: int) -> list:
        info = self.inner.orbital_info(i)
        if not self.double:
            return info
        labs = self.orbital_labels(i)
        n = len(info)
        return [{"label": labs[k], "l": info[k % n]["l"], "spin": "up" if k < n else "down"} for k in range(2 * n)]

    def orbital_centres(self, i: int) -> np.ndarray:
        c = np.asarray(self.inner.orbital_centres(i))
        return np.vstack([c, c]) if self.double else c

    @property
    def position_matrices(self):
        pm = getattr(self.inner, "position_matrices", None)
        if not pm or not self.double:
            return pm
        return {k: np.array([np.kron(np.eye(2), b[a]) for a in range(3)]) for k, b in pm.items()}

    def onsite(self, i: int) -> np.ndarray:
        h = self.inner.onsite(i)
        if self.double:
            h = np.kron(np.eye(2), h)
        if i in self.terms:
            h = h + self.terms[i]
        return h

    def hopping(self, i: int, j: int, d_vec, overlap: bool = False) -> np.ndarray:
        h = self.inner.hopping(i, j, d_vec, overlap=overlap)
        h = np.kron(np.eye(2), h) if self.double else h
        if self.rashba and not overlap:
            h = h + _rashba_block(self, i, j, d_vec, self.rashba)
        return h

    def bloch(self, k_cart, overlap: bool = False) -> np.ndarray:
        return bloch_of(self, k_cart, overlap)

    def supercell(self, matrix):
        sc = self.inner.supercell(matrix)
        par = parent_sites(self.inner, sc)
        return _Layer(sc, double=self.double, terms={s: self.terms[p] for s, p in enumerate(par) if p in self.terms},
                      step=self.step, conserves_sz=self._conserves, rashba=self.rashba, soc=self._soc)

    def to_tb_model(self) -> TBModel:
        return TBModel.from_model(self)

    def to_dict(self) -> dict:
        return self.to_tb_model().to_dict()

    def save(self, path: str) -> str:
        return self.to_tb_model().save(path)


# == the modifiers ================================================================================================
def _spin_blocks_diagonal(t: np.ndarray) -> bool:
    n = t.shape[0] // 2
    return np.abs(t[:n, n:]).max(initial=0.0) < 1e-14 and np.abs(t[n:, :n]).max(initial=0.0) < 1e-14


def _onsite_shift(m, mod, warnings):
    terms = {}
    for i in range(m.n_sites):
        if mod.get("element") and m.species[i] != mod["element"]:
            continue
        if mod.get("sites") is not None and i not in mod["sites"]:
            continue
        diag = np.zeros(m.norbs(i))
        for k, x in enumerate(m.orbital_info(i)):
            if mod.get("orbitals") is not None and strip_spin(x["label"]) not in mod["orbitals"]:
                continue
            if mod.get("l") is not None and x["l"] != mod["l"]:
                continue
            if mod.get("spin") is not None and x["spin"] != mod["spin"]:
                continue
            diag[k] = mod["shift"]
        if np.any(diag):
            terms[i] = np.diag(diag).astype(complex)
    if not terms and mod["shift"] != 0.0:
        raise ValueError(f"onsite_shift selects no orbital of the model: {mod}")
    if mod.get("spin") is not None and not m.spinful:
        raise ValueError("onsite_shift with a spin needs a spinful model (add {'kind': 'spinful'} first).")
    return _Layer(m, terms=terms, step={"step": "onsite_shift", **mod}, conserves_sz=None)


def _site_sigma(m, i):
    """sigma_a of site i's orbitals, (3, n, n): the model's spin matrices (spinor Wannier functions) or, for a spinful
    basis (spin-up orbitals first, then spin-down), the Pauli matrices."""
    sm = getattr(m, "spin_matrices", None)
    if sm:
        return np.asarray(sm[(i, i, (0, 0, 0))])
    n = m.norbs(i) // 2
    return np.array([np.kron(_SIGMA[a], np.eye(n)) for a in range(3)])


def _offsite_spin(m) -> float:
    """Largest sigma element between different sites or cells (the Zeeman term adds them as hoppings)."""
    sm = getattr(m, "spin_matrices", None)
    if not sm:
        return 0.0
    return max((float(np.abs(b).max()) for (i, j, R), b in sm.items() if i != j or any(R)), default=0.0)


def _zeeman(m, mod, warnings):
    B = np.asarray(mod["B"], dtype=float)
    has_sigma = bool(getattr(m, "spin_matrices", None))
    terms, no_L, spinor = {}, [], m.soc and getattr(m, "kind", "") == "wannier" and not has_sigma
    for i in range(m.n_sites):
        n = m.norbs(i) // 2 if m.spinful else m.norbs(i)
        labels = [strip_spin(x) for x in m.orbital_labels(i)[:n]]
        t = np.zeros((m.norbs(i), m.norbs(i)), dtype=complex)
        if mod["spin"]:                                     # apply() made the model spinful first
            sig = _site_sigma(m, i)
            t += 0.5 * mod["g"] * MU_B * sum(B[a] * sig[a] for a in range(3))
        if mod["orbital"]:
            L = L_matrices(labels)
            if L is None:
                no_L.append(i)
            else:
                LB = sum(B[a] * L[a] for a in range(3))
                t += MU_B * (np.kron(np.eye(2), LB) if m.spinful else LB)
        if np.any(t):
            terms[i] = t
    if no_L and mod["orbital"] and np.any(B):
        warnings.append(f"zeeman: no orbital (L.B) term on {len(no_L)} site(s) whose orbitals have no atomic "
                        "angular momentum (e.g. Wannier functions); only the spin term is applied there.")
    if spinor and mod["spin"] and np.any(B):
        warnings.append("zeeman: spinor Wannier functions without spin matrices are treated as pure spin up / spin "
                        "down for the spin term (a Wannier run with the .spn gives the exact sigma).")
    base_cons = getattr(m, "conserves_sz", m.spinful and not m.soc)
    cons = base_cons and m.spinful and all(_spin_blocks_diagonal(t) for t in terms.values())
    step = {"step": "zeeman", **mod}
    if warnings:
        step["warnings"] = list(warnings)
    layer = _Layer(m, terms=terms, step=step, conserves_sz=cons)
    if not (has_sigma and mod["spin"] and np.any(B) and _offsite_spin(m) > 0.0):
        return layer
    # spinor Wannier functions: sigma also couples different sites and cells - the field's term there is a hopping
    # (g muB / 2) B.sigma(i, j, R), which keeps the model exact on the Wannier run's k mesh
    out = TBModel.from_model(layer)
    hops = {key: blk.copy() for key, blk in out._hop.items()}
    for (i, j, R), b in m.spin_matrices.items():
        if i == j and not any(R):
            continue
        t = 0.5 * mod["g"] * MU_B * sum(B[a] * b[a] for a in range(3))
        hops[(i, j, R)] = hops.get((i, j, R), 0) + t
    return TBModel(lattice=out.lattice, species=out.species, frac_coords=out.frac_coords,
                   site_orbitals=out.site_orbitals, onsite=[out.onsite(i) for i in range(out.n_sites)], hoppings=hops,
                   spinful=out.spinful, soc=out.soc, is_magnetic=out.is_magnetic, efermi=out.efermi,
                   source=out.source, kind=out.kind, orbital_offsets=out.orbital_offsets, orbital_l=out.orbital_l,
                   overlaps=dict(out._ovl) if out.non_orthogonal else None, provenance=out.provenance,
                   matrices=_vector_matrices_of(m))


def _select(m, i, mod):
    """Orbital indices of site i chosen by element / sites / orbitals (spin halves both included)."""
    if mod.get("element") and m.species[i] != mod["element"]:
        return []
    if mod.get("sites") is not None and i not in mod["sites"]:
        return []
    return [k for k, x in enumerate(m.orbital_info(i))
            if mod.get("orbitals") is None or strip_spin(x["label"]) in mod["orbitals"]]


def _exchange(m, mod, warnings):
    n = np.asarray(mod["direction"], dtype=float)
    sig = sum(n[a] * _SIGMA[a] for a in range(3))
    terms = {}
    for i in range(m.n_sites):
        sel = _select(m, i, mod)
        if not sel:
            continue
        if getattr(m, "spin_matrices", None):              # spinor Wannier functions: their own sigma
            P = np.zeros((m.norbs(i), m.norbs(i)))
            P[sel, sel] = 1.0
            s3 = _site_sigma(m, i)
            terms[i] = mod["energy"] * P @ sum(n[a] * s3[a] for a in range(3)) @ P
            continue
        half = m.norbs(i) // 2
        P = np.zeros((half, half))
        for k in sel:
            P[k % half, k % half] = 1.0
        terms[i] = mod["energy"] * np.kron(sig, P)
    if not terms and mod["energy"] != 0.0:
        raise ValueError(f"exchange selects no orbital of the model: {mod}")
    base_cons = getattr(m, "conserves_sz", m.spinful and not m.soc)
    cons = base_cons and abs(n[0]) < 1e-12 and abs(n[1]) < 1e-12
    return _Layer(m, terms=terms, step={"step": "exchange", **mod}, conserves_sz=cons)


def _rashba_pairs(m, i, j, spec):
    """(a, b) orbital indices (within one spin half) of sites i, j carrying the Rashba term: the same label."""
    if spec.get("element") and (m.species[i] != spec["element"] or m.species[j] != spec["element"]):
        return []
    half_i, half_j = m.norbs(i) // 2, m.norbs(j) // 2
    li = [strip_spin(x) for x in m.orbital_labels(i)[:half_i]]
    lj = [strip_spin(x) for x in m.orbital_labels(j)[:half_j]]
    return [(a, b) for a, x in enumerate(li) for b, y in enumerate(lj) if x == y and x in spec["orbitals"]]


def _rashba_block(m, i, j, d_vec, spec) -> np.ndarray:
    d = np.asarray(d_vec, dtype=float)
    out = np.zeros((m.norbs(i), m.norbs(j)), dtype=complex)
    r = float(np.linalg.norm(d))
    if abs(r - spec["nn"]) > 1e-3 * spec["nn"]:
        return out
    pairs = _rashba_pairs(m, i, j, spec)
    if not pairs:
        return out
    e = d / r
    S = 1j * spec["lam"] * (_SIGMA[0] * e[1] - _SIGMA[1] * e[0])          # i lam (sigma x d)_z
    hi, hj = m.norbs(i) // 2, m.norbs(j) // 2
    P = np.zeros((hi, hj))
    for a, b in pairs:
        P[a, b] = 1.0
    return np.kron(S, P)


def _rashba(m, mod, warnings):
    lengths = [float(np.linalg.norm(b["d_vec"])) for b in m.bonds
               if np.linalg.norm(b["d_vec"]) > 1e-6 and _rashba_pairs(m, b["i"], b["j"], mod)]
    if not lengths:
        raise ValueError(f"rashba: no bond between {', '.join(mod['orbitals'])} orbitals"
                         + (f" of {mod['element']}" if mod.get("element") else "") + " in the model.")
    spec = {**mod, "nn": min(lengths)}
    return _Layer(m, step={"step": "rashba", **mod, "nn_distance": round(spec["nn"], 6)}, conserves_sz=False,
                  rashba=spec)


def _soc_lambdas(mod, element) -> dict:
    """{family: lambda} of an element from atomic_soc's lam ("<El>_<fam>" wins over "<El>")."""
    out = {}
    if element in mod["lam"]:
        out = {f: mod["lam"][element] for f in SOC_FAMILIES}
    for f in SOC_FAMILIES:
        if f"{element}_{f}" in mod["lam"]:
            out[f] = mod["lam"][f"{element}_{f}"]
    return out


def _atomic_soc(m, mod, warnings):
    from .orbitals import family_of
    if m.soc:
        raise ValueError("atomic_soc: the model has spin-orbit coupling already.")
    elements = set(m.species)
    unknown = sorted({k.partition("_")[0] for k in mod["lam"]} - elements)
    if unknown:
        raise ValueError(f"atomic_soc: no {', '.join(unknown)} in the model (elements: {', '.join(sorted(elements))}).")
    terms, relabelled, missing = {}, set(), set()
    for i in range(m.n_sites):
        el = m.species[i]
        lams = _soc_lambdas(mod, el)
        if not lams:
            continue
        half = m.norbs(i) // 2
        labels = [strip_spin(x) for x in m.orbital_labels(i)[:half]]
        if any(family_of(x) is None for x in labels):
            given = (mod.get("labels") or {}).get(el)
            if given is None:
                missing.add(el)
                continue
            if len(given) != half or any(family_of(x) is None for x in given):
                raise ValueError(f"atomic_soc: labels of {el} must be {half} atomic orbital names (px, py, pz, dxy, ...) "
                                 f"- one per function of the site; got {given}.")
            labels = list(given)
            relabelled.add(el)
        L = L_matrices(labels)
        fam = [family_of(x)[0] for x in labels]
        t = np.zeros((m.norbs(i), m.norbs(i)), dtype=complex)
        for f, lam in lams.items():
            idx = [a for a, g in enumerate(fam) if g == f]
            if not idx or lam == 0.0:
                continue
            P = np.zeros((half, half))
            P[idx, idx] = 1.0
            t += 0.5 * lam * sum(np.kron(_SIGMA[a], P @ L[a] @ P) for a in range(3))
        if np.any(t):
            terms[i] = t
    if missing:
        raise ValueError(f"atomic_soc: the functions of {', '.join(sorted(missing))} have no atomic label (Wannier "
                         "functions): give labels, e.g. {\"" + sorted(missing)[0] + "\": [\"pz\", \"px\", \"py\"]}, "
                         "if they are atom-centred atomic-like orbitals.")
    if not terms and any(v != 0.0 for v in mod["lam"].values()):
        raise ValueError("atomic_soc: the term vanishes - no p, d or f shell of the named elements in the model, or only "
                         "part of one (L couples the orbitals of a full shell: a lone pz has none).")
    if relabelled:
        warnings.append(f"atomic_soc: the functions of {', '.join(sorted(relabelled))} are taken as the atomic orbitals "
                        "given in labels (valid for atom-centred projected functions with the global axes).")
    step = {"step": "atomic_soc", **mod}
    return _Layer(m, terms=terms, step=step, conserves_sz=not terms, soc=bool(terms))


def default_mesh(m, n3d: int = 12, n2d: int = 18) -> list:
    per = periodic_axes(m)
    n = n2d if len(per) <= 2 else n3d
    return [n if a in per else 1 for a in range(3)]


def _eig_lowdin(H, S):
    """Eigen-decomposition of H c = E S c through S^-1/2: (E, U, S^1/2) with U the Löwdin eigenvectors."""
    if S is None:
        e, U = np.linalg.eigh(H)
        return e, U, None
    s, V = np.linalg.eigh(S)
    Sm = (V / np.sqrt(s)) @ V.conj().T
    Sp = (V * np.sqrt(s)) @ V.conj().T
    e, U = np.linalg.eigh(Sm @ H @ Sm)
    return e, U, Sp


def _scissor(m, mod, warnings):
    delta = mod["delta"]
    mesh = mod.get("mesh") or default_mesh(m)
    lat = m.lattice
    rec = 2 * np.pi * np.linalg.inv(lat).T
    dims = [m.norbs(i) for i in range(m.n_sites)]
    offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    D = int(offs[-1])
    r_orb = np.repeat(m.cart_coords, dims, axis=0)                   # bloch phases use the site positions
    need = float(np.prod(mesh)) * D * D * 16 * 2
    if need > SCISSOR_MAX_BYTES:
        raise ValueError(f"scissor: a {mesh} mesh for {D} orbitals needs about {need / 1e9:.1f} GB; use a smaller "
                         "mesh (or apply it to a smaller model).")
    grid = np.array(np.meshgrid(*[np.arange(n) / n for n in mesh], indexing="ij")).reshape(3, -1).T
    eigs = []
    for kf in grid:
        k = kf @ rec
        S = m.bloch(k, overlap=True) if m.non_orthogonal else None
        eigs.append(_eig_lowdin(m.bloch(k), S))
    E = np.array([e for e, _, _ in eigs])
    by_index = mod.get("midgap") is None and mod.get("n_valence") is not None
    direct_min = None
    if by_index:                                                     # bands above the n-th: a direct gap suffices
        nocc = int(mod["n_valence"])
        if not 0 < nocc < D:
            raise ValueError(f"scissor: n_valence must be between 1 and {D - 1}.")
        direct = E[:, nocc] - E[:, nocc - 1]
        direct_min = float(direct.min())
        if direct_min < 1e-4:
            raise ValueError(f"scissor: bands {nocc} and {nocc + 1} touch on the mesh (direct gap "
                             f"{1000 * direct_min:.2f} meV): no conduction bands to move; check n_valence.")
        vbm, cbm = E[:, nocc - 1].max(), E[:, nocc].min()
        midgap = 0.5 * (vbm + cbm) if cbm > vbm else None             # None: a semimetal (indirect overlap)
    elif mod.get("midgap") is None:
        counts = np.sum(E < E_F_TOL, axis=1)
        nocc = int(counts[0])
        if np.any(counts != nocc) or nocc in (0, D):
            lo, hi = int(counts.min()), int(counts.max())
            raise ValueError(f"scissor needs a gap at E_F = 0 at every k of the mesh: {lo} to {hi} bands lie below "
                             "E_F (a metal or a DFT semimetal there); give the number of valence bands (n_valence, "
                             "e.g. " + str(hi) + ") - a direct gap is then enough - or midgap for another gap.")
        vbm, cbm = E[:, nocc - 1].max(), E[:, nocc].min()
        if cbm <= vbm:
            raise ValueError("scissor needs a gap: the conduction band minimum is below the valence band maximum.")
        midgap = 0.5 * (vbm + cbm)
    else:
        midgap = mod["midgap"]
        counts = np.sum(E < midgap, axis=1)
        if np.any(counts != counts[0]):
            raise ValueError("scissor: bands cross the given midgap somewhere on the mesh.")
    P = np.zeros(tuple(mesh) + (D, D), dtype=complex)
    for n_k, (kf, (e, U, Sp)) in enumerate(zip(grid, eigs)):
        Uc = U[:, nocc:] if by_index else U[:, e > midgap]
        dH = delta * (Uc @ Uc.conj().T)
        if Sp is not None:
            dH = Sp @ dH @ Sp
        ph = np.exp(1j * (r_orb @ (kf @ rec)))
        idx = tuple(int(round(x * n)) for x, n in zip(kf, mesh))
        P[idx] = (ph[:, None] * dH) * ph.conj()[None, :]                # periodic gauge: D dH D*
    hR = np.fft.fftn(P, axes=(0, 1, 2)) / float(np.prod(mesh))       # aliased H(R), R mod mesh
    # spread each aliased block over its shortest images (Wigner-Seitz), equally among ties
    shifts = np.array([[a, b, c] for a in ((-1, 0, 1) if mesh[0] > 1 else (0,))
                       for b in ((-1, 0, 1) if mesh[1] > 1 else (0,))
                       for c in ((-1, 0, 1) if mesh[2] > 1 else (0,))]) * np.array(mesh)
    add = {}
    for R in np.ndindex(*mesh):
        blk = hR[R]
        if np.abs(blk).max() < 1e-10:
            continue
        for i in range(m.n_sites):
            for j in range(m.n_sites):
                sub = blk[offs[i]:offs[i + 1], offs[j]:offs[j + 1]]
                if np.abs(sub).max() < 1e-10:
                    continue
                cand = np.array(R) + shifts
                d = np.linalg.norm(m.cart_coords[j] + cand @ lat - m.cart_coords[i], axis=1)
                best = cand[d < d.min() + 1e-5]
                for Rp in best:
                    key = (i, j, tuple(int(x) for x in Rp))
                    add[key] = add.get(key, 0) + sub / len(best)
    base = TBModel.from_model(m)
    onsite = [base.onsite(i) + add.pop((i, i, (0, 0, 0)), 0) for i in range(m.n_sites)]
    hops = {key: blk.copy() for key, blk in base._hop.items()}
    for key, blk in add.items():
        if np.abs(blk).max() < 1e-9:
            continue
        hops[key] = hops.get(key, 0) + blk
    # measure the error between mesh points against the exact scissor
    rng = np.random.default_rng(11)
    per = periodic_axes(m)
    err = 0.0
    out = TBModel(lattice=base.lattice, species=base.species, frac_coords=base.frac_coords,
                  site_orbitals=base.site_orbitals, onsite=onsite, hoppings=hops, spinful=base.spinful, soc=base.soc,
                  is_magnetic=base.is_magnetic, efermi=base.efermi, source=base.source, kind=base.kind,
                  orbital_offsets=base.orbital_offsets, orbital_l=base.orbital_l,
                  overlaps=dict(base._ovl) if base.non_orthogonal else None, provenance=base.provenance,
                  matrices=_vector_matrices_of(base))
    for _ in range(6):
        kf = np.zeros(3)
        kf[per] = rng.random(len(per))
        k = kf @ rec
        S0 = m.bloch(k, overlap=True) if m.non_orthogonal else None
        e0, _, _ = _eig_lowdin(m.bloch(k), S0)
        if by_index:
            exact = np.concatenate([e0[:nocc], e0[nocc:] + delta])
        else:
            exact = np.where(e0 > midgap, e0 + delta, e0)
        S1 = out.bloch(k, overlap=True) if out.non_orthogonal else None
        e1, _, _ = _eig_lowdin(out.bloch(k), S1)
        err = max(err, float(np.abs(np.sort(exact) - e1).max()))
    step = {"step": "scissor", **mod, "mesh": list(mesh),
            "midgap": None if midgap is None else round(float(midgap), 6), "max_error_meV": round(1000 * err, 4)}
    if by_index:
        step["direct_gap_min_meV"] = round(1000 * direct_min, 4)
    if 1000 * err > SCISSOR_WARN_MEV:
        warnings.append(f"scissor: bands between mesh points are off by up to {1000 * err:.1f} meV; raise the mesh "
                        f"(now {mesh}).")
        step["warnings"] = list(warnings)
    out.provenance = list(base.provenance) + [step]
    return out


def _efield(m, mod, warnings):
    """+ E.(r - origin) (eV, E in V/Å: an electron's energy e E.r) on the model: on each orbital centre and, for a
    Wannier model with position matrices r(R) (Wannier90's _tb.dat), between the functions - the interband (Stark)
    terms - as hoppings E.r(i, j, R). Only across directions without hoppings (a layer's vacuum, across a wire):
    along a periodic direction a uniform field is not a periodic potential (a bias or gate of a device calculation)."""
    E = np.asarray(mod["E"], dtype=float)
    lat = np.asarray(m.lattice, dtype=float)
    for a in periodic_axes(m):
        if abs(E @ lat[a]) > 1e-9 * max(1.0, float(np.linalg.norm(E) * np.linalg.norm(lat[a]))):
            raise ValueError(f"efield: E has a component along lattice vector a{a + 1}, a periodic direction of the "
                             "model - a uniform field there is not periodic (use a bias or gate of a device calculation); "
                             "give E across the vacuum or the wire.")
    cent = [np.asarray(m.orbital_centres(i), float) for i in range(m.n_sites)]
    origin = np.asarray(mod["origin"], float) if mod.get("origin") is not None else np.vstack(cent).mean(axis=0)
    pm = getattr(m, "position_matrices", None)
    terms = {}
    for i in range(m.n_sites):
        t = np.diag((cent[i] - origin) @ E).astype(complex)
        if pm and (i, i, (0, 0, 0)) in pm:                 # the interband part within a site
            b = sum(E[a] * pm[(i, i, (0, 0, 0))][a] for a in range(3))
            t = t + b - np.diag(np.diag(b))
        if np.any(np.abs(t) > 0):
            terms[i] = t
    step = {"step": "efield", **mod, "origin": origin.round(6).tolist(), "position_matrices": bool(pm)}
    if not pm and np.any(E):
        warnings.append("efield: the model has no position matrices, so the field acts on the orbital centres only "
                        "(no interband Stark terms; a Wannier model from a run with its _tb.dat has them).")
        step["warnings"] = list(warnings)
    layer = _Layer(m, terms=terms, step=step, conserves_sz=None)
    off = {k: b for k, b in (pm or {}).items() if k[0] != k[1] or any(k[2])}
    if not off or not np.any(E):
        return layer
    out = TBModel.from_model(layer)
    hops = {key: blk.copy() for key, blk in out._hop.items()}
    for key, b in off.items():
        hops[key] = hops.get(key, 0) + sum(E[a] * b[a] for a in range(3))
    return TBModel(lattice=out.lattice, species=out.species, frac_coords=out.frac_coords,
                   site_orbitals=out.site_orbitals, onsite=[out.onsite(i) for i in range(out.n_sites)], hoppings=hops,
                   spinful=out.spinful, soc=out.soc, is_magnetic=out.is_magnetic, efermi=out.efermi,
                   source=out.source, kind=out.kind, orbital_offsets=out.orbital_offsets, orbital_l=out.orbital_l,
                   overlaps=dict(out._ovl) if out.non_orthogonal else None, provenance=out.provenance,
                   matrices=_vector_matrices_of(m))


_APPLY = {"onsite_shift": _onsite_shift, "zeeman": _zeeman, "exchange": _exchange, "rashba": _rashba,
          "atomic_soc": _atomic_soc,
          "scissor": _scissor, "efield": _efield}


# == the modified model ===========================================================================================
class ModifiedModel(_Delegate):
    """A base model with modifiers applied (see the module docstring). Offers the common interface (delegated to the
    effective model); ``base`` and ``modifiers`` are what its file stores."""

    def __init__(self, base, modifiers, effective, warnings):
        self.base = base
        self.modifiers = modifiers
        self.effective = effective
        self.warnings = list(warnings)

    def _target(self):
        return self.effective

    @property
    def spinful(self) -> bool:
        return self.effective.spinful

    @property
    def site_orbitals(self) -> list:
        return self.effective.site_orbitals

    @property
    def provenance(self) -> list:
        return self.effective.provenance

    def norbs(self, i):
        return self.effective.norbs(i)

    def orbital_labels(self, i):
        return self.effective.orbital_labels(i)

    def orbital_info(self, i):
        return self.effective.orbital_info(i)

    def orbital_centres(self, i):
        return self.effective.orbital_centres(i)

    def onsite(self, i):
        return self.effective.onsite(i)

    def hopping(self, i, j, d_vec, overlap=False):
        return self.effective.hopping(i, j, d_vec, overlap=overlap)

    def bloch(self, k_cart, overlap=False):
        return self.effective.bloch(k_cart, overlap)

    @property
    def conserves_sz(self) -> bool:
        e = self.effective
        return getattr(e, "conserves_sz", e.spinful and not e.soc)

    @property
    def source(self) -> dict:
        return {**self.base.source, "modifiers": self.modifiers}

    def supercell(self, matrix):
        return self.effective.supercell(matrix)

    def to_tb_model(self) -> TBModel:
        return self.effective.to_tb_model()

    def to_dict(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "base": self.base.to_dict(), "modifiers": self.modifiers}

    def save(self, path: str) -> str:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        return path


def apply(model, modifiers) -> ModifiedModel:
    """``model`` with ``modifiers`` applied in order (a list of dicts, module docstring). A ModifiedModel of a
    ModifiedModel adds to its list (the base stays the original model)."""
    if isinstance(model, ModifiedModel):
        return apply(model.base, list(model.modifiers) + list(modifiers))
    mods, cur, warnings = [], model, []
    for raw in modifiers:
        mod = check_modifier(raw)
        if mod["kind"] == "spinful":
            if not cur.spinful:
                cur = _Layer(cur, double=True, step={"step": "spinful"})
            mods.append(mod)
            continue
        if ((mod["kind"] == "zeeman" and mod["spin"]) or mod["kind"] in ("exchange", "rashba", "atomic_soc")) and not cur.spinful:
            cur = _Layer(cur, double=True, step={"step": "spinful", "auto": True})
            mods.append({"kind": "spinful", "auto": True})
        cur = _APPLY[mod["kind"]](cur, mod, warnings)
        mods.append(mod)
    return ModifiedModel(model, mods, cur, warnings)


def from_dict(data: dict) -> ModifiedModel:
    if data.get("format") != FORMAT:
        raise ValueError(f"Not a modified model file (format {data.get('format')!r}).")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError(f"Modified model file version {data.get('version')} is newer than this program.")
    from .io import model_from_dict
    base = model_from_dict(data["base"])
    mods = [m for m in data.get("modifiers", []) if not (m.get("kind") == "spinful" and m.get("auto"))]
    return apply(base, mods)
