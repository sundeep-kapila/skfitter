"""SKModel: a fitted SKFitter model as real-space blocks (one of the two model classes of skfitter.model).

The model holds what the fit means and nothing of the DFT run: the cell, the sites with their orbitals, the
model settings (neighbour shells, spin-orbit coupling, magnetism, overlaps) and the fitted parameters. It is
saved as ``<seed>_model.json`` and read back without the DFT files, so later calculations need SKFitter's saved
output only.

The blocks are computed with SKFitter's own code (``_get_submatrix_hopping``, ``build_hamiltonian``) on a
lightweight SKFitter object that carries the model's attributes but reads no DFT data, so a device built here
and the bulk model of the fit cannot drift apart.

Conventions
-----------
* Energies in eV, relative to the Fermi energy of the DFT run (as in SKFitter).
* Lengths in Å; ``lattice`` rows are the lattice vectors; ``frac_coords`` are fractional site coordinates.
* Orbital order on a site: SKFitter's sub-shell order; with spin (SOC or magnetism) all spin-up orbitals of
  the site first, then all spin-down orbitals.
* A bond (i, j, d) is the hopping <i, r_i | H | j, r_i + d>. SKFitter symmetrises H(k) as (H + H^dagger) / 2,
  so the real-space hopping of a bond is (B_ij(d) + B_ji(-d)^dagger) / 2.
* Orbital centres are the atoms (offset 0); labels follow SKFitter's sub-shell order (px, py, pz; dxy, dyz, dzx,
  dx2y2, dz2; the seven f functions in VASP's order), with ↑ / ↓ for spinful models.

The model stays parameterised (the distance scaling gives strained models); ``to_tb_model()`` writes it
out as numbers (a ``skfitter.model.TBModel`` of kind ``sk``) for consumers that need only the blocks.

"""
from __future__ import annotations

import json
import os
from functools import cached_property

import numpy as np

FORMAT = "prayog-sk-model"
FORMAT_VERSION = 1
MODEL_SUFFIX = "_model.json"


class SKModel:
    """A Slater-Koster tight-binding model in real space. Build it with :meth:`from_fitter` or :meth:`load`.
    (``load`` / ``from_dict`` also read ``prayog-tb-model`` files and return a TBModel, as
    ``skfitter.model.load_model`` does.)"""

    kind = "sk"

    def __init__(self, *, lattice, species, frac_coords, site_orbitals, eq_site_map, params, cutoff, d0,
                 n_neighbors=1, soc=False, is_magnetic=False, non_orthogonal=False, moment_signs=None,
                 efermi=0.0, parent_sites=None, source=None, hopping_mode="decay", shell_distances=None,
                 onsite_mix=None, moment_dirs=None, pair_shells=None):
        self.lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
        self.species = [str(s) for s in species]
        self.frac_coords = np.asarray(frac_coords, dtype=float).reshape(-1, 3)
        self.site_orbitals = [list(o) for o in site_orbitals]
        self.eq_site_map = [int(x) for x in eq_site_map]
        self.params = {str(k): float(v) for k, v in params.items()}
        self.cutoff = float(cutoff)
        self.d0 = float(d0)
        self.n_neighbors = int(n_neighbors)
        self.soc = bool(soc)
        self.is_magnetic = bool(is_magnetic)
        self.non_orthogonal = bool(non_orthogonal)
        # 'decay': one integral per bond type scaled by (d0/r)^eta; 'shell': one per neighbour shell (SKFitter's
        # hopping_mode), the shells at ``shell_distances`` (Å)
        self.hopping_mode = str(hopping_mode or "decay")
        self.shell_distances = [float(x) for x in (shell_distances or [])]
        # the neighbour shells of each element pair ('Ir-Mn': [r0, r1, ...] Å): the shell index of a per-shell
        # integral and the reference length d0 = r0 of a pair's distance decay (from the bonds when not given)
        self.pair_shells = {str(k): [float(x) for x in v] for k, v in (pair_shells or {}).items()} or None
        n = len(self.species)
        self.moment_signs = [int(s) for s in (moment_signs if moment_signs is not None else [1] * n)]
        self.efermi = float(efermi)
        # parent_sites[i] = the site of the fitted cell that site i copies (a supercell keeps the fit's on-site terms)
        self.parent_sites = [int(p) for p in (parent_sites if parent_sites is not None else range(n))]
        self.source = dict(source or {})
        # moment_dirs[i] = unit vector of site i's moment (exchange -(delta/2) n . sigma); None: s_i z
        self.moment_dirs = None
        if moment_dirs is not None and any(abs(float(d[2])) < 1 - 1e-6 for d in moment_dirs):
            self.moment_dirs = [[float(c) for c in d] for d in moment_dirs]
        # onsite_mix[i] = {key: matrix} - the fit's on-site mixing terms of site i (e_mix_* parameters), or None
        self.onsite_mix = None
        if onsite_mix and any(onsite_mix):
            self.onsite_mix = [{str(k): np.asarray(M, float) for k, M in (dict(site) if site else {}).items()}
                               for site in onsite_mix]
        if not (len(self.frac_coords) == len(self.site_orbitals) == len(self.eq_site_map) == n
                == len(self.moment_signs) == len(self.parent_sites)):
            raise ValueError("SKModel: species, frac_coords, site_orbitals, eq_site_map, moment_signs and "
                             "parent_sites must all have one entry per site.")

    # ------------------------------------------------------------------ construction / files
    @classmethod
    def from_fitter(cls, fitter, params=None, source=None):
        """The model of an SKFitter with its fitted parameters (``fitter.opt_params`` unless ``params`` given)."""
        params = params if params is not None else getattr(fitter, "opt_params", None)
        if params is None:
            raise ValueError("The fitter has no fitted parameters: run fit() or load a saved fit first.")
        s = fitter.structure
        signs = getattr(fitter, "site_moment_signs", None) or [1] * len(s)
        if float(getattr(fitter, "scissor", 0.0) or 0.0):          # fitted to scissor-shifted DFT bands
            source = {**(source or {}), "dft_scissor_eV": float(fitter.scissor)}
        pts = dft_path_points(fitter)
        if pts:                                                    # the DFT k-path (Properties draws bands on it)
            source = {**(source or {}), "kpath_points": pts}
        return cls(lattice=s.lattice.matrix, species=[site.specie.symbol for site in s], frac_coords=s.frac_coords,
                   site_orbitals=fitter.site_orbitals, eq_site_map=list(fitter.eq_site_map), params=params,
                   cutoff=fitter.cutoff_radius, d0=fitter.d0, n_neighbors=fitter.n_neighbors, soc=fitter.soc,
                   is_magnetic=fitter.is_magnetic, non_orthogonal=fitter.non_orthogonal, moment_signs=signs,
                   efermi=getattr(fitter, "efermi", 0.0), source=source,
                   hopping_mode=getattr(fitter, "hopping_mode", "decay"),
                   shell_distances=getattr(fitter, "shell_distances", None),
                   pair_shells=fitter._pair_shells() if hasattr(fitter, "_pair_shells") else None,
                   onsite_mix=[dict(site) for site in (getattr(fitter, "onsite_mix_terms", None) or [])] or None,
                   moment_dirs=[list(map(float, d)) for d in (getattr(fitter, "site_moment_dirs", None) or [])] or None)

    def to_dict(self) -> dict:
        settings = {"n_neighbors": self.n_neighbors, "soc": self.soc, "is_magnetic": self.is_magnetic,
                    "non_orthogonal": self.non_orthogonal, "cutoff": self.cutoff, "d0": self.d0,
                    "pair_shells": self.shells_of_pairs}
        if self.hopping_mode != "decay":              # decay-law files stay as they were
            settings.update(hopping_mode=self.hopping_mode, shell_distances=self.shell_distances)
        if self.moment_dirs:
            settings["moment_dirs"] = self.moment_dirs
        if self.onsite_mix:
            settings["onsite_mix"] = [{k: M.tolist() for k, M in site.items()} for site in self.onsite_mix]
        return {
            "format": FORMAT, "version": FORMAT_VERSION,
            "lattice": self.lattice.tolist(), "species": self.species, "frac_coords": self.frac_coords.tolist(),
            "site_orbitals": self.site_orbitals, "eq_site_map": self.eq_site_map,
            "moment_signs": self.moment_signs, "parent_sites": self.parent_sites,
            "settings": settings,
            "efermi": self.efermi, "params": self.params, "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SKModel":
        if data.get("format") == "prayog-tb-model":             # a general tight-binding model (numbers only)
            from .model import TBModel
            return TBModel.from_dict(data)
        if data.get("format") != FORMAT:
            raise ValueError(f"Not an SK model file (format {data.get('format')!r}, expected {FORMAT!r}).")
        if int(data.get("version", 0)) > FORMAT_VERSION:
            raise ValueError(f"SK model file version {data.get('version')} is newer than this program "
                             f"(reads up to {FORMAT_VERSION}).")
        st = data["settings"]
        model = cls._from_dict(data, st)
        from skfitter import core as sk                         # older files: the integral names before the element pairs
        if sk.has_legacy_hoppings(sk.migrate_legacy_keys(model.params)):
            pairs, _ = sk.neighbour_pairs(model.structure, model.cutoff)
            model.params = sk.convert_legacy_hoppings(model.params, pairs, model.species, model.site_orbitals,
                                                      model.n_neighbors, model.hopping_mode, model.shells_of_pairs)
        return model

    @classmethod
    def _from_dict(cls, data, st):
        from skfitter.core import migrate_legacy_keys
        return cls(lattice=data["lattice"], species=data["species"], frac_coords=data["frac_coords"],
                   site_orbitals=data["site_orbitals"], eq_site_map=data["eq_site_map"],
                   params=migrate_legacy_keys(data["params"]),
                   cutoff=st["cutoff"], d0=st["d0"], n_neighbors=st["n_neighbors"], soc=st["soc"],
                   is_magnetic=st["is_magnetic"], non_orthogonal=st["non_orthogonal"],
                   moment_signs=data.get("moment_signs"), efermi=data.get("efermi", 0.0),
                   parent_sites=data.get("parent_sites"), source=data.get("source"),
                   hopping_mode=st.get("hopping_mode", "decay"), shell_distances=st.get("shell_distances"),
                   onsite_mix=st.get("onsite_mix"), moment_dirs=st.get("moment_dirs"),
                   pair_shells=st.get("pair_shells"))

    def save(self, path: str) -> str:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "SKModel":
        from .io import read_json
        return cls.from_dict(read_json(path))

    # ------------------------------------------------------------------ sizes
    @property
    def n_sites(self) -> int:
        return len(self.species)

    @property
    def reach(self) -> list:
        """Largest |R| of a bond along each lattice vector (a lead must be at least that many cells long)."""
        return np.abs(np.array([b["R"] for b in self.bonds] or [[0, 0, 0]])).max(axis=0).astype(int).tolist()

    @property
    def spinful(self) -> bool:
        """True when the basis is spin-doubled (spin-orbit coupling or magnetism), as in SKFitter."""
        return self.soc or self.is_magnetic

    @property
    def shells_of_pairs(self) -> dict:
        """{element pair: shell distances (Å)} of the fitted cell (given, or from its bonds with the fit's cutoff)."""
        if self.pair_shells is None:
            from skfitter import core as sk
            pairs, _ = sk.neighbour_pairs(self.structure, self.cutoff)
            self.pair_shells = sk.pair_shell_distances(pairs, self.species)
        return self.pair_shells

    @cached_property
    def spatial_dims(self) -> list:
        from skfitter.core import SKFitter
        return [sum(SKFitter.SUB_ORBITALS[s]["dim"] for s in orbs) for orbs in self.site_orbitals]

    def norbs(self, i: int) -> int:
        """Orbitals (with spin) on site i: the size of its Kwant site."""
        return self.spatial_dims[i] * (2 if self.spinful else 1)

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    @cached_property
    def structure(self):
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice(self.lattice), self.species, self.frac_coords)

    # ------------------------------------------------------------------ the common interface (skfitter.model.protocol)
    @property
    def provenance(self) -> list:
        """Where the model came from: the SKFitter fit (and a supercell, if one was made)."""
        src = {k: v for k, v in self.source.items() if k != "supercell"}
        out = [{"step": "skfitter fit", **src}]
        if "supercell" in self.source:
            out.append({"step": "supercell", "matrix": self.source["supercell"]})
        return out

    @cached_property
    def _orbital_names(self) -> list:
        """Per site: (label, l) of each spatial orbital, in SKFitter's order."""
        from skfitter.core import SKFitter
        full = {"s": ["s"], "s*": ["s*"], "p": ["px", "py", "pz"], "d": ["dxy", "dyz", "dzx", "dx2y2", "dz2"],
                "f": ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"]}
        ell = {"s": 0, "s*": 0, "p": 1, "d": 2, "f": 3}
        out = []
        for orbs in self.site_orbitals:
            names = []
            for sub in orbs:
                fam = SKFitter.FAMILY_MAP[sub]
                info = SKFitter.SUB_ORBITALS[sub]
                if info.get("labels"):                       # groups that are combinations (cubic f)
                    names += [(lab, ell[fam]) for lab in info["labels"]]
                else:
                    names += [(full[fam][k], ell[fam]) for k in info["indices"]]
            out.append(names)
        return out

    def orbital_labels(self, i: int) -> list:
        names = [n for n, _ in self._orbital_names[i]]
        return [n + "↑" for n in names] + [n + "↓" for n in names] if self.spinful else names

    def orbital_info(self, i: int) -> list:
        """Per orbital of site i: {"label", "l", "spin" ("up"/"down", None if spinless)}."""
        names = self._orbital_names[i]
        ls = [l for _, l in names] * (2 if self.spinful else 1)
        spins = (["up"] * len(names) + ["down"] * len(names)) if self.spinful else [None] * len(names)
        return [{"label": lab, "l": l, "spin": sp} for lab, l, sp in zip(self.orbital_labels(i), ls, spins)]

    @property
    def conserves_sz(self) -> bool:
        """True when spin along z is a good quantum number (a spinful model without spin-orbit coupling)."""
        return self.spinful and not self.soc

    def orbital_centres(self, i: int) -> np.ndarray:
        """Cartesian centres (Å) of the orbitals of site i: SK orbitals sit on their atom."""
        return np.tile(self.cart_coords[i], (self.norbs(i), 1))

    def to_tb_model(self):
        """This model written out as numbers: a skfitter.model.TBModel of kind "sk" with the same blocks (and
        overlap blocks for a non-orthogonal fit), so H(k) and S(k) are the same at every k."""
        from .model import TBModel
        hops, ovl = {}, ({} if self.non_orthogonal else None)
        for b in self.bonds:
            key = (b["i"], b["j"], tuple(b["R"]))
            if key in hops:
                raise RuntimeError(f"Two bonds of the model share the hopping {key}.")
            hops[key] = self.hopping(b["i"], b["j"], b["d_vec"])
            if ovl is not None:
                ovl[key] = self.hopping(b["i"], b["j"], b["d_vec"], overlap=True)
        return TBModel(lattice=self.lattice, species=self.species, frac_coords=self.frac_coords,
                       site_orbitals=[self.orbital_labels(i) for i in range(self.n_sites)],
                       onsite=[self.onsite(i) for i in range(self.n_sites)], hoppings=hops, spinful=self.spinful,
                       soc=self.soc, is_magnetic=self.is_magnetic, efermi=self.efermi, source=self.source, kind="sk",
                       orbital_l=[[x["l"] for x in self.orbital_info(i)] for i in range(self.n_sites)],
                       overlaps=ovl, provenance=self.provenance + [{"step": "written out as numbers"}])

    # ------------------------------------------------------------------ SKFitter's block code
    @cached_property
    def fitter(self):
        """An SKFitter that carries this model's attributes and no DFT data, for SKFitter's block methods."""
        from skfitter.core import SKFitter
        f = object.__new__(SKFitter)
        f.structure = self.structure
        f.site_orbitals = self.site_orbitals
        offsets, off = [], 0
        for dim in self.spatial_dims:
            offsets.append(off)
            off += dim
        f.site_offsets = offsets
        f.site_dims = list(self.spatial_dims)
        f.base_dim = off
        f.total_dim = 2 * off if self.spinful else off
        # on-site terms are keyed by the fit's equivalent-site label of the parent site
        f.eq_site_map = self.eq_site_map
        f.soc = self.soc
        f.is_magnetic = self.is_magnetic
        f.non_orthogonal = self.non_orthogonal
        f.n_neighbors = self.n_neighbors
        f.d0 = self.d0
        f.hopping_mode = self.hopping_mode
        f.shell_distances = list(self.shell_distances)
        f.pair_shells = self.shells_of_pairs
        f.cutoff_radius = self.cutoff
        f.site_moment_signs = self.moment_signs
        f.site_moment_dirs = [np.asarray(d, float) for d in self.moment_dirs] if self.moment_dirs else None
        f.neighbor_list = []                      # build_hamiltonian then gives the on-site terms only
        f.onsite_mix_terms = [list(site.items()) for site in self.onsite_mix] if self.onsite_mix else None
        return f

    @cached_property
    def _onsite_blocks(self) -> list:
        f = self.fitter
        full = f.build_hamiltonian(np.zeros(3), self.params)          # on-site energies + SOC + exchange
        N = f.base_dim
        blocks = []
        for i in range(self.n_sites):
            idx = list(range(f.site_offsets[i], f.site_offsets[i] + self.spatial_dims[i]))
            if self.spinful:
                idx = idx + [N + k for k in idx]
            blocks.append(np.ascontiguousarray(full[np.ix_(idx, idx)]))
        return blocks

    def onsite(self, i: int) -> np.ndarray:
        """On-site block of site i (norbs x norbs): on-site energies, spin-orbit coupling and exchange."""
        return self._onsite_blocks[i].copy()

    def _raw_block(self, i, j, d_vec, overlap):
        f = self.fitter
        ei, ej = self.species[i], self.species[j]
        h = np.zeros((self.spatial_dims[i], self.spatial_dims[j]), dtype=complex)
        li = 0
        for s1 in self.site_orbitals[i]:
            d1 = f.SUB_ORBITALS[s1]["dim"]
            lj = 0
            for s2 in self.site_orbitals[j]:
                d2 = f.SUB_ORBITALS[s2]["dim"]
                h[li:li + d1, lj:lj + d2] += f._get_submatrix_hopping(ei, s1, ej, s2, np.asarray(d_vec, float),
                                                                       self.params, is_overlap=overlap)
                lj += d2
            li += d1
        return h

    def hopping(self, i: int, j: int, d_vec, overlap: bool = False) -> np.ndarray:
        """Hopping block <i| H |j at r_i + d_vec> (or the overlap block, ``overlap=True``), norbs_i x norbs_j."""
        d_vec = np.asarray(d_vec, dtype=float)
        h = 0.5 * (self._raw_block(i, j, d_vec, overlap) + self._raw_block(j, i, -d_vec, overlap).conj().T)
        return np.kron(np.eye(2), h) if self.spinful else h

    # ------------------------------------------------------------------ bonds
    @cached_property
    def bonds(self) -> list:
        """Bond templates of the fitted cell: dicts {i, j, d_vec, R} with r_j + R . lattice = r_i + d_vec.
        The same bond list SKFitter uses (neighbour_pairs with the fit's cutoff), both directions."""
        from skfitter.core import neighbour_pairs
        pairs, _ = neighbour_pairs(self.structure, self.cutoff)
        inv = np.linalg.inv(self.lattice)
        fc = self.structure.frac_coords
        out = []
        for p in pairs:
            i, j, d = p["i"], p["j"], np.asarray(p["d_vec"], float)
            R = fc[i] + d @ inv - fc[j]
            Ri = np.rint(R).astype(int)
            if np.abs(R - Ri).max() > 1e-6:
                raise ValueError(f"Bond {i}->{j} does not end on a lattice translate of site {j}.")
            out.append({"i": i, "j": j, "d_vec": d, "R": tuple(int(x) for x in Ri)})
        return out

    def bloch(self, k_cart, overlap: bool = False) -> np.ndarray:
        """H(k) (or S(k)) of the bulk model from the bond templates, sites in order, exp(i k . d) phases."""
        k = np.asarray(k_cart, dtype=float)
        dims = [self.norbs(i) for i in range(self.n_sites)]
        offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
        H = np.zeros((offs[-1], offs[-1]), dtype=complex)
        for i in range(self.n_sites):
            H[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] += np.eye(dims[i]) if overlap else self.onsite(i)
        for b in self.bonds:
            i, j = b["i"], b["j"]
            H[offs[i]:offs[i + 1], offs[j]:offs[j + 1]] += self.hopping(i, j, b["d_vec"], overlap) * np.exp(
                1j * k @ b["d_vec"])
        return 0.5 * (H + H.conj().T)

    # ------------------------------------------------------------------ supercells
    def supercell(self, matrix) -> "SKModel":
        """The same model on a supercell with lattice vectors ``matrix @ lattice`` (integer 3x3 matrix, or 3
        integers for a diagonal one). Sites keep the on-site terms and moment directions of their parent site."""
        M = np.array(matrix, dtype=int)
        if M.shape == (3,):
            M = np.diag(M)
        det = int(round(abs(np.linalg.det(M))))
        if det < 1:
            raise ValueError("Supercell matrix must be non-singular.")
        new_lat = M @ self.lattice
        inv_new = np.linalg.inv(new_lat)
        corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]) @ M
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        species, frac, orbs, eq, signs, parents, mix, dirs = [], [], [], [], [], [], [], []
        cart0 = self.cart_coords
        for n1 in range(lo[0], hi[0] + 1):
            for n2 in range(lo[1], hi[1] + 1):
                for n3 in range(lo[2], hi[2] + 1):
                    shift = np.array([n1, n2, n3]) @ self.lattice
                    for s in range(self.n_sites):
                        f = (cart0[s] + shift) @ inv_new
                        if np.all(f > -1e-8) and np.all(f < 1 - 1e-8):
                            species.append(self.species[s]); frac.append(np.clip(f, 0.0, None))
                            orbs.append(self.site_orbitals[s]); eq.append(self.eq_site_map[s])
                            signs.append(self.moment_signs[s]); parents.append(self.parent_sites[s])
                            mix.append(self.onsite_mix[s] if self.onsite_mix else None)
                            dirs.append(self.moment_dirs[s] if self.moment_dirs else None)
        if len(species) != det * self.n_sites:
            raise RuntimeError(f"Supercell enumeration found {len(species)} sites, expected {det * self.n_sites}.")
        return SKModel(lattice=new_lat, species=species, frac_coords=frac, site_orbitals=orbs, eq_site_map=eq,
                       params=self.params, cutoff=self.cutoff, d0=self.d0, n_neighbors=self.n_neighbors,
                       soc=self.soc, is_magnetic=self.is_magnetic, non_orthogonal=self.non_orthogonal,
                       moment_signs=signs, efermi=self.efermi, parent_sites=parents,
                       source={**self.source, "supercell": M.tolist()},
                       hopping_mode=self.hopping_mode, shell_distances=self.shell_distances,
                       pair_shells=self.shells_of_pairs, onsite_mix=mix if self.onsite_mix else None,
                       moment_dirs=dirs if self.moment_dirs else None)


def dft_path_points(fitter) -> list:
    """The fit's DFT k-path as [[label, k_frac], ..., None (a jump), ...], or [] when the fit has none."""
    try:
        pieces = fitter._kpath_pieces()
    except Exception:  # noqa: BLE001 - a fit without a k-path
        return []
    pts = []
    for _, _, nodes, labels in pieces:
        if pts:
            pts.append(None)
        pts += [[str(lab), [float(x) for x in node]] for node, lab in zip(nodes, labels)]
    return pts


def model_path(directory: str, seed: str) -> str:
    return os.path.join(directory, f"{seed}{MODEL_SUFFIX}")
