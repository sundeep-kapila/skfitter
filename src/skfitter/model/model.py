"""TBModel: a general real-space tight-binding model - on-site blocks per site and hopping blocks per (i, j, R).

This is the model type that holds numbers only (no parameters): SKFitter models written out as numbers (kind
``sk``) and models read from other programs. SKFitter's parameterised models keep their own class
(``skfitter.model.sk.SKModel``); both offer the same interface (``skfitter.model.protocol.TightBindingModel``).

Conventions (shared by both model classes)
------------------------------------------
* Energies in eV relative to the Fermi energy (``efermi`` records the DFT value that was subtracted).
* Lengths in Å; ``lattice`` rows are the lattice vectors; ``frac_coords`` the fractional site positions.
* Orbitals of a spinful site: all spin-up orbitals first, then all spin-down orbitals.
* A hopping (i, j, R) is <i, cell 0 | H | j, cell R>; the bond vector is r_j + R . lattice - r_i.
* Each orbital has a centre: its site position plus an offset (Å). SK orbitals sit on their atom (offset 0);
  orbitals of other models may sit elsewhere.
* Non-orthogonal models carry overlap hopping blocks S(i, j, R); the on-site overlap is the identity.

File: ``prayog-tb-model`` version 2 (version 1 files, written before October 2026, are read too). Version 2 adds
``orbitals`` (labels, angular momentum, spin, centre offsets), ``overlap``, ``provenance``, ``quality``,
``matrices`` (``spin``: sigma(i, j, R) of spinor orbitals; ``position``: r(i, j, R) = <i, 0 | r | j, R> in Å,
whose on-site diagonal is the orbital centres; both written when present) and ``modifiers``
(a modified model is saved as ``prayog-modified-model`` instead: base model + modifiers, see modifiers.py).
"""
from __future__ import annotations

import json
import os
from functools import cached_property

import numpy as np

TB_FORMAT = "prayog-tb-model"
TB_VERSION = 2
KINDS = ("wannier", "sk", "upload", "tb", "continuum")


def _spins(n: int, spinful: bool) -> list:
    if not spinful:
        return [None] * n
    if n % 2:
        raise ValueError("A spinful site needs an even number of orbitals (spin up, then spin down).")
    return ["up"] * (n // 2) + ["down"] * (n // 2)


VECTOR_MATRICES = ("spin", "position")                     # (3, ni, nj) per block: x, y, z


def _matrices_of(m):
    """The file's matrices -> {name: {(i, j, R): array}} (spin, position: (3, ni, nj) from their x, y, z parts)."""
    if not m:
        return None
    out = {}
    for name, rows in m.items():
        if name in VECTOR_MATRICES:
            try:
                out[name] = {(int(r["i"]), int(r["j"]), tuple(int(x) for x in r["R"])):
                             np.array([_cplx(r[a]) for a in "xyz"]) for r in rows}
            except (TypeError, KeyError, ValueError):
                raise ValueError(f"The model file's {name} matrices are malformed (rows {{i, j, R, x, y, z}}).") \
                    from None
        else:
            out[name] = rows
    return out


def _vector_matrices_of(m):
    """The spin and position matrices of any model of the common interface, for a copy of it (None: none)."""
    out = {}
    for name in VECTOR_MATRICES:
        v = getattr(m, f"{name}_matrices", None)
        if v:
            out[name] = v
    return out or None


def _cplx(d) -> np.ndarray:
    return np.asarray(d["re"], dtype=float) + 1j * np.asarray(d["im"], dtype=float)


def _rc(b: np.ndarray) -> dict:
    return {"re": b.real.round(10).tolist(), "im": b.imag.round(10).tolist()}


class TBModel:
    """A tight-binding model given by its blocks. ``hoppings``: {(i, j, R): block}; the reverse direction is
    added (or averaged) so that H stays Hermitian. ``overlaps`` (non-orthogonal models): the same for S."""

    def __init__(self, *, lattice, species, frac_coords, site_orbitals, onsite, hoppings, spinful=False, soc=False,
                 is_magnetic=False, efermi=0.0, source=None, kind="tb", orbital_offsets=None, orbital_l=None,
                 overlaps=None, provenance=None, quality=None, matrices=None, modifiers=None):
        self.lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
        self.species = [str(s) for s in species]
        self.frac_coords = np.asarray(frac_coords, dtype=float).reshape(-1, 3)
        self.site_orbitals = [list(o) for o in site_orbitals]
        self._onsite = [np.asarray(b, dtype=complex) for b in onsite]
        self.spinful = bool(spinful)
        self.soc = bool(soc)
        self.is_magnetic = bool(is_magnetic)
        self.efermi = float(efermi)
        self.source = dict(source or {})
        self.kind = str(kind)
        self.n_neighbors = 0
        self.provenance = [dict(p) for p in (provenance or [])]
        self.quality = dict(quality or {})
        if modifiers:
            raise ValueError("Model files with modifiers (fields, shifts) need a newer skfitter.")
        self.modifiers = []
        self.matrices = {}
        unknown = set(matrices or {}) - set(VECTOR_MATRICES)
        if unknown:
            raise ValueError(f"Model files with {', '.join(sorted(unknown))} matrices need a newer skfitter.")
        n = len(self.species)
        if not (len(self.frac_coords) == len(self.site_orbitals) == len(self._onsite) == n):
            raise ValueError("TBModel: one species, position, orbital list and on-site block per site.")
        for i, b in enumerate(self._onsite):
            if b.ndim != 2 or b.shape[0] != b.shape[1]:
                raise ValueError(f"TBModel: the on-site block of site {i} is not square.")
            self._onsite[i] = 0.5 * (b + b.conj().T)
        # orbital centres (offsets from the site) and angular momenta; spins follow from the spin-up-first order
        offs = orbital_offsets if orbital_offsets is not None else [None] * n
        self.orbital_offsets = []
        for i in range(n):
            o = np.zeros((self.norbs(i), 3)) if offs[i] is None else np.asarray(offs[i], dtype=float).reshape(-1, 3)
            if len(o) != self.norbs(i):
                raise ValueError(f"TBModel: site {i} has {self.norbs(i)} orbitals but {len(o)} centre offsets.")
            self.orbital_offsets.append(o)
        ls = orbital_l if orbital_l is not None else [None] * n
        self.orbital_l = []
        for i in range(n):
            li = [None] * self.norbs(i) if ls[i] is None else [None if x is None else int(x) for x in ls[i]]
            if len(li) != self.norbs(i):
                raise ValueError(f"TBModel: site {i} has {self.norbs(i)} orbitals but {len(li)} angular momenta.")
            self.orbital_l.append(li)
        for i in range(n):
            _spins(self.norbs(i), self.spinful)                     # checks the even count
        self._hop = self._hermitian(hoppings, "hopping")
        self.non_orthogonal = overlaps is not None
        self._ovl = self._hermitian(overlaps or {}, "overlap") if self.non_orthogonal else {}
        for key, blk in self._ovl.items():                       # every overlap pair is also a bond of H
            self._hop.setdefault(key, np.zeros_like(blk))
        if (matrices or {}).get("spin"):
            self.matrices["spin"] = self._spin_blocks(matrices["spin"])
        if (matrices or {}).get("position"):
            pos = self._spin_blocks(matrices["position"], "position")
            for i in range(n):                                   # on-site diagonal: the orbital centres (home cell)
                key = (i, i, (0, 0, 0))
                b = pos.setdefault(key, np.zeros((3, self.norbs(i), self.norbs(i)), complex)).copy()
                c = self.orbital_centres(i)
                for a in range(3):
                    np.fill_diagonal(b[a], c[:, a])
                pos[key] = b
            self.matrices["position"] = pos

    def _spin_blocks(self, blocks: dict, what: str = "spin") -> dict:
        """sigma(i, j, R) (or r(i, j, R)): (3, norbs i, norbs j) per block, on-site ones (i, i, 0) included; made
        Hermitian (each direction from the average of it and the conjugate of the other)."""
        out = {}
        for (i, j, R), blk in blocks.items():
            R = tuple(int(x) for x in R)
            b = np.asarray(blk, dtype=complex)
            if b.shape != (3, self.norbs(int(i)), self.norbs(int(j))):
                raise ValueError(f"TBModel: {what} block ({i}, {j}, {R}) has shape {b.shape}")
            out[(int(i), int(j), R)] = b
        for (i, j, R), b in list(out.items()):
            back = (j, i, tuple(-x for x in R))
            bb = out.get(back)
            other = np.zeros_like(b) if bb is None else bb.transpose(0, 2, 1).conj()
            avg = 0.5 * (b + other) if bb is not None else b
            out[(i, j, R)] = avg
            out[back] = avg.transpose(0, 2, 1).conj()
        return out

    @property
    def spin_matrices(self):
        """{(i, j, R): (3, ni, nj)} sigma blocks of spinor Wannier functions (None when the model has none: then the
        spin of a spinful model is its basis - spin-up orbitals first, then spin-down)."""
        return self.matrices.get("spin")

    @property
    def position_matrices(self):
        """{(i, j, R): (3, ni, nj)} r = <i, 0 | r | j, R> (Å) of Wannier functions (None when the model has none: then
        the position operator is diagonal, the orbital centres)."""
        return self.matrices.get("position")

    def _hermitian(self, blocks: dict, what: str) -> dict:
        out = {}
        for (i, j, R), blk in blocks.items():
            R = tuple(int(x) for x in R)
            i, j = int(i), int(j)
            if i == j and R == (0, 0, 0):
                continue
            blk = np.asarray(blk, dtype=complex)
            if blk.shape != (self.norbs(i), self.norbs(j)):
                raise ValueError(f"TBModel: {what} block ({i}, {j}, {R}) has shape {blk.shape}, expected "
                                 f"{(self.norbs(i), self.norbs(j))}.")
            out[(i, j, R)] = blk
        for (i, j, R), blk in list(out.items()):
            back = (j, i, tuple(-x for x in R))
            other = out.get(back)
            avg = blk if other is None else 0.5 * (blk + other.conj().T)
            out[(i, j, R)] = avg
            out[back] = avg.conj().T
        return out

    # -- sizes and geometry -------------------------------------------------------------------------------------
    @property
    def n_sites(self) -> int:
        return len(self.species)

    def norbs(self, i: int) -> int:
        return int(self._onsite[i].shape[0])

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    def orbital_centres(self, i: int) -> np.ndarray:
        """Cartesian centres (Å) of the orbitals of site i, in the site's home cell: (norbs, 3)."""
        return self.cart_coords[i] + self.orbital_offsets[i]

    def orbital_labels(self, i: int) -> list:
        return [str(x) for x in self.site_orbitals[i]] if len(self.site_orbitals[i]) == self.norbs(i) else \
            [f"o{k + 1}" for k in range(self.norbs(i))]

    def orbital_info(self, i: int) -> list:
        """Per orbital of site i: {"label", "l" (None if unknown), "spin" ("up"/"down", None if spinless)}."""
        return [{"label": lab, "l": l, "spin": s} for lab, l, s in
                zip(self.orbital_labels(i), self.orbital_l[i], _spins(self.norbs(i), self.spinful))]

    # -- blocks -------------------------------------------------------------------------------------------------
    def onsite(self, i: int) -> np.ndarray:
        return self._onsite[i].copy()

    def _R_of(self, i, j, d_vec):
        f = (self.cart_coords[i] + np.asarray(d_vec, float) - self.cart_coords[j]) @ np.linalg.inv(self.lattice)
        return tuple(int(x) for x in np.rint(f))

    def hopping(self, i: int, j: int, d_vec, overlap: bool = False) -> np.ndarray:
        """<i| H |j at r_i + d_vec> (or the overlap block with ``overlap=True``; zero for orthogonal models)."""
        key = (i, j, self._R_of(i, j, d_vec))
        if overlap:
            blk = self._ovl.get(key)
            return np.zeros((self.norbs(i), self.norbs(j)), dtype=complex) if blk is None else blk.copy()
        return self._hop[key].copy()

    def hopping_R(self, i: int, j: int, R, overlap: bool = False) -> np.ndarray:
        """The same block addressed by the cell R instead of the bond vector."""
        key = (int(i), int(j), tuple(int(x) for x in R))
        src = self._ovl if overlap else self._hop
        if key not in src:
            return np.zeros((self.norbs(i), self.norbs(j)), dtype=complex)
        return src[key].copy()

    @cached_property
    def bonds(self) -> list:
        cart = self.cart_coords
        return [{"i": i, "j": j, "R": R, "d_vec": cart[j] + np.array(R) @ self.lattice - cart[i]}
                for (i, j, R) in sorted(self._hop)]

    @property
    def reach(self) -> list:
        """Largest |R| of a hopping along each lattice vector (a lead must be at least that many cells long)."""
        if not self._hop:
            return [0, 0, 0]
        return np.abs(np.array([R for (_, _, R) in self._hop])).max(axis=0).astype(int).tolist()

    @property
    def n_hoppings(self) -> int:
        return len(self._hop) // 2

    def bloch(self, k_cart, overlap: bool = False) -> np.ndarray:
        """H(k) (or S(k)), sites in order, phases exp(i k . d) with d the vector between the sites."""
        k = np.asarray(k_cart, dtype=float)
        dims = [self.norbs(i) for i in range(self.n_sites)]
        offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
        if overlap and not self.non_orthogonal:
            return np.eye(offs[-1], dtype=complex)
        n = int(offs[-1])
        H = np.eye(n, dtype=complex) if overlap else np.zeros((n, n), dtype=complex)
        if not overlap:
            for i in range(self.n_sites):
                H[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] += self._onsite[i]
        idx, vals, dvec = self._bloch_plan(overlap, offs)
        if len(idx):                                         # all blocks at once (a Wannier model has thousands)
            w = vals * np.exp(1j * (dvec @ k))
            H += (np.bincount(idx, weights=w.real, minlength=n * n)
                  + 1j * np.bincount(idx, weights=w.imag, minlength=n * n)).reshape(n, n)
        return 0.5 * (H + H.conj().T)

    def _bloch_plan(self, overlap: bool, offs) -> tuple:
        """Every element of every hopping (or overlap) block as flat arrays: index row * n + col, value, and the
        bond vector d - built once (rebuilt if blocks were added)."""
        src = self._ovl if overlap else self._hop
        attr = "_plan_ovl" if overlap else "_plan_hop"
        cached = self.__dict__.get(attr)
        if cached is not None and cached[0] == len(src) and cached[1] is src:
            return cached[2]
        n = int(offs[-1])
        cart, lat = self.cart_coords, self.lattice
        idx, vals, dvec = [], [], []
        for (i, j, R), blk in src.items():
            ni, nj = blk.shape
            r = offs[i] + np.arange(ni)
            c = offs[j] + np.arange(nj)
            idx.append((r[:, None] * n + c[None, :]).ravel())
            vals.append(np.asarray(blk, complex).ravel())
            dvec.append(np.broadcast_to(cart[j] + np.asarray(R) @ lat - cart[i], (ni * nj, 3)))
        plan = ((np.concatenate(idx), np.concatenate(vals), np.concatenate(dvec)) if idx
                else (np.zeros(0, int), np.zeros(0, complex), np.zeros((0, 3))))
        self.__dict__[attr] = (len(src), src, plan)
        return plan

    def to_tb_model(self) -> "TBModel":
        """Already numbers only (the same call on an SKModel writes it out)."""
        return self

    @property
    def conserves_sz(self) -> bool:
        """True when spin along z is a good quantum number (a spinful model without spin-orbit coupling)."""
        return self.spinful and not self.soc

    @classmethod
    def from_model(cls, m, *, kind=None, provenance_step=None) -> "TBModel":
        """Any model of the common interface written out as numbers (its blocks, orbitals and provenance)."""
        hops, ovl = {}, ({} if m.non_orthogonal else None)
        for b in m.bonds:
            key = (b["i"], b["j"], tuple(int(x) for x in b["R"]))
            hops[key] = m.hopping(b["i"], b["j"], b["d_vec"])
            if ovl is not None:
                ovl[key] = m.hopping(b["i"], b["j"], b["d_vec"], overlap=True)
        n = m.n_sites
        prov = list(m.provenance) + ([provenance_step] if provenance_step else [])
        return cls(lattice=m.lattice, species=m.species, frac_coords=m.frac_coords,
                   site_orbitals=[m.orbital_labels(i) for i in range(n)], onsite=[m.onsite(i) for i in range(n)],
                   hoppings=hops, spinful=m.spinful, soc=m.soc, is_magnetic=m.is_magnetic, efermi=m.efermi,
                   source=m.source, kind=kind or m.kind,
                   orbital_offsets=[np.asarray(m.orbital_centres(i)) - m.cart_coords[i] for i in range(n)],
                   orbital_l=[[x["l"] for x in m.orbital_info(i)] for i in range(n)], overlaps=ovl,
                   provenance=prov, matrices=_vector_matrices_of(m))

    # -- supercells ---------------------------------------------------------------------------------------------
    def supercell(self, matrix) -> "TBModel":
        """The same model on the supercell with lattice vectors ``matrix @ lattice``."""
        M = np.array(matrix, dtype=int)
        if M.shape == (3,):
            M = np.diag(M)
        det = int(round(abs(np.linalg.det(M))))
        if det < 1:
            raise ValueError("Supercell matrix must be non-singular.")
        new_lat = M @ self.lattice
        inv_new = np.linalg.inv(new_lat)
        corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]) @ M
        lo, hi = corners.min(axis=0) - 1, corners.max(axis=0) + 1
        cart = self.cart_coords
        # each site's copies whose positions fall inside the new cell (as SKModel.supercell picks them), so a
        # device cut from either model has the same atoms and edges
        index, species, frac, orbs, onsite, offsets, ls = {}, [], [], [], [], [], []
        for n1 in range(lo[0], hi[0] + 1):
            for n2 in range(lo[1], hi[1] + 1):
                for n3 in range(lo[2], hi[2] + 1):
                    L = (n1, n2, n3)
                    for s in range(self.n_sites):
                        f = (cart[s] + np.array(L) @ self.lattice) @ inv_new
                        if np.all(f > -1e-8) and np.all(f < 1 - 1e-8):
                            index[(s, L)] = len(species)
                            species.append(self.species[s])
                            frac.append(np.clip(f, 0.0, None))
                            orbs.append(self.site_orbitals[s])
                            onsite.append(self._onsite[s])
                            offsets.append(self.orbital_offsets[s])
                            ls.append(self.orbital_l[s])
        if len(species) != det * self.n_sites:
            raise RuntimeError(f"Supercell enumeration found {len(species)} sites, expected {det * self.n_sites}.")

        def place(b, n):
            """Old site b in old cell n -> (new site index, new cell N)."""
            f = (cart[b] + np.array(n) @ self.lattice) @ inv_new
            N = np.floor(f + 1e-8).astype(int)
            L = tuple(int(x) for x in np.array(n) - N @ M)
            return index[(b, L)], tuple(int(x) for x in N)

        def remap(blocks):
            out = {}
            for (a, b, R), blk in blocks.items():
                for (s, L), i in index.items():
                    if s != a:
                        continue
                    j, N = place(b, np.array(L) + np.array(R))
                    out[(i, j, N)] = blk
            return out
        return TBModel(lattice=new_lat, species=species, frac_coords=frac, site_orbitals=orbs, onsite=onsite,
                       hoppings=remap(self._hop), spinful=self.spinful, soc=self.soc, is_magnetic=self.is_magnetic,
                       efermi=self.efermi, source={**self.source, "supercell": M.tolist()}, kind=self.kind,
                       orbital_offsets=offsets, orbital_l=ls,
                       overlaps=remap(self._ovl) if self.non_orthogonal else None,
                       provenance=self.provenance + [{"step": "supercell", "matrix": M.tolist()}],
                       quality=self.quality,
                       matrices={k: remap(v) for k, v in self.matrices.items() if v} or None)

    # -- files --------------------------------------------------------------------------------------------------
    def _block_list(self, blocks: dict) -> list:
        out = []
        for (i, j, R), blk in sorted(blocks.items()):
            if (i, j, R) > (j, i, tuple(-x for x in R)):
                continue                                         # one direction; the other is its conjugate
            out.append({"i": i, "j": j, "R": list(R), **_rc(blk)})
        return out

    def to_dict(self) -> dict:
        d = {"format": TB_FORMAT, "version": TB_VERSION, "kind": self.kind,
             "lattice": self.lattice.tolist(), "species": self.species, "frac_coords": self.frac_coords.tolist(),
             "site_orbitals": self.site_orbitals,
             "orbitals": [{"labels": self.orbital_labels(i), "l": self.orbital_l[i],
                           "offsets": self.orbital_offsets[i].round(8).tolist()} for i in range(self.n_sites)],
             "onsite": [_rc(b) for b in self._onsite],
             "hoppings": self._block_list(self._hop),
             "settings": {"spinful": self.spinful, "soc": self.soc, "is_magnetic": self.is_magnetic,
                          "non_orthogonal": self.non_orthogonal},
             "efermi": self.efermi, "source": self.source, "provenance": self.provenance, "quality": self.quality,
             "matrices": self._matrices_dict(), "modifiers": []}
        if self.non_orthogonal:
            d["overlap"] = {"hoppings": self._block_list(self._ovl)}
        return d

    def _matrices_dict(self) -> dict:
        out = {}
        for name in VECTOR_MATRICES:
            sp = self.matrices.get(name)
            if not sp:
                continue
            rows = []
            for (i, j, R), b in sorted(sp.items()):
                if (i, j, R) > (j, i, tuple(-x for x in R)):
                    continue
                rows.append({"i": i, "j": j, "R": list(R), **{a: _rc(b[n]) for n, a in enumerate("xyz")}})
            out[name] = rows
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "TBModel":
        if d.get("format") != TB_FORMAT:
            raise ValueError(f"Not a tight-binding model file (format {d.get('format')!r}).")
        version = int(d.get("version", 0))
        if version > TB_VERSION:
            raise ValueError(f"Model file version {version} is newer than this program (reads up to {TB_VERSION}).")
        st = d["settings"]

        def blocks(rows):
            return {(h["i"], h["j"], tuple(h["R"])): _cplx(h) for h in rows}
        orbitals = d.get("orbitals") or []                       # version 1: none (centres on the sites)
        ovl = d.get("overlap")
        if st.get("non_orthogonal") and ovl is None:
            raise ValueError("The model file says non_orthogonal but has no overlap blocks.")
        return cls(lattice=d["lattice"], species=d["species"], frac_coords=d["frac_coords"],
                   site_orbitals=d["site_orbitals"], onsite=[_cplx(b) for b in d["onsite"]],
                   hoppings=blocks(d["hoppings"]), spinful=st["spinful"], soc=st["soc"],
                   is_magnetic=st["is_magnetic"], efermi=d.get("efermi", 0.0), source=d.get("source"),
                   kind=d.get("kind", "tb"),
                   orbital_offsets=[o.get("offsets") for o in orbitals] if orbitals else None,
                   orbital_l=[o.get("l") for o in orbitals] if orbitals else None,
                   overlaps=blocks(ovl["hoppings"]) if ovl is not None else None,
                   provenance=d.get("provenance"), quality=d.get("quality"),
                   matrices=_matrices_of(d.get("matrices")), modifiers=d.get("modifiers"))

    def save(self, path: str) -> str:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        return path

    @classmethod
    def load(cls, path: str) -> "TBModel":
        from .io import read_json
        return cls.from_dict(read_json(path))
