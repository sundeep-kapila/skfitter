"""The common interface of SKFitter's tight-binding models, and a check that a model honours it.

Every model class (``SKModel``: an SKFitter fit with its parameters; ``TBModel``: numbers only - SK models
written out, models of other programs) offers what is listed in :class:`TightBindingModel`; the
properties (``skfitter.properties``) use this interface, so any model works with them.

    from skfitter.model import load_model, check_model
    model = load_model("silicon_w90_model.json")
    problems = check_model(model)          # [] when the model is consistent
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TightBindingModel(Protocol):
    """What every model offers (attributes and methods). Energies in eV (E_F = 0), lengths in Å."""

    kind: str                     # "sk" (SKFitter), "upload", "tb" (other kinds are read too)
    lattice: np.ndarray           # (3, 3), rows are the lattice vectors
    species: list                 # element of each site
    frac_coords: np.ndarray       # (n_sites, 3) fractional site positions
    site_orbitals: list           # per site: SKFitter sub-shells (SKModel) or orbital labels (TBModel)
    spinful: bool                 # spin-doubled basis: spin-up orbitals of a site first, then spin-down
    soc: bool
    is_magnetic: bool
    non_orthogonal: bool          # carries an overlap S (hopping(..., overlap=True), bloch(k, overlap=True))
    efermi: float                 # the DFT Fermi energy that was subtracted
    source: dict                  # free-form record of the inputs
    n_neighbors: int              # SK neighbour shells (0 for TBModel)

    @property
    def n_sites(self) -> int: ...

    @property
    def cart_coords(self) -> np.ndarray: ...

    @property
    def reach(self) -> list: ...

    @property
    def bonds(self) -> list: ...                                 # [{"i", "j", "R", "d_vec"}], both directions

    @property
    def provenance(self) -> list: ...                            # [{"step": ...}, ...] oldest first

    def norbs(self, i: int) -> int: ...

    def orbital_labels(self, i: int) -> list: ...

    def orbital_info(self, i: int) -> list: ...                  # [{"label", "l", "spin"}]

    def orbital_centres(self, i: int) -> np.ndarray: ...         # (norbs, 3) Cartesian, Å

    def onsite(self, i: int) -> np.ndarray: ...

    def hopping(self, i: int, j: int, d_vec, overlap: bool = False) -> np.ndarray: ...

    def bloch(self, k_cart, overlap: bool = False) -> np.ndarray: ...

    def supercell(self, matrix): ...

    def to_tb_model(self): ...                                   # the model as numbers only (a TBModel)

    def to_dict(self) -> dict: ...

    def save(self, path: str) -> str: ...


def check_model(model, k_points=None, tol: float = 1e-10) -> list:
    """Problems with a model as text ([] if none): the interface is complete, blocks have the sizes the orbitals
    say, H is Hermitian in real space and at k, labels and centres match the orbitals, and a written-out copy of
    the model (to_dict / from_dict) has the same H(k)."""
    problems = []
    if not isinstance(model, TightBindingModel):
        missing = [n for n in ("kind", "lattice", "species", "frac_coords", "site_orbitals", "spinful", "soc",
                               "is_magnetic", "non_orthogonal", "efermi", "source", "n_neighbors", "n_sites",
                               "cart_coords", "reach", "bonds", "provenance", "norbs", "orbital_labels",
                               "orbital_info", "orbital_centres", "onsite", "hopping", "bloch", "supercell",
                               "to_tb_model", "to_dict", "save") if not hasattr(model, n)]
        return [f"missing from the interface: {', '.join(missing)}"]
    n = model.n_sites
    for i in range(n):
        d = model.norbs(i)
        if model.onsite(i).shape != (d, d):
            problems.append(f"site {i}: on-site block {model.onsite(i).shape}, norbs {d}")
        elif not np.allclose(model.onsite(i), model.onsite(i).conj().T, atol=tol):
            problems.append(f"site {i}: on-site block not Hermitian")
        if len(model.orbital_labels(i)) != d:
            problems.append(f"site {i}: {len(model.orbital_labels(i))} labels for {d} orbitals")
        info = model.orbital_info(i)
        if len(info) != d or any(set(x) != {"label", "l", "spin"} for x in info):
            problems.append(f"site {i}: orbital_info malformed")
        elif model.spinful and [x["spin"] for x in info] != ["up"] * (d // 2) + ["down"] * (d // 2):
            problems.append(f"site {i}: spinful orbitals are not spin-up first, then spin-down")
        if np.asarray(model.orbital_centres(i)).shape != (d, 3):
            problems.append(f"site {i}: orbital_centres shape {np.asarray(model.orbital_centres(i)).shape}")
    for b in model.bonds:
        i, j, dv = b["i"], b["j"], np.asarray(b["d_vec"], float)
        expect = model.cart_coords[j] + np.asarray(b["R"]) @ model.lattice - model.cart_coords[i]
        if not np.allclose(expect, dv, atol=1e-6):
            problems.append(f"bond {i}->{j} R={tuple(b['R'])}: d_vec does not match the positions")
            continue
        for ovl in ((False, True) if model.non_orthogonal else (False,)):
            h = model.hopping(i, j, dv, overlap=ovl)
            if h.shape != (model.norbs(i), model.norbs(j)):
                problems.append(f"bond {i}->{j}: block shape {h.shape}")
            elif not np.allclose(h, model.hopping(j, i, -dv, overlap=ovl).conj().T, atol=tol):
                problems.append(f"bond {i}->{j} R={tuple(b['R'])}: {'S' if ovl else 'H'} not Hermitian in "
                                "real space")
    rng = np.random.default_rng(7)
    ks = np.asarray(k_points) if k_points is not None else rng.normal(size=(4, 3))
    rec = 2 * np.pi * np.linalg.inv(model.lattice).T
    from .io import model_from_dict
    try:
        copy = model_from_dict(model.to_dict())
    except Exception as exc:                                     # noqa: BLE001 - reported, not raised
        problems.append(f"to_dict / from_dict failed: {exc}")
        copy = None
    for kf in ks:
        k = np.asarray(kf, float) @ rec
        H = model.bloch(k)
        if not np.allclose(H, H.conj().T, atol=tol):
            problems.append("H(k) not Hermitian")
        if model.non_orthogonal and np.linalg.eigvalsh(model.bloch(k, overlap=True)).min() <= 0:
            problems.append("S(k) not positive definite")
        if copy is not None and not np.allclose(copy.bloch(k), H, atol=1e-8):
            problems.append("the written-out copy has a different H(k)")
            copy = None
    return problems
