"""Orbital names and angular-momentum matrices in SKFitter's real-orbital basis.

Labels are SKFitter's (``px``, ``py``, ``pz``; ``dxy``, ``dyz``, ``dzx``, ``dx2y2``, ``dz2``; the seven f functions in
VASP's order), optionally with a spin arrow (``px↑``). ``L_matrices(labels)`` gives (Lx, Ly, Lz) between the orbitals
of one site (in units of ħ), the same matrices SKFitter uses for its spin-orbit term, so an orbital Zeeman term
mu_B L.B is consistent with the fitted lambda L.S. Orbitals whose label is not an atomic one (Wannier functions,
``w1``) have no L; ``L_matrices`` then returns None.
"""
from __future__ import annotations

import numpy as np

FAMILY_BASIS = {"s": ["s"], "s*": ["s*"], "p": ["px", "py", "pz"], "d": ["dxy", "dyz", "dzx", "dx2y2", "dz2"],
                "f": ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"]}
FAMILY_L = {"s": 0, "s*": 0, "p": 1, "d": 2, "f": 3}
_WHERE = {name: (fam, k) for fam, names in FAMILY_BASIS.items() for k, name in enumerate(names)}
SPIN_MARKS = ("↑", "↓")


def strip_spin(label: str) -> str:
    return label[:-1] if label and label[-1] in SPIN_MARKS else label


def family_of(label: str):
    """(family, index in the family's basis) of an atomic orbital label, or None."""
    return _WHERE.get(strip_spin(str(label)))


def _family_L(fam: str):
    """(Lx, Ly, Lz) in the full real basis of a family (SKFitter's matrices); zeros for s."""
    if fam in ("s", "s*"):
        z = np.zeros((1, 1), dtype=complex)
        return z, z, z
    if fam == "p":
        Lx = 1j * np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        Ly = 1j * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        Lz = 1j * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        return Lx, Ly, Lz
    if fam == "d":
        r3 = np.sqrt(3)
        Lz = 1j * np.array([[0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, -1, 0, 0, 0], [-2, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        Lx = 1j * np.array([[0, 0, -1, 0, 0], [0, 0, 0, -1, -r3], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, r3, 0, 0, 0]])
        Ly = 1j * np.array([[0, 1, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 0, -1, r3], [0, 0, 1, 0, 0], [0, 0, -r3, 0, 0]])
        return Lx, Ly, Lz
    if fam == "f":
        from skfitter.core import angular_momentum_matrices         # the f matrices are built from the harmonics
        return tuple(np.asarray(m, dtype=complex) for m in angular_momentum_matrices(3))
    raise ValueError(f"unknown orbital family {fam!r}")


def L_matrices(labels):
    """(Lx, Ly, Lz) between the given spatial orbitals of one site, or None if any label is not atomic.
    Orbitals of different families (or different shells of one family) are not coupled by L."""
    where = [family_of(lab) for lab in labels]
    if any(w is None for w in where):
        return None
    n = len(labels)
    out = [np.zeros((n, n), dtype=complex) for _ in range(3)]
    full = {fam: _family_L(fam) for fam in {w[0] for w in where}}
    # orbitals of one family belong to one shell when each basis function appears once; a repeated function
    # (two p shells on one atom) starts a new shell
    shell, seen = [], {}
    for fam, k in where:
        key = (fam, k)
        seen[key] = seen.get(key, -1) + 1
        shell.append(seen[key])
    for a, (fa, ka) in enumerate(where):
        for b, (fb, kb) in enumerate(where):
            if fa == fb and shell[a] == shell[b]:
                for c in range(3):
                    out[c][a, b] = full[fa][c][ka, kb]
    return tuple(out)
