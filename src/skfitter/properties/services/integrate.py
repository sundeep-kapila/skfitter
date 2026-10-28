"""Brillouin-zone integration of band quantities: Gaussian smearing and the linear tetrahedron (3D) / triangle (2D)
method (no Blöchl corrections), on a uniform Γ-centred mesh.

``dos(E_grid, energies, ...)`` gives states per eV per cell (one band holds one state; multiply by the spin degeneracy
of a spinless model for electrons). ``weights`` (n_k, n_b) give projected DOS; the integrated DOS of every band is 1.
"""
from __future__ import annotations

import numpy as np


def gaussian_dos(E, energies, sigma, weights=None):
    """Σ_k w_k Σ_n weight_nk g_σ(E - E_nk) / N_k. ``energies`` (n_k, n_b)."""
    E = np.asarray(E, float)
    e = np.asarray(energies, float).reshape(-1)
    w = np.ones_like(e) if weights is None else np.asarray(weights, float).reshape(-1)
    nk = np.asarray(energies).shape[0]
    out = np.zeros_like(E)
    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    for s in range(0, len(e), 20000):
        x = (E[:, None] - e[None, s:s + 20000]) / sigma
        out += (np.exp(-0.5 * x * x) * norm) @ w[s:s + 20000]
    return out / nk


def _simplices(nk, dims):
    """Corner index sets of the simplices of a periodic mesh: 6 tetrahedra per cube (3D, along the main diagonal),
    2 triangles per square (2D). Returns (n_simplex, d+1) indices into the mesh flattened in 'ij' order."""
    nk = [int(n) for n in nk]
    per = [a for a in range(3) if nk[a] > 1] if dims is None else dims
    idx = np.arange(int(np.prod(nk))).reshape(nk)

    def at(i, j, k):
        return idx[i % nk[0], j % nk[1], k % nk[2]]
    I, J, Kk = np.meshgrid(*[np.arange(n) for n in nk], indexing="ij")
    I, J, Kk = I.ravel(), J.ravel(), Kk.ravel()
    if len(per) == 3:
        corners = [at(I + a, J + b, Kk + c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        c = {(a, b, cc): corners[4 * a + 2 * b + cc] for a in (0, 1) for b in (0, 1) for cc in (0, 1)}
        tets = []
        # the 6 tetrahedra sharing the diagonal (0,0,0)-(1,1,1)
        for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            for q in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                if p == q:
                    continue
                pq = tuple(min(1, x + y) for x, y in zip(p, q))
                tets.append(np.stack([c[(0, 0, 0)], c[p], c[pq], c[(1, 1, 1)]], axis=1))
        return np.concatenate(tets)
    if len(per) == 2:
        a0, a1 = per
        step = [np.zeros(3, int), np.zeros(3, int)]
        step[0][a0], step[1][a1] = 1, 1
        g = np.stack([I, J, Kk])
        def c(sx, sy):
            v = g + (sx * step[0] + sy * step[1])[:, None]
            return at(v[0], v[1], v[2])
        return np.concatenate([np.stack([c(0, 0), c(1, 0), c(1, 1)], axis=1),
                               np.stack([c(0, 0), c(0, 1), c(1, 1)], axis=1)])
    if len(per) == 1:
        a0 = per[0]
        st = np.zeros(3, int)
        st[a0] = 1
        g = np.stack([I, J, Kk])
        v = g + st[:, None]
        return np.stack([at(I, J, Kk), at(v[0], v[1], v[2])], axis=1)
    raise ValueError("the tetrahedron method needs at least one periodic direction")


def _simplex_dos(E, e):
    """DOS (per simplex, normalised to 1) of a linear band on simplices with sorted corner energies e (n, d+1)."""
    E = np.asarray(E, float)[:, None]
    d = e.shape[1] - 1
    out = np.zeros((E.shape[0], e.shape[0]))
    eps = 1e-12
    if d == 3:
        e1, e2, e3, e4 = (e[:, i][None, :] for i in range(4))
        e21, e31, e41 = np.maximum(e2 - e1, eps), np.maximum(e3 - e1, eps), np.maximum(e4 - e1, eps)
        e32, e42, e43 = np.maximum(e3 - e2, eps), np.maximum(e4 - e2, eps), np.maximum(e4 - e3, eps)
        m1 = (E >= e1) & (E < e2)
        m2 = (E >= e2) & (E < e3)
        m3 = (E >= e3) & (E < e4)
        g1 = 3 * (E - e1) ** 2 / (e21 * e31 * e41)
        g2 = (3 * e21 + 6 * (E - e2) - 3 * (e31 + e42) * (E - e2) ** 2 / (e32 * e42)) / (e31 * e41)
        g3 = 3 * (e4 - E) ** 2 / (e41 * e42 * e43)
        out = np.where(m1, g1, 0.0) + np.where(m2, g2, 0.0) + np.where(m3, g3, 0.0)
    elif d == 2:
        e1, e2, e3 = (e[:, i][None, :] for i in range(3))
        e21, e31, e32 = np.maximum(e2 - e1, eps), np.maximum(e3 - e1, eps), np.maximum(e3 - e2, eps)
        m1 = (E >= e1) & (E < e2)
        m2 = (E >= e2) & (E < e3)
        out = np.where(m1, 2 * (E - e1) / (e21 * e31), 0.0) + np.where(m2, 2 * (e3 - E) / (e31 * e32), 0.0)
    else:
        e1, e2 = e[:, 0][None, :], e[:, 1][None, :]
        out = np.where((E >= e1) & (E < e2), 1.0 / np.maximum(e2 - e1, eps), 0.0)
    return out


def tetrahedron_dos(E, energies, nk, weights=None):
    """DOS on a Γ-centred mesh of shape nk (energies (N_k, n_b) in mesh order) by linear simplices: each simplex
    carries 1/n_simplex of the zone; projected weights are averaged over the simplex corners."""
    energies = np.asarray(energies, float)
    simp = _simplices(nk, None)
    ns = len(simp)
    out = np.zeros(len(E))
    for b in range(energies.shape[1]):
        e = energies[simp, b]                                   # (ns, d+1)
        order = np.argsort(e, axis=1)
        es = np.take_along_axis(e, order, axis=1)
        g = _simplex_dos(E, es)                                 # (nE, ns)
        if weights is None:
            out += g.sum(axis=1) / ns
        else:
            w = np.asarray(weights, float)[simp, b].mean(axis=1)
            out += (g @ w) / ns
    return out


def cumulative(E, dos):
    """∫ dos dE from the bottom of the grid (trapezoid)."""
    E, dos = np.asarray(E, float), np.asarray(dos, float)
    out = np.zeros_like(dos)
    out[1:] = np.cumsum(0.5 * (dos[1:] + dos[:-1]) * np.diff(E))
    return out
