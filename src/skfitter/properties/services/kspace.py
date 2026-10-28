"""k-space sampling: paths, uniform meshes, k-planes, the first Brillouin zone and the valleys of hexagonal zones.

Fractional k are in units of the reciprocal lattice vectors (rows of ``reciprocal(model)``, with 2π); Cartesian k in
1/Å. Non-periodic directions (the vacuum of a 2D material) are never sampled.
"""
from __future__ import annotations

import numpy as np
from skfitter.model.bands import default_path, reciprocal
from skfitter.model.modifiers import periodic_axes

__all__ = ["reciprocal", "periodic_axes", "path", "mesh", "plane", "fold_to_bz", "bz_outline", "valleys",
           "dims", "check_xy_layer"]


def dims(model) -> int:
    """Number of periodic directions (1, 2 or 3)."""
    return len(periodic_axes(model))


def model_path(model) -> list:
    """The DFT k-path recorded with the model (an SKFitter fit: source.kpath_points), else the standard path of the
    lattice (skfitter.model.bands.default_path)."""
    src = getattr(model, "source", None) or {}
    pts = src.get("kpath_points")
    if pts and "supercell" not in src:
        return [None if p is None else (p[0], p[1]) for p in pts]
    return default_path(model)


def path(model, points=None, dk: float = 0.01) -> dict:
    """A path through labelled points [(label, k_frac), ..., None (a jump), ...] (default: model_path), spaced by about ``dk`` (1/Å): each segment of length L gets max(1, round(L/dk)) steps (SKFitter's rule).
    A jump takes no width; its tick reads "A|B". Returns {frac, cart, x, ticks: [(x, label)], pieces: [(start,
    stop)]}."""
    pts = points if points else model_path(model)
    rec = reciprocal(model)
    segs, cur = [], []
    for item in pts:
        if item is None:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append((str(item[0]), np.asarray(item[1], float)))
    if cur:
        segs.append(cur)
    frac, xs, ticks, pieces, x0 = [], [], [], [], 0.0
    for n_seg, seg in enumerate(segs):
        start = len(frac)
        label, a = seg[0]
        if n_seg == 0:
            ticks.append((x0, label))
        elif ticks[-1][1] != label:
            ticks[-1] = (ticks[-1][0], f"{ticks[-1][1]}|{label}")
        frac.append(a)
        xs.append(x0)
        for label, b in seg[1:]:
            length = float(np.linalg.norm((b - a) @ rec))
            n = max(1, int(round(length / float(dk))))
            for t in np.arange(1, n + 1) / n:
                frac.append(a + t * (b - a))
                xs.append(x0 + t * length)
            x0 += length
            ticks.append((x0, label))
            a = b
        pieces.append((start, len(frac)))
    frac = np.array(frac)
    return {"frac": frac, "cart": frac @ rec, "x": np.array(xs), "ticks": ticks, "pieces": pieces}


def mesh(model, nk, shift=None) -> np.ndarray:
    """Γ-centred uniform mesh (fractional, (n, 3)): nk[a] points along each periodic axis, 1 along the others.
    ``shift`` offsets the whole mesh in units of one mesh cell."""
    per = periodic_axes(model)
    nk = [int(nk[a]) if a in per else 1 for a in range(3)]
    grid = np.stack(np.meshgrid(*[np.arange(n) / n for n in nk], indexing="ij"), axis=-1).reshape(-1, 3)
    if shift is not None:
        grid = grid + np.asarray(shift, float)[None, :] / np.array(nk, float)[None, :]
    return grid


def effective_mesh(model, nk) -> tuple:
    per = periodic_axes(model)
    return tuple(int(nk[a]) if a in per else 1 for a in range(3))


def check_xy_layer(model, what: str) -> None:
    """The k-plane tools (maps, valleys, Chern numbers) need the layer in the xy plane: b1, b2 without z."""
    B = reciprocal(model)
    if np.abs(B[:2, 2]).max() > 1e-6 or periodic_axes(model)[:2] != [0, 1]:
        raise ValueError(f"{what} needs a layer cell: a1 and a2 in the xy plane (2D-type cell, as SKFitter's step 4).")


def plane(model, nk: int, k3: float = 0.0) -> np.ndarray:
    """nk x nk reduced (k1, k2) in [-1/2, 1/2) at fixed reduced k3 (a map of the kx-ky plane)."""
    check_xy_layer(model, "A k-plane map")
    u = np.arange(nk) / nk - 0.5
    U, V = np.meshgrid(u, u, indexing="ij")
    return np.stack([U.ravel(), V.ravel(), np.full(nk * nk, float(k3))], axis=1)


def fold_to_bz(model, k_cart: np.ndarray) -> np.ndarray:
    """Translate Cartesian k (n, 3) by in-plane reciprocal lattice vectors into the first Brillouin zone."""
    B = reciprocal(model)
    G = np.array([i * B[0] + j * B[1] for i in range(-2, 3) for j in range(-2, 3)])
    cand = k_cart[:, None, :] + G[None, :, :]
    best = np.argmin(np.linalg.norm(cand, axis=2), axis=1)
    return cand[np.arange(len(k_cart)), best]


def bz_outline(model) -> np.ndarray:
    """Closed polygon (m, 2) of the first Brillouin zone in the kx-ky plane."""
    from scipy.spatial import Voronoi
    B = reciprocal(model)
    G = np.array([i * B[0, :2] + j * B[1, :2] for i in range(-2, 3) for j in range(-2, 3)])
    vor = Voronoi(G)
    verts = vor.vertices[vor.regions[vor.point_region[int(np.argmin(np.linalg.norm(G, axis=1)))]]]
    verts = verts[np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))]
    return np.vstack([verts, verts[:1]])


def valleys(model, k3: float = 0.0, k_label_frac=None) -> dict:
    """K and K' of a hexagonal 2D zone (SKFitter's construction): the zone corners fall into two classes that differ
    by no reciprocal lattice vector; K is the class holding ``k_label_frac`` (a point labelled K on the DFT path)
    when given, else the corner of smallest polar angle. Returns {K, K' (Cartesian), corners, corner_valley, d_KM}."""
    check_xy_layer(model, "The valley analysis")
    B = reciprocal(model)
    verts = bz_outline(model)[:-1]
    red = verts @ np.linalg.inv(B[:2, :2])
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
    if k_label_frac is not None:
        d = red - np.asarray(k_label_frac, float)[None, :2]
        hit = np.where(np.all(np.abs(d - np.round(d)) < 1e-3, axis=1))[0]
        if len(hit):
            k_cls = int(cls[hit[0]])
    kz = float(k3) * B[2]
    rep = {}
    for name, c in (("K", k_cls), ("K'", 1 - k_cls)):
        members = np.where(cls == c)[0]
        first = members[int(np.argmin(ang[members]))]
        rep[name] = np.array([verts[first, 0], verts[first, 1], 0.0]) + kz
    edge = float(np.min(np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1)))
    return {"K": rep["K"], "K'": rep["K'"], "corners": verts,
            "corner_valley": ["K" if c == k_cls else "K'" for c in cls], "d_KM": 0.5 * edge}
