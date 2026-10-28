"""Kubo-formula building blocks of the Berry-phase family (ported from SKFitter's step 4, same conventions):

* ``curvature(H, A, B, bands)``: Ω_n = −2 Im Σ_{m≠n} <n|A|m><m|B|n>/(E_n − E_m)², pairs closer than
  ``degen_thresh`` left out; optionally averaged over degenerate groups (gauge invariant).
* ``fermi_sea_sum``: Σ_k w_k Σ_n f(E_n − E_F) Ω_n(k) on a Γ-centred mesh with adaptive refinement
  of hot spots, for many Fermi energies (and temperatures) in one pass.
"""
from __future__ import annotations

import numpy as np

AXIS = {"x": 0, "y": 1, "z": 2}
PLANE = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}          # Ω_x = Ω_yz, Ω_y = Ω_zx, Ω_z = Ω_xy


def curvature(H, A, B, bands, degen_thresh=1e-4, return_energies=False, average_degenerate=False):
    E, U = np.linalg.eigh(H)
    Ud = U.conj().transpose(0, 2, 1)
    Va = Ud @ A @ U
    Vb = Ud @ B @ U
    b = list(bands)
    rows = list(range(E.shape[1])) if average_degenerate else b
    dE = E[:, rows, None] - E[:, None, :]
    w = np.zeros_like(dE)
    ok = np.abs(dE) > degen_thresh
    w[ok] = 1.0 / dE[ok] ** 2
    term = Va[:, rows, :] * Vb[:, :, rows].transpose(0, 2, 1)
    omega = -2.0 * np.imag(np.sum(term * w, axis=2))
    if average_degenerate:
        omega = average_groups(omega, E, degen_thresh)[:, b]
    return (omega, E) if return_energies else omega


def average_groups(omega, E, degen_thresh=1e-4):
    """Ω of every band (n_k, D) replaced by its mean over each degenerate group (E ascending, (n_k, D))."""
    gid = np.concatenate([np.zeros((E.shape[0], 1), dtype=int),
                          np.cumsum(np.diff(E, axis=1) > degen_thresh, axis=1)], axis=1)
    same = gid[:, :, None] == gid[:, None, :]
    return np.einsum("kij,kj->ki", same.astype(float), omega) / same.sum(axis=2)


def operators(kernel, k_cart, alpha, beta, spin_op=None):
    """(H, A, B) at k: A = ∂_α H (or the spin current ½{σ, ∂_α H}), B = ∂_β H."""
    H = kernel.H(k_cart)
    dA = kernel.dH(k_cart, AXIS[alpha])
    dB = dA if alpha == beta else kernel.dH(k_cart, AXIS[beta])
    A = dA if spin_op is None else 0.5 * (spin_op @ dA + dA @ spin_op)
    return H, A, dB


def band_curvature(kernel, k_cart, bands, component="z", degen_thresh=1e-4, average_degenerate=True, chunk=256):
    a, b = PLANE[component]
    k_cart = np.asarray(k_cart, float).reshape(-1, 3)
    out = np.empty((len(k_cart), len(bands)))
    E = np.empty((len(k_cart), kernel.D))
    for s in range(0, len(k_cart), chunk):
        H, A, B = operators(kernel, k_cart[s:s + chunk], a, b)
        out[s:s + chunk], E[s:s + chunk] = curvature(H, A, B, bands, degen_thresh, True, average_degenerate)
    return out, E


def spin_curvature(kernel, k_cart, spin, alpha, beta, degen_thresh=1e-4, chunk=256):
    """Spin Berry curvature of every band: (omega (n_k, D), E (n_k, D))."""
    sop = kernel.spin_operators()[AXIS[spin]]
    k_cart = np.asarray(k_cart, float).reshape(-1, 3)
    D = kernel.D
    om, E = np.empty((len(k_cart), D)), np.empty((len(k_cart), D))
    for s in range(0, len(k_cart), chunk):
        H, A, B = operators(kernel, k_cart[s:s + chunk], alpha, beta, sop)
        om[s:s + chunk], E[s:s + chunk] = curvature(H, A, B, range(D), degen_thresh, True)
    return om, E


def fermi_sea_sum(kernel, recip, nk, alpha, beta, efermi, smearing, spin_op=None, chunk=256, degen_thresh=1e-4,
                  refine=True, refine_threshold=1.0, refine_levels=3, refine_factor=4, bands=None, shift=None,
                  progress=None, check=None):
    """acc(E_F) = Σ_k w_k Σ_n f(E_n − E_F) Ω_n(k) (w_k = 1 on the base mesh, split on refinement); returns
    (acc, N_k of the base mesh, points refined per level). See SKFitter._fermi_sea_sum for the method."""
    grid = np.stack(np.meshgrid(*[np.arange(n) / n for n in nk], indexing="ij"), axis=-1).reshape(-1, 3)
    if shift is not None:
        grid = grid + np.asarray(shift, float)[None, :] / np.array(nk, float)[None, :]
    n_k = len(grid)
    efermi = np.asarray(efermi, float)
    kT = np.asarray(smearing, float)
    kT = np.full(len(efermi), float(kT)) if kT.ndim == 0 else kT
    thermal = kT > 0.0
    active = np.array([n > 1 for n in nk])
    r = int(refine_factor)
    sub = [((np.arange(r) + 0.5) / r - 0.5) if act else np.zeros(1) for act in active]
    sub_off = np.stack(np.meshgrid(*sub, indexing="ij"), axis=-1).reshape(-1, 3)
    n_child = len(sub_off)
    acc = np.zeros(len(efermi))
    info = []
    sel = None if bands is None else [int(b) for b in bands]
    kfrac, weight = grid, np.ones(n_k)
    cell = 1.0 / np.array(nk, float)
    for level in range(int(refine_levels) + 1):
        flagged_k, flagged_w = [], []
        can_refine = bool(refine) and level < int(refine_levels) and n_child > 1
        d_act = int(active.sum())
        if d_act:
            rows = recip[active] * cell[active][:, None]
            h2 = np.sqrt(np.linalg.det(rows @ rows.T)) ** (2.0 / d_act)
        else:
            can_refine, h2 = False, 0.0
        m = len(kfrac)
        for s in range(0, m, chunk):
            if check:
                check()
            kf, w = kfrac[s:s + chunk], weight[s:s + chunk]
            H, A, B = operators(kernel, kf @ recip, alpha, beta, spin_op)
            if sel is None:
                sel = list(range(H.shape[1]))
            omega, E = curvature(H, A, B, sel, degen_thresh, return_energies=True)
            E = E[:, sel]
            flag = (np.abs(omega).max(axis=1) * h2 > refine_threshold) if can_refine else np.zeros(len(kf), bool)
            keep = ~flag
            if keep.any():
                x = E[keep].reshape(-1, 1) - efermi[None, :]
                occ = (x < 0.0).astype(float)
                if thermal.any():
                    occ[:, thermal] = 0.5 * (1.0 - np.tanh(x[:, thermal] / (2.0 * kT[thermal])[None, :]))
                acc += (omega[keep] * w[keep, None]).reshape(-1) @ occ
            if flag.any():
                flagged_k.append(kf[flag])
                flagged_w.append(w[flag])
            if progress and level == 0:
                progress((s + chunk) / m)
        if not flagged_k:
            break
        fk, fw = np.concatenate(flagged_k), np.concatenate(flagged_w)
        info.append(len(fk))
        kfrac = (fk[:, None, :] + sub_off[None, :, :] * cell[None, None, :]).reshape(-1, 3)
        weight = np.repeat(fw / n_child, n_child)
        cell = cell / np.where(active, r, 1)
    return acc, n_k, info
