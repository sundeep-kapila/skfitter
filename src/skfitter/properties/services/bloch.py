"""The Bloch kernel: H(k), its k-derivatives, eigenstates and operators of any common model, for many k at once.

    K = Kernel(model)
    H = K.H(k_cart)                    # (n, D, D)  in the orbital-centre gauge (below)
    dH = K.dH(k_cart, 0)               # dH/dk_x (eV Å)
    d2H = K.d2H(k_cart, 0, 1)          # d²H/dk_x dk_y (eV Å²)
    E, U = K.eig(k_cart)               # ascending; U columns in the (Löwdin-)orthonormal basis; cached

Convention (recorded in every result): phases exp(i k·(R + c_j - c_i)) with c the orbital centres (SK orbitals sit on
their atom). This is SKFitter's convention for SK models, and it is the one
in which the Berry curvature at a k-point is physical; band energies and integrated quantities do not depend on it.

H(R) is gathered once from the model's on-site and hopping blocks (the common interface), so the cost per k is a
sum over the cells R. Orthogonal models get analytic derivatives:
    H = G0,  dH_a = i (G_a + Δ_a∘G0),  d²H_ab = -(G_ab + Δ_b∘G_a + Δ_a∘G_b + Δ_aΔ_b∘G0)
with G_x... = U† (Σ_R R_x... e^{ik·R} H(R)) U, U = diag(e^{ik·c}) and Δ_a[i, j] = c_j,a - c_i,a.
Non-orthogonal models use the Löwdin-orthogonalised H = S^-1/2 H S^-1/2 (as SKFitter does)
and central finite differences (step ``delta``) for its derivatives.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

CHUNK_BYTES = 64 * 2 ** 20


class Kernel:
    def __init__(self, model, delta: float = 1e-4, cache: int = 8):
        self.model = model
        self.delta = float(delta)
        self.lattice = np.asarray(model.lattice, float)
        self.dims = [int(model.norbs(i)) for i in range(model.n_sites)]
        self.offsets = np.concatenate([[0], np.cumsum(self.dims)]).astype(int)
        self.D = int(self.offsets[-1])
        self.site_of = np.repeat(np.arange(model.n_sites), self.dims)
        self.centres = np.concatenate([np.asarray(model.orbital_centres(i), float).reshape(-1, 3)
                                       for i in range(model.n_sites)])
        self.non_orthogonal = bool(getattr(model, "non_orthogonal", False))
        self._gather()
        self.Delta = self.centres[None, :, :] - self.centres[:, None, :]        # (D, D, 3): c_j - c_i
        self.Delta = np.moveaxis(self.Delta, 2, 0)                                # (3, D, D)
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = int(cache)
        self.n_diagonalised = 0

    # -- H(R) ---------------------------------------------------------------------------------------------------
    def _gather(self):
        m, o, D = self.model, self.offsets, self.D
        cart = np.asarray(m.cart_coords, float)
        blocks, sblocks = {}, {}

        def put(store, R, i, j, blk):
            M = store.get(R)
            if M is None:
                M = store[R] = np.zeros((D, D), dtype=complex)
            M[o[i]:o[i + 1], o[j]:o[j + 1]] += blk
        for i in range(m.n_sites):
            put(blocks, (0, 0, 0), i, i, np.asarray(m.onsite(i), complex))
            if self.non_orthogonal:
                put(sblocks, (0, 0, 0), i, i, np.eye(self.dims[i]))
        inv = np.linalg.inv(self.lattice)
        for b in m.bonds:
            i, j, d = int(b["i"]), int(b["j"]), np.asarray(b["d_vec"], float)
            R = b.get("R")
            if R is None:
                R = np.rint((cart[i] + d - cart[j]) @ inv).astype(int)
            R = tuple(int(x) for x in R)
            put(blocks, R, i, j, np.asarray(m.hopping(i, j, d), complex))
            if self.non_orthogonal:
                put(sblocks, R, i, j, np.asarray(m.hopping(i, j, d, overlap=True), complex))
        self.R = np.array(sorted(blocks), dtype=int)
        self.Rcart = self.R @ self.lattice
        self.HR = np.array([blocks[tuple(r)] for r in self.R])
        self.SR = np.array([sblocks.get(tuple(r), np.zeros((D, D), complex)) for r in self.R]) \
            if self.non_orthogonal else None
        self.reach = np.abs(self.R).max(axis=0).tolist() if len(self.R) else [0, 0, 0]

    def _chunks(self, n):
        step = min(512, max(1, int(CHUNK_BYTES // (16 * 4 * self.D * self.D + 16 * len(self.R)))))
        for s in range(0, n, step):
            yield slice(s, min(n, s + step))

    def _sum(self, k, store, weight=None):
        """Σ_R w(R) e^{ik·R} X(R), then the orbital-centre gauge: (n, D, D)."""
        P = np.exp(1j * (k @ self.Rcart.T))                                   # (n, nR)
        if weight is not None:
            P = P * weight[None, :]
        G = np.einsum("kr,rij->kij", P, store)
        u = np.exp(1j * (k @ self.centres.T))                                  # (n, D)
        return u.conj()[:, :, None] * G * u[:, None, :]

    @staticmethod
    def _herm(X):
        return 0.5 * (X + X.conj().transpose(0, 2, 1))

    def _orth(self, k):
        """Löwdin-orthogonalised H (orbital-centre gauge) at k (n, 3)."""
        H = self._herm(self._sum(k, self.HR))
        if not self.non_orthogonal:
            return H
        S = self._herm(self._sum(k, self.SR))
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 1e-10)
        X = (V / np.sqrt(w)[:, None, :]) @ V.conj().transpose(0, 2, 1)
        return self._herm(X @ H @ X)

    # -- public -------------------------------------------------------------------------------------------------
    def H(self, k_cart) -> np.ndarray:
        k = np.asarray(k_cart, float).reshape(-1, 3)
        out = np.empty((len(k), self.D, self.D), complex)
        for s in self._chunks(len(k)):
            out[s] = self._orth(k[s])
        return out

    def dH(self, k_cart, a: int) -> np.ndarray:
        k = np.asarray(k_cart, float).reshape(-1, 3)
        out = np.empty((len(k), self.D, self.D), complex)
        e = np.eye(3)[a] * self.delta
        for s in self._chunks(len(k)):
            kk = k[s]
            if self.non_orthogonal:
                out[s] = (self._orth(kk + e) - self._orth(kk - e)) / (2 * self.delta)
            else:
                G0 = self._sum(kk, self.HR)
                Ga = self._sum(kk, self.HR, self.Rcart[:, a])
                out[s] = self._herm(1j * (Ga + self.Delta[a][None] * G0))
        return out

    def d2H(self, k_cart, a: int, b: int) -> np.ndarray:
        k = np.asarray(k_cart, float).reshape(-1, 3)
        out = np.empty((len(k), self.D, self.D), complex)
        for s in self._chunks(len(k)):
            kk = k[s]
            if self.non_orthogonal:
                ea, eb, h = np.eye(3)[a], np.eye(3)[b], 1e-3
                out[s] = (self._orth(kk + h * (ea + eb)) - self._orth(kk + h * (ea - eb))
                          - self._orth(kk - h * (ea - eb)) + self._orth(kk - h * (ea + eb))) / (4 * h * h)
            else:
                Ra, Rb = self.Rcart[:, a], self.Rcart[:, b]
                G0 = self._sum(kk, self.HR)
                Ga, Gb = self._sum(kk, self.HR, Ra), self._sum(kk, self.HR, Rb)
                Gab = self._sum(kk, self.HR, Ra * Rb)
                Da, Db = self.Delta[a][None], self.Delta[b][None]
                out[s] = self._herm(-(Gab + Db * Ga + Da * Gb + Da * Db * G0))
        return out

    def eig(self, k_cart, vectors: bool = True):
        """Eigenvalues (n, D) ascending and eigenvectors (n, D, D) (columns); the last few calls are cached."""
        k = np.ascontiguousarray(np.asarray(k_cart, float).reshape(-1, 3))
        key = (k.shape, hash(k.tobytes()), vectors)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        H = self.H(k)
        if vectors:
            val = np.linalg.eigh(H)
        else:
            val = np.linalg.eigvalsh(H)
        self.n_diagonalised += len(k)
        self._cache[key] = val
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return val

    def energies(self, k_cart) -> np.ndarray:
        return self.eig(k_cart, vectors=False)

    # -- operators ----------------------------------------------------------------------------------------------
    def spin_operators(self):
        """σx, σy, σz (D, D) in the model's basis (spinful models: each site's spin-up orbitals, then spin-down),
        or None for a spinless model."""
        if not self.model.spinful:
            return None
        ops = [np.zeros((self.D, self.D), complex) for _ in range(3)]
        s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
        for i, n2 in enumerate(self.dims):
            n, o = n2 // 2, self.offsets[i]
            for a in range(3):
                ops[a][o:o + n2, o:o + n2] = np.kron(s[a], np.eye(n))
        return ops

    def orbital_groups(self, by: str = "element") -> dict:
        """{group name: basis indices}, by "element", "site", "l" (s, p, d, f), "orbital" (label without spin) or
        "spin" (up / down)."""
        m = self.model
        out: dict = {}
        letters = "spdf"
        for i in range(m.n_sites):
            info = m.orbital_info(i)
            for n, x in enumerate(info):
                idx = int(self.offsets[i] + n)
                if by == "element":
                    key = m.species[i]
                elif by == "site":
                    key = f"{m.species[i]}{i + 1}"
                elif by == "l":
                    key = letters[x["l"]] if x.get("l") is not None and 0 <= x["l"] < 4 else "?"
                elif by == "orbital":
                    key = str(x["label"]).rstrip("↑↓")
                elif by == "spin":
                    key = x.get("spin") or "none"
                else:
                    raise ValueError("group by element, site, l, orbital or spin")
                out.setdefault(key, []).append(idx)
        return out
