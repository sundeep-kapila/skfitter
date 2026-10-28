"""SKFitter f orbitals: the general Slater-Koster construction (it reproduces the closed-form s, p, d blocks, so the
same code gives the f blocks), the l = 3 angular-momentum matrices and spin-orbit splitting, f in a full model (fast
Hamiltonian = reference, atomic spin-orbit limit), the jobs layer (projections, defaults, layout) and the parameter
sets (sign rule, f integrals in extended .skf files). The Eu(MnBi)2 case uses the illustrative project
tests/data/examples/EuMnBi/."""
import math
import os
import unittest

import _paths
import numpy as np

HAS = _paths.HAS_SCIENCE


def _dirs(n=150, seed=3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v = np.vstack([v, np.eye(3), -np.eye(3), [[1, 1, 1], [1, -1, 0], [0, 0, 1e-3 + 1]]])
    return v / np.linalg.norm(v, axis=1)[:, None]


@unittest.skipUnless(HAS, "needs the science stack")
class GeneralSlaterKoster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import skfitter.core as S
        cls.S, cls.F = S, S.SKFitter

    def closed_form(self, l1, l2, n, v):
        F, (a, b, c) = self.F, n
        return {(0, 0): lambda: F._sk_hopping_s_s(a, b, c, v[0]), (0, 1): lambda: F._sk_hopping_s_p(a, b, c, v[0]),
                (0, 2): lambda: F._sk_hopping_s_d(a, b, c, v[0]), (1, 1): lambda: F._sk_hopping_p_p(a, b, c, *v),
                (1, 2): lambda: F._sk_hopping_p_d(a, b, c, *v), (2, 2): lambda: F._sk_hopping_d_d(a, b, c, *v)}[(l1, l2)]()

    def test_reproduces_the_s_p_d_blocks(self):
        rng = np.random.default_rng(0)
        for l1, l2 in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
            for n in _dirs():
                v = rng.normal(size=min(l1, l2) + 1)
                new = sum(vi * U for vi, U in zip(v, self.S.sk_unit_blocks(l1, l2, n)))
                np.testing.assert_allclose(new, self.closed_form(l1, l2, n, v), atol=1e-12)

    def test_f_blocks_along_z_are_diagonal_in_m(self):
        names = [h[0] for h in self.S._harmonics(3)]
        for l1 in range(4):
            units = self.S.sk_unit_blocks(l1, 3, [0, 0, 1])
            self.assertEqual(len(units), l1 + 1)
            for m, U in enumerate(units):
                # orbital (|m|, type) couples only to the same (|m|, type), with weight 1
                H1 = self.S._harmonics(l1)
                for i, h1 in enumerate(H1):
                    for j, h3 in enumerate(self.S._harmonics(3)):
                        want = 1.0 if (h1[1] == m and h3[1] == m and h1[2] == h3[2]) else 0.0
                        self.assertAlmostEqual(U[i, j], want, places=12, msg=(l1, m, h1[0], names[j]))

    def test_rotations_keep_the_physics(self):
        """A dimer (f and p on one atom, f on the other): its levels do not depend on the bond direction, and the
        f-f block between two atoms is the transpose for the reversed bond (l1 = l2)."""
        rng = np.random.default_rng(1)
        vff, vpf = rng.normal(size=4), rng.normal(size=2)
        ref = None
        for n in _dirs(40):
            Uff = sum(v * U for v, U in zip(vff, self.S.sk_unit_blocks(3, 3, n)))
            Upf = sum(v * U for v, U in zip(vpf, self.S.sk_unit_blocks(1, 3, n)))
            np.testing.assert_allclose(Uff, sum(v * U for v, U in zip(vff, self.S.sk_unit_blocks(3, 3, -n))).T, atol=1e-12)
            H = np.zeros((17, 17))
            H[:7, 10:] = Uff
            H[7:10, 10:] = Upf
            H = H + H.T
            ev = np.linalg.eigvalsh(H)
            ref = ev if ref is None else ref
            np.testing.assert_allclose(ev, ref, atol=1e-10)
        # the invariant sum of squares: |H_ff|^2 = sum_m (2 - delta_m0) V_m^2
        U = sum(v * u for v, u in zip(vff, self.S.sk_unit_blocks(3, 3, _dirs(1)[0])))
        self.assertAlmostEqual(float((U ** 2).sum()), vff[0] ** 2 + 2 * float((vff[1:] ** 2).sum()), places=10)

    def test_angular_momentum(self):
        for fam, l in (("p", 1), ("d", 2)):              # the generated matrices = the hand-written ones
            for a, b in zip(self.S.angular_momentum_matrices(l), self.F._get_angular_momentum_matrices(fam)):
                np.testing.assert_allclose(a, b, atol=1e-13)
        Lx, Ly, Lz = self.F._get_angular_momentum_matrices("f")
        np.testing.assert_allclose(Lx @ Ly - Ly @ Lx, 1j * Lz, atol=1e-12)
        np.testing.assert_allclose(Lx @ Lx + Ly @ Ly + Lz @ Lz, 12 * np.eye(7), atol=1e-12)
        # spin-orbit lambda L.S splits f into j = 5/2 (6 states at -2 lambda) and j = 7/2 (8 states at +3/2 lambda)
        lam = 0.3
        sx, sy, sz = (np.array(m, complex) / 2 for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]))
        H = lam * (np.kron(sx, Lx) + np.kron(sy, Ly) + np.kron(sz, Lz))
        ev = np.linalg.eigvalsh(H)
        np.testing.assert_allclose(ev[:6], -2 * lam, atol=1e-12)
        np.testing.assert_allclose(ev[6:], 1.5 * lam, atol=1e-12)


@unittest.skipUnless(HAS, "needs the science stack")
class FInAModel(unittest.TestCase):
    """The graphene test data with f orbitals on carbon: unphysical, but it runs every f path of the fitter."""

    @classmethod
    def setUpClass(cls):
        from _examples_structure import project as example_project
        from skfitter import workflow as J
        cls.J = J
        cls.project, cls.dft = example_project("graphene-soc")

    def fitter(self, orbitals, soc):
        return self.J.build_fitter({"orbital_config": {"C": orbitals}, "magnetic_orbitals": {}, "n_bands_below": 1,
                                    "n_bands_above": 1, "n_neighbors": 2, "symprec": 1e-3, "fit_eta": False,
                                    "fixed_eta": 2.5, "include_soc": soc, "non_orthogonal": False, "tb_nelect": 2},
                                   self.project)

    def test_keys_and_fast_model(self):
        f = self.fitter(["pz", "f_m1", "fz3"], soc=False)
        keys = f._get_param_keys()
        for k in ("e_orbit0_C_f_m1", "e_orbit0_C_fz3"):
            self.assertIn(k, keys)
        # f split into groups: integrals per pair of groups (f_m1-f_m1, pz-fz3, ...), only those some bond uses
        self.assertTrue(any(k.startswith("v_C_fm1-C_fm1_") for k in keys))
        self.assertTrue(any(k.startswith("v_C_pz-C_fz3_") for k in keys))
        self.assertFalse(any(k.startswith(("v_C_p-C_f_", "v_C_f-C_f_")) for k in keys))
        whole = self.fitter(["p", "f"], soc=False)._get_param_keys()
        for k in ("v_C_p-C_f_pfs_0", "v_C_p-C_f_pfp_0", "v_C_f-C_f_ffs_0", "v_C_f-C_f_ffp_0", "v_C_f-C_f_ffd_0",
                  "v_C_f-C_f_fff_0"):
            self.assertIn(k, whole)
        self.assertEqual(f.base_dim, 2 * (1 + 2 + 1))
        rng = np.random.default_rng(0)
        vec = np.array([rng.normal() for _ in keys])
        self.assertTrue(f.check_fast_model(vec, n_k=6, verbose=False))

    def test_spin_orbit_atomic_limit(self):
        f = self.fitter(["f"], soc=True)
        keys = f._get_param_keys()
        self.assertIn("lambda_C_f", keys)
        lam = 0.2
        params = {k: 0.0 for k in keys}
        params["lambda_C_f"] = lam
        vec = np.array([params[k] for k in keys])
        E = np.sort(np.linalg.eigvalsh(f.build_hamiltonian(np.array([0.1, 0.2, 0.0]), f._param_vec_to_dict(vec))))
        n = len(E) // 14                                   # atoms
        np.testing.assert_allclose(E[:6 * n], -2 * lam, atol=1e-10)
        np.testing.assert_allclose(E[6 * n:], 1.5 * lam, atol=1e-10)


@unittest.skipUnless(HAS, "needs the science stack")
class JobsLayer(unittest.TestCase):
    def test_defaults_layout_projections(self):
        from skfitter import workflow as J
        self.assertEqual(J.default_orbitals(58), ["s", "d", "f"])       # Ce
        self.assertEqual(J.default_orbitals(92), ["s", "d", "f"])       # U
        self.assertEqual(J.default_orbitals(26), ["d"])
        rows = dict(J.ORBITAL_LAYOUT)
        self.assertEqual(rows["f sets"], ["f", "f_m0", "f_m1", "f_m2", "f_m3"])
        self.assertEqual(len(rows["f single"]), 7)
        self.assertEqual(J.PRESETS["s + d + f"], ["s", "d", "f"])
        SUB, FAM = J.sk_tables()
        for sub in rows["f sets"] + rows["f single"]:
            self.assertEqual(FAM[sub], "f")
            self.assertEqual(len(J.SUB_TO_PROJ[sub]), SUB[sub]["dim"])
            self.assertTrue(all(c in J.orbital_names_for(16)[0] for c in J.SUB_TO_PROJ[sub]))
        self.assertTrue(J.orbitals_conflict("f", "f_m1") and J.orbitals_conflict("f_m1", "fxz2"))
        self.assertFalse(J.orbitals_conflict("f_m1", "f_m2"))
        # bounds and starting values of the f integrals: small, usual signs
        self.assertEqual(J.default_bound("v_Ce_f-Ce_f_ffs_0"), (-2.0, 2.0))
        self.assertEqual(J.default_bound("v_O_p-Ce_f_pfp_0"), (-4.0, 4.0))
        self.assertEqual(J.default_bound("s_Ce_d-Ce_f_dfs_0"), (-0.5, 0.5))
        self.assertGreater(J.default_initial("v_O_p-Ce_f_pfs_0"), 0)
        self.assertLess(J.default_initial("v_O_p-Ce_f_pfp_0"), 0)
        self.assertLess(J.default_initial("s_O_p-Ce_f_pfs_0"), 0)
        self.assertEqual(J.default_bound("v_O_p-Ce_d_pds_0"), (-10.0, 2.0))       # unchanged


class ParameterSets(unittest.TestCase):
    def test_sign_rule_matches_harrison(self):
        from skfitter import params as P
        for k, v in P.HARRISON_ETA.items():
            if v:
                self.assertEqual(P.usual_sign(k), math.copysign(1.0, v), k)
        self.assertEqual([P.usual_sign(k) for k in ("sfs", "pfs", "pfp", "ffs", "ffp", "ffd", "fff")],
                         [1, 1, -1, 1, -1, 1, -1])
        self.assertEqual(P.integral_key("v_O_p-Ce_f_pfp_0"), "pfp")
        self.assertEqual(P.integral_key("eta_s_Ce_f-Ce_f_ffd"), "ffd")

    def test_extended_skf_with_f(self):
        """An extended (@) .skf file with f columns: the f integrals are read, their signs made the usual ones."""
        from skfitter import params as P
        n, dr = 300, 0.02
        lines = ["@ extended format", f"{dr}, {n}", "-0.2 0.0 -0.3 -0.5 0.0 " + "0.3 " * 4 + "0 1 0 2", "140.1, 19*0.0"]
        cols = P.SKF_COLUMNS_F
        for i in range(1, n + 1):
            e = math.exp(-i * dr / 2)
            H = {c: 0.0 for c in cols}
            S = {c: 0.0 for c in cols}
            H.update(sss=-0.3 * e, pfs=-0.2 * e, ffs=0.05 * e)              # pf-sigma stored with the "other" sign
            S.update(sss=0.3 * e, pfs=0.2 * e, ffs=-0.02 * e)
            lines.append(" ".join(f"{H[c]:.8e}" for c in cols) + " " + " ".join(f"{S[c]:.8e}" for c in cols))
        lines.append("Spline")
        d = P.dftb_set_from_files({"Ce-Ce.skf": "\n".join(lines)}, {"id": "t", "name": "T"})
        self.assertAlmostEqual(d["onsite"]["Ce"]["f"], -0.2 * P.HARTREE_EV, places=4)
        ps = P.ParamSet(d)
        r = 2.0
        self.assertGreater(ps.integral("Ce", "Ce", "pfs", r), 0)             # flipped to the usual sign
        self.assertGreater(ps.integral("Ce", "Ce", "ffs", r), 0)             # already usual (f-f: homonuclear, kept)
        self.assertLess(ps.integral("Ce", "Ce", "pfs", r, overlap=True), 0)


if __name__ == "__main__":
    unittest.main()
