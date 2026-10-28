"""Two-centre integrals per pair of orbital groups (skfitter: integral_keys, group_unit_blocks, check_orbital_symmetry).

MoS2 (tests/data/examples/mos2-TMDC-2D, hexagonal, c along z) with Mo d split into dz2 / d_e1 / d_e2 and S p into
pz / pxy:
* key names: standard between whole families, v_<A>.<B>.<m> otherwise, the same for either order;
* with every group-pair integral equal to the standard one, the bands are exactly those of the unsplit model;
  split_start gives those values from an unsplit fit;
* the vectorised bands agree with the reference H(k); a bond and its reverse give transposed blocks;
* integrals that vanish on every bond by symmetry are not parameters; an SKModel round trip keeps the bands;
* a split the crystal's symmetry does not allow (d_t2g / d_eg in hexagonal MoS2, or in a rotated bcc cell) is refused;
* the split-p names of older files are renamed (migrate_legacy_keys).
"""
import contextlib
import io
import json
import unittest

import _paths
import numpy as np

HAS = _paths.HAS_SCIENCE
EX = "mos2-qe"
SPLIT = {"Mo": ["dz2", "d_e1", "d_e2"], "S": ["pxy", "pz"]}
WHOLE = {"Mo": ["d"], "S": ["p"]}


def mos2(orbitals, **kw):
    from _examples_structure import cfg, fitter
    c = {**cfg(EX), "orbital_config": orbitals, "hopping_mode": "decay", "n_neighbors": 4, **kw}
    with contextlib.redirect_stdout(io.StringIO()):
        return fitter(EX, c)


def rand(f, seed=1):
    keys = f._get_param_keys()
    v = np.random.default_rng(seed).normal(0.0, 0.5, len(keys))
    for i, k in enumerate(keys):
        if k.startswith("lambda_"):
            v[i] = 0.1
    return f._param_vec_to_dict(v)


@unittest.skipUnless(HAS, "needs the science stack")
class Keys(unittest.TestCase):
    def test_names(self):
        import skfitter.core as S
        self.assertEqual(S.integral_keys("p", "d", "S", "Mo"), ["v_S_p-Mo_d_pds", "v_S_p-Mo_d_pdp"])
        self.assertEqual(S.integral_keys("d", "p", "Mo", "S"), ["v_S_p-Mo_d_pds", "v_S_p-Mo_d_pdp"])
        self.assertEqual(S.integral_keys("dz2", "pz", "Mo", "S"), ["v_S_pz-Mo_dz2_pds", "v_S_pz-Mo_dz2_pdp"])
        self.assertEqual(S.integral_keys("d_t2g", "d_eg", "Fe", "Fe"), S.integral_keys("d_eg", "d_t2g", "Fe", "Fe"))
        self.assertEqual(S.integral_keys("d", "d", "Mn", "Ir"), ["v_Ir_d-Mn_d_dds", "v_Ir_d-Mn_d_ddp",
                                                                  "v_Ir_d-Mn_d_ddd"])
        # s-p integrals are not symmetric in the elements: the s side comes first
        self.assertEqual(S.integral_keys("s", "p", "Mn", "Ir"), ["v_Mn_s-Ir_p_sps"])
        self.assertEqual(S.integral_keys("p", "s", "Mn", "Ir"), ["v_Ir_s-Mn_p_sps"])
        self.assertEqual(S.integral_keys("s", "pxy", "Bi", "Bi"), ["v_Bi_s-Bi_pxy_sps"])
        self.assertEqual(S.integral_keys("s*", "s", "Si", "Si"), ["v_Si_s-Si_s*_ss*s"])
        self.assertEqual(len(S.integral_keys("f_m1", "f", "Ce", "Ce")), 4)
        self.assertEqual(S.family_integral_key("v_S_pz-Mo_dz2_pdp_3"), "v_S_p-Mo_d_pdp_3")
        self.assertEqual(S.family_integral_key("s_Fe_dt2g-Fe_deg_ddd_0"), "s_Fe_d-Fe_d_ddd_0")
        self.assertEqual(S.family_integral_key("v_Mn_d-Mn_d_dds_0"), "v_Mn_d-Mn_d_dds_0")
        p = S.parse_integral_key("v_Ir_s-Mn_d_sds_1")
        self.assertEqual((p["pair"], p["shell"], p["btype"], p["label"]), ("Ir-Mn", 1, "Ir_s-Mn_d_sds", "sds"))
        self.assertEqual(S.bond_type("s_Ir_s-Mn_d_sds_0"), "Ir_s-Mn_d_sds")
        self.assertEqual(S.integral_pair("Ir_s-Mn_d_sds"), "Ir-Mn")
        self.assertTrue(S.is_group_integral("v_S_pz-Mo_dz2_pds_0"))
        self.assertFalse(S.is_group_integral("v_S_p-Mo_d_pds_0"))

    def test_legacy_names(self):
        import skfitter.core as S
        out = S.migrate_legacy_keys({"v_pzpzp_0": 1.0, "v_pxpxs_2": 2.0, "eta_spzs": 3.0, "v_dds_0": 4.0})
        self.assertEqual(out, {"v_pz.pz.p_0": 1.0, "v_pxy.pxy.s_2": 2.0, "v_pxy.pz.s_2": 2.0, "eta_s.pz.s": 3.0,
                               "v_dds_0": 4.0})

    def test_blocks_sum_to_the_shell_block(self):
        """The group blocks tile the whole-shell block: d split three ways, p two ways, any direction."""
        import skfitter.core as S
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = rng.normal(size=3)
            full = S.group_unit_blocks("p", "d", n)
            for m in range(2):
                tiled = np.zeros((3, 5))
                for a in ("pxy", "pz"):
                    for b in ("dz2", "d_e1", "d_e2"):
                        tiled[np.ix_(S._sub_indices(a), S._sub_indices(b))] = S.group_unit_blocks(a, b, n)[m]
                np.testing.assert_allclose(tiled, full[m], atol=1e-13)


@unittest.skipUnless(HAS, "needs the science stack and tests/data/examples/mos2-TMDC-2D")
class SplitModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.whole = mos2(WHOLE)
        cls.split = mos2(SPLIT)

    def bands(self, f, p, nk=6):
        return np.array([np.linalg.eigvalsh(f.build_hamiltonian(k, p)) for k in np.asarray(f.kpoints[:nk])])

    def test_keys_are_pruned(self):
        keys = set(self.split._get_param_keys())
        self.assertIn("v_S_pz-Mo_dz2_pds_0", keys)
        self.assertIn("v_S_pxy-S_pz_pps_0", keys)
        self.assertNotIn("v_S_p-Mo_d_pds_0", keys)
        # Mo-Mo bonds lie in the plane: d_e1 (xz, yz) and dz2 do not couple on them
        self.assertFalse(any(k.startswith("v_Mo_de1-Mo_dz2_") for k in keys))

    def test_equal_integrals_give_the_unsplit_model(self):
        import skfitter.core as S
        pw = rand(self.whole, 2)
        start = self.split.split_start(pw)
        keys = self.split._get_param_keys()
        missing = [k for k in keys if k not in start and k not in pw]
        self.assertEqual(missing, [])
        ps = self.split._param_vec_to_dict(np.array([start.get(k, pw.get(k)) for k in keys]))
        self.assertLess(np.abs(self.bands(self.whole, pw) - self.bands(self.split, ps)).max(), 1e-10)
        ps[keys[-1]] += 0.3                                   # an independent integral changes the bands
        self.assertGreater(np.abs(self.bands(self.whole, pw) - self.bands(self.split, ps)).max(), 1e-4)

    def test_fast_model_and_reverse_bonds(self):
        f = self.split
        keys = f._get_param_keys()
        vec = np.array([rand(f, 3)[k] for k in keys])
        self.assertTrue(f.check_fast_model(vec, n_k=5, verbose=False))
        p = rand(f, 4)
        rng = np.random.default_rng(5)
        for _ in range(10):
            R = rng.normal(size=3) * 3
            for a in ("pz", "pxy", "dz2", "d_e1", "d_e2"):
                for b in ("pz", "pxy", "dz2", "d_e1", "d_e2"):
                    np.testing.assert_allclose(f._get_submatrix_hopping("Mo", a, "S", b, R, p),
                                               f._get_submatrix_hopping("S", b, "Mo", a, -R, p).T, atol=1e-12)

    def test_sk_model(self):
        from skfitter.model.sk import SKModel
        p = rand(self.split, 6)
        m = SKModel.from_dict(json.loads(json.dumps(SKModel.from_fitter(self.split, p).to_dict())))
        kk = np.asarray(self.split.kpoints[:4])
        Em = np.array([np.linalg.eigvalsh(m.bloch(k)) for k in kk])
        self.assertLess(np.abs(Em - self.bands(self.split, p, 4)).max(), 1e-9)

    def test_symmetry_check(self):
        with self.assertRaises(ValueError) as ctx:
            mos2({"Mo": ["d_t2g", "d_eg"], "S": ["p"]})
        self.assertIn("symmetry", str(ctx.exception))
        with self.assertRaises(ValueError):                     # px, py apart: the 3-fold axis mixes them
            mos2({"Mo": ["d"], "S": ["px", "py", "pz"]})
        mos2({"Mo": ["dz2", "d_e1", "d_e2"], "S": ["pz"]})      # pz alone (and the hexagonal d sets) are allowed

    def test_rotated_cubic_cell(self):
        """bcc Fe in a rotated frame: t2g / e_g is not a split of its axes."""
        import skfitter.core as S
        from pymatgen.core import Lattice, Structure
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler("zyx", [20, 35, 10], degrees=True).as_matrix()
        st = Structure(Lattice(np.eye(3) * 2.84 @ rot.T), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.assertTrue(S.check_orbital_symmetry(st, [["d_t2g", "d_eg"]] * len(st)))
        R = np.eye(3)
        from pymatgen.core import Lattice, Structure
        cubic = Structure(Lattice.cubic(2.84), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.assertEqual(S.check_orbital_symmetry(cubic, [["d_t2g", "d_eg"]] * 2), [])
        self.assertTrue(R.any())


if __name__ == "__main__":
    unittest.main()
