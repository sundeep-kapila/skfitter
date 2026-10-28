"""SKFitter on the valley Berry-curvature demos made with tools/qe_2d (PBE, pslibrary PAW, no spin-orbit coupling):
monolayer hBN and graphene on hBN. Both break inversion, so the Berry curvature peaks at K and K' with opposite signs.

Also checks "fit only DFT states of the model's orbitals" (cfg model_states_only): a pz model follows the pi bands
past the sigma bands and the vacuum states of the 2D layer, and the pi electron count of a pz model (B 0, C 1, N 2).
"""
import os
import shutil
import tempfile
import unittest

import _paths
import numpy as np

FOLDER = {"hbn": "hbn-valley", "graphene_hbn": "graphene-on-hbn"}
FILES = {"qe_xml": "qe_bands.xml", "qe_bands_in": "qe_bands.in", "qe_scf": "qe_scf.out", "qe_projwfc": "qe_projwfc_up"}


def have(case):
    return all(os.path.isfile(os.path.join(_paths.example_dir(FOLDER[case]), f)) for f in FILES.values())


def cfg(orbitals, below, above, states_only=True):
    return {"orbital_config": orbitals, "magnetic_orbitals": {}, "n_bands_below": below, "n_bands_above": above,
            "n_neighbors": 3, "symprec": 1e-3, "fit_eta": True, "fixed_eta": 2.5, "include_soc": False,
            "non_orthogonal": True, "tb_nelect": None, "model_states_only": states_only}


def quiet(*_a, **_k):
    pass


class _Case(unittest.TestCase):
    case = ""

    @classmethod
    def setUpClass(cls):
        from skfitter import workflow as J
        cls.project = tempfile.mkdtemp(prefix=f"prayog_{cls.case}_")
        cls.dft = J.prepare_project(cls.project, "espresso",
                                    {r: os.path.join(_paths.example_dir(FOLDER[cls.case]), f) for r, f in FILES.items()}, quiet, quiet)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.project, ignore_errors=True)

    def fit(self, c, opts=None):
        from skfitter import workflow as J
        from skfitter import params as P
        f = J.build_fitter(c, self.project)
        sv = J.starting_values(f, c, P.bundled_sets()["harrison"], self.dft)
        o = {"method": "ls", "n_starts": 4, "refine_max_nfev": 400, "seed": 0, **(opts or {})}
        return J.run_fit_job(c, o, sv["initial"], sv["bounds"], self.project, self.case, quiet, quiet, lambda: False,
                             log=quiet, dft=self.dft)

    @staticmethod
    def gap_at(f, kf):
        B = f.structure.lattice.reciprocal_lattice.matrix
        e = np.linalg.eigvalsh(f._tb_orthogonal_hamiltonians((np.array(kf) @ B)[None, :])[0])
        return e[f.tb_vbm_index + 1] - e[f.tb_vbm_index]

    def topology(self, f, mode, **kw):
        """Wilson loops ({'K', "K'"}: Berry phase of the occupied bands on the largest loop) or Chern numbers
        ({'C', 'C_K', 'C_Kp'} of the occupied bands) through run_property."""
        from skfitter import workflow as J
        spec = {"kind": "bc", "source": "sk", "mode": mode, "component": "z", "k3": 0.0, **kw}
        out = J.run_property(f, spec, self.project, self.case, quiet, quiet, lambda: False)
        self.assertTrue(os.path.isfile(out["png"]) and out["table"]["rows"])
        if mode == "wilson":
            fig, _, res = f.compute_wilson_loops_sk(n_loop=kw.get("wn", 120), n_radii=kw.get("wnr", 30), show=False)
            return {v: float(res["valleys"][v]["occupied"][-1]) for v in ("K", "K'")}
        fig, _, res = f.compute_chern_numbers_sk(nk=kw.get("nk", 60), show=False)
        return res["occupied"]

    def berry_at_valleys(self, f, dk):
        from skfitter import workflow as J
        spec = {"kind": "bc", "source": "sk", "mode": "path", "component": "z", "dk": dk, "nk": 40, "k3": 0.0}
        out = J.run_property(f, spec, self.project, self.case, quiet, quiet, lambda: False)
        res = out["res"]
        x, om = np.asarray(res["x"]), np.asarray(res["total_occupied"])
        at = {lab: float(om[int(np.argmin(abs(x - t)))]) for t, lab in zip(*res["ticks"]) if lab in ("K", "K'")}
        return at, out


@unittest.skipUnless(_paths.HAS_SCIENCE and have("hbn"), "needs the science stack and tests/data/examples/hbn-valley")
class HexagonalBoronNitride(_Case):
    case = "hbn"

    def test_reading(self):
        m = self.dft["meta"]
        self.assertEqual((m["nelect"], m["soc"]), (8.0, False))
        self.assertEqual(self.dft["tick_labels"], ["Γ", "K", "M", "K'", "Γ"])
        E = self.dft["E"]["up"]
        self.assertAlmostEqual(E[40, 4] - E[40, 3], 4.70, delta=0.05)          # DFT gap at K (point 41)

    def test_model_states_follow_the_pi_bands(self):
        from skfitter import workflow as J
        c = cfg({"B": ["pz"], "N": ["pz"]}, 1, 1)
        f = J.build_fitter(c, self.project)
        self.assertEqual(f.tb_nelect, 2)                                        # pi electrons: B 0 + N 2
        plain = J.build_fitter(dict(c, model_states_only=False), self.project)
        self.assertGreater(f.state_filter["kpoints_changed"], 20)              # sigma / vacuum states skipped near Gamma
        # at Gamma the top valence state is sigma (about -1.3 eV); the pi state lies about 4 eV lower
        self.assertLess(f.target_dft_bands[0, 0], plain.target_dft_bands[0, 0] - 3.0)

    def test_fit_and_valley_berry_curvature(self):
        r = self.fit(cfg({"B": ["pz"], "N": ["pz"]}, 1, 1))
        f = r["fitter"]
        self.assertLess(r["rmse"] * 1000, 120.0, f"RMSE {r['rmse'] * 1000:.0f} meV")   # about 60 meV
        self.assertAlmostEqual(self.gap_at(f, [1 / 3, 1 / 3, 0]), 4.70, delta=0.2)
        at, _ = self.berry_at_valleys(f, 0.005)
        self.assertLess(at["K"] * at["K'"], 0.0)                               # opposite signs at K and K'
        self.assertAlmostEqual(abs(at["K"]), abs(at["K'"]), delta=1e-6 * abs(at["K"]) + 1e-9)
        self.assertTrue(0.5 < abs(at["K"]) < 10.0, at)                         # about 2 A^2 for a 4.7 eV gap
        # Wilson loops around K and K': opposite Berry phases; FHS valley Chern numbers of opposite sign with C = 0
        # (about +-0.34 with separate B-B and N-N integrals), which agree in sign with the Kubo curvature at K (a
        # large gap spreads the curvature, so |C_K| < 1/2)
        w = self.topology(f, "wilson", wn=60, wnr=6)
        self.assertLess(w["K"] * w["K'"], 0.0)
        self.assertAlmostEqual(w["K"], -w["K'"], delta=1e-6)
        c = self.topology(f, "chern", nk=40)
        self.assertAlmostEqual(c["C"], 0.0, delta=1e-6)
        self.assertTrue(0.1 < abs(c["C_K"]) < 0.5, c)
        self.assertAlmostEqual(c["C_K"], -c["C_Kp"], delta=1e-6)
        self.assertGreater(c["C_K"] * at["K"], 0.0)
        self.assertGreater(w["K"] * at["K"], 0.0)


@unittest.skipUnless(_paths.HAS_SCIENCE and have("graphene_hbn"), "needs the science stack and tests/data/examples/graphene-on-hbn")
class GrapheneOnBoronNitride(_Case):
    case = "graphene_hbn"

    def test_fit_and_valley_berry_curvature(self):
        c = cfg({"B": ["pz"], "N": ["pz"], "C": ["pz"]}, 2, 2)
        r = self.fit(c, {"add_weight_at_fermi_bands": True, "fermi_weight": 10.0, "fermi_window": 1.0,
                         "add_weight_at_symmetry_points": True})
        f = r["fitter"]
        self.assertEqual(f.tb_nelect, 4)                                        # N 2 + C 1 + C 1
        self.assertLess(r["report"]["near_fermi"], 30.0, r["report"])          # about 6 meV
        gap = self.gap_at(f, [1 / 3, 1 / 3, 0])
        self.assertTrue(0.015 < gap < 0.06, f"gap at K {gap * 1e3:.1f} meV")    # DFT 32 meV
        at, out = self.berry_at_valleys(f, 0.001)
        self.assertLess(at["K"] * at["K'"], 0.0)
        self.assertGreater(min(abs(at["K"]), abs(at["K'"])), 1e4)              # tall peaks: about 5e4 A^2
        self.assertTrue(os.path.isfile(out["png"]))
        # two occupied pz bands, C = 0. C_K = (graphene, +-1/2: sign of its ~30 meV sublattice mass) + (hBN band, about
        # -0.19). The band fit cannot tell the two carbons apart: swapping their on-site energies flips the mass with the
        # same RMSE, and which of the two minima the fit finds depends on the platform. So C_K is -0.69 or +0.31; the
        # graphene part follows the sign of the tall Berry-curvature peak at K, and the hBN part is the same in both.
        c = self.topology(f, "chern", nk=60)
        self.assertAlmostEqual(c["C"], 0.0, delta=1e-6)
        self.assertAlmostEqual(c["C_K"], -c["C_Kp"], delta=1e-6)
        self.assertGreater(c["C_K"] * at["K"], 0.0)
        self.assertAlmostEqual(c["C_K"] - 0.5 * np.sign(at["K"]), -0.19, delta=0.05, msg=c)


if __name__ == "__main__":
    unittest.main()
