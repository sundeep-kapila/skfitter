"""SKFitter with spin-orbit coupling on Quantum ESPRESSO data made with tools/qe_2d (graphene_soc: PBE, fully
relativistic PAW, non-collinear): reading the SOC files, on-site starting values with the C 2s band 20 eV deep and
vacuum states in the cell, an s + p fit with spin-orbit coupling, and the SOC signatures of the fitted model -
the sigma splitting at Gamma (DFT 9.2 meV), the micro-eV gap at K and the spin Hall conductivity."""
import os
import shutil
import tempfile
import unittest

import _paths
import numpy as np

DATA = _paths.example_dir("graphene-soc")
FILES = {"qe_xml": "qe_bands.xml", "qe_bands_in": "qe_bands.in", "qe_scf": "qe_scf.out", "qe_projwfc": "qe_projwfc_up"}
HAVE = all(os.path.isfile(os.path.join(DATA, f)) for f in FILES.values())

# the demo set-up (tools/qe_2d/README.md): C s + p, 3 neighbour shells, non-orthogonal, fitted distance decay,
# the 4 valence bands and pi* (x 2 for spin); lambda(C, p) fixed at 7.7 meV (reproduces the 9.2 meV of DFT)
CFG = {"orbital_config": {"C": ["s", "p"]}, "magnetic_orbitals": {}, "n_bands_below": 4, "n_bands_above": 1,
       "n_neighbors": 3, "symprec": 1e-3, "fit_eta": True, "fixed_eta": 2.5, "include_soc": True,
       "non_orthogonal": True, "tb_nelect": None}
LAMBDA = 0.0077


def quiet(*_a, **_k):
    pass


@unittest.skipUnless(_paths.HAS_SCIENCE and HAVE, "needs the science stack and tests/data/examples/graphene-soc")
class GrapheneSOC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from skfitter import workflow as J
        cls.project = tempfile.mkdtemp(prefix="prayog_gsoc_")
        cls.dft = J.prepare_project(cls.project, "espresso", {r: os.path.join(DATA, f) for r, f in FILES.items()},
                                    quiet, quiet)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.project, ignore_errors=True)

    def test_reading(self):
        m = self.dft["meta"]
        self.assertTrue(m["soc"])
        self.assertEqual((m["nelect"], m["spacegroup"]), (8.0, "P6/mmm"))
        self.assertEqual(self.dft["tick_labels"], ["Γ", "M", "K", "Γ"])
        E = self.dft["E"]["up"]
        self.assertEqual(E.shape, (113, 24))
        self.assertLess(E[64, 8] - E[64, 7], 5e-6)                         # Dirac point (K): gap < 5 micro-eV
        self.assertAlmostEqual(1e3 * (E[0, 7] - E[0, 4]), 9.2, delta=0.3)  # sigma splitting at Gamma

    def test_onsite_estimate(self):
        from skfitter import workflow as J
        f = J.build_fitter(CFG, self.project)
        est = J.estimate_onsite(self.dft, f, f._get_param_keys(), clip=False)
        # C 2s lies deep (band bottom -19.5 eV); the vacuum states above E_F must not pull it up
        self.assertLess(est["e_orbit0_C_s"], -7.0)
        self.assertLess(est["e_orbit0_C_s"], est["e_orbit0_C_p"])

    def test_fit_and_spin_orbit(self):
        from skfitter import workflow as J
        from skfitter import params as P
        f = J.build_fitter(CFG, self.project)
        sv = J.starting_values(f, CFG, P.bundled_sets()["harrison"], self.dft)
        init, bounds = dict(sv["initial"]), dict(sv["bounds"])
        init["lambda_C_p"], bounds["lambda_C_p"] = LAMBDA, (LAMBDA - 1e-5, LAMBDA + 1e-5)
        r = J.run_fit_job(CFG, {"method": "ls", "n_starts": 2, "refine_max_nfev": 300, "seed": 0}, init, bounds,
                          self.project, "gsoc", quiet, quiet, lambda: False, log=quiet, dft=self.dft)
        f = r["fitter"]
        self.assertLess(r["rmse"], 200.0, f"RMSE {r['rmse']:.0f} meV")      # about 130 meV
        B = f.structure.lattice.reciprocal_lattice.matrix

        def bands(kf):
            return np.linalg.eigvalsh(f._tb_orthogonal_hamiltonians((np.array(kf) @ B)[None, :])[0])
        g, k = bands([0, 0, 0]), bands([1 / 3, 1 / 3, 0])
        self.assertAlmostEqual(1e3 * (g[7] - g[4]), 9.2, delta=0.5)        # sigma splitting at Gamma, meV
        self.assertLess(abs(k[7]), 0.05)                                   # Dirac point at E_F
        gap = k[8] - k[7]
        self.assertTrue(0.0 < gap < 5e-6, f"gap at K {gap * 1e6:.3f} micro-eV")   # Kane-Mele gap, s + p only
        _, _, res = f.compute_spin_hall_conductivity(nk=(60, 60, 1), efermi_range=(-4.0, 1.0), num_efermi=251,
                                                     smearing=0.005, show=False)
        e, s = res["efermi"], res["sigma"]
        self.assertAlmostEqual(float(e[np.argmax(np.abs(s))]), -3.14, delta=0.1)   # at the SOC-split sigma bands
        self.assertGreater(float(np.abs(s).max()), 20.0)
        self.assertLess(abs(res["sigma_ef"]), 1.0)                          # the micro-eV gap is not resolved

        # the spin Berry curvature along the path (web: properties step), through the property runner
        spec = {"kind": "sbc", "source": "sk", "spin": "z", "alpha": "x", "beta": "y", "dk": 0.01, "efermi": 0.0,
                "smearing": 0.0, "units": "au"}
        out = J.run_property(f, spec, self.project, "gsoc", quiet, quiet, lambda: False)
        self.assertEqual(out["tag"], "sk_sbc_path_z_xy")
        self.assertTrue(os.path.isfile(out["png"]) and os.path.isfile(out["dat"]))
        self.assertEqual(out["table"]["headers"][:3], ["k_distance(1/A)", "total_all_bands", "total_fitted_bands"])
        self.assertTrue(out["res"]["paired"])                                # Kramers pairs (inversion + time reversal)
        self.assertGreater(float(np.abs(out["res"]["omega"]).max()), 10.0)  # SOC-split crossings give sharp peaks
        self.assertIn("sbc_path", [r["kind"] for r in J.scan_results(self.project, "gsoc")])


if __name__ == "__main__":
    unittest.main()
