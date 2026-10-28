"""SKFitter f orbitals on Quantum ESPRESSO data made with tools/qe_f_material (PBE, SSSP pseudopotentials, no SOC):
fcc Ce (metal, 4f band at E_F) and CeO2 (fluorite, an empty 4f band in the gap). Reading the files (bands, k-path,
Fermi level, f projections), the f band centre as on-site start, a Ce model with s, d, f, and a CeO2 fit with O p
and Ce d, f."""
import os
import tempfile
import unittest

import _paths
import numpy as np

FOLDER = {"ce": "cerium-f", "ceo2": "ceria-f"}          # the illustrative projects of the two materials
FILES = ("qe_bands.xml", "qe_bands.in", "qe_scf.out", "qe_projwfc_up")


def have(material):
    return all(os.path.isfile(os.path.join(_paths.example_dir(FOLDER[material]), f)) for f in FILES)


def load(material):
    from skfitter import workflow as J
    d = _paths.example_dir(FOLDER[material])
    project = tempfile.mkdtemp(prefix=f"prayog_{material}_")
    dft = J.prepare_project(project, "espresso", {"qe_xml": os.path.join(d, "qe_bands.xml"),
                                                  "qe_bands_in": os.path.join(d, "qe_bands.in"),
                                                  "qe_scf": os.path.join(d, "qe_scf.out"),
                                                  "qe_projwfc": os.path.join(d, "qe_projwfc_up")},
                            lambda *a: None, lambda *a: None)
    return project, dft


def f_bands(dft):
    """(band index, mean energy, mean f weight on Ce) for every band."""
    m = dft["meta"]
    cols = [i for i, n in enumerate(m["orbital_names"]) if n.startswith("f")]
    ce = [i for i, s in enumerate(m["sites"]) if s["symbol"] == "Ce"]
    P, E = dft["P"]["up"], dft["E"]["up"]
    w = P[:, :, ce, :][:, :, :, cols].sum(axis=(2, 3)).mean(axis=0)
    return [(b, float(E[:, b].mean()), float(w[b])) for b in range(E.shape[1])]


def cfg(orbitals, below, above, shells):
    return {"orbital_config": orbitals, "magnetic_orbitals": {}, "n_bands_below": below, "n_bands_above": above,
            "n_neighbors": shells, "symprec": 1e-3, "fit_eta": False, "fixed_eta": 2.5, "include_soc": False,
            "non_orthogonal": False, "tb_nelect": None}


@unittest.skipUnless(_paths.HAS_SCIENCE and have("ce"), "needs the science stack and tests/data/examples/cerium-f")
class Cerium(unittest.TestCase):
    def test_reading_and_model(self):
        import shutil
        from skfitter import workflow as J
        from skfitter import params as P
        project, dft = load("ce")
        try:
            m = dft["meta"]
            self.assertEqual((m["formula"], m["spacegroup"], m["nelect"]), ("Ce", "Fm-3m", 12.0))
            self.assertEqual(dft["tick_labels"], ["Γ", "X", "W", "K", "Γ", "L", "U", "W", "L", "K"])
            self.assertEqual(dft["P"]["up"].shape[-1], 16)                    # m-resolved s, p, d, f
            fb = [b for b in f_bands(dft) if b[2] > 0.9]
            self.assertGreaterEqual(len(fb), 5)                               # the 4f band ...
            self.assertTrue(all(-0.3 < e < 1.0 for _, e, _ in fb))           # ... at the Fermi level
            c = cfg({"Ce": ["s", "d", "f"]}, 2, 8, 2)                         # valence window (5p semicore excluded)
            f = J.build_fitter(c, project)
            keys = f._get_param_keys()
            for k in ("e_orbit0_Ce_f", "v_Ce_s-Ce_f_sfs_0", "v_Ce_d-Ce_f_dfs_0", "v_Ce_d-Ce_f_dfp_0",
                      "v_Ce_d-Ce_f_dfd_0", "v_Ce_f-Ce_f_ffs_0", "v_Ce_f-Ce_f_ffp_0", "v_Ce_f-Ce_f_ffd_0",
                      "v_Ce_f-Ce_f_fff_0"):
                self.assertIn(k, keys)
            self.assertEqual(f.tb_nelect, 4)                                  # 4f1 5d1 6s2
            init = J.default_initials(f, dft)
            self.assertTrue(0.0 < init["e_orbit0_Ce_f"] < 2.0)               # the f band centre (about 1.3 eV)
            self.assertTrue(f.check_fast_model(np.array([init[k] for k in keys]), n_k=6, verbose=False))
            sv = J.starting_values(f, c, P.bundled_sets()["nist"], dft)
            self.assertEqual(sv["source"]["e_orbit0_Ce_f"], "set")
        finally:
            shutil.rmtree(project, ignore_errors=True)


@unittest.skipUnless(_paths.HAS_SCIENCE and have("ceo2"), "needs the science stack and tests/data/examples/ceria-f")
class CeriumDioxide(unittest.TestCase):
    def test_reading_and_fit(self):
        import shutil
        from skfitter import workflow as J
        project, dft = load("ceo2")
        try:
            m = dft["meta"]
            self.assertEqual((m["formula"], m["nelect"]), ("CeO2", 24.0))
            fb = [b for b in f_bands(dft) if b[2] > 0.75]
            self.assertEqual(len(fb), 7)                                      # the empty 4f band: 7 bands ...
            self.assertTrue(all(1.3 < e < 3.0 for _, e, _ in fb))            # ... in the gap
            c = cfg({"Ce": ["d", "f"], "O": ["p"]}, 6, 7, 3)                  # O 2p valence band + Ce 4f
            f = J.build_fitter(c, project)
            keys = f._get_param_keys()
            self.assertIn("v_O_p-Ce_f_pfs_0", keys)
            self.assertIn("v_O_p-Ce_f_pfp_0", keys)
            init = J.default_initials(f, dft)
            self.assertTrue(1.3 < init["e_orbit0_Ce_f"] < 3.0)
            f.fit(maxiter=50, method="ls", n_starts=3, refine=True, refine_max_nfev=200, initial_params=init,
                  custom_bounds={k: J.default_bound(k) for k in keys}, fit_eta=False, fixed_eta=2.5, seed=0)
            rmse = J.error_report(f)["all"]
            self.assertLess(rmse, 250.0, f"RMSE {rmse:.0f} meV")               # about 100 meV with 8 starts
            self.assertLess(abs(f.opt_params["v_Ce_f-Ce_f_ffs_0"]), 1.0)                 # compact 4f: small f-f coupling
        finally:
            shutil.rmtree(project, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
