"""SKFitter: atoms whose element has no orbitals chosen are left out of the tight-binding model (allowed as long
as some element has orbitals). hBN with p_z on the boron site only (the core also takes the
set-up per site, which lets one element-like group be empty with a one-element test case)."""
import unittest

import _paths


@unittest.skipUnless(_paths.HAS_SCIENCE, "needs the science stack")
class NoOrbitals(unittest.TestCase):
    def test_site_without_orbitals(self):
        from _examples_structure import project as example_project
        from skfitter import workflow as J
        project, dft = example_project("hbn-valley")
        if True:
            sk = J.get_skfitter()
            f = sk.SKFitter(**J.fitter_source_kwargs(J.project_sources(project)), orbital_config={0: ["pz"], 1: []},
                            n_bands_below=1, n_bands_above=1, n_neighbors=2, fit_eta=False, fixed_eta=2.5)
            self.assertEqual(f.base_dim, 1)
            self.assertEqual(f.site_orbitals[1], [])
            self.assertEqual([k for k in f._get_param_keys() if k.startswith("e_orbit")], ["e_orbit0_B_pz"])
            info = J.describe_fitter(f)["info"]
            self.assertEqual(info["total_dim"], 1)
            env = J.model_env(f, {"orbital_config": {"B": ["pz"], "N": []}, "n_neighbors": 2, "fit_eta": False,
                                  "fixed_eta": 2.5, "non_orthogonal": False}, dft)
            self.assertAlmostEqual(env["d0"], 1.45, delta=0.03)


if __name__ == "__main__":
    unittest.main()
