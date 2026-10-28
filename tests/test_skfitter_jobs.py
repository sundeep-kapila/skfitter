"""skfitter_jobs: the Qt-free SKFitter layer shared by the desktop GUI and the web server.

The first group needs only numpy + matplotlib.  The second group needs skfitter.py's own dependencies
(scipy, pymatgen) and is skipped where those are missing; the GitHub test workflow has them.
A full fit on real DFT data is not tested yet (no small example data set in the repository so far).
"""
import io
import os
import sys
import tempfile
import time
import unittest

import numpy as np

import _paths
from skfitter import workflow as J

KPOINTS_LINE_MODE = """k-path for fcc
40
Line-mode
Reciprocal
0.000 0.000 0.000 ! \\Gamma
0.500 0.000 0.500 ! X

0.500 0.000 0.500 ! X
0.500 0.250 0.750 ! W
"""


class Helpers(unittest.TestCase):
    def test_parse_line_mode_kpoints(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "KPOINTS")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(KPOINTS_LINE_MODE)
            k = J.parse_kpoints_file(path)
        self.assertEqual(k["mode"], "line")
        self.assertEqual(k["npts"], 40)
        segs = k["segments"]
        self.assertEqual(len(segs), 2)
        self.assertEqual((segs[0]["l0"], segs[0]["l1"], segs[1]["l0"], segs[1]["l1"]), (J.GAMMA, "X", "X", "W"))
        self.assertEqual((segs[0]["i0"], segs[0]["i1"], segs[1]["i0"], segs[1]["i1"]), (0, 39, 40, 79))
        np.testing.assert_allclose(segs[1]["k1"], [0.5, 0.25, 0.75])

    def test_non_line_mode_kpoints(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "KPOINTS")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("mesh\n0\nGamma\n8 8 8\n0 0 0\n")
            self.assertEqual(J.parse_kpoints_file(path)["segments"], [])
            self.assertEqual(J.parse_kpoints_file(os.path.join(d, "missing"))["mode"], "unknown")

    def test_default_bounds_and_groups(self):
        self.assertEqual(J.default_bound("lambda_Si_p"), (0.0, 1.0))
        self.assertEqual(J.default_bound("e_orbit0_s_Si"), (-10.0, 15.0))
        self.assertEqual(J.default_bound("v_Si_s-Si_s_sss_1"), (-10.0, 2.0))
        self.assertEqual(J.default_bound("v_Si_p-Si_p_pps_1"), (-2.0, 8.0))
        self.assertEqual(J.default_bound("s_Si_s-Si_s_sss_1"), (-0.8, 0.8))
        self.assertEqual(J.default_initial("v_S_p-Mo_d_pds_1"), -2.0)
        self.assertEqual(J.param_group("e_orbit0_s_Si"), "On-site energy")
        self.assertEqual(J.param_group("v_Si_s-Si_p_sps_1"), "Hopping")

    def test_fit_kwargs_defaults_and_overrides(self):
        cfg = {"fit_eta": False, "fixed_eta": 2.0}
        keys = ["a", "b"]
        kw = J.fit_kwargs(cfg, None, {"a": 1, "zzz": 5}, {"b": (-1, 1)}, keys)
        self.assertEqual((kw["method"], kw["n_starts"], kw["refine_max_nfev"], kw["seed"]), ("ls", 8, 300, 0))
        self.assertEqual(kw["initial_params"], {"a": 1.0})           # keys that are not fitted are dropped
        self.assertEqual(kw["custom_bounds"], {"b": (-1.0, 1.0)})
        kw = J.fit_kwargs(cfg, {"method": "de", "n_starts": 3, "seed": ""}, {}, {}, keys)
        self.assertEqual((kw["method"], kw["n_starts"], kw["seed"]), ("de", 3, None))

    def test_captured_relays_stdout_and_stderr_by_line(self):
        lines = []
        with J.captured(lines.append):
            print("first")
            print("second part", end="")
            sys.stderr.write(" continues\n")
            print("   ")                            # blank lines are dropped
            print("last", end="")
        self.assertEqual(lines, ["first", "second part continues", "last"])

    def test_fit_progress_patterns(self):
        m = J.DE_STEP.search("differential_evolution step 12: f(x)= 3.25e-02")
        self.assertEqual((m.group(1), float(m.group(2))), ("12", 0.0325))
        m = J.LS_START.search("  start 3: loss = 1.5e+01")
        self.assertEqual((m.group(1), float(m.group(2))), ("3", 15.0))
        self.assertIsNone(J.LS_START.search("  start 4: failed").group(2))

    def test_sidecar_and_saved_fits(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(J.read_sidecar(d, "si"), {})
            J.update_sidecar(d, "si", rmse=12.5)
            self.assertEqual(J.update_sidecar(d, "si", method="ls"), {"rmse": 12.5, "method": "ls"})
            for seed, suffix in (("old", "_fit.json"), ("new", "_params.txt")):
                with open(os.path.join(d, seed + suffix), "w", encoding="utf-8") as fh:
                    fh.write("{}")
                time.sleep(0.05)
            os.utime(os.path.join(d, "old_fit.json"), (1, 1))
            self.assertEqual(J.find_saved_fits(d), ["new", "old"])

    def test_project_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(J.read_project_file(d), {})
            data = J.write_project_file(d, seedname="si", dft_code="vasp")
            self.assertIn("created", data)
            self.assertEqual(J.read_project_file(d)["seedname"], "si")

    def test_property_tags(self):
        self.assertEqual(J.property_tag({"kind": "bc", "source": "sk", "mode": "path", "component": "z"}),
                         "sk_bc_path_z")
        self.assertEqual(J.property_tag({"kind": "bc", "source": "sk", "mode": "wilson", "component": "z"}), "sk_bc_wilson")
        self.assertEqual(J.property_tag({"kind": "bc", "source": "sk", "mode": "chern", "component": "z"}), "sk_bc_chern")
        self.assertTrue(J.PROPERTY_TAG_RE.match("sk_bc_wilson") and J.PROPERTY_TAG_RE.match("sk_bc_chern"))
        self.assertEqual(J.property_tag({"kind": "ahc", "source": "wb"}), "wb_ahc")
        self.assertEqual(J.property_tag({"kind": "shc", "source": "sk", "spin": "z", "alpha": "x", "beta": "y"}),
                         "sk_shc_z_xy")
        self.assertTrue(J.PROPERTY_TAG_RE.match("sk_shc_z_xy"))

    def test_figures_render_without_a_display(self):
        from matplotlib.figure import Figure
        fig = Figure()
        J.draw_convergence(fig, [0, 1, 2], [3.0, 2.0, 1.5])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        self.assertGreater(len(buf.getvalue()), 1000)


@unittest.skipUnless(_paths.HAS_SCIENCE, "needs scipy and pymatgen (skfitter.py)")
class WithSKFitter(unittest.TestCase):
    def test_saved_fit_suffixes_match_skfitter(self):
        self.assertEqual(tuple(J.FIT_FILE_SUFFIXES), tuple(J.get_skfitter().SKFitter.PARAM_FILE_SUFFIXES))

    def test_slater_koster_blocks(self):
        S = J.get_skfitter().SKFitter
        np.testing.assert_allclose(S._sk_hopping_s_s(1, 0, 0, -1.5), [[-1.5]])
        np.testing.assert_allclose(S._sk_hopping_s_p(0, 0, 1, 2.0), [[0, 0, 2.0]])
        pp = S._sk_hopping_p_p(1.0, 0.0, 0.0, 3.0, -1.0)          # bond along x: px sigma, py/pz pi
        np.testing.assert_allclose(pp, np.diag([3.0, -1.0, -1.0]))
        l = m = n = 1 / np.sqrt(3)
        pp = S._sk_hopping_p_p(l, m, n, 3.0, -1.0)
        np.testing.assert_allclose(pp, pp.T)
        np.testing.assert_allclose(np.linalg.eigvalsh(pp), [-1.0, -1.0, 3.0], atol=1e-12)

    def test_orbital_config_checks(self):
        SUB, _ = J.sk_tables()
        orb = sorted(SUB)[0]
        self.assertIn("at least one orbital", J.validate_orbital_config({"Si": []}))
        self.assertIn("at least one orbital", J.validate_orbital_config({"Si": [], "O": []}))
        # an element without orbitals is allowed (a warning), as long as some element has one
        self.assertEqual(J.validate_orbital_config({"Si": [orb], "O": []}), "")
        self.assertIn("No orbital is chosen for O", J.orbital_warning({"Si": [orb], "O": []}))
        self.assertEqual(J.orbital_warning({"Si": [orb], "O": ["s"]}), "")
        self.assertEqual(J.orbital_warning({"Si": [], "O": []}), "")
        self.assertIn("overlap", J.validate_orbital_config({"Si": [orb, orb]}))
        self.assertEqual(J.validate_orbital_config({"Si": [orb]}), "")
        self.assertEqual(J.basis_size({"Si": [orb]}, [{"symbol": "Si"}, {"symbol": "Si"}]),
                         2 * len(SUB[orb]["indices"]))


if __name__ == "__main__":
    unittest.main()
