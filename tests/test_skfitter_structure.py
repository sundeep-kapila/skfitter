"""The structure, neighbour and parameter views (skfitter_structure) on all illustrative projects.

1. reference files      tests/baselines/structure/<example>.json (make_structure_baselines.py)
2. independent count    a brute-force numpy neighbour count (no pymatgen) gives the same shells and counts
3. textbook counts      graphene / hBN 3-6-3, bcc Fe 8-6-12, fcc Pt 12-6-24, fcc Ce 12-6, Ce in CeO2 8 O
4. the fitter           bonds == SKFitter.neighbor_list, cutoff, d0; the integrals listed for each bond are exactly
                        the ones the Hamiltonian reads for it; V(r) is the value it uses; every fitted parameter
                        appears on a site or a bond
"""
import itertools
import json
import os
import re
import unittest

import _paths
import numpy as np

from _examples_structure import cfg, example_ids, fitter, project

EXAMPLES = example_ids()
REF = os.path.join(_paths.BASELINES, "structure")


def brute_force(meta, n_shells, search=15.0, decimals=3, margin=0.1):
    """(shells, cutoff, counts[i] = {rounded dist: count}) from lattice and fractional coordinates alone."""
    lat = np.asarray(meta["lattice"], float)
    cart = np.asarray([s["frac"] for s in meta["sites"]], float) @ lat
    layer = 1.0 / np.linalg.norm(np.linalg.inv(lat).T, axis=1)
    reach = [int(np.ceil(search / h)) + 1 for h in layer]
    D = []
    for t in itertools.product(*[range(-r, r + 1) for r in reach]):
        T = np.asarray(t, float) @ lat
        D.append(np.linalg.norm(cart[None, :, :] + T - cart[:, None, :], axis=2))      # D[i, j] = |r_j + T - r_i|
    D = np.stack(D)                                                                       # (n_T, i, j)
    shells = np.sort(np.unique(np.round(D[(D > 1e-4) & (D <= search)], decimals)))
    cutoff = shells[min(n_shells, len(shells)) - 1] + margin
    counts = []
    for i in range(len(cart)):
        d = D[:, i, :]
        d = np.round(d[(d > 1e-4) & (d < cutoff)], decimals)
        u, c = np.unique(d, return_counts=True)
        counts.append({round(float(a), 3): int(b) for a, b in zip(u, c)})
    return shells, cutoff, counts


class _Recorder(dict):
    """A parameter dict that records which keys are read."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


@unittest.skipUnless(_paths.HAS_SCIENCE, "needs scipy and pymatgen")
class StructureViews(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        from skfitter import structure as S
        cls.S = S

    def views(self, ex):
        _, dft = project(ex)
        meta, c = dft["meta"], cfg(ex)
        return meta, c, self.S.structure_view(meta, c["symprec"]), self.S.neighbour_view(meta, c)

    # 1 ---------------------------------------------------------------------------------------------------------
    def test_all_examples_present(self):
        self.assertEqual(len(EXAMPLES), 10, EXAMPLES)

    def test_reference_files(self):
        import make_structure_baselines as M
        for ex in EXAMPLES:
            with self.subTest(ex):
                with open(os.path.join(REF, f"{ex}.json"), encoding="utf-8") as f:
                    ref = json.load(f)
                got = json.loads(json.dumps(M.summary(ex)))
                for k in ("formula", "spacegroup", "n_sites", "n_inequivalent", "dimensionality", "n_neighbors",
                          "n_shells", "n_pairs", "n_bonds", "hopping_keys", "warnings"):
                    self.assertEqual(got[k], ref[k], k)
                for k in ("abc", "angles", "shells"):
                    np.testing.assert_allclose(got[k], ref[k], atol=2e-3, err_msg=k)
                self.assertEqual(sorted(got["pair_shells"]), sorted(ref["pair_shells"]))      # shells per element pair
                for k in ref["pair_shells"]:
                    np.testing.assert_allclose(got["pair_shells"][k], ref["pair_shells"][k], atol=2e-3, err_msg=k)
                for k in ("volume", "cutoff", "d0"):
                    self.assertAlmostEqual(got[k], ref[k], places=2, msg=k)
                self.assertEqual([[r[0], r[2], r[3]] for s in got["sites"] for r in s["shells"]],
                                 [[r[0], r[2], r[3]] for s in ref["sites"] for r in s["shells"]])

    # 2 ---------------------------------------------------------------------------------------------------------
    def test_independent_neighbour_count(self):
        for ex in EXAMPLES:
            with self.subTest(ex):
                meta, c, _, nv = self.views(ex)
                shells, cutoff, counts = brute_force(meta, c["n_neighbors"])
                self.assertAlmostEqual(nv["cutoff"], cutoff, places=3)
                np.testing.assert_allclose(nv["shells"][:nv["n_shells"]], shells[:nv["n_shells"]], atol=1.5e-3)
                for s, ref in zip(nv["sites"], counts):
                    got = {round(r["dist"], 3): r["count"] for r in s["shells"]}
                    self.assertEqual(sorted(got.values()), sorted(ref.values()), s["label"])
                    np.testing.assert_allclose(sorted(got), sorted(ref), atol=1.5e-3)
                self.assertEqual(nv["n_pairs"], sum(sum(cn.values()) for cn in counts))

    # 3 ---------------------------------------------------------------------------------------------------------
    def test_textbook_counts(self):
        expect = {
            "graphene-soc": {"C1": [3, 6, 3]},
            "graphene-rashba": {"C1": [3, 6, 3]},
            "hbn-valley": {"B1": [3, 6, 3], "N1": [3, 6, 3]},
            "iron-soc-qe": {"Fe1": [8, 6, 12, 24]}, "platinum-soc-qe": {"Pt1": [12, 6, 24, 12]},
            "cerium-f": {"Ce1": [12, 6]},
        }
        for ex, sites in expect.items():
            with self.subTest(ex):
                *_, nv = self.views(ex)
                for s in nv["sites"]:
                    if s["label"] in sites:
                        self.assertEqual([r["count"] for r in s["shells"]], sites[s["label"]])
        *_, nv = self.views("ceria-f")
        ce = next(s for s in nv["sites"] if s["label"] == "Ce1")
        self.assertEqual(ce["shells"][0]["by_element"], {"O": 8})
        self.assertEqual(ce["missing_shells"], [2])                    # O-O distance: no Ce at 2.705 Å

    def test_structure_view(self):
        for ex in EXAMPLES:
            with self.subTest(ex):
                meta, c, sv, _ = self.views(ex)
                self.assertEqual(sv["n_sites"], len(meta["sites"]))
                self.assertEqual([e["symbol"] for e in sv["elements"]], meta["elements"])
                self.assertTrue(all(e["color"].startswith("#") and len(e["color"]) == 7 for e in sv["elements"]))
                lat = np.asarray(sv["lattice"])
                for s in sv["sites"]:
                    np.testing.assert_allclose(np.asarray(s["frac"]) @ lat, s["cart"], atol=1e-4)
                # every bond's translation reproduces its length
                cart = np.asarray([s["cart"] for s in sv["sites"]])
                for b in sv["bonds"]:
                    d = np.linalg.norm(cart[b["j"]] + np.asarray(b["image"]) @ lat - cart[b["i"]])
                    self.assertAlmostEqual(d, b["dist"], places=3)
                    self.assertAlmostEqual(sv["shells"][b["shell"] - 1], b["dist"], places=2)
        self.assertEqual(self.views("graphene-soc")[2]["dimensionality"], "2D")
        self.assertEqual(self.views("platinum-soc-qe")[2]["dimensionality"], "3D")

    def test_flags(self):
        *_, nv = self.views("mnte-qe")                        # shell 3 = 4.167 Å, shell 4 = 4.218 Å
        self.assertTrue(any("also takes in shell 4" in w for w in nv["warnings"]), nv["warnings"])
        *_, nv = self.views("graphene-soc")
        self.assertEqual(nv["warnings"], [])
        _, dft = project("graphene-soc")
        nv = self.S.neighbour_view(dft["meta"], dict(cfg("graphene-soc"), n_neighbors=60))
        self.assertTrue(nv["warnings"][0].startswith("Only "))

    # 4 ---------------------------------------------------------------------------------------------------------
    def test_bonds_are_the_fitters(self):
        for ex in EXAMPLES:
            with self.subTest(ex):
                meta, c, _, nv = self.views(ex)
                f = fitter(ex)
                self.assertAlmostEqual(nv["cutoff"], f.cutoff_radius, places=3)
                self.assertAlmostEqual(nv["d0"], f.d0, places=3)
                self.assertEqual(nv["n_pairs"], len(f.neighbor_list))
                mine = []
                lat = np.asarray(meta["lattice"])
                cart = np.asarray([s["frac"] for s in meta["sites"]]) @ lat
                for b in nv["bonds"]:
                    d = cart[b["j"]] + np.asarray(b["image"]) @ lat - cart[b["i"]]
                    mine += [(b["i"], b["j"], *d), (b["j"], b["i"], *(-d))]
                theirs = [(int(p["i"]), int(p["j"]), *p["d_vec"]) for p in f.neighbor_list]
                self.assertEqual(len(mine), len(theirs))
                left = list(theirs)
                for t in mine:                                           # one-to-one match, i, j equal, |Δd| < 1e-6
                    hit = next((n for n, u in enumerate(left) if u[:2] == t[:2] and
                                np.allclose(u[2:], t[2:], atol=1e-6)), None)
                    self.assertIsNotNone(hit, t)
                    left.pop(hit)
                self.assertEqual(sorted(k["key"] for k in nv["hopping_keys"]), sorted(f._get_required_hopping_keys()))

    def _random_params(self, f, seed=1):
        rng = np.random.default_rng(seed)
        keys = f._get_param_keys()
        vec = rng.uniform(0.3, 2.0, len(keys))
        for n, k in enumerate(keys):
            if k.startswith("eta"):
                vec[n] = rng.uniform(1.0, 4.0)
        return f._param_vec_to_dict(vec)

    def test_bond_integrals_are_what_the_hamiltonian_reads(self):
        for ex in EXAMPLES:
            with self.subTest(ex):
                meta, c, *_ = self.views(ex)
                f = fitter(ex)
                params = self._random_params(f)
                pm = self.S.param_map(meta, c, params)
                lat = np.asarray(meta["lattice"])
                cart = np.asarray([s["frac"] for s in meta["sites"]]) @ lat
                checked = 0
                for b in pm["bonds"]:
                    R = cart[b["j"]] + np.asarray(b["image"]) @ lat - cart[b["i"]]
                    ei, ej = (f.structure[b[k]].specie.symbol for k in ("i", "j"))
                    rec = _Recorder(params)
                    for s1 in f.site_orbitals[b["i"]]:
                        for s2 in f.site_orbitals[b["j"]]:
                            f._get_submatrix_hopping(ei, s1, ej, s2, R, rec)
                            f._get_submatrix_hopping(ej, s2, ei, s1, -R, rec)
                    read = {re.sub(r"_\d+$", "", k) for k in rec.read if k.startswith("v_") and re.search(r"_\d+$", k)}
                    self.assertEqual({it["key"] for it in b["integrals"]}, read, b["label"])
                    # V(r): the block with (V0, eta) equals the block with V(r) and no distance decay
                    for it in b["integrals"][:4]:
                        bt = self.S._bond_type(it["key"])
                        name = it.get("shell_key") or f"{it['key']}_0"      # per shell: the bond's own V_n
                        one = {name: it["v0"], f"eta_{bt}": params.get(f"eta_{bt}", 2.5)}
                        flat = {name: it["value"], f"eta_{bt}": 0.0}
                        for s1 in f.site_orbitals[b["i"]]:
                            for s2 in f.site_orbitals[b["j"]]:
                                np.testing.assert_allclose(f._get_submatrix_hopping(ei, s1, ej, s2, R, one),
                                                           f._get_submatrix_hopping(ei, s1, ej, s2, R, flat),
                                                           rtol=1e-10, atol=1e-12)
                        checked += 1
                self.assertGreater(checked, 0)

    def test_every_parameter_is_shown(self):
        for ex in EXAMPLES:
            with self.subTest(ex):
                meta, c, *_ = self.views(ex)
                f = fitter(ex)
                params = self._random_params(f, seed=2)
                pm = self.S.param_map(meta, c, params)
                shown = set()
                for s in pm["sites"]:
                    for row in s["onsite"] + s["soc"] + s["exchange"]:
                        self.assertIsNotNone(row["value"], row["key"])
                        shown.add(row["key"])
                for b in pm["bonds"]:
                    for it in b["integrals"]:
                        shown.add(it.get("shell_key") or f"{it['key']}_0")
                        if it["eta"] is not None:
                            shown.add(f"eta_{self.S._bond_type(it['key'])}")
                        if pm["non_orthogonal"]:
                            shown.add(f"s_{it['key'][2:]}_0")
                            if it["s_eta"] is not None:
                                shown.add(f"eta_s_{self.S._bond_type(it['key'])}")
                missing = set(f._get_param_keys()) - shown
                self.assertEqual(missing, set())
                # on-site value of every site is the one the Hamiltonian puts on its diagonal
                H = f.build_hamiltonian(np.zeros(3), {k: (v if k.startswith(("e_", "eta")) else 0.0) for k, v in params.items()})
                for i, s in enumerate(pm["sites"]):
                    o = f.site_offsets[i]
                    for row in s["onsite"]:
                        dim = f.SUB_ORBITALS[row["orbital"]]["dim"]
                        np.testing.assert_allclose(np.diag(H)[o:o + dim].real, row["value"])
                        o += dim
                for s, sign in zip(pm["sites"], f.site_moment_signs):
                    for row in s["exchange"]:
                        self.assertEqual(row["sign"], sign)


if __name__ == "__main__":
    unittest.main()
