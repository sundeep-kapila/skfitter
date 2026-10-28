"""Exchange splitting per symmetry subshell (magnetic_orbitals may name d_t2g, d_eg, pz, ... next to families).

Bcc Fe (tests/data/examples/iron-soc-qe: QE, spin-orbit, moment along z; cubic axes along x, y, z, so d can be split
into t2g / e_g - the VASP example's cell is rotated, where that split breaks the symmetry):
* ['s', 'd'] keeps one delta for all d also when d is split into t2g / e_g; ['s', 'd_t2g', 'd_eg'] gives two;
  a family next to one of its subshells covers the rest of the shell;
* p subshells work the same way (pz / pxy);
* equal split deltas reproduce the shared delta exactly; the vectorised bands agree with the reference H(k);
* a saved SKModel (no magnetic_orbitals stored) rebuilds the split exchange from its parameter keys;
* a subshell the orbital set does not have is an error (it used to be ignored silently).
"""
import contextlib
import io
import json
import os
import unittest

import _paths
import numpy as np

HAS = _paths.HAS_SCIENCE
FE_DIR = _paths.data_folder("iron-soc-qe") if HAS and os.path.isdir(_paths.data_folder("iron-soc-qe")) else ""
_TMP = None


def _qe_dir():
    """The example's QE files unpacked once into a temporary folder."""
    global _TMP
    if _TMP is None:
        import gzip
        import shutil
        import tempfile
        _TMP = tempfile.mkdtemp(prefix="fe_qe_")
        for fn in os.listdir(FE_DIR):
            if fn.startswith("qe_"):
                dst = os.path.join(_TMP, fn[:-3] if fn.endswith(".gz") else fn)
                if fn.endswith(".gz"):
                    with gzip.open(os.path.join(FE_DIR, fn)) as a, open(dst, "wb") as b:
                        shutil.copyfileobj(a, b)
                else:
                    shutil.copy(os.path.join(FE_DIR, fn), dst)
    return _TMP


def fitter(orbs, mag, **kw):
    import skfitter.core as S
    d = _qe_dir()
    args = dict(dft_code="espresso", qe_xml_path=os.path.join(d, "qe_bands.xml"),
                qe_bands_input=os.path.join(d, "qe_bands.in"), qe_scf_path=os.path.join(d, "qe_scf.out"),
                orbital_config={"Fe": orbs}, magnetic_orbitals={"Fe": mag}, n_bands_below=4, n_bands_above=1,
                n_neighbors=2, include_soc=True)
    args.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return S.SKFitter(**args)


def params(f, seed=1, **fixed):
    keys = f._get_param_keys()
    v = np.random.default_rng(seed).normal(0.0, 0.4, len(keys))
    for i, k in enumerate(keys):
        if k.startswith("eta_"):
            v[i] = 3.0
    p = f._param_vec_to_dict(v)
    p.update(fixed)
    return p


class ExchangeKey(unittest.TestCase):
    def test_resolution(self):
        if not HAS:
            self.skipTest("needs the science stack")
        import skfitter.core as S
        k = S.exchange_key
        self.assertEqual(k("Fe", "d_t2g", {"Fe": ["d"]}), "delta_spin_Fe_d")
        self.assertEqual(k("Fe", "d_t2g", {"Fe": ["d_t2g", "d_eg"]}), "delta_spin_Fe_d_t2g")
        self.assertEqual(k("Fe", "d_t2g", {"Fe": ["d", "d_eg"]}), "delta_spin_Fe_d")
        self.assertEqual(k("Fe", "d_eg", {"Fe": ["d", "d_eg"]}), "delta_spin_Fe_d_eg")
        self.assertIsNone(k("Fe", "d_t2g", {"Fe": ["d_eg"]}))
        self.assertEqual(k("Te", "pz", {"Te": ["pz"]}), "delta_spin_Te_pz")
        self.assertEqual(k("Fe", "d", {"Fe": ["d"]}), "delta_spin_Fe_d")
        self.assertEqual(S.magnetic_orbitals_from_params({"delta_spin_Fe_d_t2g": 1, "delta_spin_Fe_s": 0, "v_sss_0": 1}),
                         {"Fe": ["d_t2g", "s"]})
        errs = S.check_magnetic_orbitals({"Fe": ["d_t2g"]}, {"Fe": ["s", "d"]})
        self.assertTrue(errs and "split" in errs[0])
        self.assertFalse(S.check_magnetic_orbitals({"Fe": ["s", "d_eg"]}, {"Fe": ["s", "d_t2g", "d_eg"]}))
        self.assertTrue(S.check_magnetic_orbitals({"Fe": ["x2"]}, {"Fe": ["s"]}))


@unittest.skipUnless(HAS and FE_DIR, "needs the science stack and tests/data/examples/iron-soc-qe")
class SplitExchange(unittest.TestCase):
    SPLIT = ["s", "d_t2g", "d_eg"]

    def deltas(self, f):
        return [k for k in f._get_param_keys() if k.startswith("delta_spin_")]

    def test_keys(self):
        self.assertEqual(self.deltas(fitter(self.SPLIT, ["s", "d"])), ["delta_spin_Fe_s", "delta_spin_Fe_d"])
        self.assertEqual(self.deltas(fitter(self.SPLIT, ["s", "d_t2g", "d_eg"])),
                         ["delta_spin_Fe_s", "delta_spin_Fe_d_t2g", "delta_spin_Fe_d_eg"])
        self.assertEqual(self.deltas(fitter(self.SPLIT, ["d", "d_eg"])), ["delta_spin_Fe_d", "delta_spin_Fe_d_eg"])
        with self.assertRaises(ValueError):            # pz / pxy is not a cubic split: refused (orbital groups)
            fitter(["s", "pz", "pxy", "d"], ["pz", "d"])

    def test_bad_subshell(self):
        with self.assertRaises(ValueError) as ctx:
            fitter(["s", "d"], ["s", "d_t2g"])
        self.assertIn("split", str(ctx.exception))

    def test_equal_deltas_match_shared(self):
        shared, split = fitter(self.SPLIT, ["s", "d"]), fitter(self.SPLIT, ["s", "d_t2g", "d_eg"])
        p = params(shared, 3, delta_spin_Fe_d=2.3, delta_spin_Fe_s=0.2)
        q = {k: v for k, v in p.items() if k != "delta_spin_Fe_d"}
        q.update(delta_spin_Fe_d_t2g=2.3, delta_spin_Fe_d_eg=2.3)
        k = np.array([0.1, 0.2, 0.3])
        a = np.linalg.eigvalsh(shared.build_hamiltonian(k, p))
        b = np.linalg.eigvalsh(split.build_hamiltonian(k, q))
        self.assertLess(np.abs(a - b).max(), 1e-12)
        q["delta_spin_Fe_d_eg"] = 3.1                        # different splittings: the bands change
        c = np.linalg.eigvalsh(split.build_hamiltonian(k, q))
        self.assertGreater(np.abs(a - c).max(), 1e-3)

    def test_onsite_split(self):
        f = fitter(self.SPLIT, ["s", "d_t2g", "d_eg"])
        H = f.build_hamiltonian(np.zeros(3), {"delta_spin_Fe_d_eg": 2.0})
        N = f.base_dim
        diag = np.real(np.diag(H))[:N]
        sub = []
        for s in f.site_orbitals[0]:
            sub += [s] * f.SUB_ORBITALS[s]["dim"]
        for i, s in enumerate(sub):
            self.assertAlmostEqual(diag[i], -1.0 if s == "d_eg" else 0.0, places=12)      # spin up: majority
            self.assertAlmostEqual(np.real(H[N + i, N + i]), 1.0 if s == "d_eg" else 0.0, places=12)

    def test_fast_model(self):
        f = fitter(self.SPLIT, ["s", "d_t2g", "d_eg"])
        self.assertTrue(f.check_fast_model(np.array(list(params(f, 5).values()))[[
            list(params(f, 5)).index(k) for k in f._get_param_keys()]], verbose=False))

    def test_sk_model(self):
        from skfitter.model.sk import SKModel
        f = fitter(self.SPLIT, ["s", "d_t2g", "d_eg"])
        p = params(f, 7, delta_spin_Fe_d_t2g=2.0, delta_spin_Fe_d_eg=3.0)
        m = SKModel.from_dict(json.loads(json.dumps(SKModel.from_fitter(f, p).to_dict())))
        kk = np.asarray(f.kpoints[:4])
        Em = np.array([np.linalg.eigvalsh(m.bloch(k)) for k in kk])
        Ef = np.array([np.linalg.eigvalsh(f.build_hamiltonian(k, p)) for k in kk])
        self.assertLess(np.abs(Em - Ef).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS and FE_DIR, "needs the science stack and tests/data/examples/iron-soc-qe")
class ExchangeSignConvention(unittest.TestCase):
    """δ > 0 on a site with a moment along +z puts its spin-up orbitals below the spin-down ones (spin up is the
    majority, as in DFT); with the moment along -z (sign -1) the order is reversed."""

    def test_majority_spin_is_up_for_a_positive_moment(self):
        f = fitter(["s", "p", "d"], ["d"], include_soc=False)
        N = f.base_dim
        sub = []
        for s in f.site_orbitals[0]:
            sub += [s] * f.SUB_ORBITALS[s]["dim"]
        d = [i for i, s in enumerate(sub) if s == "d"]
        for sign in (1, -1):
            f.site_moment_signs = [sign]
            H = np.real(np.diag(f.build_hamiltonian(np.zeros(3), {"delta_spin_Fe_d": 2.0})))
            self.assertTrue(np.allclose(H[d], -sign) and np.allclose(H[[N + i for i in d]], sign), (sign, H))
