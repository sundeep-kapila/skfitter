"""SKFitter starting values from parameter sets (skfitter_params.py): reading the source formats, the shipped sets,
and mapping a set onto a model. numpy only - runs everywhere."""
import io
import math
import unittest
import zipfile

import _paths  # noqa: F401

from skfitter import params as P

# Silicon, s and p only, from the NRL file si_par.spd (Bernstein et al., PRB 62, 4477 (2000)): enough lines to test
# the reader and the formulas.
NRL_SI = """NN00001                           (New style Overlap Parameters)
Silicon (Si) -- full spd parametrization -- onsite t2g=eg -- 22 October 1999
1                                 (One atom type in this file)
12.5   0.5                        (RCUT and SCREENL for 1-1 interactions)
9                                 (Orbitals for atom 1)
28.086 14                          (Atomic Weight of Atom 1)
 2.0  2.0  0.0                    (formal spd valence occupancy for atom 1)
  -.111077868065E+01   0   1     lambda  (equation 7)
  -.555446426265E-01   0   2     a_s     (equation 9)
  -.111314406191E+01   0   3     b_s     (equation 9)
  -.732006306270E+01   0   4     c_s     (equation 9)
   .748904883145E+02   0   5     d_s     (equation 9)
   .412681593090E+00   0   6     a_p     (equation 9)
  -.907470079268E-01   0   7     b_p     (equation 9)
   .531548058609E+01   0   8     c_p     (equation 9)
  -.440416606688E+02   0   9     d_p     (equation 9)
   .234693748868E+03   0  18     e_{ss sigma}    (equation 11) (Hamiltonian)
  -.186013071183E+02   0  19     f_{ss sigma}    (equation 11) (Hamiltonian)
  -.150266106596E+02   0  20     fbar_{ss sigma} (equation 11) (Hamiltonian)
   .125021967501E+01   0  78     g_{ss sigma}    (equation 11) (Hamiltonian)
   .955551683006E+01   0  21     e_{sp sigma}    (equation 11) (Hamiltonian)
  -.412791322389E+01   0  22     f_{sp sigma}    (equation 11) (Hamiltonian)
   .249891449737E+00   0  23     fbar_{sp sigma} (equation 11) (Hamiltonian)
   .876069026130E+00   0  79     g_{sp sigma}    (equation 11) (Hamiltonian)
  -.226781629237E+02   0  24     e_{pp sigma}    (equation 11) (Hamiltonian)
   .136110406588E+01   0  25     f_{pp sigma}    (equation 11) (Hamiltonian)
   .138787213237E+01   0  26     fbar_{pp sigma} (equation 11) (Hamiltonian)
   .101654981580E+01   0  80     g_{pp sigma}    (equation 11) (Hamiltonian)
  -.159417204770E+01   0  27     e_{pp pi}       (equation 11) (Hamiltonian)
   .479138493530E+01   0  28     f_{pp pi}       (equation 11) (Hamiltonian)
  -.156927113429E+01   0  29     fbar_{pp pi}    (equation 11) (Hamiltonian)
   .110304892017E+01   0  81     g_{pp pi}       (equation 11) (Hamiltonian)
   .243937020791E+01   0  48     e_{ss sigma}    (equation 11) (Overlap)
   .909051593452E+00   0  49     f_{ss sigma}    (equation 11) (Overlap)
  -.748763316446E-01   0  50     fbar_{ss sigma} (equation 11) (Overlap)
   .105896873807E+01   0  88     g_{ss sigma}    (equation 11) (Overlap)
"""

A_SI = 5.431
SI_LATTICE = [[0, A_SI / 2, A_SI / 2], [A_SI / 2, 0, A_SI / 2], [A_SI / 2, A_SI / 2, 0]]
SI_FRAC = [[0, 0, 0], [0.25, 0.25, 0.25]]
D0 = A_SI * math.sqrt(3) / 4


def skf_text(homonuclear: bool, n=300, dr=0.02, flip_sp=False) -> str:
    """A small synthetic .skf file: ss / sp / pp integrals decaying like exp(-r), overlaps of the usual signs."""
    lines = [f"{dr}, {n}"]
    if homonuclear:
        lines.append("0.0 -0.2 -0.5 0.0 0.3 0.3 0.3 0.0 2.0 2.0")   # Ed Ep Es SPE Ud Up Us fd fp fs (Hartree)
    lines.append("12.0, 19*0.0")
    for i in range(1, n + 1):
        r = i * dr
        e = math.exp(-r / 2)
        sgn = -1 if flip_sp else 1
        H = [0, 0, 0, 0, 0, 0.3 * e, -0.1 * e, 0, sgn * 0.25 * e, -0.3 * e]   # dds ... pps ppp sds sps sss
        S = [0, 0, 0, 0, 0, -0.3 * e, 0.1 * e, 0, sgn * -0.25 * e, 0.3 * e]
        lines.append(" ".join(f"{x:.8e}" for x in H + S))
    lines.append("Spline")
    return "\n".join(lines) + "\n"


class Readers(unittest.TestCase):
    def test_skf_tables_units_and_onsite(self):
        t = P.parse_skf(skf_text(True), homonuclear=True)
        self.assertAlmostEqual(t["onsite"]["s"], -0.5 * P.HARTREE_EV)
        self.assertAlmostEqual(t["onsite"]["p"], -0.2 * P.HARTREE_EV)
        self.assertNotIn("d", t["onsite"])                           # Ed = 0: no d orbital
        self.assertAlmostEqual(t["r"][0], 0.02 * P.BOHR_A)           # the first line is r = one grid step
        r = 100 * 0.02
        self.assertAlmostEqual(t["H"]["sss"][99], -0.3 * math.exp(-r / 2) * P.HARTREE_EV, places=6)
        self.assertAlmostEqual(t["S"]["sss"][99], 0.3 * math.exp(-r / 2), places=8)

    def test_skf_compressed_numbers(self):
        self.assertEqual(P._numbers("3*0.5, 1.0 T 2"), [0.5, 0.5, 0.5, 1.0])

    def test_dftb_set_resampled_and_signs_made_standard(self):
        files = {"X-X.skf": skf_text(True), "X-Y.skf": skf_text(False, flip_sp=True), "readme.txt": "x"}
        with self.assertRaises(P.ParamSetError):                       # X is not an element
            P.dftb_set_from_files(files, {"id": "t"})
        files = {"C-C.skf": skf_text(True), "C-H.skf": skf_text(False, flip_sp=True), "notes.txt": "x"}
        d = P.dftb_set_from_files(files, {"id": "t", "name": "T"})
        self.assertEqual(d["elements"], ["H", "C"])
        self.assertEqual(sorted(d["pairs"]), ["C-C", "C-H"])
        ps = P.ParamSet(d)
        r = 1.5
        expect = 0.25 * math.exp(-(r / P.BOHR_A) / 2) * P.HARTREE_EV
        self.assertAlmostEqual(ps.integral("C", "C", "sps", r), expect, delta=expect * 2e-3)
        self.assertAlmostEqual(ps.integral("C", "H", "sps", r), expect, delta=expect * 2e-3)   # flipped back
        self.assertLess(ps.integral("C", "H", "sps", r, overlap=True), 0)
        # p on the first atom: V_ps = -V_sp of the reversed pair
        self.assertAlmostEqual(ps.integral("H", "C", "pss", r), -ps.integral("C", "H", "sps", r))
        self.assertAlmostEqual(ps.decay("C", "C", "sss", r), (r / P.BOHR_A) / 2, delta=0.01)   # -dlnV/dlnr
        self.assertTrue(any("sign" in n for n in d["import_notes"]))
        self.assertTrue(any("notes.txt" in n for n in d["import_notes"]))

    def test_nist_valence_levels(self):
        atoms = P.parse_nist_raw(open(P.os.path.join(P.BUNDLED_DIR, "nist_raw.txt"), encoding="utf-8").read())
        self.assertEqual(len(atoms), 92)
        si = P.nist_valence("Si", atoms["Si"]["orbitals"])
        self.assertAlmostEqual(si["s"], -0.399527 * P.HARTREE_EV, places=3)
        self.assertAlmostEqual(si["p"], -0.153009 * P.HARTREE_EV, places=3)
        self.assertEqual(set(P.nist_valence("K", atoms["K"]["orbitals"])), {"s"})      # 3p is semicore
        self.assertEqual(set(P.nist_valence("Fe", atoms["Fe"]["orbitals"])), {"s", "d"})
        self.assertEqual(set(P.nist_valence("Ce", atoms["Ce"]["orbitals"])), {"s", "d", "f"})
        self.assertEqual(set(P.nist_valence("U", atoms["U"]["orbitals"])), {"s", "d", "f"})
        self.assertEqual(P.occupied_families("[Ar] 3d^6 4s^2"), {"d", "s"})
        # the Sm row: the published ScRLDA column is shifted by one line; realigned here
        self.assertAlmostEqual(atoms["Sm"]["orbitals"]["6s"][1], -0.136376)
        self.assertAlmostEqual(atoms["Sm"]["orbitals"]["4f"][1], -0.102397)

    def test_harrison_formulas(self):
        nist = P.bundled_sets()["nist"]
        h = P.harrison_set(nist, {"id": "h"}, rd={"Ti": 1.08})
        ps = P.ParamSet(h)
        d = 2.35
        self.assertAlmostEqual(ps.integral("Si", "Si", "sss", d), -1.32 * 7.62 / d ** 2)
        self.assertAlmostEqual(ps.integral("Si", "Si", "ppp", d), -0.63 * 7.62 / d ** 2)
        self.assertIsNone(ps.integral("Si", "Si", "sds", d))                 # no r_d for Si
        self.assertAlmostEqual(ps.integral("C", "Ti", "pds", d), -2.95 * 7.62 * 1.08 ** 1.5 / d ** 3.5)
        self.assertAlmostEqual(ps.integral("Ti", "Ti", "dds", d), -16.2 * 7.62 * 1.08 ** 3 / d ** 5)
        self.assertAlmostEqual(ps.decay("Si", "Si", "pps", d), 2.0, places=6)
        self.assertIsNone(ps.integral("Si", "Si", "sss", d, overlap=True))
        self.assertAlmostEqual(ps.onsite("Si", "pz"), nist["onsite"]["Si"]["p"])

    def test_harrison_table(self):
        rd, on = P.parse_harrison_table("Ti, 1.08\nV 0.98\n")
        self.assertEqual(rd, {"Ti": 1.08, "V": 0.98})
        rd, on = P.parse_harrison_table("element,r_d,e_s,e_d\nTi,1.08,-6.0,-11.04\n")
        self.assertEqual(on, {"Ti": {"s": -6.0, "d": -11.04}})
        for bad in ("Xx, 1\n", "Ti, 9\n", "element,foo\nTi,1\n"):
            with self.assertRaises(P.ParamSetError):
                P.parse_harrison_table(bad)

    def test_nrl_reader_and_formulas(self):
        e = P.parse_nrl(NRL_SI)
        self.assertEqual((e["element"], e["rcut"], e["screenl"], e["style"]), ("Si", 12.5, 0.5, "NN00001"))
        self.assertEqual(sorted(e["H"]), ["ppp", "pps", "sps", "sss"])
        d = P.nrl_set([e], {"id": "nrl"})
        ps = P.ParamSet(d)
        r = D0
        rb = r / P.BOHR_A
        poly = 234.693748868 - 18.6013071183 * rb - 15.0266106596 * rb ** 2
        want = poly * math.exp(-1.25021967501 ** 2 * rb) * float(P.cutoff_nrl(rb, 12.5, 0.5)) * P.RYDBERG_EV
        self.assertAlmostEqual(ps.integral("Si", "Si", "sss", r), want)
        self.assertTrue(-3.0 < want < -1.0)                                # about -1.9 eV, as usual for Si
        # new-style overlap for like atoms: (1 + e R + f R^2 + fbar R^3) exp(-g^2 R) F
        s = ps.integral("Si", "Si", "sss", r, overlap=True)
        so = (1 + 2.43937020791 * rb + 0.909051593452 * rb ** 2 - 0.0748763316446 * rb ** 3) * \
            math.exp(-1.05896873807 ** 2 * rb) * float(P.cutoff_nrl(rb, 12.5, 0.5))
        self.assertAlmostEqual(s, so)
        # the on-site energy depends on the neighbours
        shells = P.neighbour_shells(SI_LATTICE, ["Si", "Si"], SI_FRAC, cutoff=9.0)
        rho = ps.density("Si", shells[0])
        es = ps.onsite("Si", "s", rho)
        a, b, c, dd = e["onsite"]["s"]
        self.assertAlmostEqual(es, (a + b * rho ** (2 / 3) + c * rho ** (4 / 3) + dd * rho ** 2) * P.RYDBERG_EV)
        self.assertIsNone(ps.onsite("Si", "d_t2g", rho))
        for bad in ("hello", NRL_SI.replace("(Si)", "(Qq)").replace("Silicon", "Nothing"),
                    NRL_SI.replace("\n1   ", "\n2   ", 1)):
            with self.assertRaises(P.ParamSetError):
                P.parse_nrl(bad)

    def test_neighbour_shells(self):
        sh = P.neighbour_shells(SI_LATTICE, ["Si", "Si"], SI_FRAC, cutoff=4.0)
        self.assertEqual(sh[0][0], ["Si", round(D0, 3), 4])
        self.assertEqual(sh[0][1][2], 12)                                  # second shell


class ShippedSets(unittest.TestCase):
    def test_all_sets_load(self):
        sets = P.bundled_sets()
        self.assertEqual(sorted(sets), ["dftb-3ob", "dftb-matsci", "dftb-mio", "dftb-pbc", "harrison", "nist"])
        for d in sets.values():
            self.assertTrue(d["name"] and d["licence"] and d["citation"], d["id"])
        self.assertEqual(len(sets["nist"]["elements"]), 92)
        self.assertIn("Ti", sets["dftb-matsci"]["elements"])

    def test_graphene_integrals_have_the_usual_signs(self):
        for sid in ("dftb-mio", "dftb-3ob", "dftb-matsci", "dftb-pbc", "harrison"):
            ps = P.ParamSet(P.bundled_sets()[sid])
            v = {n: ps.integral("C", "C", n, 1.42) for n in ("sss", "sps", "pps", "ppp")}
            self.assertTrue(v["sss"] < 0 < v["sps"] and v["ppp"] < 0 < v["pps"], (sid, v))
            self.assertTrue(-4.5 < v["ppp"] < -2.0, (sid, v))          # graphene's pp-pi: about -3 eV


class StartingValues(unittest.TestCase):
    KEYS = ["e_orbit0_Si_s", "e_orbit0_Si_p", "e_orbit0_Si_s*", "v_Si_s-Si_s_sss_0", "v_Si_s-Si_p_sps_0", "v_Si_p-Si_p_pps_0", "v_Si_p-Si_p_ppp_0",
            "v_Si_s-Si_s*_ss*s_0", "s_Si_s-Si_s_sss_0", "s_Si_s-Si_p_sps_0", "eta_Si_s-Si_s_sss", "lambda_Si_p"]

    def env(self, centres=None, n_neighbors=1):
        return P.model_environment(SI_LATTICE, ["Si", "Si"], SI_FRAC, [0, 0], {"Si": ["s", "p", "s*"]}, D0,
                                   n_neighbors, False, 2.0, True, centres)

    def run_set(self, sid, **kw):
        fi = {k: 0.123 for k in self.KEYS}
        fb = {k: (-8.0, 8.0) for k in self.KEYS}
        return P.starting_values(P.bundled_sets()[sid], self.env(**kw), self.KEYS, fi, fb)

    def test_shift_to_the_dft_centres(self):
        r = self.run_set("harrison", centres={"e_orbit0_Si_s": -7.0, "e_orbit0_Si_p": 1.0})
        on = P.bundled_sets()["nist"]["onsite"]["Si"]
        shift = ((-7.0 - on["s"]) + (1.0 - on["p"])) / 2
        self.assertAlmostEqual(r["shift"], shift, places=3)
        self.assertEqual(r["alignment"], "dft")
        self.assertAlmostEqual(r["initial"]["e_orbit0_Si_s"], on["s"] + shift, places=4)
        self.assertAlmostEqual(r["initial"]["e_orbit0_Si_s*"], on["s"] + shift + 6.0, places=4)
        self.assertAlmostEqual(r["initial"]["v_Si_s-Si_s_sss_0"], -1.32 * 7.62 / D0 ** 2, places=3)   # D0 to 1 mA
        self.assertEqual(r["source"]["v_Si_s-Si_s*_ss*s_0"], "default")               # no set has s*
        self.assertEqual(r["initial"]["v_Si_s-Si_s*_ss*s_0"], 0.123)
        self.assertEqual(r["source"]["s_Si_s-Si_s_sss_0"], "default")                # Harrison: no overlaps
        self.assertEqual(r["source"]["lambda_Si_p"], "default")
        self.assertEqual(r["coverage"]["hopping"], [4, 5])
        for k in self.KEYS:
            lo, hi = r["bounds"][k]
            self.assertTrue(lo <= r["initial"][k] <= hi, k)

    def test_without_projections_highest_occupied_level_below_ef(self):
        r = self.run_set("nist")
        self.assertEqual(r["alignment"], "fermi")
        self.assertAlmostEqual(r["initial"]["e_orbit0_Si_p"], -1.0, places=6)   # Si 3p: the highest occupied
        self.assertEqual(r["coverage"]["hopping"], [0, 5])

    def test_dftb_values_and_decay(self):
        r = self.run_set("dftb-pbc", centres={"e_orbit0_Si_s": -7.0})
        ps = P.ParamSet(P.bundled_sets()["dftb-pbc"])
        self.assertAlmostEqual(r["initial"]["v_Si_p-Si_p_pps_0"], ps.integral("Si", "Si", "pps", round(D0, 3)), places=4)
        self.assertAlmostEqual(r["initial"]["s_Si_s-Si_p_sps_0"], ps.integral("Si", "Si", "sps", round(D0, 3), True), places=4)
        self.assertLess(r["initial"]["s_Si_s-Si_p_sps_0"], 0)
        self.assertEqual(r["source"]["eta_Si_s-Si_s_sss"], "set")

    def test_more_shells_refer_to_d0(self):
        r1 = self.run_set("harrison", n_neighbors=1)
        r2 = self.run_set("harrison", n_neighbors=2)
        self.assertAlmostEqual(r1["initial"]["v_Si_s-Si_s_sss_0"], r2["initial"]["v_Si_s-Si_s_sss_0"], places=3)   # the bonds are at d0

    def test_set_without_the_element(self):
        r = self.run_set("dftb-mio", centres={"e_orbit0_Si_s": -7.0})
        self.assertEqual(r["coverage"]["onsite"], [0, 3])
        self.assertEqual(r["coverage"]["hopping"], [0, 5])
        self.assertTrue(any("no on-site energies for Si" in n for n in r["notes"]))


class Imports(unittest.TestCase):
    def test_dftb_zip_merge_and_replace(self):
        d = P.empty_set("dftb", {"id": "mine", "name": "Mine"})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("set/C-C.skf", skf_text(True))
        d = P.import_files(d, {"c.zip": buf.getvalue()})
        self.assertEqual(sorted(d["pairs"]), ["C-C"])
        d = P.import_files(d, {"H-H.skf": skf_text(True).encode()})
        self.assertEqual(sorted(d["pairs"]), ["C-C", "H-H"])
        d = P.import_files(d, {"H-H.skf": skf_text(True).encode()}, replace=True)
        self.assertEqual(sorted(d["pairs"]), ["H-H"])
        with self.assertRaises(P.ParamSetError):
            P.import_files(d, {"bad.zip": b"not a zip"})

    def test_other_kinds(self):
        h = P.import_files(P.empty_set("harrison", {"id": "h"}), {"rd.csv": b"element,r_d,e_d\nTi,1.08,-11.04\n"})
        self.assertEqual(h["harrison"]["rd"], {"Ti": 1.08})
        self.assertEqual(h["onsite"]["Ti"], {"d": -11.04})
        n = P.import_files(P.empty_set("nrl", {"id": "n"}), {"si_par.spd": NRL_SI.encode(), "x.html": b"<html>"})
        self.assertEqual(n["elements"], ["Si"])
        self.assertIn("x.html", n["import_notes"][0])
        t = P.import_files(P.empty_set("nist", {"id": "t"}), {"t.txt": b"14|[Ne] 3s^2 3p^2|3s=-0.4,-0.4;3p=-0.15,-0.15\n"})
        self.assertAlmostEqual(t["onsite"]["Si"]["p"], -0.15 * P.HARTREE_EV, places=3)

    def test_edit_onsite(self):
        d = P.edit_onsite(P.bundled_sets()["nist"], {"Si": {"s": -9.5, "f": None}})
        self.assertEqual(d["onsite"]["Si"]["s"], -9.5)
        for bad in ({"Xx": {"s": 1}}, {"Si": {"q": 1}}, {"Si": {"s": float("nan")}}):
            with self.assertRaises(P.ParamSetError):
                P.edit_onsite(d, bad)

    def test_file_round_trip(self):
        import tempfile, os
        d = P.bundled_sets()["harrison"]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.json.gz")
            P.save_file(d, p)
            self.assertEqual(P.load_file(p)["onsite"], d["onsite"])
            self.assertEqual(P.dump_bytes(d), P.dump_bytes(d))       # the same data gives the same bytes


if __name__ == "__main__":
    unittest.main()
