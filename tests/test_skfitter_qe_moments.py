"""Site moments from a pw.x SCF output (read_qe_site_moments): the up / down direction of every site's exchange
splitting for Quantum ESPRESSO data, as the MAGMOM signs are for VASP. Without them an antiferromagnet (MnTe,
EuMn2Bi2) would be fitted with all moments parallel."""
import os
import tempfile
import unittest

import _paths  # noqa: F401  (puts skfitter/ on the path)

COLLINEAR = """
     Magnetic moment per site  (integrated on atomic sphere of radius R)
     atom   1 (R=0.290)  charge= 13.5000  magn=  4.1000
     atom   2 (R=0.290)  charge= 13.5000  magn= -4.1000
     atom   3 (R=0.290)  charge= 13.8000  magn=  0.0000
     atom   4 (R=0.290)  charge= 13.8000  magn= -0.0000

     total cpu time spent up to now is      100.0 secs

     Magnetic moment per site  (integrated on atomic sphere of radius R)
     atom   1 (R=0.290)  charge= 13.6029  magn=  4.5837
     atom   2 (R=0.290)  charge= 13.6029  magn= -4.5837
     atom   3 (R=0.290)  charge= 13.8718  magn= -0.0000
     atom   4 (R=0.290)  charge= 13.8718  magn= -0.0000
"""


def _noncollinear(vectors):
    out = []
    for i, (mx, my, mz) in enumerate(vectors, 1):
        out.append(f""" ==============================================================================
     atom number   {i:2d} relative position :    0.0000   0.0000   {i:.4f}
     charge :    15.064750  (integrated on a sphere of radius 0.293)
     magnetization :     {mx:12.6f} {my:12.6f} {mz:12.6f}
     magnetization/charge:    0.000000    0.000000    0.788574
     polar coord.: r, theta, phi [deg] :    11.879664    0.000000  360.000000
""")
    return "\n".join(out)


class QeSiteMoments(unittest.TestCase):
    def setUp(self):
        import skfitter.core as S
        self.S = S
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text):
        path = os.path.join(self.tmp.name, "scf.out")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_collinear_takes_the_last_block(self):
        m = self.S.read_qe_site_moments(self.write(COLLINEAR), 4)
        self.assertEqual(m, [4.5837, -4.5837, -0.0, -0.0])

    def test_noncollinear_along_z_keeps_the_signs(self):
        # EuMn2Bi2: Eu +/-, Mn up up down down, Bi small; the start of the run and then a later iteration
        first = [(0, 0, 11.88), (0, 0, -11.88), (0, 0, 8.0), (0, 0, 8.0), (0, 0, -8.0), (0, 0, -8.0)]
        last = [(0, 0, 6.9), (0, 0, -6.9), (0, 0, 4.7), (0, 0, 4.7), (0, 0, -4.7), (0, 0, -4.7)]
        m = self.S.read_qe_site_moments(self.write(_noncollinear(first) + "\n iteration #  2\n" + _noncollinear(last)), 6)
        self.assertEqual([round(v, 3) for v in m], [6.9, -6.9, 4.7, 4.7, -4.7, -4.7])

    def test_noncollinear_along_x_projects_on_that_axis(self):
        m = self.S.read_qe_site_moments(self.write(_noncollinear([(3.0, 0, 0), (-3.0, 0, 0)])), 2)
        self.assertEqual([round(v, 6) for v in m], [3.0, -3.0])

    def test_wrong_site_count_or_no_moments_is_none(self):
        self.assertIsNone(self.S.read_qe_site_moments(self.write(COLLINEAR), 5))
        self.assertIsNone(self.S.read_qe_site_moments(self.write("     the Fermi energy is  9.78 ev\n"), 4))
        self.assertIsNone(self.S.read_qe_site_moments(os.path.join(self.tmp.name, "missing.out"), 4))


@unittest.skipUnless(_paths.HAS_SCIENCE, "needs the science stack")
class IronQeExampleStaysFerromagnetic(unittest.TestCase):
    """The ferromagnetic QE example: every moment up, as before."""

    def test_iron_signs(self):
        import skfitter.core as S
        here = _paths.data_folder("iron-soc-qe")
        scf = [os.path.join(here, f) for f in os.listdir(here) if "scf" in f and not f.endswith(".gz")]
        if not scf:
            self.skipTest("no plain SCF output in the example")
        m = S.read_qe_site_moments(scf[0])
        if m is not None:
            self.assertTrue(all(v >= 0 for v in m))


if __name__ == "__main__":
    unittest.main()
