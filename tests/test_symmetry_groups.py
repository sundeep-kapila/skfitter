"""symmetry_groups (skfitter character --suggest): the finest orbital groups the crystal symmetry allows for an
element's shell, the on-site terms its site symmetry still allows, and the site point group."""
import pytest

import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs scipy and pymatgen")


def _l12():
    """Mn3Ir (L1_2): Ir at the cube corner (m-3m), Mn at the face centres (4/mmm, axis along x, y or z)."""
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(3.785), ["Ir", "Mn", "Mn", "Mn"],
                     [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def test_l12_mn_d_is_t2g_eg_with_its_tetragonal_crystal_field():
    from skfitter.core import symmetry_groups
    st = _l12()
    mn = symmetry_groups(st, "Mn", "d")
    assert mn["groups"] == ["d_t2g", "d_eg"] and mn["site_symmetry"] == ["4/mmm"]
    assert mn["mixing"] == 2                     # t2g -> b2g + eg and eg -> a1g + b1g about each site's own axis
    ir = symmetry_groups(st, "Ir", "d")
    assert ir == {"groups": ["d_t2g", "d_eg"], "mixing": 0, "site_symmetry": ["m-3m"]}
    assert symmetry_groups(st, "Mn", "p")["groups"] == ["p"]
    assert symmetry_groups(st, "Mn", "p")["mixing"] == 1          # pz' apart from px', py' on a 4/mmm site


def test_hexagonal_and_cubic_examples():
    import _examples_structure as X
    from skfitter.core import symmetry_groups
    st = X.fitter("mnte-qe").structure
    assert symmetry_groups(st, "Mn", "d")["groups"] == ["dz2", "d_e1", "d_e2"]
    assert symmetry_groups(st, "Te", "p")["groups"] == ["pz", "pxy"]
    fe = X.fitter("iron-soc-qe").structure
    assert symmetry_groups(fe, "Fe", "d") == {"groups": ["d_t2g", "d_eg"], "mixing": 0, "site_symmetry": ["m-3m"]}


def test_valence_shells_without_a_projection():
    """Mn3Ir's pseudopotentials carry no Mn 4p or Ir 6p wavefunction; --suggest names them in its note."""
    from skfitter.core import unprojected_valence_shells
    got = unprojected_valence_shells({"Mn": ["3S", "3P", "3D", "4S"], "Ir": ["5S", "5P", "5D", "6S"]})
    assert got == {"Mn": ["4p"], "Ir": ["6p"]}
    assert unprojected_valence_shells({"Fe": ["3S", "3P", "3D", "4S", "4P"]}) == {}
    assert unprojected_valence_shells({"Eu": ["5S", "5P", "5D", "4F", "6S", "0P"]}) == {}   # n = 0 counts as present
    assert unprojected_valence_shells({"Ce": ["5S", "5P", "5D", "6S", "6P"]}) == {"Ce": ["4f"]}
    assert unprojected_valence_shells({"C": ["2S", "2P"], "H": ["1S"]}) == {}               # no 1p shell
