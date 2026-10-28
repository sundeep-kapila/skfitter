"""General moment directions: the exchange term is -(delta/2) n_i . sigma with n_i the direction of site i's moment
(read from a non-collinear DFT run, or s_i z for a collinear one)."""
import os

import numpy as np
import pytest

import _paths
from skfitter.core import read_qe_site_moments


def test_qe_moment_vectors(tmp_path):
    p = tmp_path / "scf.out"
    p.write_text("     atom number     1 relative position :    0.0000   0.0000   0.0000\n"
                 "     charge :   13.5\n     magnetization :    1.0000   0.0000   2.0000\n"
                 "     atom number     2 relative position :    0.5000   0.5000   0.5000\n"
                 "     charge :   13.5\n     magnetization :   -1.0000   0.0000  -2.0000\n", encoding="utf-8")
    v = read_qe_site_moments(str(p), 2, vectors=True)
    assert np.allclose(v[0], [1, 0, 2]) and np.allclose(v[1], [-1, 0, -2])
    s = read_qe_site_moments(str(p), 2)                 # the components along the common axis (signs)
    assert s[0] > 0 > s[1]
    p.write_text("     Magnetic moment per site  (integrated on atomic sphere of radius R)\n"
                 "     atom   1 (R=0.290)  charge= 13.6  magn=  4.58\n     atom   2 (R=0.290)  charge= 13.6  magn= -4.58\n", encoding="utf-8")
    assert np.allclose(read_qe_site_moments(str(p), 2, vectors=True)[1], [0, 0, -4.58])


def _mnte(dirs=None):
    import _examples_structure as X
    cfg = {**X.cfg("mnte-qe"), "orbital_config": {"Mn": ["s", "d"], "Te": ["p"]},
           "magnetic_orbitals": {"Mn": ["d"]}, "include_soc": False}
    f = X.fitter("mnte-qe", cfg)
    if dirs is not None:
        f.site_moment_dirs = [np.asarray(d, float) for d in dirs]
    return f


def test_spin_rotation_invariance_without_soc():
    """Without SOC only the relative directions matter: rotating every moment by the same rotation leaves the
    bands unchanged, and the exchange term with n = +-z is the collinear one."""
    if not _paths.HAS_SCIENCE or not os.path.isfile(os.path.join(_paths.EXAMPLES, "mnte-qe", "example.json")):
        pytest.skip("MnTe example not available")
    f = _mnte()
    assert all(abs(d[2]) > 0.999 for d in f.site_moment_dirs)
    keys = f._get_param_keys()
    rng = np.random.default_rng(2)
    p = {k: 0.5 * rng.normal() for k in keys}
    p.update({k: 6.0 for k in keys if k.startswith("delta_spin")})
    k = 0.2 * rng.normal(size=3) @ f.structure.lattice.reciprocal_lattice.matrix
    e_z = np.linalg.eigvalsh(f.build_hamiltonian(k, p))
    a, b = 0.7, 1.1
    Rm = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ \
        np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    f.site_moment_dirs = [Rm @ d for d in f.site_moment_dirs]
    H = f.build_hamiltonian(k, p)
    assert np.abs(H[:f.base_dim, f.base_dim:]).max() > 1e-3          # spin up and down now mix
    assert np.allclose(np.linalg.eigvalsh(H), e_z, atol=1e-10)
    from skfitter.model.sk import SKModel
    m = SKModel.from_fitter(f, params=p)
    m2 = SKModel.from_dict(m.to_dict())
    assert m2.moment_dirs is not None
    for i in range(m.n_sites):
        assert np.allclose(m.onsite(i), m2.onsite(i))
    sc = m.supercell([1, 1, 2])
    for i in range(sc.n_sites):
        assert np.allclose(sc.onsite(i), m.onsite(sc.parent_sites[i]))
