"""On-site mixing (onsite_mixing): the on-site terms beyond one energy per orbital group that the site symmetry
allows - mixing of groups of the same symmetry type, of families on sites without inversion, and the crystal field
of a group the site splits further."""
import os

import numpy as np
import pytest

import _paths
from skfitter.core import _site_representation, onsite_mixing_terms

pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402


def nias():
    """NiAs-type MnTe (P6_3/mmc): Mn on D3d sites, Te on D3h sites."""
    return Structure(Lattice.hexagonal(4.15, 6.71), ["Mn", "Mn", "Te", "Te"],
                     [[0, 0, 0], [0, 0, 0.5], [1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])


def wurtzite():
    """Wurtzite ZnO (P6_3mc): both sites C3v, no inversion."""
    return Structure.from_spacegroup("P6_3mc", Lattice.hexagonal(3.25, 5.21), ["Zn", "O"],
                                     [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.382]])


def terms(st, orbs, elements=None):
    eq = SpacegroupAnalyzer(st, symprec=1e-3).get_symmetry_dataset().equivalent_atoms
    site_orbs = [orbs[s.specie.symbol] for s in st]
    return st, site_orbs, eq, onsite_mixing_terms(st, site_orbs, eq, elements)


@pytest.mark.parametrize("orbs, expect", [
    ({"Mn": ["s", "dz2", "d_e1", "d_e2"], "Te": ["pz", "pxy"]}, {"Mn": 2, "Te": 0}),   # s-dz2, E1-E2
    ({"Mn": ["d"], "Te": ["p"]}, {"Mn": 3, "Te": 1}),            # trigonal d: a1g + 2 eg; D3h p: pz / pxy
    ({"Mn": ["f"], "Te": ["s", "p"]}, {"Mn": 6, "Te": 1}),       # D3d f: a1u + 2 a2u + 2 eu; s-p: none, pz/pxy
])
def test_counts_nias(orbs, expect):
    st, _, _, (out, keys) = terms(nias(), orbs)
    for el, n in expect.items():
        assert sum(k.split("_")[2] == el for k in keys) == n, (el, keys)


def test_counts_cubic_and_polar():
    fcc = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    _, _, _, (_, keys) = terms(fcc, {"Fe": ["s", "p", "d_t2g", "d_eg"]})
    assert keys == []                                           # O_h: every group is its own irreducible type
    _, _, _, (_, keys) = terms(wurtzite(), {"Zn": ["s", "pz", "pxy", "dz2", "d_e1", "d_e2"], "O": ["s", "pz", "pxy"]})
    zn = [k for k in keys if "_Zn_" in k]; o = [k for k in keys if "_O_" in k]
    # Zn (C3v): a1 = s, pz, dz2 (3 pairs); e = pxy, E1, E2 (3 pairs)
    assert len(zn) == 6 and len(o) == 1                          # O: s - pz
    _, _, _, (_, keys) = terms(wurtzite(), {"Zn": ["s", "pz"], "O": ["s", "pz"]}, elements=["O"])
    assert len(keys) == 1 and "_O_" in keys[0]


@pytest.mark.parametrize("make, orbs", [(nias, {"Mn": ["s", "dz2", "d_e1", "d_e2"], "Te": ["s", "p"]}),
                                        (wurtzite, {"Zn": ["s", "p", "d"], "O": ["s", "p"]})])
def test_matrices_symmetric_and_consistent(make, orbs):
    """Every matrix is invariant under the operations that keep its site, and the matrix of a site equals the
    representative's moved there by any operation that maps the two (not only the one used)."""
    st, site_orbs, eq, (out, keys) = terms(make(), orbs)
    sga = SpacegroupAnalyzer(st, symprec=1e-3)
    oc, of = sga.get_symmetry_operations(cartesian=True), sga.get_symmetry_operations(cartesian=False)
    fr = st.frac_coords
    def maps(o, i, j):
        d = o.operate(fr[i]) - fr[j]
        return np.abs(d - np.round(d)).max() < 1e-3
    for i, site in enumerate(out):
        for key, M in site:
            assert np.allclose(M, M.T)
            assert np.isclose(np.abs(np.linalg.eigvalsh(M)).max(), 1.0)
            rep = next(r for r in range(len(st)) if eq[r] == eq[i])
            Mr = dict(out[rep])[key]
            for c, f in zip(oc, of):
                if maps(f, rep, i):
                    D, _ = _site_representation(site_orbs[rep], np.asarray(c.rotation_matrix, float))
                    assert np.allclose(D @ Mr @ D.T, M, atol=1e-8)


def test_model_with_mixing_keeps_the_crystal_symmetry():
    """MnTe example with on-site mixing (random values, exchange off): E(Wk) = E(k) for every point operation;
    the SK model keeps the terms through a file round trip and a supercell."""
    if not _paths.HAS_SCIENCE or not os.path.isfile(os.path.join(_paths.EXAMPLES, "mnte-qe", "example.json")):
        pytest.skip("MnTe example not available")
    import _examples_structure as X
    from skfitter.model.sk import SKModel
    cfg = {**X.cfg("mnte-qe"), "orbital_config": {"Mn": ["s", "dz2", "d_e1", "d_e2"], "Te": ["pz", "pxy"]},
           "magnetic_orbitals": {"Mn": ["dz2", "d_e1", "d_e2"]}, "onsite_mixing": True}
    f = X.fitter("mnte-qe", cfg)
    keys = f._get_param_keys()
    mk = [k for k in keys if k.startswith("e_mix")]
    assert len(mk) == 2
    rng = np.random.default_rng(3)
    p = {k: (0.0 if k.startswith("delta_spin") else 0.3 * rng.normal()) for k in keys}
    ops = SpacegroupAnalyzer(f.structure, symprec=1e-3).get_point_group_operations(cartesian=True)
    rec = f.structure.lattice.reciprocal_lattice.matrix
    for _ in range(3):
        k = 0.3 * rng.normal(size=3) @ rec
        e0 = np.linalg.eigvalsh(f.build_hamiltonian(k, p))
        for op in ops:
            assert np.allclose(np.linalg.eigvalsh(f.build_hamiltonian(op.rotation_matrix @ k, p)), e0, atol=1e-4)   # the cell is hexagonal to ~1e-5
    p0 = dict(p); p0.update({k: 0.0 for k in mk})
    assert not np.allclose(f.build_hamiltonian(np.zeros(3), p), f.build_hamiltonian(np.zeros(3), p0))
    m = SKModel.from_fitter(f, params=p)
    m2 = SKModel.from_dict(m.to_dict())
    for i in range(m.n_sites):
        assert np.allclose(m.onsite(i), m2.onsite(i))
        a = f.site_offsets[i]
        idx = list(range(a, a + m.spatial_dims[i]))
        idx = idx + [f.base_dim + x for x in idx] if m.spinful else idx
        assert np.allclose(m.onsite(i), f.build_hamiltonian(np.zeros(3), {**p, **{k: 0.0 for k in keys if k.startswith("v_")}})[np.ix_(idx, idx)], atol=1e-10)
    sc = m.supercell([1, 1, 2])
    for i in range(sc.n_sites):
        assert np.allclose(sc.onsite(i), m.onsite(sc.parent_sites[i]))
