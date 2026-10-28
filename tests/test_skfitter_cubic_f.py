"""Cubic f groups (f_a2u, f_t1u, f_t2u): combinations of the tesseral f orbitals, split by an O_h site symmetry."""
import numpy as np
import pytest

from skfitter.core import (SKFitter, _CUBIC_F, coarser_integral_keys, check_orbital_symmetry, group_unit_blocks,
                      groups_overlap, sk_unit_blocks, sub_basis)

CUBIC = ["f_a2u", "f_t1u", "f_t2u"]


def test_cubic_basis_is_orthonormal_and_complete():
    B = np.vstack([_CUBIC_F[g] for g in CUBIC])
    assert B.shape == (7, 7)
    assert np.allclose(B @ B.T, np.eye(7)) and np.allclose(B.T @ B, np.eye(7))
    assert [SKFitter.SUB_ORBITALS[g]["dim"] for g in CUBIC] == [1, 3, 3]


@pytest.mark.parametrize("other", ["s", "p", "d_t2g", "d_eg", "f_a2u", "f_t1u", "f_t2u"])
def test_group_blocks_add_up_to_the_full_block(other):
    """Summed over the cubic groups (back in the tesseral basis), the group blocks give the plain f blocks: with
    equal integrals for every group pair the cubic model is the plain f model."""
    rng = np.random.default_rng(1)
    fam = other if other in ("s", "p") else other[0]
    partners = [other] if other in ("s", "p") else (CUBIC if fam == "f" else ["d_t2g", "d_eg"])
    for _ in range(5):
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        full = group_unit_blocks("f", fam, n)
        tot = [np.zeros_like(U) for U in full]
        for g in CUBIC:
            for h in partners:
                for m, U in enumerate(group_unit_blocks(g, h, n)):
                    tot[m] += sub_basis(g).T @ U @ sub_basis(h)
        for a, b in zip(tot, full):
            assert np.allclose(a, b, atol=1e-12)


def test_reverse_direction_consistent():
    n = np.array([0.3, -0.5, 0.81]); n /= np.linalg.norm(n)
    for g in CUBIC:
        for h in ["d_t2g", "p", "f_t2u"]:
            A = group_unit_blocks(g, h, n); Bk = group_unit_blocks(h, g, -n)
            for a, b in zip(A, Bk):
                assert np.allclose(a, b.T, atol=1e-12)


def test_overlap_and_coarser_keys():
    assert groups_overlap("f", "f_t1u") and groups_overlap("f_m0", "f_t1u") and groups_overlap("fxyz", "f_a2u")
    assert not groups_overlap("f_t1u", "f_t2u") and not groups_overlap("f_a2u", "f_t1u")
    assert not groups_overlap("fxyz", "f_t1u") and not groups_overlap("d_t2g", "f_t1u")
    assert coarser_integral_keys("v_Ce_deg-Ce_ft1u_dfd_0") == ["v_Ce_d-Ce_ft1u_dfd_0", "v_Ce_deg-Ce_f_dfd_0", "v_Ce_d-Ce_f_dfd_0"]
    assert coarser_integral_keys("v_Ce_f-Ce_f_ffs_0") == []


def test_symmetry_check():
    from pymatgen.core import Lattice, Structure
    fcc = Structure(Lattice([[0, 2.58, 2.58], [2.58, 0, 2.58], [2.58, 2.58, 0]]), ["Ce"], [[0, 0, 0]])
    flu = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.41), ["Ce", "O"], [[0, 0, 0], [.25, .25, .25]])
    hcp = Structure(Lattice.hexagonal(3.6, 5.8), ["Ce", "Ce"], [[1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75]])
    c = np.cos(np.pi / 4); Rz = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1]])
    rot = Structure(Lattice(fcc.lattice.matrix @ Rz.T), ["Ce"], [[0, 0, 0]])

    def errs(st, subs):
        return check_orbital_symmetry(st, [subs if x.specie.symbol == "Ce" else ["p"] for x in st])
    assert errs(fcc, CUBIC) == [] and errs(flu, CUBIC) == []
    assert errs(hcp, CUBIC) and errs(rot, CUBIC)               # no cubic axes along x, y, z
    assert errs(hcp, ["f_m0", "f_m1", "f_m2", "f_m3"]) == []
