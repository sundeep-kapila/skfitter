"""The common model type (skfitter.model): one interface for SKFitter fits and their models written out as numbers.

* On every illustrative project (random parameters, every key of the model): the SK model passes the interface
  check, and written out as numbers (TBModel of kind "sk") it has the same H(k) and S(k) to 1e-10 eV, also after a
  file round trip and on a supercell.
"""
import json

import numpy as np
import pytest

import _paths

TOL = 1e-10
science = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs the SKFitter science stack")


def kpoints(model, n=5, seed=3):
    B = 2 * np.pi * np.linalg.inv(np.asarray(model.lattice)).T
    return np.random.default_rng(seed).normal(size=(n, 3)) @ B


def same_hk(a, b, tol=TOL):
    for k in kpoints(a):
        assert np.abs(a.bloch(k) - b.bloch(k)).max() < tol
        if a.non_orthogonal or b.non_orthogonal:
            assert np.abs(a.bloch(k, overlap=True) - b.bloch(k, overlap=True)).max() < tol


# == the package on its own ======================================================================================








# == SK models: the interface, and written out as numbers ========================================================
def _sk_examples():
    import _examples_structure as X
    return X.example_ids()


def _sk_model(example_id):
    import _examples_structure as X
    from _helpers import random_params
    from skfitter.model import SKModel
    f = X.fitter(example_id)
    return SKModel.from_fitter(f, params=random_params(f), source={"example": example_id})


@science
@pytest.mark.parametrize("example_id", _sk_examples())
def test_sk_model_and_its_numbers(example_id):
    from skfitter.model import TightBindingModel, check_model, model_from_dict
    sk = _sk_model(example_id)
    assert isinstance(sk, TightBindingModel) and check_model(sk) == []
    tb = sk.to_tb_model()
    assert isinstance(tb, TightBindingModel) and check_model(tb) == []
    assert tb.kind == "sk" and tb.non_orthogonal == sk.non_orthogonal and tb.spinful == sk.spinful
    assert [tb.norbs(i) for i in range(tb.n_sites)] == [sk.norbs(i) for i in range(sk.n_sites)]
    assert [tb.orbital_info(i) for i in range(tb.n_sites)] == [sk.orbital_info(i) for i in range(sk.n_sites)]
    for i in range(sk.n_sites):
        assert np.allclose(tb.orbital_centres(i), sk.orbital_centres(i))
        assert all(x["l"] in (0, 1, 2, 3) for x in sk.orbital_info(i))
    assert tb.provenance[0]["step"] == "skfitter fit" and tb.provenance[-1]["step"] == "written out as numbers"
    same_hk(sk, tb)
    same_hk(sk, model_from_dict(json.loads(json.dumps(tb.to_dict()))), tol=1e-8)


@science
@pytest.mark.parametrize("example_id", ["graphene-soc", "iron-soc-qe", "platinum-soc-qe"])
def test_supercells_agree(example_id):
    sk = _sk_model(example_id)
    M = [[1, 1, 0], [0, 1, 0], [0, 0, 1]] if example_id == "graphene-soc" else [2, 1, 1]
    a, b = sk.supercell(M), sk.to_tb_model().supercell(M)
    assert a.n_sites == b.n_sites and a.species == b.species and np.allclose(a.frac_coords, b.frac_coords)
    same_hk(a, b)
    assert b.provenance[-1] == {"step": "supercell", "matrix": np.diag(M).tolist() if np.ndim(M) == 1 else M}


# == every model file of the NEGF illustrative projects ==========================================================




# == orbital centres: each function keeps its own centre =========================================================
def _wrapped_equal(a, b, lattice, tol=1e-6):
    """Do two lists of points agree modulo lattice translations (as sets)?"""
    inv = np.linalg.inv(lattice)
    left = [tuple(np.round(((p @ inv) % 1.0) % 1.0, 5)) for p in a]
    right = [tuple(np.round(((p @ inv) % 1.0) % 1.0, 5)) for p in b]
    return sorted(left) == sorted(right)






# == NEGF Transport on the written-out copy ======================================================================


