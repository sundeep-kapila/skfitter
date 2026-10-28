"""Atomic spin-orbit coupling as a model modifier (prayog_tb ``atomic_soc``).

* A spin-less p atom plus atomic_soc lambda is the textbook spinful atom with lambda L.S (SKFitter's term): the same
  on-site matrix; levels j = 3/2 at +lambda/2 (4) and j = 1/2 at -lambda (2). A d shell: j = 5/2 at +lambda (6), j = 3/2
  at -3 lambda / 2 (4). "<El>_<family>" picks one shell, "<El>" every p / d / f shell, s is untouched.
* Wannier-like functions (labels w1, w2, ...) need ``labels``; with them, in any order of the functions, the spectrum
  is the atomic one. A chain of p atoms keeps Kramers pairs E(k) = E(-k) and loses S_z conservation.
* The modified model is marked spin-orbit coupled, survives its file, and is refused on a model with SOC already;
  unknown elements, missing or wrong labels and malformed lam are refused.
"""
import numpy as np
import pytest

import _paths  # noqa: F401


def spinless(labels=("px", "py", "pz"), element="Te", onsite=None):
    from skfitter.model.model import TBModel
    from skfitter.model.orbitals import FAMILY_L, family_of
    n = len(labels)
    ell = [FAMILY_L[family_of(x)[0]] if family_of(x) else 1 for x in labels]
    return TBModel(lattice=np.eye(3) * 30.0, species=[element], frac_coords=[[0, 0, 0]], site_orbitals=[list(labels)],
                   onsite=[np.zeros((n, n), complex) if onsite is None else onsite], hoppings={}, kind="tb",
                   orbital_l=[ell])


def levels(m):
    return np.linalg.eigvalsh(m.onsite(0))


def test_matches_the_textbook_atom_and_levels():
    from skfitter.model import apply
    from skfitter.model.textbook import atom
    lam = 0.49
    m = apply(spinless(), [{"kind": "atomic_soc", "lam": {"Te_p": lam}}])
    assert m.spinful and m.soc and not m.conserves_sz
    assert np.allclose(m.onsite(0), atom(("px", "py", "pz"), lam).onsite(0))
    assert np.allclose(levels(m), [-lam, -lam] + [lam / 2] * 4)
    d = apply(spinless(("dxy", "dyz", "dzx", "dx2y2", "dz2"), "W"), [{"kind": "atomic_soc", "lam": {"W": 0.2}}])
    assert np.allclose(levels(d), [-0.3] * 4 + [0.2] * 6)


def test_family_keys_and_s_untouched():
    from skfitter.model import apply
    labs = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2y2", "dz2")
    both = apply(spinless(labs, "Bi"), [{"kind": "atomic_soc", "lam": {"Bi": 0.1, "Bi_p": 0.6}}])
    e = levels(both)
    assert np.allclose(e, sorted([0.0, 0.0] + [-0.6] * 2 + [0.3] * 4 + [-0.15] * 4 + [0.1] * 6))
    only_d = apply(spinless(labs, "Bi"), [{"kind": "atomic_soc", "lam": {"Bi_d": 0.1}}])
    assert np.allclose(levels(only_d), sorted([0.0] * 8 + [-0.15] * 4 + [0.1] * 6))


def test_wannier_functions_need_labels_in_any_order():
    from skfitter.model import apply
    w = spinless(("w1", "w2", "w3"))
    with pytest.raises(ValueError, match="no atomic label"):
        apply(w, [{"kind": "atomic_soc", "lam": {"Te_p": 0.4}}])
    with pytest.raises(ValueError, match="3 atomic orbital names"):
        apply(w, [{"kind": "atomic_soc", "lam": {"Te_p": 0.4}, "labels": {"Te": ["px", "py"]}}])
    a = apply(w, [{"kind": "atomic_soc", "lam": {"Te_p": 0.4}, "labels": {"Te": ["pz", "px", "py"]}}])
    assert np.allclose(levels(a), [-0.4, -0.4] + [0.2] * 4) and any("labels" in x for x in a.warnings)
    # the functions in Wannier90's order (pz, px, py) equal the atom with the same labels
    from skfitter.model.textbook import atom
    assert np.allclose(a.onsite(0), atom(("pz", "px", "py"), 0.4).onsite(0))


def test_chain_kramers_and_file(tmp_path):
    from skfitter.model import apply
    from skfitter.model.model import TBModel
    from skfitter.model.modifiers import from_dict
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3)) * 0.3
    h0 = np.diag([0.0, 0.1, -0.2]).astype(complex)
    chain = TBModel(lattice=np.diag([3.0, 30.0, 30.0]), species=["Te"], frac_coords=[[0, 0, 0]],
                    site_orbitals=[["px", "py", "pz"]], onsite=[h0], kind="tb", orbital_l=[[1, 1, 1]],
                    hoppings={(0, 0, (1, 0, 0)): A.astype(complex), (0, 0, (-1, 0, 0)): A.T.astype(complex)})
    m = apply(chain, [{"kind": "atomic_soc", "lam": {"Te_p": 0.3}}])
    for k in (0.3, 0.7, 1.1):
        Hp, Hm = m.bloch([k, 0, 0]), m.bloch([-k, 0, 0])
        assert np.allclose(Hp, Hp.conj().T)
        assert np.allclose(np.linalg.eigvalsh(Hp), np.linalg.eigvalsh(Hm))
    back = from_dict(m.to_dict()) if hasattr(m, "to_dict") else None
    if back is not None:
        assert back.soc and np.allclose(back.bloch([0.4, 0, 0]), m.bloch([0.4, 0, 0]))
    path = m.save(str(tmp_path / "chain_soc_model.json"))
    from skfitter.model.io import load_model
    again = load_model(path)
    assert again.soc and np.allclose(again.bloch([0.4, 0, 0]), m.bloch([0.4, 0, 0]))


def test_refusals():
    from skfitter.model import apply
    from skfitter.model.textbook import atom
    with pytest.raises(ValueError, match="already"):
        apply(atom(("px", "py", "pz"), 0.2), [{"kind": "atomic_soc", "lam": {"X_p": 0.1}}])
    with pytest.raises(ValueError, match="no Bi in the model"):
        apply(spinless(), [{"kind": "atomic_soc", "lam": {"Bi_p": 0.1}}])
    for bad in ({}, {"Te_q": 0.1}, {"Te_p": "a"}, 0.4):
        with pytest.raises(ValueError, match="atomic_soc"):
            apply(spinless(), [{"kind": "atomic_soc", "lam": bad}])
    with pytest.raises(ValueError, match="vanishes"):
        apply(spinless(("s",)), [{"kind": "atomic_soc", "lam": {"Te": 0.1}}])
