"""Two-centre integrals per element pair: names v_<El1>_<orb1>-<El2>_<orb2>_<integral>_<shell>, shells counted per
element pair (0 = the pair's nearest bond), the decay law referred to each pair's nearest bond, and the conversion
of parameter sets saved with the older element-independent names (rc4), which must give exactly the same bands."""
import contextlib
import io
import json
import os

import numpy as np
import pytest

import _examples_structure as X
import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs scipy and pymatgen")
LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "legacy_mnte_params.json")


def _fitter(**kw):
    c = {**X.cfg("mnte-qe"), "orbital_config": {"Mn": ["s", "d"], "Te": ["s", "p"]},
         "magnetic_orbitals": {"Mn": ["d"]}, "fit_eta": True, **kw}
    with contextlib.redirect_stdout(io.StringIO()):
        return X.fitter("mnte-qe", c)


def test_pairs_sharing_a_distance_get_their_own_integrals():
    f = _fitter(hopping_mode="shell", n_neighbors=3)
    sh = f._pair_shells()
    assert set(sh) == {"Mn-Mn", "Mn-Te", "Te-Te"}
    # a = 4.2177 A is a Mn-Mn and a Te-Te distance: one shell of the cell, two element pairs
    assert any(abs(r - 4.2177) < 1e-3 for r in sh["Mn-Mn"]) and any(abs(r - 4.2177) < 1e-3 for r in sh["Te-Te"])
    keys = set(f._get_param_keys())
    n_mn = next(n for n, r in enumerate(sh["Mn-Mn"]) if abs(r - 4.2177) < 1e-3)
    n_te = next(n for n, r in enumerate(sh["Te-Te"]) if abs(r - 4.2177) < 1e-3)
    assert f"v_Mn_s-Mn_s_sss_{n_mn}" in keys and f"v_Te_s-Te_s_sss_{n_te}" in keys
    # s-p integrals depend on which element carries the s orbital
    assert "v_Mn_s-Te_p_sps_0" in keys and "v_Te_s-Mn_p_sps_0" not in keys        # no Mn p in the model
    assert any(k.startswith("v_Te_s-Te_p_sps_") for k in keys)


def test_integrals_of_one_pair_act_only_on_its_bonds():
    f = _fitter(hopping_mode="shell", n_neighbors=3)
    keys = f._get_param_keys()
    p = f._param_vec_to_dict(np.random.default_rng(3).normal(0, 0.4, len(keys)))
    k = np.array([0.21, -0.13, 0.37])
    H = f.build_hamiltonian(k, p)
    q = dict(p)
    for name in [x for x in keys if x.startswith("v_Te_p-Te_p_")]:
        q[name] += 0.5
    dH = np.abs(f.build_hamiltonian(k, q) - H)
    N = f.base_dim
    te = [i for i, s in enumerate(f.structure) if s.specie.symbol == "Te"]
    rows = np.zeros(N, bool)
    for i in te:
        rows[f.site_offsets[i]:f.site_offsets[i] + f.site_dims[i]] = True
    rows = np.concatenate([rows, rows]) if H.shape[0] == 2 * N else rows
    assert dH.max() > 1e-3
    assert dH[~rows].max() < 1e-14 and dH[:, ~rows].max() < 1e-14         # Mn rows and columns untouched


def test_decay_refers_to_each_pairs_nearest_bond():
    f = _fitter(hopping_mode="decay", n_neighbors=3)
    for pair, sh in f._pair_shells().items():
        assert abs(f._pair_d0(pair) - sh[0]) < 1e-12
    bt = "Te_p-Te_p_pps"
    p = {f"v_{bt}_0": -1.3, f"eta_{bt}": 3.0}
    d0 = f._pair_d0("Te-Te")
    assert d0 > f.d0 + 0.5                                         # Te-Te bonds are longer than the cell's nearest
    assert f._integral_value(f"v_{bt}", bt, d0, p) == pytest.approx(-1.3)
    assert f._integral_value(f"v_{bt}", bt, 1.1 * d0, p) == pytest.approx(-1.3 / 1.1 ** 3)


@pytest.mark.parametrize("case", [0, 1], ids=["decay", "shell"])
def test_legacy_parameter_sets_give_the_same_bands(case):
    with open(LEGACY, encoding="utf-8") as fh:
        c = json.load(fh)["cases"][case]
    f = _fitter(hopping_mode=c["hopping_mode"], n_neighbors=c["n_neighbors"], orbital_config=c["orbital_config"],
                magnetic_orbitals=c["magnetic_orbitals"])
    from skfitter import core as sk
    assert sk.has_legacy_hoppings(c["params"])
    new = f.convert_legacy_params(c["params"], {"hopping_mode": c["hopping_mode"]})
    assert not sk.has_legacy_hoppings(new)
    keys = f._get_param_keys()
    assert set(keys) <= set(new)
    p = f._param_vec_to_dict(np.array([new[k] for k in keys]))
    E = np.array([np.linalg.eigvalsh(f.build_hamiltonian(f.kpoints[i], p)) for i in c["k_indices"]])
    np.testing.assert_allclose(E, np.array(c["bands"]), atol=1e-9)
    assert f.check_fast_model(np.array([new[k] for k in keys]), n_k=4, verbose=False)


def test_legacy_model_file_is_converted():
    from skfitter.model.sk import SKModel
    with open(LEGACY, encoding="utf-8") as fh:
        c = json.load(fh)["cases"][1]
    f = _fitter(hopping_mode="shell", n_neighbors=c["n_neighbors"])
    new = f.convert_legacy_params(c["params"], {"hopping_mode": "shell"})
    p = f._param_vec_to_dict(np.array([new[k] for k in f._get_param_keys()]))
    d = SKModel.from_fitter(f, p).to_dict()
    assert "Mn-Te" in d["settings"]["pair_shells"]
    old = dict(d, params=c["params"])                                  # the same model saved with the old names
    old["settings"] = {k: v for k, v in d["settings"].items() if k != "pair_shells"}
    m = SKModel.from_dict(json.loads(json.dumps(old)))
    k = np.asarray(f.kpoints[c["k_indices"][2]])
    np.testing.assert_allclose(np.linalg.eigvalsh(m.bloch(k)), c["bands"][2], atol=1e-9)
