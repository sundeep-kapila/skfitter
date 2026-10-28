"""[model] soc_orbitals (which orbital families carry on-site spin-orbit coupling) and model_states_only for magnetic
data: a non-collinear spin-orbit run (bcc Fe) and a collinear antiferromagnet (MnTe, spin-up and spin-down states
merged in energy order)."""
import numpy as np
import pytest

import _examples_structure as X
import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs scipy and pymatgen")


def _cfg(example, **kw):
    c = dict(X.cfg(example))
    c.update(kw)
    return c


def _start(fitter):
    from skfitter import workflow as J
    return {k: J.default_initial(k) for k in fitter._get_param_keys()}


def test_soc_orbitals_keys_and_hamiltonian():
    full = X.fitter("iron-soc-qe")
    only_d = X.fitter("iron-soc-qe", _cfg("iron-soc-qe", soc_orbitals={"Fe": ["d"]}))
    kf, kd = set(full._get_param_keys()), set(only_d._get_param_keys())
    assert {"lambda_Fe_p", "lambda_Fe_d"} <= kf
    assert "lambda_Fe_d" in kd and "lambda_Fe_p" not in kd
    assert kf - kd == {"lambda_Fe_p"}
    p = _start(full)
    p["lambda_Fe_p"], p["lambda_Fe_d"] = 0.3, 0.06
    k = np.array([0.31, -0.17, 0.42])
    H_d = only_d.build_hamiltonian(k, p)                       # lambda_Fe_p is ignored: Fe p carries no SOC
    p0 = dict(p, lambda_Fe_p=0.0)
    assert np.allclose(H_d, full.build_hamiltonian(k, p0), atol=1e-12)
    assert not np.allclose(H_d, full.build_hamiltonian(k, p), atol=1e-6)


@pytest.mark.parametrize("bad, words", [({"Fe": ["s"]}, "only p, d and f"), ({"Fe": ["f"]}, "no f orbitals"),
                                        ({"Xx": ["d"]}, "not in the model")])
def test_soc_orbitals_errors(bad, words):
    with pytest.raises(ValueError, match="soc_orbitals"):
        try:
            X.fitter("iron-soc-qe", _cfg("iron-soc-qe", soc_orbitals=bad))
        except ValueError as exc:
            assert words in str(exc)
            raise


def test_model_states_noncollinear():
    from skfitter import workflow as J
    _, dft = X.project("iron-soc-qe")
    spd = X.fitter("iron-soc-qe")
    d_only = X.fitter("iron-soc-qe", _cfg("iron-soc-qe", orbital_config={"Fe": ["d"]}, magnetic_orbitals={"Fe": ["d"]}))
    m_spd, m_d = J.model_state_mask(dft, spd), J.model_state_mask(dft, d_only)
    assert m_spd is not None and m_d is not None
    assert m_spd.shape == spd._all_dft_energies.shape == m_d.shape
    assert m_d.sum() < m_spd.sum()                             # s- and p-like states drop out of a d-only model
    assert m_d.mean() > 0.2


def test_model_states_collinear_merged():
    from skfitter import workflow as J
    _, dft = X.project("mnte-qe")
    f = X.fitter("mnte-qe")
    m = J.model_state_mask(dft, f)
    assert m is not None and m.shape == f._all_dft_energies.shape
    te_only = X.fitter("mnte-qe", _cfg("mnte-qe", orbital_config={"Mn": ["s"], "Te": ["p"]}, magnetic_orbitals={}))
    m_te = J.model_state_mask(dft, te_only)
    vbm = int(f.dft_vbm_index)
    # the Te p valence bands are kept; the majority Mn d states deep below them are not
    assert m_te[:, vbm].mean() > 0.5 and m_te.sum() < m.sum()


def test_model_states_only_fit_window_for_a_magnet():
    """With model_states_only the fitter rebuilds the window from the kept states (no fall-back to the plain window)."""
    f = X.fitter("iron-soc-qe", _cfg("iron-soc-qe", orbital_config={"Fe": ["d"]}, magnetic_orbitals={"Fe": ["d"]},
                                     model_states_only=True))
    assert getattr(f, "state_filter", None) is not None
    assert f.state_filter["n_removed"] > 0
