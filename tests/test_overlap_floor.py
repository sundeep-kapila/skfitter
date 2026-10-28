"""Non-orthogonal fits: the overlap_floor penalty keeps the smallest eigenvalue of S(k) above a floor, so a fit cannot
make S nearly singular at a few k-points (Mn3Ir at Gamma), and overlap_conditioning reports it."""
import contextlib
import io

import numpy as np
import pytest

import _examples_structure as X
import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs scipy and pymatgen")


def _fitter():
    c = {**X.cfg("mnte-qe"), "orbital_config": {"Mn": ["s", "d"], "Te": ["s", "p"]},
         "magnetic_orbitals": {"Mn": ["d"]}, "hopping_mode": "shell", "n_neighbors": 2, "non_orthogonal": True}
    with contextlib.redirect_stdout(io.StringIO()):
        return X.fitter("mnte-qe", c)


def test_penalty_acts_only_below_the_floor():
    f = _fitter()
    keys = f._get_param_keys()
    x = np.zeros(len(keys))
    f._set_active_k(None)
    nk, nb = len(f.kpoints), len(f.target_tb_band_indices)
    f.overlap_floor = 0.05
    assert len(f._residuals(x)) == nk * nb + nk                     # one penalty residual per k-point
    assert not f._overlap_penalty(f._param_vec_to_dict(x)).any()    # S = 1
    ov = f.overlap_conditioning(f._param_vec_to_dict(x), n_random=8)
    assert ov["path"] == pytest.approx(1.0) and ov["bz"] == pytest.approx(1.0)
    # a large Mn s - Te s overlap (six Te around each Mn) drives S towards singular at Gamma
    i = keys.index(next(k for k in keys if k == "s_Mn_s-Te_s_sss_0"))
    x[i] = 0.18
    p = f._param_vec_to_dict(x)
    lam = f._overlap_min_eig(p, f._get_active()["phase"])
    pen = f._overlap_penalty(p)
    assert lam.min() < 0.05 and pen.max() == pytest.approx(f.OVERLAP_PENALTY * (0.05 - lam.min()))
    assert np.all(pen[lam >= 0.05] == 0)
    f.overlap_floor = 0.0
    assert len(f._residuals(x)) == nk * nb                          # floor 0: no penalty residuals


def test_orthogonal_model_has_no_conditioning_report():
    c = {**X.cfg("mnte-qe"), "orbital_config": {"Mn": ["d"], "Te": ["p"]}, "magnetic_orbitals": {"Mn": ["d"]}}
    with contextlib.redirect_stdout(io.StringIO()):
        f = X.fitter("mnte-qe", c)
    assert f.overlap_conditioning({}) is None


def test_fast_model_check_ignores_states_sent_far_away_by_a_singular_overlap():
    """With S(k) nearly singular a few states sit at ~1e8 eV, where the two eigensolvers differ by eV; the check
    compares the fitted bands only, relative to their size, so the fit keeps the fast model."""
    f = _fitter()
    keys = f._get_param_keys()
    x = np.zeros(len(keys))
    x[keys.index("s_Mn_s-Te_s_sss_0")] = 0.18
    assert f.check_fast_model(x, verbose=False)
