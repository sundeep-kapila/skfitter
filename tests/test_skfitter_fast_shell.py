"""Per-shell models on a cell symmetric only to symprec: bonds slightly off their shell distance get V_n (r_n/r)^eta
in the reference path, and the fast path must do the same (otherwise check_fast_model fails and every fit falls back
to the slow reference evaluation)."""
import os

import numpy as np
import pytest

import _paths


def test_fast_matches_reference_off_shell_distance():
    if not _paths.HAS_SCIENCE or not os.path.isfile(os.path.join(_paths.EXAMPLES, "mnte-qe", "example.json")):
        pytest.skip("MnTe example not available")
    import _examples_structure as X
    cfg = {**X.cfg("mnte-qe"), "hopping_mode": "shell", "n_neighbors": 5,
           "orbital_config": {"Mn": ["s", "dz2", "d_e1", "d_e2"], "Te": ["pz", "pxy"]},
           "magnetic_orbitals": {"Mn": ["dz2", "d_e1", "d_e2"]}}
    f = X.fitter("mnte-qe", cfg)
    r = np.array([np.linalg.norm(b["d_vec"]) for b in f.neighbor_list])
    el = f._site_elements()
    rn = np.array([f._shell_of(np.linalg.norm(b["d_vec"]), f"{'-'.join(sorted((el[b['i']], el[b['j']])))}")[1]
                   for b in f.neighbor_list])
    assert np.abs(r - rn).max() > 1e-5                  # the example cell has bonds off their shell distance
    keys = f._get_param_keys()
    x = np.random.default_rng(1).normal(scale=0.5, size=len(keys))
    assert f.check_fast_model(x, verbose=False)
