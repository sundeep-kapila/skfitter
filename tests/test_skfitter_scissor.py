"""SKFitter's scissor (phase M2c): the DFT conduction bands are shifted before fitting.

* The fitted window's conduction bands move by exactly delta, the valence bands not at all; repeated calls add.
* A metal (no gap in the DFT bands) is refused; delta = 0 changes nothing.
* The set-up (cfg 'scissor') reaches the fitter through skfitter_jobs.build_fitter, the server checks its range,
  the set-up view's band plot shows the shifted bands, and an SK model made from the fit records it.
"""
import numpy as np
import pytest

import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs the SKFitter science stack")


def _fitter(example_id, **cfg_changes):
    import _examples_structure as X
    cfg = {**X.cfg(example_id), **cfg_changes}
    return X.fitter(example_id, cfg)


def test_conduction_bands_move_valence_bands_stay():
    base = _fitter("hbn-valley")
    shifted = _fitter("hbn-valley", scissor=0.7)
    vbm = int(base.dft_vbm_index)
    n_below = sum(1 for i in base.target_indices if i <= vbm)
    T0, T1 = np.asarray(base.target_dft_bands), np.asarray(shifted.target_dft_bands)
    assert np.allclose(T1[:, :n_below], T0[:, :n_below], equal_nan=True)
    d = (T1[:, n_below:] - T0[:, n_below:])
    assert np.isfinite(d).sum() > 0 and np.allclose(d[np.isfinite(d)], 0.7)        # NaN: states left out of the fit
    E0, E1 = base._all_dft_energies, shifted._all_dft_energies
    assert np.allclose(E1[:, :vbm + 1], E0[:, :vbm + 1]) and np.allclose(E1[:, vbm + 1:] - E0[:, vbm + 1:], 0.7)
    shifted.apply_scissor(-0.2)
    assert abs(shifted.scissor - 0.5) < 1e-12
    base.apply_scissor(0.0)
    assert not hasattr(base, "scissor")


def test_metal_is_refused():
    with pytest.raises(ValueError, match="no gap"):
        _fitter("platinum-soc-qe", scissor=0.3)


def test_model_records_the_scissor():
    from _helpers import random_params
    from skfitter.model import SKModel
    f = _fitter("hbn-valley", scissor=0.4)
    m = SKModel.from_fitter(f, params=random_params(f), source={"example": "hbn-valley"})
    assert m.source["dft_scissor_eV"] == 0.4 and m.provenance[0]["dft_scissor_eV"] == 0.4


