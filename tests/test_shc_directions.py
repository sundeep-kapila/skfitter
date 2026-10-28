"""Spin Hall conductivity with directions given as Cartesian vectors (e.g. the [111] frame of a kagome plane):
a vector along an axis gives the same result as the axis name, and the spin operator is linear in the direction."""
import contextlib
import io

import numpy as np
import pytest

import _examples_structure as X
import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs scipy and pymatgen")


def _shc(f, spin, alpha, beta):
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, r = f.compute_spin_hall_conductivity(nk=(4, 4, 4), spin=spin, alpha=alpha, beta=beta,
                                                   efermi_range=(-0.5, 0.5), num_efermi=5, refine=False, show=False)
    return r


def test_vector_directions_match_axis_names_and_are_linear():
    f = X.fitter("platinum-soc-qe")
    f.opt_params = f._param_vec_to_dict(np.random.default_rng(1).normal(0, 0.3, len(f._get_param_keys())))
    from skfitter import core
    ref = _shc(f, "z", "x", "y")
    assert np.abs(ref["sigma"]).max() > 1.0                              # a non-trivial SHC
    vec = _shc(f, [0, 0, 2], [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(vec["sigma"], ref["sigma"], rtol=1e-10, atol=1e-8)
    assert vec["spin"] == "[002]" and vec["alpha"] == "[100]"             # labels as given
    sx, sy = _shc(f, "x", "x", "y")["sigma"], _shc(f, "y", "x", "y")["sigma"]
    mixed = _shc(f, [1, 1, 0], "x", "y")["sigma"]                     # spin along [110]: (sx + sy)/sqrt(2)
    np.testing.assert_allclose(mixed, (sx + sy) / np.sqrt(2), rtol=1e-8, atol=1e-6)
    assert core.direction_label([1, -1, 0]) == "[1-10]"
    with pytest.raises(ValueError):
        core._cart_direction("w")
