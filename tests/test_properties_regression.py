"""Properties (phase M3): the migrated calculators give SKFitter's step-4 numbers on the same models.

Each illustrative project's fitter gets random parameters (the same model for both codes); SKFitter's own methods
(``plot_berry_curvature_sk``, ``compute_wilson_loops_sk``, ``compute_chern_numbers_sk``,
``plot_spin_berry_curvature_sk``, ``compute_anomalous_hall_conductivity``, ``compute_spin_hall_conductivity`` and
``..._vs_temperature``) are compared with ``prayog_properties.run`` on ``SKModel.from_fitter``. SKFitter differentiates
H by central differences, the Properties kernel analytically: agreement to ~1e-6 relative (refinement is switched off
in the Hall sums so that both visit the same k-points).
"""
import numpy as np
import pytest

import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs the SKFitter science stack")


def setup_module():
    import os
    import sys
    pass


def _fitter(example_id, seed=0, **cfg):
    import _examples_structure as X
    from _helpers import random_params
    f = X.fitter(example_id, {**X.cfg(example_id), **cfg})
    f.opt_params = random_params(f, seed)
    return f


def _model(f):
    from skfitter.model import SKModel
    return SKModel.from_fitter(f, source={"test": True})


def _close(a, b, rel=1e-6, abs_=1e-8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    scale = max(1.0, float(np.max(np.abs(b))))
    assert np.max(np.abs(a - b)) <= rel * scale + abs_, (np.max(np.abs(a - b)), scale)


@pytest.fixture(scope="module")
def hbn():
    f = _fitter("hbn-valley", seed=3)
    return f, _model(f)


def test_berry_curvature_path_and_map(hbn):
    import skfitter.properties as P
    f, m = hbn
    idx, nocc = list(f.target_tb_band_indices), int(f.tb_vbm_index) + 1
    _, _, ref = f.plot_berry_curvature_sk(mode="path", dk=0.02, show=False)
    r = P.run(m, "geometric.berry_curvature", {"mode": "path", "dk": 0.02, "bands": idx, "n_occupied": nocc})
    _close(r["data"]["omega"], ref["omega"])
    _close(r["data"]["total_occupied"], ref["total_occupied"])
    assert r["views"][0]["ticks"][0][1] == ref["ticks"][1][0]
    _, _, ref = f.plot_berry_curvature_sk(mode="map", nk=16, show=False)
    r = P.run(m, "geometric.berry_curvature", {"mode": "map", "nk": 16, "bands": idx, "n_occupied": nocc})
    _close(r["data"]["omega"], ref["omega"])


def test_wilson_loops_and_chern_numbers(hbn):
    import skfitter.properties as P
    f, m = hbn
    idx, nocc = list(f.target_tb_band_indices), int(f.tb_vbm_index) + 1
    _, _, ref = f.compute_wilson_loops_sk(n_loop=60, n_radii=6, show=False)
    r = P.run(m, "geometric.wilson", {"n_loop": 60, "n_radii": 6, "bands": idx, "n_occupied": nocc})
    for v in ("K", "K'"):
        _close(r["data"]["valleys"][v]["occupied"], ref["valleys"][v]["occupied"])
    _, _, ref = f.compute_chern_numbers_sk(nk=24, show=False)
    r = P.run(m, "geometric.chern", {"nk": 24, "bands": idx, "n_occupied": nocc})
    o, ro = r["data"]["occupied"], ref["occupied"]
    assert abs(o["C"] - ro["C"]) < 1e-8 and abs(o["C_K"] - ro["C_K"]) < 1e-8 and abs(o["C_Kp"] - ro["C_Kp"]) < 1e-8
    for i in idx:
        assert abs(r["data"]["per_band"][str(i)]["C"] - ref["per_band"][i]["C"]) < 1e-8


def test_spin_berry_curvature():
    import skfitter.properties as P
    f = _fitter("graphene-soc", seed=1)
    m = _model(f)
    idx = list(f.target_tb_band_indices)
    _, _, ref = f.plot_spin_berry_curvature_sk(dk=0.03, efermi=0.1, smearing=0.02, show=False)
    r = P.run(m, "geometric.spin_berry", {"dk": 0.03, "efermi": 0.1, "smearing": 0.02, "bands": idx})
    _close(r["data"]["total"], ref["total"], rel=1e-5)
    _close(r["data"]["omega"], ref["omega"], rel=1e-5)
    assert r["data"]["paired"] == ref["paired"]


@pytest.fixture(scope="module")
def iron():
    f = _fitter("iron-soc-qe", seed=2, include_soc=True)
    return f, _model(f)


def test_anomalous_hall(iron):
    import skfitter.properties as P
    f, m = iron
    _, _, ref = f.compute_anomalous_hall_conductivity(nk=(6, 6, 6), efermi_range=(-1, 1), num_efermi=11, smearing=0.05,
                                                      refine=False, show=False)
    r = P.run(m, "geometric.ahc", {"mesh": [6, 6, 6], "energies": {"min": -1, "max": 1, "n": 11}, "smearing": 0.05,
                                   "refine": False})
    assert np.allclose(r["data"]["efermi"], ref["efermi"])
    _close(r["data"]["sigma"], ref["sigma"], rel=1e-5)
    assert np.max(np.abs(ref["sigma"])) > 1.0                 # a non-trivial check: magnetism + SOC


def test_spin_hall_and_temperature(iron):
    import skfitter.properties as P
    f, m = iron
    _, _, ref = f.compute_spin_hall_conductivity(nk=(5, 5, 5), efermi_range=(-1, 1), num_efermi=5, smearing=0.05,
                                                 refine=False, show=False)
    r = P.run(m, "geometric.shc", {"mesh": [5, 5, 5], "energies": {"min": -1, "max": 1, "n": 5}, "smearing": 0.05,
                                   "refine": False})
    _close(r["data"]["sigma"], ref["sigma"], rel=1e-5)
    _, _, ref = f.compute_spin_hall_conductivity_vs_temperature(temperatures=[10, 300], nk=(4, 4, 4), n_shifts=2,
                                                                refine=False, show=False)
    r = P.run(m, "geometric.shc", {"mesh": [4, 4, 4], "mode": "temperature", "temperatures": [10, 300], "n_shifts": 2,
                                   "refine": False})
    _close(r["data"]["sigma_meshes"], ref["sigma_meshes"], rel=1e-5)


def test_refinement_agrees_within_its_own_noise(iron):
    """With refinement on, the two codes may refine slightly different points (analytic vs finite-difference
    derivatives near the threshold); the totals still agree closely."""
    import skfitter.properties as P
    f, m = iron
    _, _, ref = f.compute_anomalous_hall_conductivity(nk=(4, 4, 4), efermi_range=(0, 0), num_efermi=1, smearing=0.05,
                                                      refine_levels=1, show=False)
    r = P.run(m, "geometric.ahc", {"mesh": [4, 4, 4], "energies": {"min": 0, "max": 0, "n": 1}, "smearing": 0.05,
                                   "refine_levels": 1})
    assert abs(r["data"]["sigma"][0] - ref["sigma"][0]) <= 1e-3 * max(1.0, abs(ref["sigma"][0]))


def test_model_carries_the_dft_path(hbn):
    from skfitter.properties.services import kspace
    f, m = hbn
    pts = m.source["kpath_points"]
    assert pts and pts[0] is not None and len(pts[0][1]) == 3
    kp = kspace.path(m, None, 0.02)
    assert np.allclose(kp["x"], f._dense_kpath(0.02)["x"])


def test_threads_and_refinement_window(iron):
    """skfitter berry --workers: the k-point blocks on several threads give the same sum. refine_window > 0 refines
    only where bands near the scanned Fermi energies have a large curvature: a window wider than every band is the
    full refinement, a narrow one refines fewer points (at 40^3 the Fe AHC changes by about 2 %)."""
    from skfitter.core import SKFitter
    f, _ = iron
    kw = dict(nk=(5, 5, 5), efermi_range=(-0.5, 0.5), num_efermi=11, smearing=0.05, refine=True, refine_levels=1,
              show=False)

    def run(workers=1, window=0.0):
        seen = []
        try:
            SKFitter.k_workers, SKFitter.refine_window = workers, window
            SKFitter.k_progress = lambda level, done, total: seen.append((level, done, total))
            _, _, res = f.compute_anomalous_hall_conductivity(**kw)
        finally:
            SKFitter.k_workers, SKFitter.k_progress, SKFitter.refine_window = 1, None, 0.0
        return res["sigma"], sum(t for lv, d, t in seen if d == 0 and lv > 0), seen

    one, n_one, seen = run()
    many, n_many, _ = run(workers=3)
    wide, n_wide, _ = run(window=1e3)
    narrow, n_narrow, _ = run(window=0.05)
    np.testing.assert_allclose(many, one, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(wide, one, rtol=1e-10, atol=1e-8)
    assert n_many == n_wide == n_one > 0 and n_narrow < n_one
    assert seen[0] == (0, 0, 125) and (0, 125, 125) in seen
