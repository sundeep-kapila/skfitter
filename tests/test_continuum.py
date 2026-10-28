"""Continuum (k·p) wire models on a grid (prayog_tb.continuum, phase SC3).

* Bulk: the discretized bands converge to the k·p bands as the grid is refined (error ~ (k a)^2, quartered when a
  is halved); mass anisotropy and the band sign (valence) are kept.
* Spin-orbit: Rashba E = E0 + hbar^2 k^2 / 2m -+ alpha k, the radial (Te) term splits by 2 beta |k| in every
  direction - against kp_energies, the undiscretized formula.
* Wire subbands: a W x W hard-wall rectangle on the grid gives exactly 2 c (1 - cos(pi / (N + 1))) per direction
  (N + 1 = W / a), and converges to hbar^2 pi^2 (2 / W^2) / 2m; a circle converges to the Bessel zero j_0,1 = 2.4048.
* The file format: the parameters only, read back to the same model (load_model); the model passes check_model.
* fit_continuum recovers masses, the band sign and the k-linear term from a model (a bulk continuum model with
  known parameters, at k0 = 0).
* Split pair (Te's H4 / H5): E = E0 - hbar^2 k^2/2m -+ sqrt(Delta^2 + (beta k_z)^2) with the camel back of the top
  band; two valleys that are time-reversal partners (H(-k)* of each other); the fit recovers Delta, beta_zz and the
  masses; after a scissor with n_valence, "vbm" follows its valence bands although bands cross E_F = 0.
"""
import json
import os

import numpy as np
import pytest

import _paths  # noqa: F401  (puts the packages on the path)


def bulk(**kw):
    from skfitter.model import ContinuumModel
    return ContinuumModel({"shape": "bulk", **kw})


def energies(m, k_per_nm):
    from skfitter.model.bands import eigen
    return np.sort(np.real(eigen(m, np.asarray(k_per_nm, float) / 10.0))) * 1e3          # meV


def test_bulk_bands_converge_to_kp():
    from skfitter.model.continuum import kp_energies
    p = {"masses": [0.05, 0.1, 0.2], "E0_meV": 5.0}
    k = np.array([0.1, -0.05, 0.08])                                       # 1/nm
    exact = kp_energies(p, k)
    errs = [abs(energies(bulk(grid_nm=a, **p), k)[0] - exact[0]) for a in (1.0, 0.5, 0.25)]
    assert errs[0] < 0.5 and errs[1] / errs[0] == pytest.approx(0.25, rel=0.05) and errs[2] / errs[1] == pytest.approx(0.25, rel=0.05)
    v = energies(bulk(grid_nm=0.25, band="valence", **p), k)
    assert v[0] < 5.0 and abs(v[0] - kp_energies({**p, "band": "valence"}, k)[0]) < 0.05


@pytest.mark.parametrize("kind", ["rashba", "radial"])
def test_k_linear_spin_orbit(kind):
    from skfitter.model.continuum import kp_energies, soc_preset
    p = {"masses": [0.05, 0.05, 0.05], "soc_meV_nm": soc_preset(kind, 20.0, "x")}
    for k in ([0, 0, 0.05], [0.04, 0.03, 0.0], [0.02, -0.03, 0.04]):
        e = energies(bulk(grid_nm=0.2, **p), k)
        assert np.allclose(e, kp_energies(p, k), atol=0.02), (kind, k, e, kp_energies(p, k))


def test_wire_subbands_exact_and_converged():
    from skfitter.model import ContinuumModel, check_model
    from skfitter.model.continuum import HBAR2_2M0
    W, m = 30.0, 0.1
    for a in (3.0, 1.5):
        w = ContinuumModel({"cross_section": "rectangle", "width_nm": [W, W], "grid_nm": a, "masses": [m, m, m]})
        assert check_model(w) == [] and w.reach == [1, 0, 0]
        N = int(round(W / a)) - 1
        c = HBAR2_2M0 / (m * (10 * a) ** 2) * 1e3
        e = energies(w, [0, 0, 0])
        assert e[0] == pytest.approx(2 * 2 * c * (1 - np.cos(np.pi / (N + 1))), abs=1e-9) and e[1] == pytest.approx(e[0])
    cont = 2 * HBAR2_2M0 * 1e3 * np.pi ** 2 / (m * (10 * W) ** 2)
    assert abs(e[0] - cont) / cont < 0.01
    D = 30.0
    c = ContinuumModel({"cross_section": "circle", "diameter_nm": D, "grid_nm": 1.0, "masses": [m, m, m]})
    bessel = HBAR2_2M0 * 1e3 * (2.404826 / (10 * D / 2)) ** 2 / m
    assert abs(energies(c, [0, 0, 0])[0] - bessel) / bessel < 0.05


def test_file_round_trip(tmp_path):
    from skfitter.model import ContinuumModel, load_model
    w = ContinuumModel({"diameter_nm": 20, "grid_nm": 4, "E0_meV": -3, "g": np.diag([1, 2, 30]).tolist()})
    p = os.path.join(str(tmp_path), "w_model.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(w.to_dict(), fh)
    r = load_model(p)
    assert r.kind == "continuum" and r.n_sites == w.n_sites and np.allclose(r.g_tensor, np.diag([1, 2, 30]))
    assert np.allclose(energies(r, [0, 0, 0.1]), energies(w, [0, 0, 0.1]))
    with pytest.raises(ValueError):
        ContinuumModel({"cross_section": "rectangle", "width_nm": [4, 4], "grid_nm": 4})   # no grid point inside


def test_fit_recovers_the_parameters():
    from skfitter.model import fit_continuum
    from skfitter.model.continuum import soc_preset
    m = bulk(grid_nm=0.25, masses=[0.08, 0.12, 0.3], band="valence", E0_meV=-7.0, soc_meV_nm=soc_preset("radial", 15.0))
    f = fit_continuum(m, [0, 0, 0], band=0)
    assert f["band"] == "valence" and f["E0_meV"] == pytest.approx(-7.0, abs=1e-6)
    assert np.allclose(f["masses"], [0.08, 0.12, 0.3], rtol=0.01)
    assert np.allclose(np.diag(f["soc_meV_nm"]), 15.0, rtol=0.01)


def te_like(**kw):
    """Te's top valence pair at H in the textbook form (masses 0.11 / 0.26, Delta 63 meV); beta_zz 150 meV nm makes
    beta^2 / 2 Delta (179 meV nm^2) exceed hbar^2 / 2 m_z (147 meV nm^2): the camel back of the top band."""
    p = {"band": "valence", "E0_meV": -40.0, "masses": [0.11, 0.11, 0.26], "splitting_meV": 63.0, "valleys": 2,
         "soc_meV_nm": [[0, 0, 0], [0, 0, 0], [0, 0, 150.0]], "grid_nm": 0.2}
    p.update(kw)
    return p


def one_valley(p):
    """The valley-k0 block alone (what an atomistic model shows near H: H4 and H5, H' elsewhere in the zone)."""
    from skfitter.model.continuum import _blocks, check_params
    from skfitter.model.model import TBModel
    q = check_params({**p, "shape": "bulk"})
    onsite, hop = _blocks({**q, "valleys": 1})
    a = 10.0 * q["grid_nm"]
    return TBModel(lattice=np.eye(3) * a, species=["W"], frac_coords=[[0, 0, 0]], site_orbitals=[["s"]],
                   onsite=[onsite], hoppings={(0, 0, tuple(int(d == k) for k in range(3))): hop[d] for d in range(3)},
                   spinful=True, soc=True)


def test_split_pair_bands_camel_back_and_time_reversal():
    from skfitter.model.continuum import HBAR2_2M0_MEV_NM2, kp_energies
    p = te_like()
    m = bulk(**p)
    assert m.norbs(0) == 4 and m.spinful
    for k in ([0, 0, 0.03], [0.02, -0.01, 0.0], [0.01, 0.02, -0.04]):
        e = energies(m, k)
        exact = kp_energies(p, k)
        assert np.allclose(e[::2], exact, atol=0.05) and np.allclose(e[1::2], exact, atol=0.05)   # both valleys
    top = lambda kz: p["E0_meV"] - HBAR2_2M0_MEV_NM2 * kz ** 2 / 0.26 + np.sqrt(63.0 ** 2 + (150.0 * kz) ** 2)
    kz = np.linspace(0, 0.6, 601)
    assert kz[np.argmax(top(kz))] == pytest.approx(0.293, abs=0.01)     # the camel back: the top is off k0
    T = np.kron(np.array([[0, 1], [-1, 0]]), np.eye(2))                    # i sigma_y over the valley index
    for k in ([0.01, 0.02, 0.03], [0.0, 0.0, 0.05]):
        H, Hm = m.bloch(np.asarray(k) / 10.0), m.bloch(-np.asarray(k) / 10.0)
        assert np.allclose(T @ Hm.conj() @ T.T, H, atol=1e-12)             # time reversal across the valleys
    with pytest.raises(ValueError, match="valleys"):
        bulk(**{**p, "valleys": 1})


def test_fit_recovers_a_split_pair():
    from skfitter.model import fit_continuum
    p = te_like(grid_nm=0.1, soc_meV_nm=[[0, 0, 0], [0, 0, 0], [0, 0, 40.0]])
    f = fit_continuum(one_valley(p), [0, 0, 0], band=0)
    assert f["band"] == "valence" and f["valleys"] == 2 and f["fit"]["split_pair"]
    assert f["splitting_meV"] == pytest.approx(63.0, rel=1e-3) and f["E0_meV"] == pytest.approx(-40.0, abs=0.05)
    assert np.allclose(f["masses"], [0.11, 0.11, 0.26], rtol=0.01)
    assert f["soc_meV_nm"][2][2] == pytest.approx(40.0, rel=0.01)
    assert max(f["fit"]["transverse_beta_meV_nm"]) < 0.5
    m = bulk(**{k: v for k, v in f.items() if k != "fit"}, grid_nm=0.1)   # the fitted model rebuilds
    assert np.allclose(energies(m, [0, 0, 0.05])[::2], energies(one_valley(p), [0, 0, 0.05]), atol=0.05)


def test_fit_follows_the_scissor_valence_bands():
    """A model whose bands at k0 sit on both sides of E_F = 0 (a DFT semimetal): after a scissor with n_valence,
    "vbm" is the top valence pair, not the pair below E_F."""
    from skfitter.model import apply, fit_continuum
    from skfitter.model.model import TBModel
    from skfitter.model.continuum import _blocks, check_params
    q = check_params({**te_like(grid_nm=0.2, E0_meV=+30.0), "shape": "bulk"})   # H4 at +93 meV, H5 at -33 meV
    on, hop = _blocks({**q, "valleys": 1})
    a = 10.0 * q["grid_nm"]
    z = np.zeros((2, 2), complex)
    cb = 0.2 * np.eye(2)                                                     # a conduction pair 0.2 eV up
    m0 = TBModel(lattice=np.eye(3) * a, species=["W"], frac_coords=[[0, 0, 0]], site_orbitals=[["s", "p"]],
                 onsite=[np.block([[on, z], [z, cb + 0 * on]])],
                 hoppings={(0, 0, tuple(int(d == k) for k in range(3))): np.block([[hop[d], z], [z, -0.001 * np.eye(2)]])
                           for d in range(3)}, spinful=True)
    m = apply(m0, [{"kind": "scissor", "delta": 0.3, "n_valence": 2, "mesh": [4, 4, 4]}])
    f = fit_continuum(m, [0, 0, 0], band="vbm")
    assert f["fit"]["band_index"] == 0 and f["splitting_meV"] == pytest.approx(63.0, rel=1e-3)


def test_fit_with_n_valence_on_the_model_as_built():
    """The same pair from the unmodified semimetal model when the number of valence bands is given."""
    from skfitter.model import fit_continuum
    from skfitter.model.model import TBModel
    from skfitter.model.continuum import _blocks, check_params
    q = check_params({**te_like(grid_nm=0.2, E0_meV=+30.0), "shape": "bulk"})
    on, hop = _blocks({**q, "valleys": 1})
    a = 10.0 * q["grid_nm"]
    m0 = TBModel(lattice=np.eye(3) * a, species=["W"], frac_coords=[[0, 0, 0]], site_orbitals=[["s"]], onsite=[on],
                 hoppings={(0, 0, tuple(int(d == k) for k in range(3))): hop[d] for d in range(3)}, spinful=True)
    f = fit_continuum(m0, [0, 0, 0], band="vbm", n_valence=2)
    assert f["fit"]["band_index"] == 0 and f["splitting_meV"] == pytest.approx(63.0, rel=1e-3)
