"""Properties (phase M3, prayog_properties): framework, services and the physics of each calculator against exact
results.

* Framework: the registry describes every calculator as JSON; unsuitable models are refused with the reason (spin
  quantities on a spinless model, valleys on a non-hexagonal or 3D cell); unknown inputs refused; results save and
  load; a shared kernel is reused by several calculators.
* Services: analytic ∂H/∂k and ∂²H/∂k² against finite differences (SK, textbook TB, orbital centres off the atoms,
  non-orthogonal); the tetrahedron / triangle DOS integrates to the number of states; moving orbital centres (the
  Bloch convention) changes Ω(k) but not the anomalous Hall conductivity.
* Electronic: bands = eigenvalues of bloch(k); simple cubic s band: m* = ħ²/(2ta²) at Γ (exact); DOS of MoS2 holds
  2 electrons below E_F.
* Geometric: Haldane model C = −1 and σ_xy = +e²/h (= −C e²/h) in the gap, C = 0 for the trivial insulator; two-band
  Berry curvature of the lower band Ω = +½ d̂·(∂x d̂ × ∂y d̂); Kane-Mele quantum spin Hall σ^z_xy = −e²/h per sheet (in (ħ/e) S);
  gapless graphene: Berry phase π around K.
"""
import json
import os
import sys

import numpy as np
import pytest

import _paths

sys.path.insert(0, os.path.join(_paths.ROOT, "properties"))
GRAPHENE = os.path.join(_paths.ROOT, "tests", "data", "models", "graphene_pz_model.json")


@pytest.fixture(scope="module")
def P():
    import skfitter.properties as prayog_properties
    return prayog_properties


def _sheet(model, sigma):
    from skfitter.properties.services.units import E2_OVER_H, sheet_factor
    return sigma * sheet_factor(model) * 1e-8 / E2_OVER_H


# -- framework -------------------------------------------------------------------------------------------------
def test_registry_and_refusals(P):
    from skfitter.model import textbook as T
    reg = P.registry()
    json.dumps(reg)
    ids = {c["id"] for c in reg["calculators"]}
    assert {"electronic.bands", "electronic.dos", "electronic.edges", "electronic.masses", "electronic.compare",
            "geometric.berry_curvature", "geometric.spin_berry", "geometric.wilson", "geometric.chern",
            "geometric.ahc", "geometric.shc"} <= ids
    assert [f["id"] for f in reg["families"] if f["installed"]] == ["electronic", "geometric"]
    assert all(c["formula"] and c["help"] for c in reg["calculators"])      # the web typesets every formula (KaTeX)
    h = T.haldane()
    el = P.eligible(h)
    assert any(i["level"] == "error" and "spin" in i["message"] for i in el["geometric.shc"])
    assert not [i for i in el["geometric.chern"] if i["level"] == "error"]
    with pytest.raises(ValueError, match="spin operator"):
        P.run(h, "geometric.spin_berry")
    with pytest.raises(ValueError, match="2D"):
        P.run(T.cubic_s(), "geometric.wilson")
    with pytest.raises(ValueError, match="unknown input"):
        P.run(h, "geometric.chern", {"colour": "red"})
    with pytest.raises(ValueError, match="nk"):
        P.run(h, "geometric.chern", {"nk": 2})
    assert any("time-reversal" in i["message"] for i in P.eligible(T.mos2_three_band())["geometric.ahc"])


def test_result_file_round_trip_and_shared_kernel(P, tmp_path):
    from skfitter.model import textbook as T
    m = T.mos2_three_band()
    rs = P.run_many(m, [("electronic.bands", {"weights": "orbital"}), ("geometric.berry_curvature", {})])
    assert rs[0]["model"]["hash"] == rs[1]["model"]["hash"] and rs[0]["format"] == P.FORMAT
    files = P.save_result(rs[0], str(tmp_path))
    back = P.load_result(files["json"])
    assert back["calculator"]["id"] == "electronic.bands" and files["csv"] and os.path.getsize(files["csv"][0]) > 100
    v = back["views"][0]
    assert v["type"] == "bands" and set(v["weights"]) == {"dz2", "dxy", "dx2y2"}
    assert back["conventions"]["bloch_phase"].startswith("exp(i k·(R + c_j − c_i))")
    c = P.prepare(m, "geometric.ahc", {"mesh": [10, 10, 10]})[3]
    assert c.kpoints == 100 and c.matrix == 6                  # a 2D model is sampled in its plane only


# -- services ----------------------------------------------------------------------------------------------------
def _derivative_errors(model):
    from skfitter.properties.services.bloch import Kernel
    K = Kernel(model)
    k = np.random.default_rng(1).normal(size=(4, 3))
    h = 1e-5
    e1 = max(np.abs((K.H(k + h * np.eye(3)[a]) - K.H(k - h * np.eye(3)[a])) / (2 * h) - K.dH(k, a)).max()
             for a in range(3))
    e2 = max(np.abs((K.dH(k + h * np.eye(3)[b], a) - K.dH(k - h * np.eye(3)[b], a)) / (2 * h) - K.d2H(k, a, b)).max()
             for a in range(3) for b in range(3))
    ev = np.abs(np.linalg.eigvalsh(K.H(k)) - np.array([np.linalg.eigvalsh(model.bloch(x)) for x in k])).max()
    return e1, e2, ev


def _off_centre_haldane(shift):
    from skfitter.model import TBModel, textbook as T
    h = T.haldane(M=0.1)
    d = h.to_dict()
    m = TBModel.from_dict(d)
    m.orbital_offsets = [np.array([shift]), np.zeros((1, 3))]          # move orbital A off its atom
    return m


def test_kernel_derivatives(P):
    from skfitter.model import load_model, textbook as T
    for m in (load_model(GRAPHENE), T.mos2_three_band(), T.kane_mele(lam_v=0.05), _off_centre_haldane([0.3, -0.2, 0.1])):
        e1, e2, ev = _derivative_errors(m)
        assert e1 < 1e-6 and e2 < 1e-5 and ev < 1e-10, (e1, e2, ev)


@pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs the SKFitter science stack")
def test_kernel_non_orthogonal():
    from skfitter.model import SKModel
    from skfitter.properties.services.bloch import Kernel
    d = SKModel.load(GRAPHENE).to_dict()
    d["settings"]["non_orthogonal"] = True
    d["params"].update({"s_pzpzp_0": 0.1, "s_pzpzs_0": 0.0, "eta_s_pzpzp": 2.5, "eta_s_pzpzs": 2.5})
    m = SKModel.from_dict(d)
    K = Kernel(m)
    k = np.random.default_rng(2).normal(size=(3, 3))
    import scipy.linalg as sla
    ref = np.array([sla.eigh(m.bloch(x), m.bloch(x, overlap=True), eigvals_only=True) for x in k])
    assert np.abs(np.linalg.eigvalsh(K.H(k)) - ref).max() < 1e-10


def test_dos_integrates_to_the_states(P):
    from skfitter.model import textbook as T
    for m, mesh in ((T.cubic_s(), [12, 12, 12]), (T.mos2_three_band(), [24, 24, 1])):
        for method in ("tetrahedron", "gaussian"):
            r = P.run(m, "electronic.dos", {"mesh": mesh, "method": method, "energies": {"min": -9, "max": 9, "n": 9001},
                                           "projections": "none"})
            N = r["data"]["N"]
            states = (1 if m.spinful else 2) * sum(m.norbs(i) for i in range(m.n_sites))
            assert abs(N[-1] - states) < 2e-3 * states, (method, N[-1])
    r = P.run(T.mos2_three_band(), "electronic.dos", {"mesh": [24, 24, 1]})
    assert "2.0000" in dict(r["summary"])["electrons per cell below E_F"]


def test_bloch_convention_changes_omega_not_ahc(P):
    from skfitter.model import textbook as T
    a, b = T.haldane(M=0.1), _off_centre_haldane([0.4, 0.3, 0.0])
    ka = P.run(a, "geometric.berry_curvature", {"dk": 0.05})["data"]["total_occupied"]
    kb = P.run(b, "geometric.berry_curvature", {"dk": 0.05})["data"]["total_occupied"]
    assert np.abs(np.array(ka) - np.array(kb)).max() > 1e-3          # Ω(k) depends on the convention
    p = {"mesh": [48, 48, 1], "energies": {"min": 0, "max": 0, "n": 1}, "smearing": 0.0}
    sa = P.run(a, "geometric.ahc", p)["data"]["sigma"][0]
    sb = P.run(b, "geometric.ahc", p)["data"]["sigma"][0]
    assert abs(sa - sb) < 1e-6 * abs(sa)                             # the integral does not


# -- electronic --------------------------------------------------------------------------------------------------
def test_bands_and_masses(P):
    from skfitter.model import textbook as T
    from skfitter.properties.services import kspace
    m = T.mos2_three_band()
    r = P.run(m, "electronic.bands", {"dk": 0.05, "emin": -20, "emax": 20})
    kp = kspace.path(m, None, 0.05)
    ref = np.array([np.linalg.eigvalsh(m.bloch(k)) for k in kp["cart"]])
    assert np.abs(np.array(r["data"]["energies"]) - ref).max() < 1e-10
    assert [t[1] for t in r["views"][0]["ticks"]] == ["Γ", "M", "K", "Γ", "K'"]
    t, a = 0.7, 2.5
    c = T.cubic_s(t=t, a=a, onsite=5.0)
    for method in ("perturbation", "finite_difference"):
        r = P.run(c, "electronic.masses", {"where": "point", "k": [0, 0, 0], "band": 0, "method": method})
        masses = r["data"]["point"]["masses"]
        assert np.allclose(masses, 7.619964231 / (2 * t * a * a), rtol=1e-6 if method == "perturbation" else 1e-4)
    r = P.run(m, "electronic.masses", {})
    assert r["data"]["VBM"]["masses"].__len__() == 2 and all(x < 0 for x in r["data"]["VBM"]["masses"])


def test_edges_and_compare(P, tmp_path):
    from skfitter.model import apply, textbook as T
    m = T.mos2_three_band()
    r = P.run(m, "electronic.edges", {})
    assert abs(r["data"]["edges"]["gap"] - 1.5898) < 1e-3 and set(r["data"]["valleys"]) == {"K", "K'"}
    other = apply(m, [{"kind": "onsite_shift", "shift": 0.01}])
    path = str(tmp_path / "shifted_model.json")
    other.save(path)
    r = P.run(m, "electronic.compare", {"models": [path], "labels": ["shifted"], "dk": 0.05})
    assert len([v for v in r["views"] if v["type"] == "bands"]) == 2
    assert abs(r["views"][-1]["rows"][0][1] - 10.0) < 1e-6           # every band 10 meV up


# -- geometric ---------------------------------------------------------------------------------------------------
def test_haldane_chern_and_ahc(P):
    from skfitter.model import textbook as T
    h = T.haldane(t2=0.1, M=0.1)
    o = P.run(h, "geometric.chern", {"nk": 30})["data"]["occupied"]
    assert abs(o["C"] + 1) < 1e-9
    s = P.run(h, "geometric.ahc", {"mesh": [60, 60, 1], "energies": {"min": 0, "max": 0, "n": 1}, "smearing": 0.0})
    assert abs(_sheet(h, s["data"]["sigma"][0]) - 1.0) < 1e-3       # σ_xy = −C e²/h
    trivial = T.haldane(t2=0.02, M=0.5)
    assert abs(P.run(trivial, "geometric.chern", {"nk": 30})["data"]["occupied"]["C"]) < 1e-9


def test_two_band_berry_curvature(P):
    from skfitter.model import textbook as T
    from skfitter.properties.geometric import kubo
    from skfitter.properties.services.bloch import Kernel
    h = T.haldane(t2=0.1, M=0.2)
    K = Kernel(h)
    k = np.random.default_rng(4).normal(size=(6, 3)) * 0.8
    k[:, 2] = 0
    om, _ = kubo.band_curvature(K, k, [0], "z", average_degenerate=False)
    s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]

    def dhat(kk):
        H = K.H(kk[None])[0]
        d = np.array([np.real(np.trace(H @ x)) / 2 for x in s])
        return d / np.linalg.norm(d)
    eps = 1e-5
    for n, kk in enumerate(k):
        dx = (dhat(kk + [eps, 0, 0]) - dhat(kk - [eps, 0, 0])) / (2 * eps)
        dy = (dhat(kk + [0, eps, 0]) - dhat(kk - [0, eps, 0])) / (2 * eps)
        ref = 0.5 * dhat(kk) @ np.cross(dx, dy)                      # lower band: Ω₋ = +½ d̂·(∂x d̂ × ∂y d̂)
        assert abs(om[n, 0] - ref) < 1e-5 * max(1, abs(ref)), (om[n, 0], ref)


def test_kane_mele_quantum_spin_hall(P):
    from skfitter.model import textbook as T
    km = T.kane_mele(lam_so=0.1)
    r = P.run(km, "geometric.shc", {"mesh": [60, 60, 1], "energies": {"min": 0, "max": 0, "n": 1}, "smearing": 0.0})
    assert abs(_sheet(km, r["data"]["sigma"][0]) + 1.0) < 1e-3
    trivial = T.kane_mele(lam_so=0.02, lam_v=0.5)
    r = P.run(trivial, "geometric.shc", {"mesh": [40, 40, 1], "energies": {"min": 0, "max": 0, "n": 1},
                                         "smearing": 0.0})
    assert abs(_sheet(trivial, r["data"]["sigma"][0])) < 0.05


def test_graphene_berry_phase_pi(P):
    from skfitter.model import textbook as T
    g = T.haldane(t2=0.0, M=0.0)
    r = P.run(g, "geometric.wilson", {"n_radii": 4, "n_occupied": 1})
    for v in ("K", "K'"):
        assert abs(abs(r["data"]["valleys"][v]["occupied"][-1]) - np.pi) < 1e-6
    assert set(r["data"]["gapless"]) == {"K", "K'"}


def test_occupied_bands_counted_with_the_kernel(P):
    """M3f: the bands below E_F come from the kernel's eigenvalues on the whole mesh (fast), with count_valence's rule:
    the same count as the model's own Bloch sums, the same refusal for a metal, remembered on the kernel."""
    from skfitter.model import SKModel, textbook
    from skfitter.model.bands import count_valence
    from skfitter.properties.services.bloch import Kernel
    from skfitter.properties.services.occupation import n_occupied
    for m in (textbook.mos2_three_band(), textbook.haldane(t2=0.1, M=0.1), SKModel.load(GRAPHENE)):
        K = Kernel(m)
        assert n_occupied(m, K) == count_valence(m) and K._memo
    metal = textbook.cubic_s()
    with pytest.raises(ValueError, match="metal"):
        n_occupied(metal)
    with pytest.raises(ValueError, match="metal"):
        count_valence(metal)


def test_graphene_quantum_anomalous_hall(P):
    """M3h: Rashba plus an exchange field make graphene a quantum anomalous Hall insulator with C = 2 (Qiao et al.,
    PRB 82, 161414 (2010)); either alone does not."""
    from skfitter.model import apply, textbook
    g = textbook.graphene()
    m = apply(g, [{"kind": "rashba", "lam": 0.1}, {"kind": "exchange", "energy": 0.2}])
    r = P.run(m, "geometric.chern", {"nk": 90})
    assert r["data"]["occupied"]["C"] == pytest.approx(2.0, abs=1e-6)
    a = P.run(m, "geometric.ahc", {"mesh": [150, 150, 1], "energies": {"min": 0, "max": 0, "n": 1}, "smearing": 0})
    assert _sheet(m, a["data"]["sigma"][0]) == pytest.approx(-2.0, abs=0.02)
    a = P.run(apply(g, [{"kind": "exchange", "energy": 0.2}]), "geometric.ahc",
              {"mesh": [60, 60, 1], "energies": {"min": 0, "max": 0, "n": 1}, "smearing": 0, "refine": False})
    assert abs(a["data"]["sigma"][0]) < 1e-8                 # no spin-orbit coupling: no Hall effect
