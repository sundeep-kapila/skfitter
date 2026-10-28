"""Model-level modifiers and the band-edge tools (prayog_tb, phase M2a), checked against exact statements.

* Angular momentum: our L matrices are SKFitter's (p, d, f), obey [Lx, Ly] = i Lz and L^2 = l(l+1).
* An isolated p atom: without spin-orbit coupling E = mu_B B (m_l + g m_s); with lambda L.S the weak-field
  splitting follows the Landé factors g_J = 4/3 (j = 3/2) and 2/3 (j = 1/2), for B along any direction.
* Graphene p_z: a sublattice mass opens a direct gap of exactly delta at K; the spin Zeeman term splits every band
  by g mu_B B; spin along z stays conserved only for B along z.
* Monolayer MoS2 (three-band model of Liu et al. 2013, d_z2, d_xy, d_x2-y2 with spin-orbit coupling): out of plane
  the valence-band top moves with g = +3 in one valley and -3 in the other (orbital m = ±2 plus spin), the
  conduction band (m = 0) splits with g = ±1, so the gap changes with g = -4 / +2 per valley; in plane the valence
  band does not move to first order (spins locked out of plane) while the conduction band splits.
* Scissor: the gap grows by exactly delta, valence bands do not move (orthogonal and non-orthogonal models), the
  error between mesh points is small and recorded; metals are refused.
* Exchange and Rashba (phase M3h): an exchange field of E eV splits every band of graphene by exactly 2E along any
  direction (S_z conserved only along z); Rashba lam on graphene's nearest-neighbour bonds puts the four Dirac states
  at K at 0, 0, ±3 lam (Kane and Mele) and breaks S_z; both are Hermitian, survive the file and fold in supercells.
* Files: a modified model saves its base and modifiers and reads back to the same H(k); an SK base stays an SK
  model; supercells of a modified model fold its bands; NEGF Transport reads modified models.
"""
import json
import os

import numpy as np
import pytest

import _paths

MU = 5.7883818060e-5
GRAPHENE = os.path.join(_paths.ROOT, "tests", "data", "models", "graphene_pz_model.json")
science = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs the SKFitter science stack")


# == models ======================================================================================================
def p_atom(lam=0.0):
    from skfitter.model.textbook import atom
    return atom(("px", "py", "pz"), lam)


def mos2(soc=True, shift=-0.8):
    from skfitter.model.textbook import mos2_three_band
    return mos2_three_band(soc=soc, shift=shift)


VALLEYS = {"K": [2 / 3, 1 / 3, 0], "K'": [1 / 3, 2 / 3, 0]}           # = prayog_tb.textbook.MOS2_VALLEYS


def graphene():
    from skfitter.model import load_model
    return load_model(GRAPHENE)


# == angular momentum ============================================================================================
@science
@pytest.mark.parametrize("fam, labels", [("p", ["px", "py", "pz"]), ("d", ["dxy", "dyz", "dzx", "dx2y2", "dz2"]),
                                         ("f", ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"])])
def test_L_matrices(fam, labels):
    from skfitter.core import SKFitter
    from skfitter.model.orbitals import L_matrices
    L = L_matrices(labels)
    ref = SKFitter._get_angular_momentum_matrices(fam)
    for a in range(3):
        assert np.allclose(L[a], ref[a])
    ell = {"p": 1, "d": 2, "f": 3}[fam]
    assert np.allclose(L[0] @ L[1] - L[1] @ L[0], 1j * L[2])
    assert np.allclose(sum(x @ x for x in L), ell * (ell + 1) * np.eye(len(labels)))


def test_L_needs_atomic_labels():
    from skfitter.model.orbitals import L_matrices
    assert L_matrices(["w1", "w2"]) is None
    Lz = L_matrices(["px", "py", "pz", "s"])[2]                 # s is not coupled to p
    assert np.allclose(Lz[3], 0) and np.allclose(Lz[:, 3], 0)
    two = L_matrices(["px", "py", "pz", "px", "py", "pz"])[2]    # two p shells: no coupling between them
    assert np.allclose(two[:3, 3:], 0) and np.allclose(two[:3, :3], two[3:, 3:])


# == an isolated atom ============================================================================================
@pytest.mark.parametrize("direction", [[0, 0, 1], [1, 0, 0], [1, 1, 1]])
def test_p_atom_without_soc(direction):
    from skfitter.model import apply, bands as B
    u = np.array(direction, float) / np.linalg.norm(direction)
    Bt = 3.0
    m = apply(p_atom(0.0), [{"kind": "zeeman", "B": (Bt * u).tolist(), "g": 2.0}])
    e = np.sort(B.eigen(m, np.zeros(3)))
    expect = np.sort([MU * Bt * (ml + 2.0 * ms) for ml in (-1, 0, 1) for ms in (-0.5, 0.5)])
    assert np.allclose(e, expect, atol=1e-12)


@pytest.mark.parametrize("direction", [[0, 0, 1], [0.3, -0.4, 0.5]])
def test_p_atom_lande(direction):
    from skfitter.model import apply, bands as B
    lam, Bt = 0.1, 1.0
    u = np.array(direction, float) / np.linalg.norm(direction)
    e = np.sort(B.eigen(apply(p_atom(lam), [{"kind": "zeeman", "B": (Bt * u).tolist(), "g": 2.0}]), np.zeros(3)))
    j12 = [-lam + (2 / 3) * mj * MU * Bt for mj in (-0.5, 0.5)]
    j32 = [lam / 2 + (4 / 3) * mj * MU * Bt for mj in (-1.5, -0.5, 0.5, 1.5)]
    assert np.allclose(e, np.sort(j12 + j32), atol=1e-8)          # second order ~ (mu_B B)^2 / lambda ~ 3e-8 eV


# == graphene ====================================================================================================
@science
def test_sublattice_mass_opens_the_gap():
    from skfitter.model import apply, check_model, sublattice_mass, bands as B
    g = graphene()
    e0 = B.band_edges(g)
    assert e0["gap"] < 1e-6
    for delta in (0.1, 0.3):
        m = apply(g, sublattice_mass(delta, [0], [1]))
        assert check_model(m) == []
        e = B.band_edges(m)
        assert abs(e["gap"] - delta) < 1e-6 and e["gap_type"] == "direct"
        a = np.linalg.norm(g.lattice[0])                        # the gap sits at a corner K of the zone
        kf = (np.array(e["vbm"]["k_frac"]) + 0.5) % 1.0 - 0.5
        ks = [np.linalg.norm((kf + [i, j, 0]) @ B.reciprocal(g)) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        assert abs(min(ks) - 4 * np.pi / (3 * a)) < 1e-4


@science
def test_spin_zeeman_on_graphene():
    from skfitter.model import apply, check_model, bands as B
    g = graphene()
    k = np.array([0.11, 0.27, 0.0]) @ B.reciprocal(g)
    e0 = B.eigen(g, k)
    for Bvec, cons in (([0, 0, 8], True), ([8, 0, 0], False)):
        m = apply(g, [{"kind": "zeeman", "B": Bvec, "g": 2.0}])
        assert m.spinful and not g.spinful and m.modifiers[0] == {"kind": "spinful", "auto": True}
        assert m.conserves_sz is cons and check_model(m) == []
        e = B.eigen(m, k)
        expect = np.sort(np.concatenate([e0 - MU * 8, e0 + MU * 8]))
        assert np.allclose(e, expect, atol=1e-12)
        assert m.warnings == []                                 # p_z: L.B is zero but defined


@science
def test_field_sweep_slopes_on_graphene():
    from skfitter.model import sublattice_mass, apply, bands as B
    gapped = apply(graphene(), sublattice_mass(0.2, [0], [1]))
    s = B.field_sweep(gapped, [0, 5, 10], direction="out-of-plane", g=2.0)
    assert s["n_valence"] == 2 and [r["B"] for r in s["rows"]] == [0, 5, 10]
    # spin only (p_z has no orbital moment): VBM up and CBM down by mu_B B, the gap closes at 2 mu_B B
    assert abs(s["slopes"]["vbm"]["g"] - 1.0) < 1e-3 and abs(s["slopes"]["cbm"]["g"] + 1.0) < 1e-3
    assert abs(s["slopes"]["gap"]["g"] + 2.0) < 1e-3


# == monolayer MoS2: valley Zeeman, out of plane against in plane =================================================
def test_mos2_model_is_symmetric():
    from skfitter.model import check_model, bands as B
    m = mos2(soc=False, shift=0.0)
    assert check_model(m) == []
    rec = B.reciprocal(m)
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    C3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for k in np.random.default_rng(0).normal(size=(4, 3)) * [1, 1, 0]:
        assert np.allclose(B.eigen(m, k), B.eigen(m, C3 @ k), atol=1e-12)
    eK = B.eigen(m, np.array(VALLEYS["K"]) @ rec)
    assert abs((eK[1] - eK[0]) - 1.663) < 2e-3                  # the direct gap at K of the GGA model


def test_mos2_valley_zeeman_out_of_plane():
    from skfitter.model import bands as B
    s = B.field_sweep(mos2(), [0, 2, 4, 6], direction="out-of-plane", valleys=VALLEYS)
    sl = s["slopes"]
    gK, gKp = sl["K:vb"]["g"], sl["K':vb"]["g"]
    assert abs(abs(gK) - 3.0) < 0.02 and abs(gK + gKp) < 1e-3     # +3 in one valley, -3 in the other
    assert abs(sl["K:cb"]["g"] + 1.0) < 0.02 and abs(sl["K':cb"]["g"] + 1.0) < 0.02   # lower spin branch
    assert {round(sl["K:gap"]["g"]), round(sl["K':gap"]["g"])} == {-4, 2}
    row = s["rows"][-1]
    assert abs(row["K:cb_levels"][1] - row["K:cb_levels"][0] - 2 * MU * 2.0023 * 6 / 2) < 1e-6   # CB split g mu_B B
    assert all(abs(abs(x[2]) - 1) < 1e-6 for x in row["K:vb_spins"])                       # VB spins along z


def test_mos2_in_plane_field_leaves_the_valence_band():
    from skfitter.model import bands as B
    s = B.field_sweep(mos2(), [0, 5, 10], direction="in-plane", valleys=VALLEYS)
    sl = s["slopes"]
    assert abs(sl["K:vb"]["g"]) < 0.02 and abs(sl["K':vb"]["g"]) < 0.02
    assert abs(sl["K:cb"]["g"] + 1.0) < 0.02
    rot = B.direction_sweep(mos2(), 10.0, angles=[0, 45, 90], valleys=VALLEYS)
    splitting = [r["K:vb"] - r["K':vb"] for r in rot["rows"]]
    assert abs(splitting[0]) > 1e-3 and abs(splitting[2]) < 1e-5
    assert abs(splitting[1] - splitting[0] * np.cos(np.radians(45))) < 2e-5   # valley splitting ~ B cos(theta)


def test_plane_axes_and_csv(tmp_path):
    from skfitter.model import bands as B
    m = mos2()
    ax = B.plane_axes(m)
    assert ax["two_d"] and np.allclose(ax["normal"], [0, 0, 1]) and np.allclose(ax["in_plane_1"], [1, 0, 0])
    assert np.allclose(B.field_direction(m, theta=90, phi=90), ax["in_plane_2"])
    s = B.field_sweep(m, [0, 1], valleys=VALLEYS)
    p = B.write_csv(s["rows"], str(tmp_path / "s.csv"))
    head = open(p, encoding="utf-8").readline()
    assert head.startswith("vbm,") or "B" in head
    bs = B.bands(m, [("Γ", [0, 0, 0]), ("K", VALLEYS["K"]), ("M", [0.5, 0, 0]), ("Γ", [0, 0, 0])],
                 window=(-1.0, 1.0))
    assert bs["energies"].shape[1] == len(bs["band_index"]) == 4 and bs["spin"].shape[2] == 3
    assert [t[1] for t in bs["ticks"]] == ["Γ", "K", "M", "Γ"]


# == scissor =====================================================================================================
@pytest.mark.parametrize("soc", [False, True])
def test_scissor_shifts_the_gap(soc):
    from skfitter.model import apply, check_model, bands as B
    m0 = mos2(soc=soc)
    for delta in (0.4, -0.3):
        m = apply(m0, [{"kind": "scissor", "delta": delta}])
        assert check_model(m) == []
        step = m.provenance[-1]
        assert step["step"] == "scissor" and step["max_error_meV"] < 1.0 and not m.warnings
        nv = 2 if soc else 1
        for kf in np.random.default_rng(5).random((4, 3)) * [1, 1, 0]:
            k = kf @ B.reciprocal(m0)
            e0, e1 = B.eigen(m0, k), B.eigen(m, k)
            assert np.abs(e1[:nv] - e0[:nv]).max() < 1e-3 and np.abs(e1[nv:] - e0[nv:] - delta).max() < 1e-3
        assert abs(B.band_edges(m)["gap"] - B.band_edges(m0)["gap"] - delta) < 1e-6   # exact on the mesh points


@science
def test_scissor_non_orthogonal_and_metal():
    from skfitter.model import SKModel, apply, sublattice_mass, bands as B
    d = SKModel.load(GRAPHENE).to_dict()
    d["settings"]["non_orthogonal"] = True
    d["params"].update({"s_pzpzp_0": 0.1, "s_pzpzs_0": 0.0, "eta_s_pzpzp": 2.5, "eta_s_pzpzs": 2.5})
    base = apply(SKModel.from_dict(d), sublattice_mass(0.4, [0], [1]))
    g0 = B.band_edges(base)["gap"]
    m = apply(base, [{"kind": "scissor", "delta": 0.25, "mesh": [24, 24, 1]}])
    assert m.non_orthogonal and abs(B.band_edges(m)["gap"] - g0 - 0.25) < 1e-6
    with pytest.raises(ValueError, match="gap"):
        apply(graphene(), [{"kind": "scissor", "delta": 0.2}])


def semimetal_chain(gap=1.6, t=0.5):
    """Two decoupled bands along x: E1 = 2t cos k and E2 = gap + 2t cos(k - pi/2) (hopping t e^{i pi/2}): a direct gap
    of at least gap - 2 sqrt(2) t at every k, but overlapping across the zone (VBM 2t above CBM gap - 2t) for
    gap < 4t - a DFT semimetal."""
    from skfitter.model.model import TBModel
    lat = np.diag([3.0, 20.0, 20.0])
    return TBModel(lattice=lat, species=["X"], frac_coords=[[0, 0, 0]], site_orbitals=[["s", "p"]],
                   onsite=[np.diag([0.0, gap]).astype(complex)],
                   hoppings={(0, 0, (1, 0, 0)): np.diag([t, t * 1j]).astype(complex)})


def test_scissor_opens_a_semimetal_with_n_valence():
    from skfitter.model import apply, bands as B
    from skfitter.model.modifiers import default_mesh
    m0 = semimetal_chain()
    assert B.band_edges(m0, n_valence=1)["overlap"]
    with pytest.raises(ValueError, match="n_valence"):                 # E_F = 0 cuts the overlapping bands
        apply(m0, [{"kind": "scissor", "delta": 0.4}])
    m = apply(m0, [{"kind": "scissor", "delta": 0.4, "n_valence": 1}])
    step = m.provenance[-1]
    assert step["midgap"] is None and step["direct_gap_min_meV"] > 150.0
    for kx in np.linspace(-0.5, 0.5, 7):
        k = np.array([kx, 0, 0]) @ B.reciprocal(m0)
        assert np.allclose(B.eigen(m, k), B.eigen(m0, k) + [0.0, 0.4], atol=1e-9)
    with pytest.raises(ValueError, match="touch"):
        apply(semimetal_chain(gap=1.0), [{"kind": "scissor", "delta": 0.4, "n_valence": 1}])


def test_scissor_default_mesh_holds_h():
    """The automatic mesh (12 per direction in 3D) holds 1/2 and 1/3 - H (1/3, 1/3, 1/2) of a hexagonal cell - so the
    scissor is exact there."""
    from skfitter.model.modifiers import default_mesh
    from skfitter.model.model import TBModel
    m = TBModel(lattice=np.eye(3) * 3, species=["X"], frac_coords=[[0, 0, 0]], site_orbitals=[["s"]],
                onsite=[np.zeros((1, 1), complex)],
                hoppings={(0, 0, tuple(np.eye(3, dtype=int)[d])): np.ones((1, 1), complex) for d in range(3)})
    mesh = default_mesh(m)
    assert mesh == [12, 12, 12] and all((n * f) % 1 == 0 for n in mesh for f in (1 / 2, 1 / 3, 1 / 4, 1 / 6))


# == files, supercells, NEGF =====================================================================================
@science
def test_files_and_supercells(tmp_path):
    from skfitter.model import SKModel, apply, check_model, load_model, read_json, sublattice_mass
    g = graphene()
    m = apply(g, sublattice_mass(0.2, [0], [1]) + [{"kind": "zeeman", "B": [1, 0, 5]}])
    p = m.save(str(tmp_path / "g_mod_model.json"))
    raw = read_json(p)
    assert raw["format"] == "prayog-modified-model" and raw["base"]["format"] == "prayog-sk-model"
    back = load_model(p)
    assert isinstance(back.base, SKModel) and [x["kind"] for x in back.modifiers] == [
        "onsite_shift", "onsite_shift", "spinful", "zeeman"]
    for k in np.random.default_rng(2).normal(size=(3, 3)):
        assert np.allclose(back.bloch(k), m.bloch(k), atol=1e-12)
    more = apply(back, [{"kind": "onsite_shift", "shift": 0.05, "element": "C"}])
    assert more.base is back.base and len(more.modifiers) == 5
    sc = m.supercell([2, 1, 1])
    assert check_model(sc) == []
    from skfitter.model import bands as B
    k = np.array([0.13, 0.31, 0.0]) @ B.reciprocal(m)
    b1 = B.reciprocal(m)[0]
    folded = np.sort(np.concatenate([B.eigen(m, k), B.eigen(m, k + b1 / 2)]))
    assert np.allclose(B.eigen(sc, k), folded, atol=1e-10)
    with pytest.raises(ValueError, match="selects no orbital"):
        apply(g, [{"kind": "onsite_shift", "shift": 0.1, "element": "Si"}])
    with pytest.raises(ValueError, match="unknown field"):
        apply(g, [{"kind": "zeeman", "Bz": 1}])
    assert json.loads(json.dumps(m.to_dict()))["modifiers"][-1]["B"] == [1.0, 0.0, 5.0]




@science
@pytest.mark.parametrize("example_id", ["graphene-soc", "platinum-soc-qe", "iron-soc-qe", "cerium-f"])
def test_zeeman_on_skfitter_models(example_id):
    from test_tb_model import _sk_model
    from skfitter.model import apply, check_model, model_from_dict
    sk = _sk_model(example_id)
    m = apply(sk, [{"kind": "zeeman", "B": [3.0, -2.0, 5.0]}])
    assert check_model(m) == [] and not m.warnings
    back = model_from_dict(json.loads(json.dumps(m.to_dict())))
    k = np.array([0.1, 0.2, 0.3])
    assert np.allclose(back.bloch(k), m.bloch(k), atol=1e-12)
    # the field adds mu_B (L + 2 S).B on each site: Hermitian, and it changes H only on the sites
    dH = m.onsite(0) - (sk.onsite(0) if sk.spinful else np.kron(np.eye(2), sk.onsite(0)))
    assert np.allclose(dH, dH.conj().T) and np.abs(dH).max() > 0




def test_command_line(tmp_path):
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=os.path.join(_paths.ROOT, "src"), PYTHONIOENCODING="utf-8")
    out = tmp_path / "b.csv"
    r = subprocess.run([sys.executable, "-m", "skfitter.model", "field", "textbook:mos2", "--B", "0,5,10", "--csv",
                        str(out)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "K':gap" in r.stdout and "g =" in r.stdout and out.exists()
    r = subprocess.run([sys.executable, "-m", "skfitter.model", "angle", "textbook:mos2", "--B", "10", "--angles",
                        "0,90"], capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    assert r.returncode == 0 and "theta" in r.stdout, r.stderr


def test_exchange_and_rashba():
    from skfitter.model import apply, bands as B, check_model, load_model, textbook
    g = textbook.graphene()
    rec = B.reciprocal(g)
    K = np.array([2 / 3, 1 / 3, 0]) @ rec
    k = np.array([0.11, 0.27, 0.0]) @ rec
    for n in ([0, 0, 1], [1, 0, 0], [1, 1, 1]):
        m = apply(g, [{"kind": "exchange", "energy": 0.1, "direction": n}])
        assert m.spinful and [x["kind"] for x in m.modifiers] == ["spinful", "exchange"]
        e0 = B.eigen(g, k)
        assert np.allclose(B.eigen(m, k), np.sort(np.concatenate([e0 - 0.1, e0 + 0.1])), atol=1e-12)
        assert m.conserves_sz == (n == [0, 0, 1])
    for lam in (0.03, 0.1):
        m = apply(g, [{"kind": "rashba", "lam": lam}])
        assert np.allclose(B.eigen(m, K), [-3 * lam, 0, 0, 3 * lam], atol=1e-10) and not m.conserves_sz
        H = m.bloch(k)
        assert np.abs(H - H.conj().T).max() < 1e-14
    m = apply(g, [{"kind": "rashba", "lam": 0.1}, {"kind": "exchange", "energy": 0.2}])
    assert m.provenance[-2]["nn_distance"] == pytest.approx(2.46 / np.sqrt(3), rel=1e-6)
    assert check_model(m) == []
    sc = m.supercell([2, 1, 1])                          # the term is on bonds, so it folds
    b1 = rec[0]
    assert np.allclose(B.eigen(sc, k), np.sort(np.concatenate([B.eigen(m, k), B.eigen(m, k + b1 / 2)])), atol=1e-10)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        back = load_model(m.save(os.path.join(tmp, "g_qah_model.json")))
    assert np.allclose(back.bloch(k), m.bloch(k), atol=1e-12)
    with pytest.raises(ValueError, match="no bond between"):
        apply(g, [{"kind": "rashba", "lam": 0.1, "orbitals": ["s"]}])
    with pytest.raises(ValueError, match="selects no orbital"):
        apply(g, [{"kind": "exchange", "energy": 0.1, "element": "Si"}])
    with pytest.raises(ValueError, match="direction"):
        apply(g, [{"kind": "exchange", "energy": 0.1, "direction": [0, 0, 0]}])
