"""Orbital character of the fitted bands at one k-point (set-up step, 'Orbital character' tab)."""
import numpy as np
import pytest

import _paths  # noqa: F401  (puts the packages on the path)
from skfitter import workflow as J


def _dft(magnetic=False):
    """A two-site (A, B) model of a DFT path: 5 k-points along kx, 4 bands per spin, s p d projections (3 columns)."""
    nk, nb = 5, 4
    lat = np.eye(3) * 2.0
    frac = np.array([[0.1 * i, 0, 0] for i in range(nk)])
    kcart = frac @ J.reciprocal_matrix(lat)
    E_up = np.tile(np.array([-2.0, -1.0, 0.5, 1.5]), (nk, 1)) + 0.01 * np.arange(nk)[:, None]
    P = np.zeros((nk, nb, 2, 3), np.float32)
    P[:, 0, 0, 0] = 1.0                       # band 0: A s
    P[:, 1, 1, 1] = 0.7; P[:, 1, 0, 2] = 0.2  # band 1: B p + A d (0.1 unassigned)
    P[:, 2, 0, 2] = 0.99                      # band 2: A d
    P[:, 3, 1, 0] = 0.5; P[:, 3, 1, 1] = 0.5  # band 3: B s + B p
    meta = {"lattice": lat.tolist(), "sites": [{"symbol": "A"}, {"symbol": "B"}], "elements": ["A", "B"],
            "family_cols": {"s": [0], "p": [1], "d": [2]}, "has_projection": True, "magnetic": magnetic,
            "nelect": 4.0, "soc": False}
    dft = {"meta": meta, "kcart": kcart, "E": {"up": E_up, "down": None}, "P": {"up": P},
           "x": np.linspace(0, 1, nk), "ticks": np.array([0.0, 1.0]), "tick_labels": ["G", "X"]}
    if magnetic:
        dft["E"]["down"] = E_up + 0.25
        dft["P"]["down"] = P.copy()
    return dft


def test_points_and_nearest():
    dft = _dft()
    pts = J.kpath_points(dft)
    assert [(p["label"], p["index"]) for p in pts] == [("G", 0), ("X", 4)]
    assert np.allclose(pts[1]["frac"], [0.4, 0, 0])
    j, d = J.nearest_kpoint(dft, [0.21, 0.0, 0.0])
    assert j == 2 and d == pytest.approx(0.01 * 2 * np.pi / 2.0, rel=1e-6)
    j, _ = J.nearest_kpoint(dft, [1.3, 1.0, -1.0])            # equivalent to (0.3, 0, 0)
    assert j == 3


def test_character_by_element_and_family():
    dft = _dft()
    r = J.orbital_character(dft, [0, 1, 2, 3], 2)
    assert r["channels"][-1] == "unassigned"
    w = {row["band"]: row["weights"] for row in r["rows"]}
    assert w[0]["A s"] == pytest.approx(1.0) and w[0]["unassigned"] == pytest.approx(0.0)
    assert w[1]["B p"] == pytest.approx(0.7) and w[1]["A d"] == pytest.approx(0.2)
    assert w[1]["unassigned"] == pytest.approx(0.1, abs=1e-4)
    assert w[3]["B s"] == pytest.approx(0.5) and w[3]["B p"] == pytest.approx(0.5)
    assert [row["E"] for row in r["rows"]] == pytest.approx([-1.98, -0.98, 0.52, 1.52])


def test_merged_spin_channels():
    """A collinear magnet: the fitted bands are spin up and down merged and sorted, each with its spin's projections."""
    dft = _dft(magnetic=True)
    r = J.orbital_character(dft, list(range(8)), 0)
    assert [row["spin"] for row in r["rows"]] == ["up", "down", "up", "down", "up", "down", "up", "down"]
    assert [row["E"] for row in r["rows"]] == pytest.approx([-2.0, -1.75, -1.0, -0.75, 0.5, 0.75, 1.5, 1.75])


def test_no_projections():
    dft = _dft()
    dft["P"] = {}
    dft["meta"]["has_projection"] = False
    with pytest.raises(ValueError):
        J.orbital_character(dft, [0], 0)
