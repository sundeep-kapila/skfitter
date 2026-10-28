"""Hybrid Wannier charge centres, Chern numbers and Z2 (skfitter.properties, calculator geometric.wcc) on textbook
models with known answers: the Haldane model (Chern ±1 / 0), the Kane-Mele model (Z2 = 1 / 0) and the atomic limit
(the centre sits on the orbital that holds the electron, which checks the gauge closing the loops)."""
import numpy as np
import pytest

from skfitter.model import textbook as T
from skfitter.properties.core.runner import run
from skfitter.properties.geometric.wcc import wilson_centres, z2_reference_lines
from skfitter.properties.services import kspace
from skfitter.properties.services.bloch import Kernel


def centres(model, nocc, full=True, n_scan=31, n_loop=60):
    scan = np.linspace(0.0, 1.0 if full else 0.5, n_scan)
    w, gap = wilson_centres(Kernel(model), kspace.reciprocal(model), "k3", 0.0, "k2", scan, n_loop, nocc)
    return scan, w, gap


@pytest.mark.parametrize("M, chern", [(0.0, 1), (0.4, 1), (3.0, 0)])
def test_haldane_chern(M, chern):
    d = run(T.haldane(M=M), "geometric.wcc", {"n_scan": 31, "n_loop": 60})["data"]
    assert abs(abs(d["chern"]) - chern) < 1e-6
    assert d["gap"] > 0.05


@pytest.mark.parametrize("lam_v, z2", [(0.0, 1), (0.1, 1), (0.5, 0)])
def test_kane_mele_z2(lam_v, z2):
    d = run(T.kane_mele(lam_v=lam_v), "geometric.wcc", {"full": False, "n_scan": 41, "n_loop": 60})["data"]
    assert d["z2"] == z2 and d["z2_converged"]
    assert d["z2_lines"] == [z2, z2, z2]
    assert d["chern"] is None


def test_atomic_limit_centre():
    """A large sublattice potential puts the electron on one site; its centre is that site's coordinate along a2."""
    for M, x in ((20.0, 1.0 / 3.0), (-20.0, 0.0)):
        _, w, _ = centres(T.haldane(t2=0.0, M=M), 1)
        dist = np.abs(np.mod(w[:, 0] - x + 0.5, 1.0) - 0.5)
        assert dist.max() < 0.01, (M, w[:3, 0])


def test_loop_mesh_independence():
    _, w1, _ = centres(T.kane_mele(), 2, full=False, n_scan=11, n_loop=40)
    _, w2, _ = centres(T.kane_mele(), 2, full=False, n_scan=11, n_loop=160)
    d = np.abs(np.mod(w1 - w2 + 0.5, 1.0) - 0.5)
    assert d.max() < 2e-3


def test_reference_lines_on_a_trivial_set():
    w = np.tile([[0.1, 0.1]], (11, 1))
    assert z2_reference_lines(w) == [0, 0, 0]
