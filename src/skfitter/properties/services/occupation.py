"""Occupations: Fermi-Dirac at a temperature, the number of occupied bands, carriers from a density of states."""
from __future__ import annotations

import numpy as np

from .units import KB_EV


def fermi(E, mu=0.0, kT=0.0):
    """f((E - mu)/kT); kT <= 0 gives the step function (1 below mu). Broadcasts."""
    x = np.asarray(E, float) - mu
    if np.all(np.asarray(kT) <= 0):
        return (x < 0).astype(float)
    return 0.5 * (1.0 - np.tanh(0.5 * x / kT))


def kT_of(temperature_K: float) -> float:
    return KB_EV * float(temperature_K)


def n_occupied(model, kernel=None, mesh: int = 12) -> int:
    """Bands below E_F = 0 (the count most k-points agree on; ValueError for a metal): skfitter.model.bands.count_valence's
    rule, with the eigenvalues of the kernel on the whole mesh at once (the SK model's own Bloch sums, one k-point at a
    time, take tens of seconds for a large spinful model). Remembered on the kernel."""
    from skfitter.model.bands import count_valence
    if kernel is None:
        from .bloch import Kernel
        kernel = Kernel(model)
    key = ("n_occupied", int(mesh))
    memo = kernel.__dict__.setdefault("_memo", {})
    if key not in memo:
        try:
            memo[key] = count_valence(model, mesh=mesh, energies=kernel.energies)
        except ValueError as exc:
            memo[key] = exc
    if isinstance(memo[key], Exception):
        raise memo[key]
    return memo[key]


def spin_degeneracy(model) -> int:
    """2 for a spinless model (each band holds two electrons), 1 for a spinful one."""
    return 1 if model.spinful else 2
