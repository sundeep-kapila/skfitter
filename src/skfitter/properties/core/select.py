"""Shared input handling: which bands, how many are occupied, the k-path of a calculation."""
from __future__ import annotations

from .inputs import Bands, KPath, Number


def occupied(model, n_occupied: int, kernel=None) -> int:
    """n_occupied when given (> 0), else the bands below E_F = 0 (ValueError for a metal: give it); counted with
    ``kernel`` (the calculation's, or a new one)."""
    if n_occupied and int(n_occupied) > 0:
        return int(n_occupied)
    from ..services.occupation import n_occupied as count
    try:
        return count(model, kernel)
    except ValueError:
        raise ValueError("The model is a metal at E_F = 0, so 'occupied' is not defined by a gap: give the number "
                         "of occupied bands (n_occupied).") from None


def band_list(sel, D: int, n_occ_fn) -> list:
    """Band indices for a Bands input value ("all", "occupied", "edges", or a list); n_occ_fn() gives the occupied
    count when needed."""
    if sel == "all":
        return list(range(D))
    if sel == "occupied":
        return list(range(n_occ_fn()))
    if sel == "edges":
        n = n_occ_fn()
        return [b for b in (n - 1, n) if 0 <= b < D]
    bad = [b for b in sel if b >= D]
    if bad:
        raise ValueError(f"bands: the model has {D} bands (0 to {D - 1}); {bad} do not exist")
    return list(sel)


def path_points(p_path):
    return None if p_path == "auto" else p_path["points"]


BANDS = Bands("Bands", "all", help="all, occupied, edges (top valence + bottom conduction) or band indices (0 = lowest)")
N_OCC = Number("Occupied bands (0: those below E_F)", 0, integer=True, min=0, advanced=True,
               help="Needed for a metal; the bands below E_F otherwise.")
PATH = KPath("k-path", "auto", help="auto: the standard path of the lattice; or labelled points (fractional k)")
DK = Number("Path spacing", 0.01, unit="1/Å", min=1e-4, max=1.0)
