"""Physical constants and units of the Properties module: one place for every number (CODATA 2018).

Energies in eV (relative to E_F = 0 of the model), lengths in Å, k in 1/Å (including 2π), Berry curvature in Å²,
conductivities in S/cm for 3D cells; for a 2D material the per-volume value uses the cell volume (vacuum included),
and sheet values are also given in units of e²/h.
"""
E2_OVER_HBAR = 2.434134807e-4        # e²/ħ in siemens
E2_OVER_H = 3.874045865e-5           # e²/h in siemens
KB_EV = 8.617333262e-5               # Boltzmann constant, eV/K
HBAR_EVS = 6.582119569e-16           # ħ, eV s
HBAR2_OVER_ME = 7.619964231          # ħ²/m_e in eV Å² (so E = ħ²k²/2m* -> m*/m_e = 7.62 / (d²E/dk²))
MU_B_EV_T = 5.7883818060e-5          # Bohr magneton, eV/T
BOHR_ANGSTROM = 0.529177210903
ANGSTROM_PER_CM = 1e8


def sheet_factor(model) -> float:
    """Height (Å) of the cell along its non-periodic direction (2D material), to turn S/cm into a sheet value."""
    import numpy as np
    from skfitter.model.modifiers import periodic_axes
    per = periodic_axes(model)
    if len(per) != 2:
        return float("nan")
    lat = np.asarray(model.lattice, float)
    n = np.cross(lat[per[0]], lat[per[1]])
    free = [a for a in range(3) if a not in per][0]
    return float(abs(lat[free] @ n) / np.linalg.norm(n))
