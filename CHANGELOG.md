# Changelog

## 1.0.0rc5 (2026-10-25, review version)

- Hopping and overlap integrals are resolved by element pair: `v_<El1>_<orbital1>-<El2>_<orbital2>_<integral>_<shell>`
  (e.g. `v_Ir_d-Mn_d_dds_0`). Bonds of different element pairs at the same distance (Mn-Mn and Ir-Mn in Mn3Ir,
  Mn-Mn and Te-Te in MnTe) no longer share integrals, and s-p (s-d, p-d, ...) integrals depend on which element
  carries which orbital.
- Neighbour shells are counted per element pair (`_0` = the pair's nearest bond); the decay law refers to each pair's
  nearest bond, with one exponent per pair and integral. `skfitter info` lists the shells of every pair.
- Fits and model files saved with the earlier names (`v_dds_0`, `v_pz.dz2.s_1`) are converted on reading
  (`start_from`, `skfitter check`, model files) and give the same bands.
- `skfitter info`: non-collinear magnetic data are reported as such, with the moment direction of each site.
- `skfitter character --suggest` adds a column with the finest orbital groups the crystal symmetry allows for every
  p, d and f shell, the on-site crystal-field / mixing terms the site symmetry still allows (`onsite_mixing`) and the
  site point group, and prints the model's orbitals split that way (e.g. Mn3Ir: Mn d -> d_t2g, d_eg + 2 on-site
  terms on the 4/mmm sites; Ir d -> d_t2g, d_eg on the m-3m site).
- `skfitter character --suggest` lists the valence shells that have no projection state (no wavefunction in the
  pseudopotential, e.g. Mn 4p and Ir 6p) and, when the weight on no projected orbital is large (at least 3% per
  state and half of what the model cannot carry), notes that the missing character may sit in them and names the
  families to try (Mn3Ir with s, d: 10% per state, "try adding Ir p, Mn p").
- Non-orthogonal fits keep the overlap matrix well conditioned: new `[fit]` key `overlap_floor` (0.05) adds a
  penalty where the smallest eigenvalue of S(k) drops below it. Without it the Mn3Ir fit made S singular at Gamma
  (eigenvalue 4e-9), which pushed four states to ~1e8 eV and out of the fitted window, and `skfitter berry` failed
  on it. The fit report gives the smallest eigenvalue of S(k) (`overlap_min_eig`, on the fitted k-points and on
  random k-points of the zone), and a singular S now gives an error that says what to change. The check on the random k-points is vectorised (Mn3Ir: seconds instead of minutes after the fit).
- The fast-model check at the start of a fit compares the fitted bands only, relative to their size. A nearly
  singular overlap matrix puts a few states at ~1e8 eV, where the two eigensolvers differ by eV; the check then
  failed and the fit fell back to the reference path, some 300 times slower (Mn3Ir non-orthogonal restart). The
  fallback, if it happens, is now also printed on the terminal.
- `skfitter fit`: Ctrl+C stops the fit at the next band evaluation and saves the best parameters found so far (fit
  files, bands and report, marked `"stopped": true`); a second Ctrl+C aborts without saving.
- `start_from` takes precedence over `[fit.initial]`: a warm start keeps every fitted value, and `[fit.initial]` only
  starts the parameters the earlier fit lacks (a note lists the entries it ignores). Before, `[fit.initial]` reset
  them (the Mn3Ir restarts began with four on-site energies back at their initial guesses).
- Spin Hall conductivity in any frame: `spin`, `alpha` and `beta` of a `shc` property take Cartesian vectors as
  well as x / y / z (e.g. spin = [1, 1, 1], alpha = [1, -1, 0], beta = [1, 1, -2] for the frame of a (111) kagome
  plane, as used for Mn3Ir in the literature).
- `skfitter berry --workers N`: the Brillouin-zone sums of the conductivities run on N threads, with a progress line
  for the uniform mesh and for each refinement level (elapsed time, time left). New `[[property]]` key
  `refine_window` (ahc, shc): only bands within that many eV of the scanned Fermi energies trigger the adaptive
  refinement (Mn3Ir 12^3: 55k instead of millions of refined points).
- Examples: the decay-law and per-shell steps use 8 least-squares starts (with 2, MoS2 and MnTe ended in poorer
  minima once the integrals were per element pair).

## 1.0.0rc4 (2026-10-25, review version)

- `[model] soc_orbitals`: choose the orbital families that carry on-site spin-orbit coupling (default: every p, d
  and f family, as before), like `magnetic_orbitals` for the exchange.
- `model_states_only` now works for magnetic data: non-collinear and spin-orbit runs, and collinear magnets with
  spin-up and spin-down projections merged in energy order. `character --suggest` follows it.

## 1.0.0rc3 (2026-10-24, review version)

- `skfitter character CONFIG --suggest [--cutoff W]`: the orbital families the model lacks, ranked over all k-points
  of the fitted bands by their largest weight (absolute), their largest band-averaged weight (weighted) and their
  share of the weight the model cannot carry, with a starting on-site energy, the occupied weight and the parameter
  count of each; prints `orbitals = { ... }` lines to paste into the configuration (nothing is changed automatically).

## 1.0.0rc2 (2026-10-24, review version)

- New property `wcc`: hybrid Wannier charge centres of a plane (Wilson loops along a reciprocal lattice vector), with
  the Chern number of the plane and its Z2 index (Soluyanov-Vanderbilt largest-gap method, cross-checked with fixed
  reference lines). Tested on the Haldane model (C = ±1 / 0), the Kane-Mele model (Z2 = 1 / 0) and the atomic limit.
- `skfitter.properties`: importing one family module directly no longer hides the other calculators.

## 1.0.0rc1 (2026-10-23, review version)

Release candidate for the co-authors' review before submission.

## 1.0.0 (unreleased)

First public release: the `skfitter` command line (info, character, check, fit, bands, berry, run, params), the
`skfitter` Python package (fitting engine, model classes, properties), the eight benchmark examples of the paper
and the scripts that regenerate its figures.
