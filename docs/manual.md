# SKFitter manual

## Commands

```text
skfitter info       CONFIG [--params]                 DFT data, crystal, neighbour shells, model size
skfitter character  CONFIG [--k K] [--bands fit|i-j]  orbital character of DFT bands at one k-point
skfitter character  CONFIG --suggest [--cutoff W]     orbital families the model lacks, over all k-points
skfitter check      CONFIG [--tolerance MEV]          the published parameters (start_from) on the DFT bands, no fit;
                                                      compares with reference/<name>_report.json (exit 1 if they differ)
skfitter fit        CONFIG [--workers N]              fit the model
skfitter bands      CONFIG                            bands of the saved fit (no refit)
skfitter berry      CONFIG [--only TYPE ...] [--index N ...] [--workers N]  the [[property]] calculations on the saved fit
skfitter run        CONFIG                            fit, then every [[property]]
skfitter params                                       the bundled parameter sets
```

`-v` prints the fitter's log. `python -m skfitter` is the same command. The exit status is 0 on success, 2 for an
error in the configuration and 1 for a failed calculation.

Ctrl+C during `skfitter fit` stops the fit at the next band evaluation (with `--workers`, after the current
Jacobian) and saves the best parameters found so far, with the bands and the report as usual; the report is marked
`"stopped": true`. A second Ctrl+C aborts at once without saving.

`--k` takes a label of the DFT path (`Γ` or `G`, `K`, `M`, ...) or three fractional coordinates (`"0.5 0 0"`); the
orbital projections exist on the DFT path only, so the nearest path point is used. `--bands fit` (default) shows the
bands of the fitting window.

`--suggest` ranks the orbital families (per element: s, p, d, f) that the model does not have, from the DFT
projections of the fitted bands at every k-point of the path (degenerate states averaged; each state weighted
1 + (fermi_weight − 1) exp(−(E/fermi_window)²) as in the fit). For each family it prints

* **absolute**: its largest weight in any state of the window;
* **weighted**: the largest, over the bands of the window, of its weighted average over k;
* **average** and **missing**: its weighted average over all states, and its share of the weight that the model's
  own orbitals cannot carry (what adding it would explain);
* **centre**: its projection-weighted energy (from E_F; deep semicore states left out), a starting on-site energy;
* **electrons**, **filled**: its occupied weight per cell and that as a share of the family's capacity 2(2l+1) per
  site. A nearly full family (such as Bi 6s) adds 2(2l+1) electrons per site to `[model] electrons`, a nearly empty
  one none;
* **+params**: the parameters the model gains with it;
* **symmetry groups (site)**: the finest split of a p, d or f shell into SKFitter's orbital groups that the crystal
  symmetry allows on all sites of the element (in the structure's Cartesian axes), the number of on-site terms the
  site symmetry still allows inside the shell (`+n on-site`: a crystal field within a group or a mixing between
  groups, added with `onsite_mixing`), and the site point group. For Mn3Ir (L1₂): Mn d → `d_t2g d_eg +2 on-site
  (4/mmm)`; the t2g/eg groups and the two on-site terms together give the tetragonal (D4h) splitting about each
  Mn site's own axis. Ir d → `d_t2g d_eg (m-3m)`.

A family is suggested by a method when its value reaches `--cutoff` (0.1). The command prints the
`orbitals = { ... }` line of the model with each method's suggestions added, to paste into `[model]`, and writes
`<name>_orbital_suggestions.json`. It also prints the model's orbitals with every whole shell split as the
symmetry allows (and the `onsite_mixing` line when on-site terms remain). Nothing in the configuration is changed. Projections never add up to 1: the
weight on no atomic orbital ("projection spilling") is printed too. Part of it is plane-wave character no atomic
orbital carries, but part can belong to a valence shell the pseudopotential has no wavefunction for, so projwfc.x
cannot project on it (Mn 4p and Ir 6p in Mn3Ir). The command lists such shells (`valence shells without a
projection`: p at the outermost s shell, d one shell lower for transition metals and lanthanides / actinides, f two
lower for lanthanides / actinides) and, when the unassigned weight is at least 3% per state and at least half of
what the model cannot carry, prints a note that names them as candidates (Mn3Ir with s and d: 10% per state, "try
adding Ir p, Mn p"). Their share cannot be ranked from the projections; a fit with them added is the test. The
shells are read from the projection file's state labels (QE); VASP PROCAR files carry no such labels.

## Configuration file

A TOML file. Paths are relative to the configuration file.

### Top level

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | file name | stem of every output file |
| `output` | `"out"` | output folder; it also holds the copies of the DFT inputs and a cache (`dft_cache.npz`) |

### `[dft]`

| Key | Meaning |
| --- | --- |
| `code` | `"espresso"` (default) or `"vasp"` |
| `xml` | QE: the `pw.x` XML file of the non-self-consistent bands run (required) |
| `bands_in` | QE: the input of the bands run (k-path and labels) |
| `scf` | QE: the SCF output (Fermi level, magnetic moments) |
| `projwfc` | QE: the `projwfc.x` `filproj` file (orbital projections); the spin-down file `..._down` next to an `..._up` file is read too |
| `vasprun`, `kpoints` | VASP: `vasprun.xml` (with `LORBIT = 11` for projections) and the line-mode `KPOINTS` |

Files may be gzipped (`.gz`).

### `[model]`

| Key | Default | Meaning |
| --- | --- | --- |
| `orbitals` | required | orbitals per element, e.g. `{ Fe = ["s", "p", "d_t2g", "d_eg"] }` |
| `magnetic_orbitals` | `{}` | orbital groups with an exchange splitting Δ, e.g. `{ Fe = ["d_t2g", "d_eg"] }` |
| `bands_below`, `bands_above` | 4, 4 | DFT bands fitted below / above the Fermi level (per spin with SOC) |
| `neighbours` | 3 | number of neighbour shells of the cell (sets the bond cutoff: shell N + 0.1 Å) |
| `hopping` | `"decay"` | `"decay"`: V(r) = V₀ (d₀/r)^η per element pair (d₀ = the pair's nearest bond); `"shell"`: an independent integral per shell of each element pair |
| `soc` | `false` | on-site spin–orbit coupling λ per element and orbital family |
| `soc_orbitals` | all | the p, d, f families that carry λ when `soc = true`, e.g. `{ Bi = ["p"], Eu = ["d"] }` (Eu 4f left out, as for a scalar-relativistic pseudopotential); elements not listed get none |
| `overlap` | `false` | non-orthogonal model with fitted overlap integrals |
| `electrons` | from DFT | number of electrons in the model (sets the Fermi level of the model) |
| `fit_eta` | `true` | fit the decay exponents η (decay law; for shells, the scaling of bonds off their shell distance) |
| `eta` | 2.5 | fixed η when `fit_eta = false` |
| `symprec` | 0.001 | symmetry tolerance (Å) for space group, equivalent sites and orbital groups |
| `model_states_only` | `false` | fit only the DFT states with at least half their weight on the model's orbitals (needs the projections; non-magnetic, non-collinear and collinear magnetic data); the window is rebuilt from those states at every k-point |
| `scissor` | 0 | shift of the DFT conduction bands (eV) before the fit |
| `onsite_mixing` | none | `true` or a list of elements: on-site couplings between groups allowed by the site symmetry |

Orbital names: `s`, `s*`, `p`, `px`/`py`/`pz`, `pxy`, `d`, `d_t2g`/`d_eg`, `dz2`/`d_e1`/`d_e2`, `f`, `f_a2u`/`f_t1u`/`f_t2u`
and the other f groups. A split must be allowed by the site symmetry; SKFitter checks it and says which groups mix.

#### Parameter names

Every two-centre integral belongs to one pair of (element, orbital group) and is named

    v_<El1>_<orbital1>-<El2>_<orbital2>_<integral>_<shell>

for example `v_Mn_d-Mn_d_dds_0`, `v_Ir_d-Mn_d_ddp_1`, `v_Ir_s-Mn_d_sds_0` or, between symmetry groups,
`v_Bi_pz-Mn_dz2_pds_0` (group names without underscores: `d_t2g` → `dt2g`). The pair is written with the lower
l first, then in element order, then by group; `<integral>` is the two families and σ, π, δ, φ (`sss`, `sps`,
`pds`, `ddd`, `ss*s`, ...). An s–p integral therefore depends on which element carries the s orbital
(`v_Ir_s-Mn_p_sps` and `v_Mn_s-Ir_p_sps` are different integrals). The suffix counts the neighbour shells of that
element pair: `_0` is the pair's nearest bond, `_1` its next distance, and so on (`skfitter info` lists the shells
of every pair). With `hopping = "decay"`, `_0` is V₀ at the pair's nearest bond d₀ and
`eta_<El1>_<orbital1>-<El2>_<orbital2>_<integral>` its exponent. Overlaps are `s_...` and `eta_s_...`. On-site
energies are `e_orbit<n>_<El>_<orbital>` (n = the equivalence class of the site), spin–orbit couplings
`lambda_<El>_<family>` and exchange splittings `delta_spin_<El>_<orbital>`.

Fits saved before version 1.0.0rc5 used one integral per orbital pair for all element pairs (`v_dds_0`,
`v_pz.dz2.s_1`, shells counted over the whole cell). `start_from`, `skfitter check` and model files read them and
rename the integrals for the element pairs with the same bands.

### `[fit]`

| Key | Default | Meaning |
| --- | --- | --- |
| `method` | `"ls"` | `"ls"` bounded least squares (multi-start), `"de"` differential evolution |
| `starts` | 8 | random starts of the least-squares fit |
| `workers` | 1 | threads for the finite-difference Jacobian of each least-squares step (`skfitter fit --workers N`) |
| `tolerance` | 1e-10 | relative change of the loss (and of the parameters) at which a least-squares start stops; a looser value such as 1e-6 ends long fits much earlier - compare its result with the default before relying on it |
| `overlap_floor` | 0.05 | non-orthogonal models: the smallest eigenvalue the overlap matrix S(k) may take at the fitted k-points; below it a penalty is added. A nearly singular S sends states to very high energies, which a fit can exploit to drop unwanted states from the fitted window at a few k-points (an unphysical model; `skfitter berry` then fails with 'not positive definite'). The fit reports the smallest eigenvalue on the fitted k-points and on random k-points of the zone; 0 switches the penalty off |
| `seed` | 0 | random seed |
| `refine_max_nfev` | 300 | evaluations of the final refinement |
| `maxiter`, `popsize` | 200, 15 | differential evolution |
| `k_stride` | 1 | use every n-th k-point |
| `fermi_weight`, `fermi_window` | 1, 0.5 | weight w_F of bands near E_F: w = 1 + (w_F − 1) exp(−(E/ε_F)²) |
| `symmetry_weight`, `symmetry_width` | 1, 2 | weight w_S at the high-symmetry points (and that many neighbours) |
| `start_set` | `"harrison"` | starting values and bounds: a bundled set (`skfitter params`) or `"default"` |
| `start_from` | none | a `*_fit.json` / `*_fit.txt` of an earlier fit (a warm start; a decay-law fit can start a shell model, a whole-shell fit a split model). Its values take precedence over `[fit.initial]`, which then only starts the parameters the earlier fit lacks (a note lists any it ignores); `[fit.bounds]` are widened to include the start values; `[fit.fix]` still fixes |
| `[fit.initial]` | | starting values, `key = value` |
| `[fit.bounds]` | | bounds, `key = [low, high]` |
| `[fit.fix]` | | fixed parameters, `key = value` |

`skfitter info CONFIG --params` lists the parameter names with their starting values and bounds.

The default bounds of the exchange splittings `delta_spin_*` are ±5 eV, enough for Fe (Δ ≈ 2–3 eV) but not for
large moments: set them in `[fit.bounds]`, as the MnTe and EuMn₂Bi₂ examples do (Mn 3d 3–12 eV, Eu 4f 8–12 eV).
A fit warns about every parameter that ends at a bound.

The loss is L = (1/N) Σ w_nk (E^TB_nk − E^DFT_nk)², with the weights above.

### `[[property]]`

One table per calculation; `type` selects it. The keys and defaults:

| `type` | Keys (defaults) |
| --- | --- |
| `berry_path` | `component` (`"z"`), `dk` (0.01 Å⁻¹) |
| `berry_map` | `component`, `nk` (60), `k3` (0, fractional third coordinate) |
| `wilson` | `wn` (120 points per loop), `wnr` (30 radii), `wr` (0: automatic radius), `k3` |
| `chern` | `nk` (60), `k3` |
| `wcc` | `fixed` (`"k3"`), `fixed_value` (0), `loop` (`"k2"`), `n_loop` (60), `n_scan` (41), `full` (true), `n_occupied` (0: the bands below E_F) |
| `spin_berry_path` | `spin`, `alpha`, `beta` (`"z"`, `"x"`, `"y"`), `dk`, `efermi` (0), `smearing` (0), `units` (`"au"`) |
| `ahc` | `component`, `nk` ([40, 40, 40]), `emin`, `emax` (−1, 1 eV), `num_efermi` (201), `smearing` (0.01 eV), `refine` (true), `refine_window` (0: all bands), `bands` (`"all"` or `"fitted"`) |
| `shc` | `spin`, `alpha`, `beta` (`"z"`, `"x"`, `"y"`; or Cartesian vectors such as `[1, 1, 1]`), `nk`, `emin`, `emax`, `num_efermi`, `smearing`, `refine`, `refine_window` |
| `shc_temperature` | `spin`, `alpha`, `beta`, `nk` ([30, 30, 30]), `temperatures` ([0, 100, 200, 300] K), `efermi`, `n_shifts` (4), `refine` |

The conductivities sum over a uniform k-mesh and then refine, up to three times, the cells where a band's Berry
curvature is large (near band crossings), each into 4³ smaller cells. In a metal with many bands near each other
(magnetic, with SOC) most cells can qualify and the refinement then dominates the time; `refine_window = W` (eV)
lets only the bands within W of the scanned Fermi energies decide (a pair of occupied bands cancels in the sum),
which in Fe changes the AHC by about 2 % at a fraction of the cost. `skfitter berry --workers N` evaluates the
k-points on N threads (set `OMP_NUM_THREADS=1`), and prints the progress of the uniform mesh and of each
refinement level with the time left.

The spin Hall conductivity σ^spin_αβ is the spin current along `alpha`, polarised along `spin`, for a field along
`beta`. Each may be a Cartesian vector (normalised internally), which gives any component in a rotated frame; for
example spin = [1, 1, 1], alpha = [1, −1, 0], beta = [1, 1, −2] is σ^z_xy in a frame with z normal to a (111) plane.
The output files are then named `..._shc_s111_a1-10_b11-2`.

Wilson loops and Chern numbers are for two-dimensional hexagonal zones (valleys K and K′). Conductivities are in
S/cm ((ħ/e) S/cm for the spin Hall conductivity).

`wcc` gives the hybrid Wannier charge centres of the occupied bands on a plane of the zone: the reduced coordinate
`fixed` is held at `fixed_value`, Wilson loops run along `loop`, and the third coordinate is scanned from 0 to 1
(`full = true`) or to 1/2. The centres are fractions of the real lattice vector along `loop`. From them come the
Chern number of the plane (winding of the summed centres, full scan) and its Z2 index (Soluyanov-Vanderbilt
largest-gap method over half the plane, cross-checked with fixed reference lines). Z2 needs time-reversal symmetry;
for a 3D insulator the strong index is (Z2(k3 = 0) + Z2(k3 = 1/2)) mod 2, from two `wcc` entries. The occupied bands
must be separated from the rest on the whole plane (a metal needs `n_occupied`).

## Output files

See the README. The `.dat` files are plain text with a header line naming the columns.
