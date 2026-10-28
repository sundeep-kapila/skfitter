# SKFitter

Symmetry-resolved Slater–Koster tight-binding models fitted to DFT band structures, and their Berry-phase
properties.

SKFitter reads the bands, the k-path, the Fermi level, the magnetic moments and the orbital projections of a
Quantum ESPRESSO or VASP run, builds a Slater–Koster model for the orbitals you choose (s, p, d and f, split into
the groups the site symmetry allows), and fits its parameters to the DFT bands. The fitted model gives band
structures, Berry curvature, anomalous and spin Hall conductivities, Wilson loops, hybrid Wannier centres, Chern
numbers and Z2 indices.

* **Orbitals:** s, p, d, f, and symmetry-resolved groups (p_z / p_xy; d t2g / eg or A1 / E1 / E2; f a2u / t1u /
  t2u). Two-centre integrals for every pair of groups, checked against the crystal's symmetry, with integrals that
  vanish by symmetry left out.
* **Spin:** on-site spin–orbit coupling for every orbital family. Exchange splitting per orbital group along each
  site's moment direction, read from DFT, for collinear and non-collinear magnets (ferromagnets,
  antiferromagnets, altermagnets).
* **Distance dependence:** integrals per element pair (`v_Ir_d-Mn_d_dds_0`), with a decay law V(r) = V₀ (d₀/r)^η
  from each pair's nearest bond, or independent integrals per neighbour shell of each pair.
* **Fitting:** bounded least squares (or differential evolution) with Fermi-level and high-symmetry-point weights,
  warm starts from an earlier fit, starting values from bundled parameter sets (Harrison, NIST, DFTB), and
  selection of the DFT states that the model's orbitals describe.
* **Properties:** Berry curvature (path and map), spin Berry curvature, anomalous Hall conductivity, spin Hall
  conductivity (also against temperature), Wilson loops and Chern / valley Chern numbers, hybrid Wannier charge
  centres with the Chern and Z2 invariants of a plane.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/sundeep-kapila/skfitter.git
cd skfitter
pip install .            # or: pip install -e ".[test]" for development
```

This installs the `skfitter` command and the Python package `skfitter`. Dependencies: numpy, scipy, matplotlib,
pymatgen (and tomli on Python 3.10).

## Quick start

Every example folder holds the DFT data of one material and a configuration `skfitter.toml`:

```bash
cd examples/graphene
skfitter info skfitter.toml                 # the DFT data, the crystal and the model
skfitter check skfitter.toml                # the published parameters on the DFT bands: the paper's errors
skfitter character skfitter.toml --k K      # orbital character of the fitted DFT bands at K
skfitter character skfitter.toml --suggest  # orbitals the model lacks, ranked over all k-points
skfitter fit skfitter.toml                  # fit; results in out/
skfitter bands skfitter.toml                # bands of the saved fit, no refit
skfitter berry skfitter.toml                # the [[property]] calculations of the configuration
```

A configuration names the DFT files, the model and the fit:

```toml
name = "mos2"
output = "out"

[dft]
code = "espresso"                 # or "vasp" (vasprun = "...", kpoints = "...")
xml = "data/qe_bands.xml.gz"      # pw.x XML of the bands run (gzip is read directly)
bands_in = "data/qe_bands.in"     # the bands input (k-path labels)
scf = "data/qe_scf.out"           # SCF output (Fermi level, moments)
projwfc = "data/qe_projwfc_up.gz" # projwfc.x filproj (orbital projections), optional

[model]
orbitals = { Mo = ["dz2", "d_e1", "d_e2"], S = ["pz", "pxy"] }
bands_below = 4
bands_above = 2
neighbours = 9
hopping = "shell"
soc = true

[fit]
start_set = "harrison"
fermi_weight = 60
fermi_window = 0.7

[[property]]
type = "shc"
nk = [40, 40, 10]
```

The full list of keys is in [docs/manual.md](docs/manual.md).

## Outputs

`skfitter fit` writes into the output folder:

| File | Content |
| --- | --- |
| `<name>_fit.json`, `<name>_fit.txt` | fitted parameters, RMSE and model settings |
| `<name>_bands.dat`, `<name>_bands.png` | DFT and tight-binding bands along the path |
| `<name>_report.json` | RMSE (all bands, near E_F, at high-symmetry points), parameter count, wall time |
| `<name>_settings.json` | the configuration, starting values and bounds of the fit |

`skfitter berry` writes one `.dat` and one `.png` per property; `skfitter character` writes a `.json` table and a
bar chart.

## Examples and test runs

`examples/` holds the eight benchmark systems of the SKFitter paper (graphene, MoS₂, Pt, Fe, γ-Ce, CeO₂, MnTe,
EuMn₂Bi₂) with Quantum ESPRESSO data. A quick test run (a few minutes in all) evaluates the published parameters of
every system and compares the errors with the paper:

```bash
python paper/run_examples.py --only check
```

Each configuration also fits from the published parameters (`skfitter fit`); `examples/README.md` lists the expected
errors and running times (from seconds for graphene to hours for EuMn₂Bi₂). `python paper/run_examples.py` runs every
calculation behind the paper (all fits, the decay-law / per-shell models, the EuMn₂Bi₂ build-up and the Berry-phase
properties; several hours on a multi-core PC) and `python paper/make_figures.py` then draws Figs. 2–5 and Table 1.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests use the Quantum ESPRESSO data in `tests/data/examples` and `examples/`.

## Python use

```python
from skfitter import SKFitter, load_model
import skfitter.properties as P
```

`skfitter.workflow` exposes the steps of the command line (reading DFT data, building and fitting a model, property
runs) as functions.

## Citing

If you use SKFitter, please cite the paper (see `CITATION.cff`):

S. Kapila, B. Muralidharan and S. Bhattacharjee, *SKFitter: symmetry-resolved Slater–Koster tight-binding models
from first principles for s, p, d and f electrons, spin–orbit coupling and magnetism* (2026).

The software itself is archived on Zenodo: [doi:10.5281/zenodo.23719846](https://doi.org/10.5281/zenodo.23719846).

The bundled DFTB parameter sets are adapted from DFTB.org under CC BY-SA 4.0; please cite the references in
`src/skfitter/data/param_sets/licences/` when you start fits from them.

## Licence

The SKFitter program is free software under the GNU Affero General Public License v3.0 only (AGPL-3.0-only), see
[LICENSE](LICENSE). For use under other terms, a commercial licence is available from the authors
(sundeep.kapila@swasth.org).

The DFT data and configurations of the examples (`examples/`) and the test data (`tests/data/`) are licensed under
the Creative Commons Attribution 4.0 International licence (CC BY 4.0), see
[examples/LICENSE.md](examples/LICENSE.md). The bundled DFTB parameter sets keep their CC BY-SA 4.0 licence
(`src/skfitter/data/param_sets/`), and the reference curves in `paper/reference_data/` are digitized from the
publications cited there.
