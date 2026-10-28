# Examples

The eight benchmark systems of the SKFitter paper, with the Quantum ESPRESSO data of the paper (PBE; settings
in the paper's Table 3). Each folder holds

* `data/`: the `pw.x` XML of the bands run, the bands input, the SCF output and the `projwfc.x` projections;
* `skfitter.toml`: the model of the paper (Table 1), starting from the published parameters in `reference/`;
* `steps/`: the decay-law and per-shell models with whole orbital families, fitted from scratch in a chain
  (and, for EuMn₂Bi₂, the orbital-by-orbital build-up of Fig. 5).

Run one example from its folder:

```bash
cd examples/mos2
skfitter check skfitter.toml        # the published parameters, no fit (seconds)
skfitter fit skfitter.toml
skfitter fit steps/decay.toml && skfitter fit steps/shell.toml
```

or everything behind the paper's figures with `python paper/run_examples.py` and then
`python paper/make_figures.py`.

## Expected results (test run)

RMSE over all fitted bands and within E_F ± 1 eV (meV), free parameters, and the wall time of `skfitter fit`
on one core of a laptop (Intel Core Ultra 7 165H, 64 GB), with up to 12 runs in parallel (numbers within about 1 meV of these are
expected on other machines; the time of a fit started from the published parameters).

| Example | Model | Param. | RMSE all | ±1 eV | Time (s) |
| --- | --- | --- | --- | --- | --- |
| `graphene` Graphene | C s p, SOC, non-orthogonal | 27 | 68.3 | 11.0 | 75 |
| `mos2` 2H-MoS₂ | Mo d (A₁, E′, E″), S p_z / p_xy, SOC | 87 | 25.8 | 10.4 | 4137 |
| `platinum` Pt (fcc) | s p d, SOC | 45 | 27.5 | 22.5 | 55 |
| `iron` Fe (bcc) | s p d (t₂g, e_g), Δ per group, SOC, ferromagnet | 72 | 126.5 | 14.6 | 144 |
| `cerium` γ-Ce (fcc) | s, d (t₂g, e_g), f (a₂u, t₁u, t₂u) | 125 | 31.4 | 9.2 | 117 |
| `ceria` CeO₂ | Ce d (t₂g, e_g), f (a₂u, t₁u, t₂u), O p | 63 | 52.6 | 16.8 | 99 |
| `mnte` MnTe | Mn s d (A₁, E₁, E₂), Te p_z / p_xy, altermagnet | 63 | 71.6 | 41.1 | 774 |
| `eumn2bi2` EuMn₂Bi₂ | Eu s d f, Mn s d (A₁, E₁, E₂), Bi s p_z / p_xy, AFM, SOC on Bi 6p | 216 | 51.3 | 31.8 | 42753 |

EuMn₂Bi₂ was fitted with two Jacobian threads (`skfitter fit --workers 2`), five fits in parallel.

Model classes (`steps/`; the decay-law and per-shell fits start from Harrison's values with 8 least-squares starts,
2 for EuMn₂Bi₂), error within E_F ± 1 eV (meV):

| Example | decay law | per shell | split (`skfitter.toml` or `steps/split.toml`) |
| --- | --- | --- | --- |
| `graphene` | 17.2 | 10.7 | -- |
| `mos2` | 105.4 | 61.5 | 10.4 |
| `platinum` | 93.6 | 22.5 | 13.7 |
| `iron` | 265.9 | 126.7 | 14.6 |
| `cerium` | 51.1 | 31.4 | 9.2 |
| `ceria` | 61.5 | 32.9 | 16.8 |
| `mnte` | 187.9 | 123.8 | 41.1 |
| `eumn2bi2` | 324.4 | 285.1 | 31.8 |

EuMn₂Bi₂ build-up (Fig. 5b; `steps/` in `eumn2bi2/`): the symmetry-split model, each orbital that
`skfitter character --suggest` ranks (`suggest.toml`, `paper/table_suggest.py`) added alone, and the final model with
all four (spin–orbit coupling on Bi 6p only). RMSE all / within E_F ± 1 eV (meV):

| Step | Param. | RMSE all | ±1 eV |
| --- | --- | --- | --- |
| decay law (`steps/decay.toml`) | 40 | 340.7 | 324.4 |
| per shell (`steps/shell.toml`) | 43 | 255.4 | 285.1 |
| symmetry-split groups (`steps/split.toml`) | 122 | 101.1 | 119.9 |
| + Eu 5d (`steps/eu_d.toml`) | 155 | 77.5 | 80.5 |
| + Mn 4s (`steps/mn_s.toml`) | 141 | 83.8 | 118.8 |
| + Eu 6s (`steps/eu_s.toml`) | 135 | 81.7 | 102.5 |
| + Bi 6s (`steps/bi_s.toml`) | 139 | 96.8 | 105.2 |
| final: + Eu 5d, Mn 4s, Eu 6s, Bi 6s (`skfitter.toml`) | 216 | 51.3 | 31.8 |

The property runs of Fig. 3 (`skfitter berry skfitter.toml` in `iron/` and `platinum/`) take longer: the
Fe anomalous Hall conductivity on a 40³ mesh, twelve times for the spin–orbit scan, and the Pt spin Hall
conductivity on an 80³ mesh. Lower `nk` in the configuration for a quick check.

## Data

The DFT data were computed for the paper with Quantum ESPRESSO 7.5. The configurations use only these files; no
DFT code is needed to run the examples.

The data and configurations in this folder are licensed under CC BY 4.0 (see `LICENSE.md`); the program is
AGPL-3.0.
