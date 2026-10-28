# SKFitter starting-value parameter sets

Data files read by `skfitter.params` (starting values and bounds of a fit: `start_set` in the configuration).
They are data, not code, and keep the licences of their sources.

| File | Source | Licence |
| --- | --- | --- |
| `dftb-mio.json.gz` | DFTB.org mio-1-1, github.com/dftbparams/mio | CC BY-SA 4.0 - cite the references in `licences/README-mio.txt` |
| `dftb-3ob.json.gz` | DFTB.org 3ob-3-1, github.com/dftbparams/3ob | CC BY-SA 4.0 - `licences/README-3ob.txt` |
| `dftb-pbc.json.gz` | DFTB.org pbc-0-3, github.com/dftbparams/pbc | CC BY-SA 4.0 - `licences/README-pbc.txt` |
| `dftb-matsci.json.gz` | DFTB.org matsci-0-3, github.com/dftbparams/matsci | CC BY-SA 4.0 - `licences/README-matsci.txt` |
| `nist_raw.txt`, `nist.json.gz` | NIST Standard Reference Database 141, Atomic Reference Data for Electronic Structure Calculations (math.nist.gov/DFTdata), neutral atoms H-U | public data; cite S. Kotochigova et al., Phys. Rev. A 55, 191 (1997) |
| `harrison.json.gz` | Harrison's universal coefficients (Elementary Electronic Structure, 1999, as tabulated by L. Shi and D. A. Papaconstantopoulos, Phys. Rev. B 70, 205101 (2004)) with the NIST on-site energies | published formulas |

The DFTB files are **adapted material** under CC BY-SA 4.0 (`licences/LICENSE-CC-BY-SA-4.0.txt`): the tables were
resampled every 0.05 Å from 1.0 to 5.0 Å, converted to eV and Å, and the sign of each integral table was brought to the
usual Slater-Koster convention (the sign of an atomic basis function is a free choice; the overlap, where it is
largest, then has the opposite sign of Harrison's coefficient). The adapted files are shared under the same licence.
The rights holders require that the references listed in each README are cited in publications that use the data.

`nist_raw.txt`: one line per element, `Z|configuration|orbital=LDA,ScRLDA;...` (Hartree), read from the NIST tables
on 26 September 2026. On the samarium page the ScRLDA column is shifted down by one line; the values here are
realigned (1s = -1704.693193 ... 6s = -0.136376).

