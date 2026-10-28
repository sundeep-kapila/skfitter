# Contributing to SKFitter

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/sundeep-kapila/skfitter/issues).

## Reporting a problem

Please include the `skfitter --version` output, the configuration file, the command and its full output (run with
`-v` for the fitter's log), and, if you can share them, the DFT files.

## Pull requests

* Add or update tests in `tests/` for every change of behaviour; `pytest` must pass.
* Keep the command line and the configuration keys backwards compatible, or document the change in
  `CHANGELOG.md`.
* Do not add DFT data that you may not redistribute.

## Licence of contributions

SKFitter is distributed under the GNU Affero General Public License v3.0 only, and the authors also offer it under
a commercial licence. To keep both possible, contributions are accepted under a Contributor Licence Agreement:
before a pull request is merged you will be asked to confirm that you wrote the contribution (or have the right
to submit it) and that you grant the SKFitter authors the right to distribute it under the AGPL-3.0 and under
other licences. Contact sundeep.kapila@swasth.org with questions.
