"""The suggestion table of the paper: the orbitals the symmetry-split EuMn2Bi2 model lacks, as ``skfitter character --suggest``
ranks them (examples/eumn2bi2/suggest.toml), next to the error of the split model with each orbital added alone
(steps/<orbital>.toml) and of the final model with all of them (skfitter.toml). Writes paper/figures/table_suggest.tex
(the rows of the paper's table) and table_suggest.md."""
import json
import os

from _common import EXAMPLES, FIGURES, read_report

SUGGEST = os.path.join(EXAMPLES, "eumn2bi2", "out_suggest", "eumn2bi2_split_orbital_suggestions.json")
STEP = {"Eu d": ("eumn2bi2_eu_d", "Eu 5d"), "Mn s": ("eumn2bi2_mn_s", "Mn 4s"), "Eu s": ("eumn2bi2_eu_s", "Eu 6s"),
        "Bi s": ("eumn2bi2_bi_s", "Bi 6s")}


def main():
    if not os.path.isfile(SUGGEST):
        raise SystemExit(f"{SUGGEST} is missing: run  skfitter character suggest.toml --suggest  in examples/eumn2bi2")
    with open(SUGGEST, encoding="utf-8") as f:
        sug = json.load(f)
    split = read_report("eumn2bi2", "eumn2bi2_split")
    final = read_report("eumn2bi2")
    rows = [c for c in sug["channels"] if not c["in_model"] and c["channel"] in STEP]
    rows.sort(key=lambda c: -(c["missing_share"] or 0.0))
    md = ["| Orbital | missing share | absolute | weighted | centre (eV) | filled | +param. | all (meV) | ±1 eV (meV) |",
          "|" + " --- |" * 9,
          f"| split model | | | | | | {split['n_params']} | {split['rmse_meV']:.1f} | {split['near_fermi_1eV_meV']:.1f} |"]
    tex = [f"symmetry-split model & & & & & & {split['n_params']} & {split['rmse_meV']:.0f} & "
           f"{split['near_fermi_1eV_meV']:.0f} \\\\", "\\midrule"]
    for c in rows:
        name, label = STEP[c["channel"]]
        r = read_report("eumn2bi2", name)
        cells = [f"{100 * c['missing_share']:.0f}\\%", f"{c['absolute']:.2f}", f"{c['weighted']:.2f}",
                 f"{c['centre_eV']:+.1f}", f"{100 * c['filled']:.0f}\\%", f"+{r['n_params'] - split['n_params']}",
                 f"{r['rmse_meV']:.0f}", f"{r['near_fermi_1eV_meV']:.0f}"]
        tex.append(f"+ {label} & " + " & ".join(cells) + " \\\\")
        md.append(f"| + {label} | " + " | ".join(x.replace("\\%", "%") for x in cells[:5]) +
                  f" | {cells[5]} | {r['rmse_meV']:.1f} | {r['near_fermi_1eV_meV']:.1f} |")
    tex += ["\\midrule", f"final: all four & & & & & & +{final['n_params'] - split['n_params']} & "
            f"{final['rmse_meV']:.0f} & {final['near_fermi_1eV_meV']:.0f} \\\\"]
    md.append(f"| final: all four | | | | | | +{final['n_params'] - split['n_params']} | {final['rmse_meV']:.1f} | "
              f"{final['near_fermi_1eV_meV']:.1f} |")
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(FIGURES, "table_suggest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(FIGURES, "table_suggest.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))
    print("wrote paper/figures/table_suggest.tex")


if __name__ == "__main__":
    main()
