"""Table 1 of the paper from the command-line reports: the model shown in Fig. 2 (skfitter.toml), its RMSE over all
fitted bands and within E_F +- 1 eV, and the +-1 eV error of the three model classes (decay law, per shell,
symmetry-split groups; steps/*.toml). Writes paper/figures/table1.tex (the rows of the paper's table) and table1.md."""
import os

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    import tomli as tomllib

from _common import EXAMPLES, FIGURES, class_errors, read_report

# label and "orbitals and terms" column of the paper's table
ROWS = [
    ("graphene", "(a) Graphene", "C s p, SOC, non-orthogonal"),
    ("mos2", "(b) MoS$_2$", "Mo d ($A_1$, $E'$, $E''$), S $p_z/p_{xy}$, SOC"),
    ("platinum", "(c) Pt", "s p d, SOC"),
    ("iron", "(d) Fe", "s p d ($t_{2g}$, $e_g$), $\\Delta_{t2g}/\\Delta_{eg}$, SOC, FM"),
    ("cerium", "(e) $\\gamma$-Ce", "Ce s, d ($t_{2g}$, $e_g$), f ($a_{2u}$, $t_{1u}$, $t_{2u}$)"),
    ("ceria", "(f) CeO$_2$", "Ce d ($t_{2g}$, $e_g$), f ($a_{2u}$, $t_{1u}$, $t_{2u}$), O p"),
    ("mnte", "(g) MnTe", "Mn s d ($A_1$, $E_1$, $E_2$), Te $p_z/p_{xy}$, AFM"),
    ("eumn2bi2", "EuMn$_2$Bi$_2$ (Fig.~\\ref{fig:eu})",
     "Eu s d f, Mn s d ($A_1$, $E_1$, $E_2$), Bi s $p_z/p_{xy}$, AFM, SOC on Bi 6p"),
]


def main():
    md = ["| Material | Shells | Param. | all (meV) | ±1 eV (meV) | decay | shell | split |", "|" + " --- |" * 8]
    tex = []
    for ex, title, orbitals in ROWS:
        with open(os.path.join(EXAMPLES, ex, "skfitter.toml"), "rb") as f:
            conf = tomllib.load(f)
        r = read_report(ex)
        c = class_errors(ex)
        fmt = lambda v: "--" if v is None else f"{v:.0f}"  # noqa: E731
        vals = [str(conf["model"]["neighbours"]), str(r["n_params"]), f"{r['rmse_meV']:.0f}",
                f"{r['near_fermi_1eV_meV']:.0f}", fmt(c["decay"]), fmt(c["shell"]), fmt(c["split"])]
        md.append(f"| {ex} | " + " | ".join(vals) + " |")
        tex.append(f"{title} & {orbitals} & " + " & ".join(vals) + " \\\\")
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(FIGURES, "table1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(FIGURES, "table1.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
