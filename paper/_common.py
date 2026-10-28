"""Shared helpers of the figure scripts: read the outputs that ``skfitter fit`` / ``berry`` / ``character`` wrote into
examples/<id>/out/."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
FIGURES = os.path.join(ROOT, "paper", "figures")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_DECAY, C_SHELL, C_SPLIT, C_ORB = "#2a78d6", "#eb6834", "#1f9d74", "#8a5cc7"


def out_file(example, name):
    path = os.path.join(EXAMPLES, example, "out", name)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run  skfitter fit examples/{example}/skfitter.toml  first "
                         "(or paper/run_examples.py)")
    return path


def read_bands(example, name=None):
    """x, DFT bands, TB bands, tick positions and labels of <name>_bands.dat."""
    path = out_file(example, f"{name or example}_bands.dat")
    with open(path, encoding="utf-8") as f:
        head = [next(f) for _ in range(2)]
    ticks = head[0].split("k-path ticks:", 1)[1].split()
    labels = [t.rsplit("=", 1)[0] for t in ticks]
    pos = [float(t.rsplit("=", 1)[1]) for t in ticks]
    cols = head[1].lstrip("#").split()
    data = np.loadtxt(path, encoding="utf-8")
    dft = [i for i, c in enumerate(cols) if c.startswith("dft_")]
    tb = [i for i, c in enumerate(cols) if c.startswith("tb_")]
    return data[:, 0], data[:, dft], data[:, tb], pos, labels


def read_spin_bands(example, name=None):
    """x, DFT up / down, TB up / down, ticks and labels of <name>_bands_spin.dat (skfitter bands --spin)."""
    path = out_file(example, f"{name or example}_bands_spin.dat")
    with open(path, encoding="utf-8") as f:
        head = [next(f) for _ in range(2)]
    ticks = head[0].split("k-path ticks:", 1)[1].split()
    labels = [t.rsplit("=", 1)[0] for t in ticks]
    pos = [float(t.rsplit("=", 1)[1]) for t in ticks]
    cols = head[1].lstrip("#").split()
    data = np.loadtxt(path, encoding="utf-8")
    pick = lambda pre: data[:, [i for i, c in enumerate(cols) if c.startswith(pre)]]  # noqa: E731
    return data[:, 0], pick("dft_up_"), pick("dft_dn_"), pick("tb_up_"), pick("tb_dn_"), pos, labels


def read_report(example, name=None):
    with open(out_file(example, f"{name or example}_report.json"), encoding="utf-8") as f:
        return json.load(f)


FIG_W = 5.6        # inches: about the text width of the elsarticle preprint (390 pt = 5.4 in), so sizes print ~1:1


def paper_style(base=9.0):
    """Matplotlib settings shared by Figs. 2-5: figures are drawn FIG_W wide, close to the text width, so the
    sizes below are about the printed sizes. Black, thick plot borders and ticks; bold axis labels and tick values."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": base, "axes.titlesize": base + 0.5, "axes.labelsize": base + 0.5, "axes.labelweight": "bold",
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5, "legend.fontsize": base - 1.0,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 1.3, "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 4.0,
        "ytick.major.size": 4.0, "xtick.minor.width": 0.9, "ytick.minor.width": 0.9, "axes.titleweight": "bold",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, stem, bold_ticks=True):
    """Write paper/figures/<stem>.png (600 dpi) and .pdf (vector, fonts embedded); tick values in bold."""
    if bold_ticks:
        for ax in fig.axes:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontweight("bold")
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, stem + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, stem + ".pdf"), bbox_inches="tight")
    print("wrote", os.path.join("paper", "figures", stem + ".png"))


# Table 1 / Fig. 2(h): the three model classes. The final model (skfitter.toml) is the split class unless its orbitals
# are whole families (graphene has no split model; Pt's split model is steps/split.toml).
SYSTEMS = [("graphene", "Graphene"), ("mos2", "MoS$_2$"), ("platinum", "Pt"), ("iron", "Fe"), ("cerium", "$\\gamma$-Ce"),
           ("ceria", "CeO$_2$"), ("mnte", "MnTe"), ("eumn2bi2", "EuMn$_2$Bi$_2$")]
FINAL_CLASS = {"graphene": "shell", "platinum": "shell"}


def class_errors(example):
    """{'decay', 'shell', 'split'}: error within E_F +- 1 eV (meV) of each model class, None if not run."""
    out = {}
    for c in ("decay", "shell", "split"):
        try:
            out[c] = read_report(example, f"{example}_{c}")["near_fermi_1eV_meV"]
        except SystemExit:
            out[c] = None
    out[FINAL_CLASS.get(example, "split")] = read_report(example)["near_fermi_1eV_meV"]
    if example == "graphene":
        out["split"] = None
    return out
