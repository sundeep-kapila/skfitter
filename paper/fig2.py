"""Fig. 2 of the paper: Slater-Koster fits (model of Table 1) against the QE bands of seven systems, and the error
within E_F +- 1 eV of the three model classes. Needs the outputs of  skfitter fit  for the examples."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402

from _common import (FIG_W, C_DECAY, C_SHELL, C_SPLIT, GRID, INK, INK2, MUTED, SYSTEMS, class_errors,  # noqa: E402
                     paper_style, read_bands, read_report, save)

PANELS = [("graphene", "Graphene"), ("mos2", "MoS$_2$"), ("platinum", "Pt"), ("iron", "Fe"),
          ("cerium", "$\\gamma$-Ce"), ("ceria", "CeO$_2$"), ("mnte", "MnTe")]
# error of the three model classes within E_F +- 1 eV (meV), from the command-line reports
SUMMARY = {title: class_errors(ex) for ex, title in SYSTEMS}


def lab(s):
    return s.replace("G", "Γ") if s.startswith("G") else s


paper_style()
fig = plt.figure(figsize=(FIG_W, 6.4))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.36, left=0.09, right=0.98, top=0.96, bottom=0.13)
band_axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
last = gs[2, :].subgridspec(1, 2, width_ratios=[1, 1.6], wspace=0.62)
band_axes.append(fig.add_subplot(last[0, 0]))
for i, (ax, (ex, name)) in enumerate(zip(band_axes, PANELS)):
    x, E, T, ticks, labels = read_bands(ex)
    ax.plot(x, E, color=INK, lw=1.3)
    ax.plot(x, T, color=C_SHELL, lw=1.3, ls=(0, (4, 2)))
    for t in ticks:
        ax.axvline(t, color=GRID, lw=0.8, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(ticks, [lab(l) for l in labels])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.nanmin(E) - 0.3, np.nanmax(E) + 0.3)
    r = read_report(ex)
    ax.set_title(f"({chr(97 + i)}) {name}", loc="left", color=INK)
    ax.set_title(f"{r['rmse_meV']:.0f}/{r['near_fermi_1eV_meV']:.0f} meV", loc="right", color=INK2,
                 fontsize=6.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7.5)
    if i % 3 == 0:
        ax.set_ylabel("E − E$_F$ (eV)")
ax = fig.add_subplot(last[0, 1])
names = list(SUMMARY)
y = np.arange(len(names))
h = 0.26
for off, key, col in ((-h, "decay", C_DECAY), (0.0, "shell", C_SHELL), (h, "split", C_SPLIT)):
    ax.barh(y + off, [np.nan if SUMMARY[n][key] is None else SUMMARY[n][key] for n in names], h, color=col, edgecolor="white", linewidth=1)
ax.axvline(25, color=INK, lw=1.0, ls="--")
ax.text(28, -0.85, "25 meV target", fontsize=7.5, color=INK, va="center")
ax.set_yticks(y, names)
ax.set_ylim(len(names) - 0.4, -1.2)
ax.set_xlabel("RMSE within E$_F$ ± 1 eV (meV)")
ax.set_title("(h) Fit error within E$_F$ ± 1 eV", loc="left", color=INK)
ax.grid(axis="x", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.set_xlim(0, max(175, 1.08 * max(v for d in SUMMARY.values() for v in d.values() if v is not None)))
fig.legend(handles=[Line2D([0], [0], color=INK, lw=1.3, label="DFT (Quantum ESPRESSO)"),
                    Line2D([0], [0], color=C_SHELL, lw=1.3, ls=(0, (4, 2)), label="Slater–Koster fit (Table 1 model)"),
                    Patch(color=C_DECAY, label="decay law"), Patch(color=C_SHELL, label="per shell"),
                    Patch(color=C_SPLIT, label="symmetry-split groups")],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7)
save(fig, "fig2")
