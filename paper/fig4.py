"""Fig. 4: MnTe altermagnet - spin-resolved bands (a) and the spin splitting of the top valence band (b), DFT
against the final (symmetry-split, 5-shell) model and the unsplit per-shell model. Needs
skfitter bands --spin  for examples/mnte/skfitter.toml and examples/mnte/steps/shell.toml."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

from _common import FIG_W, read_report, read_spin_bands, paper_style, save  # noqa: E402

UP, DN = "#2563b0", "#c2410c"
x, du, dd, tu, td, ticks, labels = read_spin_bands("mnte")
_, _, _, su, sd, _, _ = read_spin_bands("mnte", "mnte_shell")
lab = [l.replace("'", "′").replace("G", "Γ") for l in labels]
nf, ns = read_report("mnte")["n_params"], read_report("mnte", "mnte_shell")["n_params"]
paper_style()
fig, (a, b) = plt.subplots(1, 2, figsize=(FIG_W, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
for E, col, ls, lw, zo in [(du, UP, "-", 1.5, 2), (dd, DN, "-", 1.5, 2), (tu, UP, (0, (4, 2)), 1.1, 3),
                           (td, DN, (0, (4, 2)), 1.1, 3)]:
    for i in range(E.shape[1]):
        a.plot(x, E[:, i], color=col, ls=ls, lw=lw, zorder=zo)
for t in ticks:
    a.axvline(t, color="#c8c8c8", lw=0.6, zorder=1)
    b.axvline(t, color="#c8c8c8", lw=0.6, zorder=1)
a.axhline(0, color="#888", lw=0.6, ls=":")
a.set_ylim(-3.0, 2.0)
a.set_xlim(x[0], x[-1])
a.set_xticks(ticks, lab)
a.set_ylabel("E − E$_F$ (eV)")
a.set_title("(a) Spin-resolved bands, k$_z$ = 0.3", loc="left")
a.legend(handles=[Line2D([], [], color=UP, lw=1.5, label="spin ↑"), Line2D([], [], color=DN, lw=1.5, label="spin ↓"),
                  Line2D([], [], color="#444", lw=1.5, label="DFT"),
                  Line2D([], [], color="#444", lw=1.1, ls=(0, (4, 2)), label="tight binding")],
         loc="lower left", fontsize=7.5, frameon=True, framealpha=0.92, ncol=2, handlelength=1.6, columnspacing=0.8)


def top(E):
    return np.where(E <= 0.05, E, -1e9).max(axis=1)


b.axhline(0, color="#888", lw=0.7)
curves = [1e3 * (top(u) - top(d)) for u, d in ((du, dd), (tu, td), (su, sd))]
b.plot(x, curves[0], color="#1a1a1a", lw=1.8, label="DFT")
b.plot(x, curves[1], color="#d6601f", lw=1.4, ls=(0, (4, 2)),
       label=f"TB, split groups ({nf} p)")
b.plot(x, curves[2], color="#7a7a7a", lw=1.1, ls=":", label=f"TB, whole d, p; per shell ({ns} p)")
b.set_xlim(x[0], x[-1])
b.set_xticks(ticks, lab)
lo = min(-80.0, 1.1 * min(c.min() for c in curves))
b.set_ylim(lo, max(1000.0, 1.65 * max(c.max() for c in curves)))
b.set_ylabel("E$_↑$ − E$_↓$, top valence band (meV)")
b.set_title("(b) Altermagnetic spin splitting", loc="left")
b.legend(loc="upper left", fontsize=7.5, frameon=False)
fig.tight_layout()
save(fig, "fig4")
