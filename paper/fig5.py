"""Fig. 5: EuMn2Bi2 - building the model up orbital by orbital. (a) orbital character of the DFT states at Γ
(skfitter character); (b) error of every step (skfitter fit steps/*.toml and skfitter.toml); (c) DFT bands with the
decay-law and the final model."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

from _common import FIG_W, EXAMPLES, GRID, INK, INK2, read_bands, read_report, paper_style, save  # noqa: E402

ACC, GREY, BASIC = "#d6601f", "#9a9992", "#2a78d6"
paper_style()
STEPS = [("eumn2bi2_decay", "decay law"), ("eumn2bi2_shell", "per shell"),
         ("eumn2bi2_split", "split groups"), ("eumn2bi2_eu_d", "+ Eu 5d"),
         ("eumn2bi2_mn_s", "+ Mn 4s"), ("eumn2bi2_eu_s", "+ Eu 6s"), ("eumn2bi2_bi_s", "+ Bi 6s"),
         ("eumn2bi2", "final (all 4)")]
rows = []
for name, label in STEPS:
    r = read_report("eumn2bi2", name)
    rows.append((label, r["n_params"], r["rmse_meV"], r["near_fermi_1eV_meV"]))

fig = plt.figure(figsize=(FIG_W, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.0], height_ratios=[1.15, 1], hspace=0.42, wspace=0.85)

# (a) orbital character at Γ
ax = fig.add_subplot(gs[0, 0])
for char in (os.path.join(EXAMPLES, "eumn2bi2", d, "eumn2bi2_character_Γ.json") for d in ("out_character", "reference")):
    if os.path.isfile(char):                        # skfitter character character.toml --k Γ, else its stored result
        break
if os.path.isfile(char):
    with open(char, encoding="utf-8") as f:
        Q = json.load(f)
    near = [r for r in Q["rows"] if abs(r["E"]) <= 1.0]
    chans = [c for c in Q["channels"] if c not in ("unassigned",)]
    cmap = matplotlib.colormaps["tab10"]
    left = np.zeros(len(near))
    for i, ch in enumerate(chans):
        v = np.array([r["weights"].get(ch, 0.0) for r in near])
        ax.barh(np.arange(len(near)), v, left=left, color=cmap(i % 10), label=ch, height=0.7, edgecolor="white",
                linewidth=0.6)
        left += v
    ax.set_yticks(np.arange(len(near)), [f"{r['E']:+.2f}" for r in near], fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, handlelength=1.2)
else:
    ax.text(0.5, 0.5, "needs the projwfc.x file\n(skfitter character skfitter.toml --k Γ)", ha="center", va="center",
            transform=ax.transAxes, color=INK2)
ax.set_ylabel("state at Γ, E − E$_F$ (eV)")
ax.set_xlabel("orbital weight (QE projections)")
ax.set_xlim(0, 1.05)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.set_title("(a) Orbital character at Γ", loc="left")

# (b) error per step
ax = fig.add_subplot(gs[0, 1])
y = np.arange(len(rows))
h = 0.36
ax.barh(y - h / 2, [r[2] for r in rows], h, color="#b9b8b0", label="all fitted bands", edgecolor="white", linewidth=1)
ax.barh(y + h / 2, [r[3] for r in rows], h, color=INK, label="within E$_F$ ± 1 eV", edgecolor="white", linewidth=1)
for i, r in enumerate(rows):
    ax.text(max(r[2], r[3]) + 3, i, f"{r[1]} p", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
ax.set_ylim(len(rows) - 0.4, -0.6)
ax.set_xlabel("RMSE (meV)")
ax.set_xlim(0, max(200, 1.4 * max(max(r[2], r[3]) for r in rows)))
ax.legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.2)
ax.set_title("(b) Fit error per step", loc="left")
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)

# (c) DFT, decay-law and final bands
ax = fig.add_subplot(gs[1, :])
x, D, T, ticks, labels = read_bands("eumn2bi2")
xa, _, A, _, _ = read_bands("eumn2bi2", "eumn2bi2_decay")
ax.plot(x, D, color=GREY, lw=2.4, alpha=0.8, zorder=1)
ax.plot(xa, A, color=BASIC, lw=1.2, ls=(0, (2, 1.6)), zorder=2)
ax.plot(x, T, color=ACC, lw=1.4, zorder=3)
for tk in ticks:
    ax.axvline(tk, color=GRID, lw=0.6, zorder=0)
ax.axhline(0, color=GRID, lw=0.6, ls=":")
ax.set_xticks(ticks, [l.replace("G", "Γ") for l in labels])
ax.set_xlim(x[0], x[-1])
ax.set_ylim(-1.5, 1.5)
ax.set_ylabel("E − E$_F$ (eV)")
ra, rf = rows[0], rows[-1]
ax.legend(handles=[Line2D([], [], color=GREY, lw=2.4, label="DFT (Quantum ESPRESSO)"),
                   Line2D([], [], color=BASIC, lw=1.2, ls=(0, (2, 1.6)),
                          label=f"SK, basic: decay law, Eu f, Mn d, Bi p ({ra[1]} p, ±1 eV: {ra[3]:.0f} meV)"),
                   Line2D([], [], color=ACC, lw=1.4,
                          label=f"SK, final: split groups + Eu 5d, Mn 4s, Eu 6s, Bi 6s ({rf[1]} p, ±1 eV: {rf[3]:.0f} meV)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=7.5)
ax.set_title("(c) Bands near E$_F$: DFT, basic and final SK models", loc="left")
save(fig, "fig5")
