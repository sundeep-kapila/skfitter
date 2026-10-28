"""Fig. 3: Berry-phase properties of the fitted models against first-principles references.
(a) Fe anomalous Hall conductivity against E; (b) Fe AHC against the spin-orbit strength, with Yao et al. (2004);
(c) Pt spin Hall conductivity with Guo et al. (2008). Needs  skfitter berry  for examples/iron and examples/platinum.
Fe: plotted as -sigma_xy of the model (majority spin along +z), the convention of Yao et al. (magnetisation along +z)."""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

from _common import FIG_W, EXAMPLES, GRID, INK, INK2, out_file, paper_style, save  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ACC, REF = "#d6601f", "#2a6fb0"
paper_style()
fig = plt.figure(figsize=(FIG_W, 5.3))
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.42)
a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# (a) Fe AHC against E
E, s = np.loadtxt(out_file("iron", "iron_sk_ahc_z.dat"), encoding="utf-8").T
s = -s
a.plot(E, s, color=INK, lw=1.6, label="SKFitter model (this work)")
a.axhline(751, color=REF, lw=0.9, ls=":")
a.text(1.95, 800, "Yao et al., 751 S/cm", fontsize=7.5, color=REF, fontweight="bold", ha="right")
a.axvline(0, color=GRID, lw=0.8)
a.set_xlabel("E − E$_F$ (eV)")
a.set_ylabel("σ$_{xy}$ (S/cm)")
a.set_title("(a) Fe: AHC vs energy", loc="left")
a.set_ylim(-600, 1950)
a.legend(frameon=False, fontsize=7.5, loc="upper right")

# (b) Fe AHC against lambda(Fe, d) / fitted value
with open(os.path.join(EXAMPLES, "iron", "reference", "iron_fit.json"), encoding="utf-8") as f:
    LAM0 = json.load(f)["parameters"]["lambda_Fe_d"]
pts = {}
for path in glob.glob(os.path.join(EXAMPLES, "iron", "out", "iron_lambda_*_sk_ahc_z.dat")):
    lam = float(re.search(r"iron_lambda_([0-9.e-]+)_sk_ahc", os.path.basename(path)).group(1))
    e, v = np.loadtxt(path, encoding="utf-8").T
    pts[lam] = float(np.mean(-v[np.abs(e) <= 0.1 + 1e-9]))
E0, s0 = np.loadtxt(out_file("iron", "iron_sk_ahc_z.dat"), encoding="utf-8").T     # the fitted lambda (a)
pts[LAM0] = float(np.mean(-s0[np.abs(E0) <= 0.1 + 1e-9]))
x = np.array(sorted(pts)) / LAM0
y = np.array([pts[k] for k in sorted(pts)])
with open(os.path.join(HERE, "reference_data", "yao2004_fig4_digitized.json"), encoding="utf-8") as f:
    yao = json.load(f)
Y = np.array(yao["points"])
YC = np.array([[0, 0]] + [q for q in yao["curve"] if q[0] > 0.12])
b.plot(YC[:, 0], YC[:, 1], "-", color=REF, lw=1.0, alpha=0.7)
b.plot(Y[:, 0], Y[:, 1], "o", mfc="white", mec=REF, ms=5, label="Yao et al. (first principles)")
b.plot(x, y, "-", color=INK, lw=1.0)
b.plot(x, y, "s", color=INK, ms=3.8, label="SKFitter model (this work)")
i0 = int(np.argmin(np.abs(x - 1.0)))
b.plot([x[i0]], [y[i0]], "s", color=ACC, ms=6, zorder=3)
b.set_xlabel(f"ξ/ξ$_0$  (λ$_d$/λ$_{{d,0}}$, λ$_{{d,0}}$ = {1e3 * LAM0:.0f} meV)")
b.set_ylabel("σ$_{xy}$ at E$_F$ ± 0.1 eV (S/cm)")
b.set_title("(b) Fe: AHC vs SOC strength", loc="left")
b.set_xlim(-0.1, 4.2)
b.set_ylim(0, max(1750, 1.1 * y.max()))
b.legend(frameon=False, fontsize=7, loc="lower right")

# (c) Pt SHC with Guo et al.
E, s = np.loadtxt(out_file("platinum", "platinum_sk_shc_z_xy.dat"), encoding="utf-8").T[:2]
with open(os.path.join(HERE, "reference_data", "guo2008_fig1b_digitized.json"), encoding="utf-8") as f:
    G = np.array(json.load(f))
c.plot(G[:, 0], G[:, 1], color=REF, lw=1.3, label="Guo et al. (first principles)")
c.plot(E, s, color=INK, lw=1.2, label="SKFitter model (this work)")
c.axvline(0, color=GRID, lw=0.8)
c.axhline(0, color=GRID, lw=0.8)
c.set_xlabel("E − E$_F$ (eV)")
c.set_ylabel("σ$^{z}_{xy}$ ((ħ/e) S/cm)")
c.set_xlim(-10, 4)
c.set_title("(c) Pt spin Hall conductivity", loc="left")
c.legend(frameon=False, fontsize=7.5, loc="upper left")
for ax in (a, b, c):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
save(fig, "fig3")
