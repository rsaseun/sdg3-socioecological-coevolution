"""Figure 4 (RQ3). Trajectories, distribution dynamics and convergence of coupling coordination, 2000-2023."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import PROC, RES, W2, MM, SUBCOL, SUBSYSTEMS, CLASSES, save_fig, panel_label  # noqa: E402

CC = pd.read_csv(PROC / "ccd_panel.csv")
S = json.load(open(RES / "rq3_summary.json"))
kde = pd.read_csv(RES / "rq3_kde.csv", index_col=0)
REGCOL = {"Central": "#E69F00", "East": "#009E73", "Southern": "#0072B2", "West": "#CC79A7"}

fig, axs = plt.subplots(2, 2, figsize=(W2, 118 * MM))
plt.subplots_adjust(hspace=0.45, wspace=0.28, left=0.07, right=0.98, top=0.96, bottom=0.08)

# a. trajectories
a = axs[0, 0]
for s in SUBSYSTEMS + ["D"]:
    col = f"U_{s}" if s != "D" else "D"
    g = CC.groupby("year")[col]
    m, se = g.mean(), g.std() / np.sqrt(g.count())
    c = SUBCOL.get(s, "#1d1d1f")
    a.plot(m.index, m, color=c, lw=1.6 if s == "D" else 1.1, label="Coordination D" if s == "D" else s)
    a.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.12, lw=0)
pw = CC.groupby("year").apply(lambda g: np.average(g.D, weights=g.POP))
a.plot(pw.index, pw, color="#1d1d1f", lw=1.0, ls="--", label="D, population-weighted")
a.set_ylabel("Index (0–1)"); a.set_xlim(2000, 2023); a.set_ylim(0.1, 0.8)
a.legend(frameon=False, ncol=2, fontsize=5.6, loc="upper left", handlelength=1.6)
panel_label(a, "a", x=-0.1)

# b. kernel densities
b = axs[0, 1]
cols = {"2000": "#c6dbef", "2008": "#6baed6", "2016": "#2171b5", "2023": "#08306b"}
for y, c in cols.items():
    b.plot(kde.index, kde[y], color=c, lw=1.3, label=y)
    b.fill_between(kde.index, 0, kde[y], color=c, alpha=0.08)
for lo, hi, name in CLASSES[1:]:
    b.axvline(lo, color="#999999", lw=0.5, ls=":")
ymax = b.get_ylim()[1]
for lo, hi, name in CLASSES:
    mid = (max(lo, 0.3) + min(hi, 1.0)) / 2
    short = {"Imbalanced": "Imbal.", "Near imbalance": "Near", "Barely coordinated": "Barely",
             "Primary coordination": "Primary", "Intermediate or better": "Intermediate+"}[name]
    b.text(mid, ymax * 1.0, short, ha="center", va="bottom", fontsize=5.4, color="#555555")
b.set_xlabel("Coupling coordination degree D"); b.set_ylabel("Density"); b.set_xlim(0.3, 1.0)
b.legend(frameon=False, fontsize=5.8, loc="center right")
panel_label(b, "b", x=-0.1)

# c. beta convergence
c = axs[1, 0]
Dw = CC.pivot(index="iso3", columns="year", values="D")
reg = CC[CC.year == 2023].set_index("iso3").region
name = CC[CC.year == 2023].set_index("iso3").country
x = np.log(Dw[2000]); y = 100 * (np.log(Dw[2023]) - np.log(Dw[2000])) / 23
for r, col in REGCOL.items():
    k = reg[reg == r].index
    c.scatter(x[k], y[k], s=14, color=col, edgecolor="white", lw=0.4, label=f"{r} Africa")
bb = S["beta_abs"]["D"]
xx = np.linspace(x.min(), x.max(), 50)
c.plot(xx, 100 * (np.polyfit(x, y / 100, 1)[1] + bb["beta"] * xx), color="#1d1d1f", lw=1)
for k in ["RWA", "SLE", "TZA", "SOM", "SSD", "CAF", "MUS", "SYC", "ETH"]:
    c.annotate(name[k].replace("Central African Republic", "CAR"), (x[k], y[k]), fontsize=5.2, xytext=(3, 2),
               textcoords="offset points")
c.set_xlabel("ln D in 2000"); c.set_ylabel("Annual growth of D, 2000–2023 (%)")
c.text(0.98, 0.95, f"β = {bb['beta']:.4f} (p < 0.001)\nhalf-life ≈ {bb['half_life']:.0f} years",
       transform=c.transAxes, ha="right", va="top", fontsize=5.8)
c.legend(frameon=False, fontsize=5.4, loc="upper right", bbox_to_anchor=(1.0, 0.8), ncol=1, handletextpad=0.1)
panel_label(c, "c", x=-0.1)

# d. sigma convergence
d = axs[1, 1]
for s in SUBSYSTEMS + ["D"]:
    col = f"U_{s}" if s != "D" else "D"
    g = CC.groupby("year")[col]
    cv = g.std() / g.mean()
    d.plot(cv.index, cv, color=SUBCOL.get(s, "#1d1d1f"), lw=1.6 if s == "D" else 1.1,
           label="Coordination D" if s == "D" else s)
d.set_yscale("log")
from matplotlib.ticker import NullFormatter, FixedLocator  # noqa: E402
d.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1.0])); d.set_yticklabels(["0.2", "0.3", "0.5", "1.0"])
d.yaxis.set_minor_formatter(NullFormatter())
d.set_ylabel("Coefficient of variation (log scale)"); d.set_xlim(2000, 2023)
sg = S["sigma"]
d.text(0.98, 0.04, f"D: CV {sg['cv_2000']:.2f} → {sg['cv_2023']:.2f}", transform=d.transAxes, ha="right",
       fontsize=5.8)
d.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper right")
panel_label(d, "d", x=-0.1)
save_fig(fig, "Figure_4")
