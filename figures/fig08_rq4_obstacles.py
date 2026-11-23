"""Figure 8 (RQ4). Obstacle degrees in 2023: which indicators hold back coordination in each country."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RES, PROC, W2, MM, SUBCOL, SUBSYSTEMS, save_fig  # noqa: E402

O = pd.read_csv(RES / "rq4_obstacle_indicator_2023.csv", index_col=0)
Osub = pd.read_csv(RES / "rq4_obstacle_subsystem_2023.csv", index_col=0)
W = pd.read_csv(RES / "weights.csv").query("scheme=='entropy'")
CC = pd.read_csv(PROC / "ccd_panel.csv")
d23 = CC[CC.year == 2023].set_index("iso3")
order = d23.D.sort_values().index
ind = list(W.indicator)
SHORT = {"GDP": "GDP per capita", "TFR": "Fertility", "ABR": "Adolescent births", "PoU": "Undernourishment",
         "FLFP": "Female labour force", "GE": "Gov. effectiveness", "CC": "Control of corruption", "RL": "Rule of law",
         "RQ": "Regulatory quality", "VA": "Voice & accountability", "PV": "Political stability", "WAT": "Basic water",
         "SAN": "Basic sanitation", "FUEL": "Clean cooking fuel", "ELEC": "Electricity", "CHE": "Health spending",
         "GHP": "Gov. health priority", "OOP": "Out-of-pocket share", "DTP3": "DTP3", "MCV1": "Measles (MCV1)"}
lab = SHORT
M = O.loc[order, ind] * 100

fig = plt.figure(figsize=(W2, 175 * MM))
ax = fig.add_axes([0.13, 0.13, 0.6, 0.82])
im = ax.imshow(M.values, cmap="OrRd", vmin=0, vmax=30, aspect="auto")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(d23.loc[order, "country"].str.replace("Central African Republic", "CAR")
                   .str.replace("São Tomé and Príncipe", "São Tomé & Príncipe"), fontsize=5.4)
ax.set_xticks(range(len(ind))); ax.set_xticklabels([lab[i] for i in ind], rotation=50, ha="right", fontsize=5.6, rotation_mode="anchor")
start = 0
for s in SUBSYSTEMS:
    n = int((W.subsystem == s).sum())
    ax.add_patch(plt.Rectangle((start - 0.5, -1.9), n, 1.0, color=SUBCOL[s], clip_on=False))
    ax.text(start - 0.5 + n / 2, -1.4, s, ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
    if start:
        ax.axvline(start - 0.5, color="white", lw=1.2)
    start += n
ax.set_ylim(len(order) - 0.5, -2.0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
cax = fig.add_axes([0.76, 0.028, 0.2, 0.008])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Obstacle degree (% of country's total shortfall)",
                                                                       fontsize=6)
cb.ax.tick_params(labelsize=5.6)
ax.text(-0.02, 1.012, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right")

# b. subsystem obstacle shares and D
bx = fig.add_axes([0.76, 0.13, 0.11, 0.82], sharey=ax)
left = np.zeros(len(order))
for s in SUBSYSTEMS:
    v = Osub.loc[order, s].values * 100
    bx.barh(range(len(order)), v, left=left, color=SUBCOL[s], height=0.8, label=s)
    left += v
bx.set_xlim(0, 100); bx.set_xlabel("Subsystem share (%)", fontsize=6)
bx.tick_params(axis="y", left=False, labelleft=False); bx.tick_params(axis="x", labelsize=5.6)
bx.text(0.0, 1.012, "b", transform=bx.transAxes, fontsize=8, fontweight="bold", ha="right")
cx = fig.add_axes([0.89, 0.13, 0.09, 0.82], sharey=ax)
cx.scatter(d23.loc[order, "D"], range(len(order)), s=8, color="#1d1d1f")
cx.set_xlim(0.35, 1.0); cx.set_xlabel("D, 2023", fontsize=6)
cx.tick_params(axis="y", left=False, labelleft=False); cx.tick_params(axis="x", labelsize=5.6)
for v in (0.5, 0.6, 0.7):
    cx.axvline(v, color="#cccccc", lw=0.5, ls=":")
cx.text(0.0, 1.012, "c", transform=cx.transAxes, fontsize=8, fontweight="bold", ha="right")
bx.legend(loc="upper left", bbox_to_anchor=(-0.1, -0.045), ncol=2, frameon=False, fontsize=5.6)
save_fig(fig, "Figure_8")
