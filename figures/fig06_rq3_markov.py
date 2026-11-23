"""Figure 6 (RQ3). Persistence and spatial contagion of coordination states (Markov and spatial Markov chains)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RES, W2, MM, save_fig, panel_label  # noqa: E402

P = pd.read_csv(RES / "rq3_markov_P.csv", index_col=0)
T = pd.read_csv(RES / "rq3_markov_counts.csv", index_col=0)
SP = pd.read_csv(RES / "rq3_spatial_markov_P.csv", index_col=[0, 1])
S = json.load(open(RES / "rq3_summary.json"))
names = list(P.index)
ab = ["Imbal.", "Near", "Barely", "Primary", "Interm.+"]

fig = plt.figure(figsize=(W2, 72 * MM))
a = fig.add_axes([0.07, 0.2, 0.25, 0.68])
im = a.imshow(P.values, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        v, n = P.iloc[i, j], T.iloc[i, j]
        if n > 0:
            a.text(j, i, f"{v:.3f}\n({n})", ha="center", va="center", fontsize=5.0, linespacing=1.0,
                   color="white" if v > 0.6 else "#1d1d1f")
a.set_xticks(range(5)); a.set_xticklabels(ab, fontsize=5.8, rotation=30, ha="right")
a.set_yticks(range(5)); a.set_yticklabels(ab, fontsize=5.8)
a.set_xlabel("Class in year t+1"); a.set_ylabel("Class in year t")
for sp in a.spines.values():
    sp.set_visible(False)
panel_label(a, "a", x=-0.28, y=1.04)

# b. spatial Markov: upgrade / downgrade by neighbourhood
b = fig.add_axes([0.42, 0.2, 0.3, 0.68])
nb = ["Low-coordination neighbours", "Middle", "High-coordination neighbours"]
nbl = ["Low (<0.55)", "Middle", "High (≥0.65)"]
cols = ["#B2182B", "#BDBDBD", "#2166AC"]
up = S["upgrade_probability_by_neighbourhood"]
dn = S["downgrade_probability_by_neighbourhood"]
groups = [("Up from\nNear", "up", "Near imbalance"), ("Up from\nBarely", "up", "Barely coordinated"),
          ("Up from\nPrimary", "up", "Primary coordination"), ("Down from\nPrimary", "down", "Primary coordination")]
x = np.arange(len(groups))
for k, (n, l, c) in enumerate(zip(nb, nbl, cols)):
    vals = [(up if kind == "up" else dn)[n].get(cls) for _, kind, cls in groups]
    vals = [np.nan if v is None else v for v in vals]
    b.bar(x + (k - 1) * 0.26, vals, width=0.25, color=c, label=l)
    for xi, v in zip(x + (k - 1) * 0.26, vals):
        if np.isfinite(v):
            b.text(xi, v + 0.004, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=5.0)
b.set_xticks(x); b.set_xticklabels([g[0] for g in groups], fontsize=5.8)
b.set_ylabel("Annual transition probability")
b.legend(title="Neighbours' mean D", title_fontsize=5.8, frameon=False, fontsize=5.6, loc="upper right")
h = S["spatial_homogeneity"]
b.text(0.02, 0.97, f"Spatial homogeneity rejected:\nQ = {h['Q']:.1f}, df = {h['Q_dof']}, p < 0.001\n"
       f"LR = {h['LR']:.1f}, p = {h['LR_p']:.3f}", transform=b.transAxes, va="top", fontsize=5.6)
b.set_ylim(0, 0.25)
b.text(0.99, 0.6, "bar labels = %", transform=b.transAxes, ha="right", fontsize=5.3, color="#555555")
panel_label(b, "b", x=-0.16, y=1.04)

# c. observed vs ergodic distribution
c = fig.add_axes([0.80, 0.2, 0.19, 0.68])
obs0 = pd.Series(S["class_2000"]).reindex(names).fillna(0) / 48
obs1 = pd.Series(S["class_2023"]).reindex(names).fillna(0) / 48
erg = pd.Series(S["steady_state"]).reindex(names)
xx = np.arange(5)
for k, (v, l, col) in enumerate([(obs0, "2000", "#c6dbef"), (obs1, "2023", "#2171b5"),
                                 (erg, "Long-run (ergodic)", "#08306b")]):
    c.bar(xx + (k - 1) * 0.27, v.values, width=0.26, color=col, label=l)
c.set_xticks(xx); c.set_xticklabels(ab, fontsize=5.6, rotation=30, ha="right")
c.set_ylabel("Share of countries"); c.legend(frameon=False, fontsize=5.5, loc="upper left")
panel_label(c, "c", x=-0.3, y=1.04)
save_fig(fig, "Figure_6")
