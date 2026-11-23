"""Supplementary Figures S1-S4.

S1  Data provenance: share of panel cells observed / interpolated / kNN-imputed by country and indicator.
S2  RQ1 accuracy and stability: case-dropping stability and bootstrapped edge intervals.
S3  Subsystem indices by country, 2023 (maps).
S4  Country trajectories of the four subsystem indices and D, 2000-2023 (small multiples).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RAW, PROC, RES, W2, MM, SUBCOL, SUBSYSTEMS, PANEL, save_fig, panel_label  # noqa: E402

CC = pd.read_csv(PROC / "ccd_panel.csv")
names = CC.drop_duplicates("iso3").set_index("iso3").country

# ---------------------------------------------------------------- S1 provenance
F = pd.read_csv(PROC / "panel_flags.csv")
cols = [v[0] for v in PANEL.values()] + ["LE", "U5MR", "MMR", "NMR", "STUNT"]
share = F.groupby("iso3")[cols].agg(lambda s: (s != "observed").mean() * 100)
knn = F.groupby("iso3")[cols].agg(lambda s: (s == "kNN-imputed").any())
fig, ax = plt.subplots(figsize=(W2, 165 * MM))
im = ax.imshow(share.values, cmap="Greys", vmin=0, vmax=100, aspect="auto")
for i, j in zip(*np.where(knn.values)):
    ax.text(j, i, "k", ha="center", va="center", fontsize=5, color="#D55E00", fontweight="bold")
ax.set_yticks(range(len(share))); ax.set_yticklabels(share.index.map(names), fontsize=5.2)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=5.6)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("% of 2000–2023 cells not directly observed",
                                                                    fontsize=6)
cb.ax.tick_params(labelsize=5.5)
ax.set_title("Grey shade: interpolated or edge-carried cells; k: series imputed by year-wise kNN", fontsize=6.4)
save_fig(fig, "Figure_S1")

# ---------------------------------------------------------------- S2 network stability and edge accuracy
S = json.load(open(RES / "rq1_summary.json"))
E = pd.read_csv(RES / "rq1_edges.csv").head(30)
fig = plt.figure(figsize=(W2, 105 * MM))
a = fig.add_axes([0.07, 0.14, 0.3, 0.78])
for m, c in [("strength", "#0072B2"), ("bridge", "#D55E00")]:
    st = pd.DataFrame(S["stability"][m])
    x = 100 * (1 - st["drop"])
    a.plot(x, st["mean"], "o-", color=c, ms=3, lw=1.1, label=f"{m.capitalize()} (CS = {S['CS'][m]:.1f})")
    a.fill_between(x, st.q05, st.q95, color=c, alpha=0.15, lw=0)
a.axhline(0.7, color="#999999", lw=0.6, ls="--")
a.invert_xaxis(); a.set_xlabel("% of countries retained"); a.set_ylabel("Correlation with full-sample ranking")
a.legend(frameon=False, fontsize=5.8, loc="lower left")
panel_label(a, "a", x=-0.15)
b = fig.add_axes([0.58, 0.08, 0.4, 0.86])
y = np.arange(len(E))[::-1]
b.hlines(y, E.ci_lo, E.ci_hi, color="#999999", lw=0.8)
b.scatter(E.weight, y, s=10, color=np.where(E.weight > 0, "#2166AC", "#B2182B"), zorder=3)
for yi, inc in zip(y, E.inclusion):
    b.text(0.66, yi, f"{inc:.2f}", fontsize=5.0, va="center")
b.axvline(0, color="#999999", lw=0.6)
b.set_yticks(y); b.set_yticklabels([f"{x1} – {x2}" for x1, x2 in zip(E.label_a, E.label_b)], fontsize=5.2)
b.set_xlabel("Partial correlation (95% bootstrap interval, 1,000 resamples)"); b.set_xlim(-0.5, 0.72)
b.text(0.66, len(E) - 0.2, "incl.", fontsize=5.2, fontweight="bold")
panel_label(b, "b", x=-0.55, y=1.0)
save_fig(fig, "Figure_S2")

# ---------------------------------------------------------------- S3 subsystem maps 2023
geo = gpd.read_file(RAW / "ssa48_boundaries.geojson").to_crs("ESRI:102022")
d23 = CC[CC.year == 2023].set_index("iso3")
fig, axs = plt.subplots(1, 4, figsize=(W2, 62 * MM))
plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.9, bottom=0.14)
for ax, s in zip(axs, SUBSYSTEMS):
    g = geo.copy(); g["v"] = g.iso3.map(d23[f"U_{s}"])
    base = plt.get_cmap("Greys")(np.linspace(0.05, 1, 256))
    col = np.array(plt.matplotlib.colors.to_rgba(SUBCOL[s]))
    cm = ListedColormap(np.array([np.linspace(1, c, 256) for c in col[:3]]).T.tolist())
    g.plot(column="v", ax=ax, cmap=cm, vmin=0, vmax=1, edgecolor="#bbbbbb", linewidth=0.2)
    ax.set_title(f"{s} (mean {d23[f'U_{s}'].mean():.2f})", fontsize=6.6, color=SUBCOL[s], fontweight="bold")
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.85)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1]); cb.ax.tick_params(labelsize=5.4)
save_fig(fig, "Figure_S3")

# ---------------------------------------------------------------- S4 small multiples
order = d23.D.sort_values(ascending=False).index
fig, axs = plt.subplots(8, 6, figsize=(W2, 230 * MM), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.45, wspace=0.08, left=0.05, right=0.99, top=0.96, bottom=0.05)
for ax, iso in zip(axs.ravel(), order):
    g = CC[CC.iso3 == iso]
    for s in SUBSYSTEMS:
        ax.plot(g.year, g[f"U_{s}"], color=SUBCOL[s], lw=0.7)
    ax.plot(g.year, g.D, color="#1d1d1f", lw=1.3)
    ax.set_title(names[iso].replace("Central African Republic", "CAR").replace("São Tomé and Príncipe", "São Tomé"),
                 fontsize=5.6, pad=1.5)
    ax.set_ylim(0, 1); ax.set_xlim(2000, 2023)
    ax.set_xticks([2000, 2010, 2020]); ax.set_xticklabels(["00", "10", "20"], fontsize=5)
    ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=5, length=2)
handles = [plt.Line2D([0], [0], color=SUBCOL[s], lw=1, label=s) for s in SUBSYSTEMS]
handles.append(plt.Line2D([0], [0], color="#1d1d1f", lw=1.5, label="Coordination D"))
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.0))
save_fig(fig, "Figure_S4")
