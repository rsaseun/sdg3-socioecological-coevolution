"""Figure 3 (RQ2). Lead-lag co-evolution: temporal network, drivers and panel Granger tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RES, W2, MM, SUBCOL, SUBSYSTEMS, save_fig  # noqa: E402

E = pd.read_csv(RES / "rq2_temporal_edges.csv")
C = pd.read_csv(RES / "rq2_temporal_centrality.csv", index_col=0)
DH = pd.read_csv(RES / "rq2_dumitrescu_hurlin.csv")
order = []
for g in SUBSYSTEMS + ["Outcome"]:
    order += list(C.index[C.group == g])
n = len(order)
ang = {v: np.pi / 2 - 2 * np.pi * (i + 0.5) / n for i, v in enumerate(order)}
pos = {v: np.array([np.cos(a), np.sin(a)]) for v, a in ang.items()}
short = {"GDP": "GDP", "TFR": "Fertility\ndecline", "PoU": "Less under-\nnourishment", "FLFP": "Female\nlabour",
         "GE": "Gov.\neffectiveness", "CC": "Control of\ncorruption", "PV": "Political\nstability",
         "WAT": "Basic\nwater", "SAN": "Basic\nsanitation", "FUEL": "Clean\nfuel", "ELEC": "Electricity",
         "CHE": "Health\nspending", "OOP": "Lower OOP\nshare", "DTP3": "DTP3", "LE": "Life\nexpectancy",
         "U5MR": "Child\nsurvival", "MMR": "Maternal\nsurvival"}

fig = plt.figure(figsize=(W2, 100 * MM))
ax = fig.add_axes([0.0, 0.02, 0.42, 0.92])
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.5, 1.5)
off = E[~E.autoregressive]
for _, e in off[off.inclusion < 0.9].iterrows():
    ax.add_patch(FancyArrowPatch(pos[e.source] * 0.93, pos[e.target] * 0.93, arrowstyle="-", color="#d9d9d9",
                                 lw=0.35, connectionstyle="arc3,rad=0.18", zorder=1))
for _, e in off[off.inclusion >= 0.9].iterrows():
    ax.add_patch(FancyArrowPatch(pos[e.source] * 0.93, pos[e.target] * 0.93,
                                 arrowstyle="-|>,head_length=4,head_width=2.2",
                                 color="#2166AC" if e.beta > 0 else "#B2182B", lw=0.8 + 9 * abs(e.beta),
                                 connectionstyle="arc3,rad=0.18", zorder=2, shrinkA=6, shrinkB=7))
for v in order:
    g = C.loc[v, "group"]
    s = 40 + 900 * C.loc[v, "out_strength"]
    ax.scatter(*pos[v], s=s, color=SUBCOL[g], edgecolor="white", lw=0.8, zorder=3)
    tp = pos[v] * (1.24 + 0.12 * C.loc[v, "out_strength"] / C.out_strength.max())
    ax.text(*tp, short[v], ha="center", va="center", fontsize=5.4, linespacing=0.95)
fig.text(0.005, 0.955, "a", fontsize=8, fontweight="bold")
leg = [Patch(fc=SUBCOL[k], label=k) for k in SUBSYSTEMS + ["Outcome"]]
leg += [Line2D([0], [0], color="#2166AC", lw=1.6, label="Stable +"), Line2D([0], [0], color="#B2182B", lw=1.6,
                                                                           label="Stable −"),
        Line2D([0], [0], color="#d9d9d9", lw=1.0, label="Unstable")]
ax.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, -0.075),
          handlelength=1.1, columnspacing=0.8)

# ---------------------------------------------------------------- b. drivers vs responders
bx = fig.add_axes([0.53, 0.40, 0.20, 0.55])
c = C.sort_values("net_driver")
yy = np.arange(len(c))
bx.barh(yy, c.out_strength, color=[SUBCOL[g] for g in c.group], height=0.7, label="Out (drives)")
bx.barh(yy, -c.in_strength, color=[SUBCOL[g] for g in c.group], height=0.7, alpha=0.4, label="In (responds)")
bx.scatter(c.out_strength_robust, yy, marker="|", s=40, color="#1d1d1f", lw=1.0, zorder=3)
bx.axvline(0, color="#1d1d1f", lw=0.6)
bx.set_yticks(yy); bx.set_yticklabels([s.replace("-\n", "").replace("\n", " ") for s in c.index.map(short)], fontsize=5.6)
bx.set_xlabel("← in-strength   out-strength →")
bx.set_xlim(-0.6, 0.45)
bx.text(0.98, 0.02, "| = stable edges only", transform=bx.transAxes, ha="right", fontsize=5.3, color="#444444")
fig.text(0.44, 0.955, "b", fontsize=8, fontweight="bold")

# ---------------------------------------------------------------- c. Dumitrescu-Hurlin tests
cx = fig.add_axes([0.80, 0.40, 0.19, 0.50])
drv = ["U_Social", "U_Institutional", "U_Environmental", "U_Health system", "D"]
outs = ["LE", "U5MR", "MMR"]
k1 = DH[DH.K == 1]
M = np.zeros((len(drv), 6)); Pm = np.ones_like(M)
for i, x in enumerate(drv):
    for j, y in enumerate(outs):
        r = k1[(k1.x == x) & (k1.y == y)].iloc[0]; M[i, j], Pm[i, j] = r.Ztilde, r.p_Ztilde
        r = k1[(k1.x == y) & (k1.y == x)].iloc[0]; M[i, 3 + j], Pm[i, 3 + j] = r.Ztilde, r.p_Ztilde
im = cx.imshow(M, cmap="Purples", vmin=0, vmax=8, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        st = "***" if Pm[i, j] < 0.001 else "**" if Pm[i, j] < 0.01 else "*" if Pm[i, j] < 0.05 else ""
        cx.text(j, i, f"{M[i, j]:.1f}{st}", ha="center", va="center", fontsize=5.0,
                color="white" if M[i, j] > 5 else "#1d1d1f")
cx.set_yticks(range(len(drv)))
cx.set_yticklabels(["Social", "Institutional", "Environmental", "Health system", "Coordination D"], fontsize=5.8)
cx.set_xticks(range(6)); cx.set_xticklabels(["LE", "U5", "MM", "LE", "U5", "MM"], fontsize=5.8)
cx.axvline(2.5, color="white", lw=2)
cx.text(1, -0.85, "subsystem →\noutcome", ha="center", fontsize=5.6, va="bottom")
cx.text(4, -0.85, "outcome →\nsubsystem", ha="center", fontsize=5.6, va="bottom")
cb = fig.colorbar(im, ax=cx, orientation="horizontal", fraction=0.06, pad=0.1)
cb.set_label("Dumitrescu–Hurlin Z̃ (lag 1)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
for sp in cx.spines.values():
    sp.set_visible(False)
fig.text(0.75, 0.955, "c", fontsize=8, fontweight="bold")

# ---------------------------------------------------------------- d. subsystem lead-lag matrix (stable edges)
G = pd.read_csv(RES / "rq2_group_leadlag.csv", index_col=0)
dx = fig.add_axes([0.53, 0.06, 0.20, 0.22])
gl = ["Social", "Institutional", "Environmental", "Health system", "Outcome"]
im2 = dx.imshow(G.loc[gl, gl].values, cmap="Greys", vmin=0, vmax=0.4)
for i in range(5):
    for j in range(5):
        v = G.loc[gl[i], gl[j]]
        if v > 0:
            dx.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2, color="white" if v > 0.25 else "#1d1d1f")
lab = ["Soc", "Inst", "Env", "Hlth", "Out"]
dx.set_xticks(range(5)); dx.set_xticklabels(lab, fontsize=5.6); dx.set_yticks(range(5)); dx.set_yticklabels(lab, fontsize=5.6)
dx.set_xlabel("responds (t)", fontsize=6); dx.set_ylabel("leads (t−1)", fontsize=6)
for sp in dx.spines.values():
    sp.set_visible(False)
fig.text(0.44, 0.29, "d", fontsize=8, fontweight="bold")
ex = fig.add_axes([0.78, 0.06, 0.21, 0.22]); ex.axis("off")
ex.text(0, 1.0, "Reading the panels", fontsize=6.2, fontweight="bold", va="top")
ex.text(0, 0.82, "a: arrows = lagged change in one condition\n   predicting later change in another\n   (3-year periods; "
        "stable = kept in ≥90%\n   of 500 country-block bootstraps);\n   node size = out-strength.\n"
        "d: summed |β| of stable edges between\n   subsystems.  *p<0.05 **p<0.01 ***p<0.001",
        fontsize=5.5, va="top", linespacing=1.3)
save_fig(fig, "Figure_3")
