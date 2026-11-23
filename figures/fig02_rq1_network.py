"""Figure 2 (RQ1). Cross-level socioecological network, bridge conditions and module recovery."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RES, W2, MM, LEVCOL, SEED, save_fig, panel_label  # noqa: E402
from sehs.methods import modularity_of  # noqa: E402

W = pd.read_csv(RES / "rq1_network_pcor.csv", index_col=0)
cent = pd.read_csv(RES / "rq1_centrality.csv", index_col=0)
E = pd.read_csv(RES / "rq1_edges.csv")
S = json.load(open(RES / "rq1_summary.json"))
nodes = list(W.index)

G = nx.Graph()
G.add_nodes_from(nodes)
for _, e in E.iterrows():
    G.add_edge(e.a, e.b, weight=abs(e.weight), w=e.weight, inc=e.inclusion)
# layout: modules on a ring; members on a small circle around each module centre
mods = cent.module
ms = sorted(mods.unique())
pos, mcent, mrad = {}, {}, {}
for k, m in enumerate(ms):
    th = np.pi / 2 - 2 * np.pi * k / len(ms)
    c = np.array([np.cos(th), np.sin(th)]) * 1.42
    mem = list(mods.index[mods == m])
    r = 0.13 + 0.035 * len(mem)
    mcent[m], mrad[m] = c, r
    for i, n in enumerate(mem):
        a2 = th + np.pi + 2 * np.pi * (i + 0.5) / len(mem)   # start facing the ring centre
        pos[n] = c + r * np.array([np.cos(a2), np.sin(a2)])
fig = plt.figure(figsize=(W2, 118 * MM))
ax = fig.add_axes([0.0, 0.03, 0.56, 0.93])
ax.set_aspect("equal"); ax.axis("off")
for m in ms:
    ax.add_patch(plt.Circle(mcent[m], mrad[m] + 0.11, fc="#f3f3f3", ec="#c4c4c4", lw=0.6, ls="--", zorder=0))
    ax.text(*mcent[m], f"M{m}", ha="center", va="center", fontsize=7.2, color="#555555", fontweight="bold",
            zorder=5, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="#c4c4c4", lw=0.5))
for u, v, d in G.edges(data=True):
    x = [pos[u][0], pos[v][0]]; y = [pos[u][1], pos[v][1]]
    ax.plot(x, y, color="#2166AC" if d["w"] > 0 else "#B2182B", lw=0.4 + 5.5 * abs(d["w"]),
            alpha=0.25 + 0.6 * d["inc"] ** 2, zorder=1, solid_capstyle="round")
for n in nodes:
    lev = cent.loc[n, "level"]
    r = 0.035 + 0.04 * cent.loc[n, "predictability"]
    ax.add_patch(plt.Circle(pos[n], r, fc=LEVCOL[lev], ec="white", lw=0.8, zorder=3))
    m = mods[n]
    v = pos[n] - mcent[m]
    v = v / np.linalg.norm(v)
    tp = pos[n] + v * (r + 0.025)
    ha = "left" if v[0] > 0.25 else ("right" if v[0] < -0.25 else "center")
    va = "bottom" if v[1] > 0.5 else ("top" if v[1] < -0.5 else "center")
    ax.text(tp[0], tp[1], cent.loc[n, "label"].replace(" by women", "\nby women").replace("UHC service coverage", "UHC service\ncoverage"),
            ha=ha, va=va, fontsize=5.3, zorder=4, color="#1d1d1f", linespacing=0.95,
            bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.2))
xs = np.array(list(pos.values()))
ax.set_xlim(xs[:, 0].min() - 0.42, xs[:, 0].max() + 0.42); ax.set_ylim(xs[:, 1].min() - 0.3, xs[:, 1].max() + 0.3)
fig.text(0.005, 0.975, "a", fontsize=8, fontweight="bold")
leg = [Patch(fc=LEVCOL[k], label=k) for k in ["Household", "Community", "Health system", "Governance", "Ecological",
                                             "Outcome"]]
leg += [Line2D([0], [0], color="#2166AC", lw=2, label="Positive (synergy)"),
        Line2D([0], [0], color="#B2182B", lw=2, label="Negative")]
ax.legend(handles=leg, loc="lower left", ncol=4, frameon=False, fontsize=5.6, handlelength=1.2,
          bbox_to_anchor=(0.0, -0.06), columnspacing=0.9)
ax.text(0.99, -0.02, f"{S['n_edges']} edges, {100 * S['share_cross']:.0f}% across levels\nnode size = predictability"
        "\nM1–M6 = empirical modules", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color="#444444")

# ---------------------------------------------------------------- b. bridge strength
bx = fig.add_axes([0.70, 0.36, 0.29, 0.62])
c = cent.sort_values("bridge")
yy = np.arange(len(c))
bx.barh(yy, c.bridge, color=[LEVCOL[l] for l in c.level], height=0.7)
bx.errorbar(c.bridge, yy, xerr=[c.bridge - c.bridge_lo, c.bridge_hi - c.bridge], fmt="none", ecolor="#333333",
            lw=0.5, capsize=0)
bx.scatter(c.predictability, yy, marker="D", s=7, color="white", edgecolor="#1d1d1f", lw=0.5, zorder=3,
           label="Predictability (R²)")
bx.set_yticks(yy); bx.set_yticklabels(c.label, fontsize=5.6)
bx.set_xlabel("Bridge strength (95% bootstrap CI)")
bx.legend(loc="lower right", frameon=False, fontsize=5.6, handletextpad=0.2)
bx.set_xlim(0, max(c.bridge_hi.max(), 1.5))
fig.text(0.585, 0.975, "b", fontsize=8, fontweight="bold")

# ---------------------------------------------------------------- c. module recovery
cx = fig.add_axes([0.70, 0.07, 0.29, 0.19])
rng = np.random.default_rng(SEED)
Gp = nx.Graph()
Gp.add_nodes_from(range(len(nodes)))
Wv = W.values
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        if Wv[i, j] > 0:
            Gp.add_edge(i, j, weight=Wv[i, j])
theo = pd.factorize(cent.level.values)[0]
null = [modularity_of(Gp, rng.permutation(theo)) for _ in range(2000)]
cx.hist(null, bins=40, color="#cccccc", edgecolor="white", lw=0.3)
cx.axvline(S["modularity_theoretical"], color="#D55E00", lw=1.1)
cx.axvline(S["modularity_empirical"], color="#1d1d1f", lw=1.1, ls="--")
cx.text(S["modularity_theoretical"], cx.get_ylim()[1] * 0.95, " theoretical\n levels", color="#D55E00", fontsize=5.5,
        va="top")
cx.text(S["modularity_empirical"], cx.get_ylim()[1] * 0.95, "empirical \nmodules ", color="#1d1d1f", fontsize=5.5,
        va="top", ha="right")
cx.set_xlabel("Modularity Q (grey: 2,000 random partitions)")
cx.set_yticks([])
cx.spines["left"].set_visible(False)
cx.text(0.0, -0.62, f"ARI(levels, modules) = {S['ari_levels_vs_modules']:.2f} (p = {S['p_ari']:.2f})",
        transform=cx.transAxes, fontsize=5.8)
fig.text(0.585, 0.255, "c", fontsize=8, fontweight="bold")
save_fig(fig, "Figure_2")
