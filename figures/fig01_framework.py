"""Figure 1. Conceptual framework and study design (four research questions)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import W2, MM, save_fig, LEVCOL, SUBCOL  # noqa: E402

fig = plt.figure(figsize=(W2, 112 * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 190); ax.set_ylim(0, 112); ax.axis("off")


def box(x, y, w, h, fc, ec, title, body=None, tsize=7.2, bsize=6.2, lw=0.9, tcol=None, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold",
                color=tcol or ec)
    else:
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=tsize, fontweight="bold",
                color=tcol or ec)
        ax.text(x + w / 2, y + (h - 4.2) / 2, body, ha="center", va="center", fontsize=bsize, linespacing=1.3,
                color="#222222")


def arrow(p0, p1, c="#444444", lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"{style},head_length=3.2,head_width=1.6", color=c, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def tag(x, y, s, c="#1d1d1f"):
    ax.add_patch(FancyBboxPatch((x - 5, y - 2.3), 10, 4.6, boxstyle="round,pad=0,rounding_size=2.2", fc=c,
                                ec="none"))
    ax.text(x, y, s, ha="center", va="center", fontsize=6.4, color="white", fontweight="bold")


# ---------------------------------------------------------------- A. nested socioecological levels
ax.text(2, 108, "Socioecological levels (cross-section, 2019–2023)", fontsize=7.4, fontweight="bold", va="center")
cx, cy = 33, 72
levels = [("Ecological", 30, 30), ("Governance", 24.5, 24.5), ("Health system", 19, 19), ("Community", 13.5, 13.5),
          ("Household", 8.2, 8.2)]
for name, rx, ry in levels:
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry * 0.98, fc=LEVCOL[name], alpha=0.17, ec=LEVCOL[name], lw=0.9))
for name, rx, ry in levels[:-1]:
    ax.text(cx, cy + ry - 2.8, name, ha="center", va="center", fontsize=6.3, color=LEVCOL[name], fontweight="bold")
ax.text(cx, cy + 1.2, "Household", ha="center", va="center", fontsize=6.3, color=LEVCOL["Household"], fontweight="bold")
ax.text(cx, cy - 2.4, "SDG 3\noutcomes", ha="center", va="center", fontsize=5.6, color="#222222", linespacing=1.0)
ax.text(cx, cy - 34.5, "21 determinants + 3 outcomes, 48 countries", ha="center", va="center", fontsize=6.0,
        color="#444444", style="italic")
tag(cx, 34, "RQ1", "#0072B2")
ax.text(cx, 28.7, "How are the levels connected,\nand do they form real modules?", ha="center", va="center",
        fontsize=6.1, linespacing=1.25)

# ---------------------------------------------------------------- B. four subsystems over time
ax.text(70, 108, "Co-evolving subsystems (panel, 2000–2023)", fontsize=7.4, fontweight="bold", va="center")
subs = [("Social", "income, fertility, nutrition,\nwomen's work"), ("Institutional", "six governance\ndimensions"),
        ("Environmental", "water, sanitation,\nclean fuel, electricity"),
        ("Health system", "spending, priority,\nprotection, immunisation")]
ys = [91, 77, 63, 49]
for (s, b), y in zip(subs, ys):
    box(70, y, 36, 11.5, "white", SUBCOL[s], s, b, tsize=6.8, bsize=5.7)
# lead-lag dynamics among subsystems
arrow((65.0, 55), (65.0, 97), c="#888888", lw=0.9, style="<|-|>")
ax.text(67.5, 76, "lead\u2013lag dynamics", rotation=90, ha="center", va="center", fontsize=6.0, color="#666666")
tag(88, 42.5, "RQ2", "#CC79A7")
ax.text(88, 36.3, "Which subsystems lead and\nwhich follow over time?", ha="center", va="center", fontsize=6.1,
        linespacing=1.25)

# ---------------------------------------------------------------- C. coupling coordination
box(116, 62, 30, 28, "#F4F6FB", "#1d1d1f", "Coupling coordination",
    "C = balance\nT = development\nD = √(C × T)\n5 classes, lagging\nsubsystem", tsize=6.8, bsize=6.0)
for y in ys:
    arrow((106, y + 5.7), (116, 76), c=SUBCOL[subs[ys.index(y)][0]], lw=1.0)
tag(131, 55.5, "RQ3", "#D55E00")
ax.text(131, 49.2, "Has coordination improved,\nconverged, and is it\nspatially contagious?", ha="center", va="center",
        fontsize=6.1, linespacing=1.25)

# ---------------------------------------------------------------- D. health and well-being
box(155, 62, 32, 28, "#EEF6F1", "#009E73", "Health and well-being",
    "life expectancy\nunder-five mortality\nmaternal mortality\nchild stunting", tsize=6.8, bsize=6.0)
arrow((146, 76), (155, 76), c="#1d1d1f", lw=1.4)
tag(171, 55.5, "RQ4", "#009E73")
ax.text(171, 49.2, "Is there a dividend of\nbalanced development\nbeyond its level?", ha="center", va="center",
        fontsize=6.1, linespacing=1.25)

# ---------------------------------------------------------------- E. methods strip
ax.plot([2, 188], [22.5, 22.5], color="#bbbbbb", lw=0.6)
ax.text(13, 15.5, "Methods", fontsize=7.0, fontweight="bold", va="center", alpha=0)
meth = [("#0072B2", "RQ1", "Nonparanormal graphical lasso; bridge\ncentrality; predictability; module recovery\n(ARI, modularity null); bootstrap stability"),
        ("#CC79A7", "RQ2", "Regularised first-difference panel VAR\n(lead–lag network, country-block bootstrap);\nDumitrescu–Hurlin panel Granger tests"),
        ("#D55E00", "RQ3", "Entropy-weighted coupling coordination;\nσ- and β-convergence; Markov and spatial\nMarkov chains (Rey homogeneity tests)"),
        ("#009E73", "RQ4", "Two-way FE with Driscoll–Kraay errors;\nwithin–between (Mundlak) model;\ncounterfactual dividend; obstacle degrees")]
for i, (c, t, s) in enumerate(meth):
    x = 2 + i * 47.5
    tag(x + 5, 15.5, t, c)
    ax.text(x, 11.6, s, fontsize=5.7, va="top", linespacing=1.3)
ax.text(95, 1.4, "Data: World Bank WDI and Worldwide Governance Indicators (48 countries × 24 years, 20 indicators "
        "+ 5 outcomes); WHO, UN IGME, JMP and other sources for the cross-section", ha="center", fontsize=5.8,
        color="#444444", style="italic")
save_fig(fig, "Figure_1")
