"""Figure 7 (RQ4). The coordination dividend: balance effects, robustness, heterogeneity and counterfactual gains."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RES, W2, MM, save_fig, panel_label  # noqa: E402

R = pd.read_csv(RES / "robustness.csv")
ME = pd.read_csv(RES / "rq4_marginal_effects.csv")
CF = pd.read_csv(RES / "rq4_counterfactual.csv")
outs = [("LE", "Life expectancy\n(years)", False), ("U5MR", "Under-five mortality\n(% change)", True),
        ("MMR", "Maternal mortality\n(% change)", True), ("STUNT", "Child stunting\n(percentage points)", False)]
checks = list(dict.fromkeys(R.check))
lab = {c: c.replace("Main (entropy weights, classical C, lag 1)", "Main model").replace("R1 ", "").replace(
    "R2 ", "").replace("R3 ", "").replace("R4 ", "").replace("R5 ", "").replace("R6 ", "").replace(
    " weights", " weights").replace("balance = -SD of subsystem indices", "Balance = −SD of indices").replace(
    "excluding countries with imputed series", "Excluding imputed series").replace("modified coupling degree",
                                                                                     "Modified coupling degree")
       for c in checks}
lab = {k: (v[0].upper() + v[1:]).replace("Critic", "CRITIC").replace("Pca", "PCA") for k, v in lab.items()}

fig = plt.figure(figsize=(W2, 150 * MM))
y = np.arange(len(checks))[::-1]
for k, (o, title, pct) in enumerate(outs):
    ax = fig.add_axes([0.2 + k * 0.2, 0.47, 0.17, 0.48])
    r = R[R.outcome == o].set_index("check").loc[checks]
    est, lo, hi = r.coef_C, r.coef_C - 1.96 * r.se_C, r.coef_C + 1.96 * r.se_C
    if pct:
        est, lo, hi = [100 * (np.exp(v) - 1) for v in (est, lo, hi)]
    sig = r.p_C < 0.05
    ax.axvline(0, color="#999999", lw=0.6)
    ax.hlines(y, lo, hi, color=np.where(sig, "#1d1d1f", "#aaaaaa"), lw=0.9)
    ax.scatter(est, y, s=[22 if i == 0 else 12 for i in range(len(y))], color=np.where(sig, "#009E73", "white"),
               edgecolor=np.where(sig, "#009E73", "#888888"), zorder=3, lw=0.7)
    ax.set_title(title, fontsize=6.6)
    ax.set_yticks(y); ax.set_yticklabels([lab[c] for c in checks] if k == 0 else [], fontsize=5.8)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.axhspan(y[0] - 0.5, y[0] + 0.5, color="#EEF6F1", zorder=0)
    for yy in [y[0] - 0.5, y[3] - 0.5, y[6] - 0.5, y[9] - 0.5]:
        ax.axhline(yy, color="#e0e0e0", lw=0.5)
    ax.set_ylim(-0.6, len(y) - 0.4)
    if k == 0:
        panel_label(ax, "a", x=-1.0, y=1.06)
fig.text(0.6, 0.415, "Within-country effect of +1 SD balance (coupling C) at a given development level T; "
         "filled = p < 0.05; bars = 95% CI (Driscoll–Kraay)", ha="center", fontsize=5.8, color="#444444")

# b. heterogeneity: marginal effect of balance across development
b = fig.add_axes([0.08, 0.07, 0.36, 0.27])
for o, col, nm in [("MMR", "#CC79A7", "Maternal mortality (ln)"), ("U5MR", "#0072B2", "Under-five mortality (ln)")]:
    m = ME[ME.outcome == o]
    b.plot(m["T"], m.effect, color=col, lw=1.3, label=nm)
    b.fill_between(m["T"], m.lo, m.hi, color=col, alpha=0.15, lw=0)
b.axhline(0, color="#999999", lw=0.6)
b.set_xlabel("Comprehensive development T"); b.set_ylabel("Effect of +1 SD balance (log points)")
b.legend(frameon=False, fontsize=5.8, loc="upper left")
b.set_xlim(0.2, 0.75)
panel_label(b, "b", x=-0.12, y=1.04)

# c. counterfactual dividend by country
c = fig.add_axes([0.58, 0.07, 0.4, 0.27])
top = CF.sort_values("dLE", ascending=False).head(14)
xx = np.arange(len(top))
c.bar(xx, top.dLE, color="#009E73", width=0.65, label="Life expectancy gain (years)")
c.set_xticks(xx); c.set_xticklabels(top.country.str.replace("Central African Republic", "CAR").str.replace(
    "Congo, Dem. Rep.", "DR Congo").str.replace("South Sudan", "S. Sudan"), rotation=45, ha="right", fontsize=5.6)
c.set_ylabel("Life-expectancy gain (years)")
c2 = c.twinx()
c2.plot(xx, -top.dU5MR_pct, "o", color="#0072B2", ms=3.2, label="Under-five mortality reduction (%)")
c2.set_ylabel("Under-five mortality reduction (%)", color="#0072B2"); c2.tick_params(axis="y", colors="#0072B2")
c2.spines["right"].set_visible(True); c2.spines["right"].set_color("#0072B2")
c2.set_ylim(0, None)
h1, l1 = c.get_legend_handles_labels(); h2, l2 = c2.get_legend_handles_labels()
c.legend(h1 + h2, l1 + l2, frameon=False, fontsize=5.6, loc="upper right")
panel_label(c, "c", x=-0.1, y=1.04)
save_fig(fig, "Figure_7")
