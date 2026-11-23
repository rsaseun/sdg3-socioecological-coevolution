"""Step 4 (RQ2). Lead-lag co-evolution of socioecological conditions and health, 2000-2023.

A. Regularised first-difference panel VAR (lead-lag network).
   Indicators are averaged over eight 3-year periods (2000-02 ... 2021-23), which damps interpolation artefacts;
   period-to-period changes (Delta) are standardised and demeaned by period (period fixed effects). Each change
   is regressed on all lagged changes with the lasso (penalty by 5-fold cross-validation grouped by country).
   A directed edge i -> j means that earlier change in i predicts later change in j, net of all other lagged
   changes. Country-block bootstrap (500 resamples of countries) gives inclusion rates and 95% intervals; edges
   retained in at least 90% of resamples are called stable.
   The contemporaneous network is the graphical lasso of the VAR residuals.
B. Dumitrescu-Hurlin (2012) panel Granger non-causality tests on annual first differences of the four subsystem
   indices, D and the outcomes (lags 1 and 2), in both directions.
Outputs: outputs/results/rq2_*.csv|json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import PROC, RES, PANEL, SEED, PERIODS, SUBSYSTEMS  # noqa: E402
from sehs.methods import cv1se_penalty, glasso_net, normal_scores, dumitrescu_hurlin, ips_like  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
P = pd.read_csv(PROC / "panel.csv")
CC = pd.read_csv(PROC / "ccd_panel.csv")

NODES = ["GDP", "TFR", "PoU", "FLFP", "GE", "CC", "PV", "WAT", "SAN", "FUEL", "ELEC", "CHE", "OOP", "DTP3",
         "LE", "U5MR", "MMR"]
SIGN = {"TFR": -1, "PoU": -1, "OOP": -1, "U5MR": -1, "MMR": -1}  # orient: increase = improvement
GROUP = {v[0]: v[2] for v in PANEL.values()}
GROUP.update({"LE": "Outcome", "U5MR": "Outcome", "MMR": "Outcome"})
LABEL = {v[0]: v[1].replace(" (log)", "") for v in PANEL.values()}
LABEL.update({"LE": "Life expectancy", "U5MR": "Child survival", "MMR": "Maternal survival",
              "TFR": "Fertility decline", "PoU": "Less undernourishment", "OOP": "Lower out-of-pocket share"})

# ------------------------------------------------------------------ A. period panel
P["period"] = np.searchsorted([b for a, b in PERIODS], P.year, side="left")
per = P.groupby(["iso3", "period"])[NODES].mean()
for c, s in SIGN.items():
    per[c] = s * per[c]
d = per.groupby(level=0).diff()                      # change between consecutive periods
d = (d - d.mean()) / d.std()
lagd = d.groupby(level=0).shift(1)
frame = pd.concat({"y": d, "x": lagd}, axis=1).dropna()
frame = frame.groupby(level=1).transform(lambda v: v - v.mean())    # period fixed effects
Y, X = frame["y"][NODES].values, frame["x"][NODES].values
groups = frame.index.get_level_values(0).values
p = len(NODES)

B = np.zeros((p, p))          # B[j, i]: lagged change in i -> change in j
alphas = np.zeros(p)
resid = np.zeros_like(Y)
folds = list(GroupKFold(n_splits=5).split(X, Y[:, 0], groups))   # country-grouped folds
for j in range(p):
    m = LassoCV(cv=folds, alphas=60, max_iter=20000).fit(X, Y[:, j])
    B[j], alphas[j] = m.coef_, m.alpha_
    resid[:, j] = Y[:, j] - m.predict(X)

# country-block bootstrap
isos = np.unique(groups)
NB = 500
bootB = np.zeros((NB, p, p))
for b in range(NB):
    pick = rng.choice(isos, len(isos), replace=True)
    rows = np.concatenate([np.where(groups == g)[0] for g in pick])
    for j in range(p):
        bootB[b, j] = Lasso(alpha=alphas[j], max_iter=20000).fit(X[rows], Y[rows, j]).coef_
incl = (np.abs(bootB) > 1e-10).mean(0)
lo, hi = np.percentile(bootB, [2.5, 97.5], axis=0)

edges = []
for j in range(p):
    for i in range(p):
        if abs(B[j, i]) > 1e-10:
            edges.append(dict(source=NODES[i], target=NODES[j], source_label=LABEL[NODES[i]],
                              target_label=LABEL[NODES[j]], source_group=GROUP[NODES[i]],
                              target_group=GROUP[NODES[j]], beta=B[j, i], ci_lo=lo[j, i], ci_hi=hi[j, i],
                              inclusion=incl[j, i], autoregressive=i == j))
E = pd.DataFrame(edges).sort_values("beta", key=np.abs, ascending=False)
E.to_csv(RES / "rq2_temporal_edges.csv", index=False)
off = ~np.eye(p, dtype=bool)
robust = (incl >= 0.9) & off & (np.abs(B) > 1e-10)
cent = pd.DataFrame({"label": [LABEL[n] for n in NODES], "group": [GROUP[n] for n in NODES],
                     "out_strength": np.where(off, np.abs(B), 0).sum(0),
                     "in_strength": np.where(off, np.abs(B), 0).sum(1),
                     "out_expected_influence": np.where(off, B, 0).sum(0),
                     "out_strength_robust": np.where(robust, np.abs(B), 0).sum(0),
                     "autoregressive": np.diag(B)}, index=NODES)
cent["net_driver"] = cent.out_strength - cent.in_strength
cent.to_csv(RES / "rq2_temporal_centrality.csv")

# subsystem-level lead-lag matrix (sum of |beta| from group to group, robust edges only)
grp = [GROUP[n] for n in NODES]
gnames = SUBSYSTEMS + ["Outcome"]
G = pd.DataFrame(0.0, index=gnames, columns=gnames)  # rows = source, cols = target
for j in range(p):
    for i in range(p):
        if robust[j, i]:
            G.loc[grp[i], grp[j]] += abs(B[j, i])
G.to_csv(RES / "rq2_group_leadlag.csv")

# contemporaneous network of residuals
Zr = normal_scores(resid)
lam_c, _ = cv1se_penalty(Zr, SEED)
PCc, _, r2c = glasso_net(Zr, lam_c)
pd.DataFrame(PCc, index=NODES, columns=NODES).round(4).to_csv(RES / "rq2_contemporaneous_pcor.csv")

# ------------------------------------------------------------------ B. Dumitrescu-Hurlin tests (annual differences)
A = CC[["iso3", "year", "U_Social", "U_Institutional", "U_Environmental", "U_Health system", "D"]].merge(
    P[["iso3", "year", "LE", "U5MR", "MMR"]], on=["iso3", "year"]).sort_values(["iso3", "year"])
A["U5MR"], A["MMR"] = -A["U5MR"], -A["MMR"]   # survival orientation
drivers = ["U_Social", "U_Institutional", "U_Environmental", "U_Health system", "D"]
outs = ["LE", "U5MR", "MMR"]
dA = A.copy()
for c in drivers + outs:
    dA[c] = A.groupby("iso3")[c].diff()
dA = dA.dropna()
dh = []
for K in (1, 2):
    for x in drivers:
        for y in outs:
            dh.append(dict(direction="driver->outcome", x=x, y=y, **dumitrescu_hurlin(dA, y, x, K=K)))
            dh.append(dict(direction="outcome->driver", x=y, y=x, **dumitrescu_hurlin(dA, x, y, K=K)))
DH = pd.DataFrame(dh)
DH.to_csv(RES / "rq2_dumitrescu_hurlin.csv", index=False)
ur = pd.DataFrame([dict(var=c, level_mean_adf=ips_like(A, c)[0], level_share_reject=ips_like(A, c)[1],
                        diff_mean_adf=ips_like(dA, c)[0], diff_share_reject=ips_like(dA, c)[1])
                   for c in drivers + outs])
ur.to_csv(RES / "rq2_unitroot_summary.csv", index=False)

summary = dict(n_obs=int(len(frame)), n_countries=int(len(isos)), n_nodes=p,
               n_edges=int((np.abs(B[off]) > 1e-10).sum()), n_robust_edges=int(robust.sum()),
               n_cross_group_robust=int(sum(robust[j, i] and grp[i] != grp[j] for j in range(p) for i in range(p))),
               top_drivers=cent.sort_values("out_strength_robust", ascending=False).head(6).label.tolist(),
               contemporaneous_edges=int((np.abs(np.triu(PCc, 1)) > 0).sum()),
               var_r2=dict(zip(NODES, [float(1 - resid[:, j].var() / Y[:, j].var()) for j in range(p)])),
               dh_significant_K1=DH.query("K==1 and p_Ztilde<0.05")[["direction", "x", "y", "Ztilde"]]
               .round(2).values.tolist())
json.dump(summary, open(RES / "rq2_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "var_r2"}, indent=1))
print(cent.round(3).sort_values("out_strength_robust", ascending=False).to_string())
print(E[(E.inclusion >= 0.5) & (~E.autoregressive)].round(3).head(25).to_string())
print(G.round(3))
print(DH[DH.K == 1].round(3).to_string())
