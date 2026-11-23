"""Step 2 (RQ1). Cross-level structure of the socioecological system (latest year, 48 countries).

* 21 determinants at five levels + 3 SDG 3 outcomes, all oriented so that higher = more favourable.
* Nonparanormal graphical lasso; penalty by 10x repeated 5-fold cross-validation with the one-standard-error
  rule; partial correlations below |0.05| treated as absent.
* Node strength, bridge strength (links to other levels) and predictability (node R^2).
* Do the theoretical levels exist as empirical modules? Louvain modules on positive edges, adjusted Rand index
  with the five theoretical levels, and modularity of the theoretical partition against 2,000 random partitions
  of equal sizes.
* Accuracy and stability: 1,000 nonparametric bootstrap edge intervals and inclusion rates; case-dropping
  correlation-stability (CS) coefficients for strength and bridge strength.
Outputs: outputs/results/rq1_*.csv|json
"""
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RAW, RES, LEVELS, XOUT, SEED  # noqa: E402
from sehs.methods import normal_scores, cv1se_penalty, glasso_net, bridge_strength, louvain_modules, modularity_of  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
cs = pd.read_csv(RAW / "cross_section_48.csv").set_index("iso3")
cs["lnHEPC"] = np.log(cs.pop("HEPC"))
nodes, level, sign, label = [], [], [], []
for lev, dd in LEVELS.items():
    for c, (lab, s) in dd.items():
        nodes.append(c); level.append(lev); sign.append(s); label.append(lab)
for c, lab in XOUT.items():
    nodes.append(c); level.append("Outcome"); sign.append(+1 if c == "SCI" else -1)
    label.append({"U5MR": "Child survival", "MMR": "Maternal survival", "SCI": "UHC service coverage"}[c])
X = cs[nodes].copy()
X["U5MR"], X["MMR"] = np.log(X["U5MR"]), np.log(X["MMR"])
miss = int(X.isna().sum().sum())
X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X), index=X.index, columns=nodes)
X = X * np.array(sign)  # orient: higher = more favourable
Z = normal_scores(X.values)
p = len(nodes)

lam, lam_best = cv1se_penalty(Z, SEED)
PC, prec, r2 = glasso_net(Z, lam)
W = pd.DataFrame(PC, index=nodes, columns=nodes)
W.round(4).to_csv(RES / "rq1_network_pcor.csv")
lev = np.array(level)
cent = pd.DataFrame({"label": label, "level": level, "strength": np.abs(PC).sum(0),
                     "expected_influence": PC.sum(0), "bridge": bridge_strength(PC, lev),
                     "predictability": r2, "degree": (np.abs(PC) > 0).sum(0)}, index=nodes)

edges = [(nodes[i], nodes[j], PC[i, j]) for i in range(p) for j in range(i + 1, p) if PC[i, j] != 0]
n_cross = sum(lev[nodes.index(a)] != lev[nodes.index(b)] for a, b, _ in edges)

# ---- module recovery: are the theoretical levels empirical communities?
det = lev != "Outcome"
mod_lab, Q_emp, G = louvain_modules(PC, seed=SEED)
cent["module"] = mod_lab + 1
ari = adjusted_rand_score(lev[det], mod_lab[det])
theo = pd.factorize(lev)[0]
Q_theo = modularity_of(G, theo)
null = []
for _ in range(2000):
    null.append(modularity_of(G, rng.permutation(theo)))
null = np.array(null)
p_theo = float((np.sum(null >= Q_theo) + 1) / (len(null) + 1))
# ARI null for the empirical-vs-theoretical agreement
ari_null = np.array([adjusted_rand_score(rng.permutation(lev[det]), mod_lab[det]) for _ in range(2000)])
p_ari = float((np.sum(ari_null >= ari) + 1) / (len(ari_null) + 1))

# ---- accuracy: nonparametric bootstrap of edges (penalty fixed at the full-sample EBIC value)
B = 1000
boot = np.zeros((B, p, p))
for b in range(B):
    idx = rng.integers(0, len(Z), len(Z))
    Zb = normal_scores(X.values[idx] + rng.normal(0, 1e-9, (len(idx), p)))
    boot[b] = glasso_net(Zb, lam)[0]
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
incl = (np.abs(boot) > 0).mean(0)
etab = pd.DataFrame([dict(a=a, b=b, label_a=label[nodes.index(a)], label_b=label[nodes.index(b)],
                          level_a=lev[nodes.index(a)], level_b=lev[nodes.index(b)], weight=w,
                          ci_lo=lo[nodes.index(a), nodes.index(b)], ci_hi=hi[nodes.index(a), nodes.index(b)],
                          inclusion=incl[nodes.index(a), nodes.index(b)],
                          cross_level=lev[nodes.index(a)] != lev[nodes.index(b)]) for a, b, w in edges])
etab = etab.reindex(etab.weight.abs().sort_values(ascending=False).index)
etab.to_csv(RES / "rq1_edges.csv", index=False)
# bootstrapped strength / bridge intervals
bs_str = np.abs(boot).sum(1)
bs_bri = np.array([bridge_strength(boot[b], lev) for b in range(B)])
cent["strength_lo"], cent["strength_hi"] = np.percentile(bs_str, [2.5, 97.5], axis=0)
cent["bridge_lo"], cent["bridge_hi"] = np.percentile(bs_bri, [2.5, 97.5], axis=0)

# ---- stability: case-dropping subset bootstrap
drops = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
stab = {"strength": [], "bridge": []}
for d in drops:
    rs = {"strength": [], "bridge": []}
    for _ in range(200):
        idx = rng.choice(len(Z), int(round(len(Z) * (1 - d))), replace=False)
        Wb = glasso_net(normal_scores(X.values[idx]), lam)[0]
        rs["strength"].append(stats.spearmanr(np.abs(Wb).sum(0), cent["strength"]).statistic)
        rs["bridge"].append(stats.spearmanr(bridge_strength(Wb, lev), cent["bridge"]).statistic)
    for m in stab:
        v = np.nan_to_num(np.array(rs[m]), nan=0.0)
        stab[m].append(dict(drop=d, mean=float(v.mean()), q05=float(np.percentile(v, 5)),
                            q95=float(np.percentile(v, 95)), share_above_07=float((v >= 0.7).mean())))
CS = {m: max([s["drop"] for s in stab[m] if s["share_above_07"] >= 0.95] or [0.0]) for m in stab}
cent.to_csv(RES / "rq1_centrality.csv")

summary = dict(n_countries=len(X), n_nodes=p, missing_cells_imputed=miss, penalty_1se=float(lam), penalty_best=float(lam_best),
               n_edges=len(edges), possible=p * (p - 1) // 2, density=len(edges) / (p * (p - 1) / 2),
               n_cross_level=int(n_cross), share_cross=n_cross / len(edges),
               n_negative=int(sum(w < 0 for *_, w in edges)),
               n_modules=int(mod_lab.max() + 1), modularity_empirical=float(Q_emp),
               modularity_theoretical=float(Q_theo), modularity_null_mean=float(null.mean()),
               modularity_null_q95=float(np.percentile(null, 95)), p_theoretical=p_theo,
               ari_levels_vs_modules=float(ari), p_ari=p_ari,
               mean_predictability_outcomes=float(cent.loc[cent.level == "Outcome", "predictability"].mean()),
               mean_predictability_all=float(cent["predictability"].mean()),
               edges_ci_excluding_zero=int(((etab.ci_lo > 0) | (etab.ci_hi < 0)).sum()),
               CS=CS, stability=stab, bootstrap_B=B)
json.dump(summary, open(RES / "rq1_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "stability"}, indent=1))
print(cent.sort_values("bridge", ascending=False).round(3).head(10))
print(cent.groupby("module").apply(lambda g: list(g.label)).to_string())
