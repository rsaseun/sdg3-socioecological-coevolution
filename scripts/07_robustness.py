"""Step 7. Robustness and sensitivity analyses (Table 6 and Supplementary Tables S3-S4).

R1  Alternative weights for the subsystem indices (equal, CRITIC, PCA): rank agreement of D and the
    within-country coefficient of balance (C) in model M2.
R2  Modified coupling degree (Wang et al., 2021) in place of the classical C.
R3  Imbalance as the standard deviation of the four subsystem indices.
R4  Excluding countries with any whole-series imputation (ERI, SSD, BDI, GNQ, LSO, SYC).
R5  Lag length 0, 2 and 3 years.
R6  Leave-one-region-out.
R7  RQ1 network under the cross-validation-optimal (denser) penalty: correlation of strength and bridge ranks.
Outputs: outputs/results/robustness.csv, outputs/results/robustness_network.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import PROC, RES, RAW, LEVELS, XOUT, SEED  # noqa: E402

warnings.filterwarnings("ignore")
P = pd.read_csv(PROC / "panel.csv")
CC = pd.read_csv(PROC / "ccd_panel.csv")
df = CC.merge(P[["iso3", "year", "LE", "U5MR", "MMR", "STUNT"]], on=["iso3", "year"]).sort_values(["iso3", "year"])
OUTS = ["LE", "U5MR", "MMR", "STUNT"]


def z(s):
    return (s - s.mean()) / s.std()


def m2(data, tcol="T", ccol="C", lag=1, outs=OUTS, label=""):
    d = data.copy()
    d["Tz"], d["Cz"] = z(d[tcol]), z(d[ccol])
    if lag:
        d["Tz"] = d.groupby("iso3").Tz.shift(lag)
        d["Cz"] = d.groupby("iso3").Cz.shift(lag)
    rows = []
    for y in outs:
        e = d.dropna(subset=[y, "Tz", "Cz"]).set_index(["iso3", "year"])
        m = PanelOLS(e[y], e[["Tz", "Cz"]], entity_effects=True, time_effects=True).fit(cov_type="kernel")
        rows.append(dict(check=label, outcome=y, coef_C=m.params["Cz"], se_C=m.std_errors["Cz"], p_C=m.pvalues["Cz"],
                         coef_T=m.params["Tz"], n=int(m.nobs)))
    return rows


rows = m2(df, label="Main (entropy weights, classical C, lag 1)")
d23 = df[df.year == 2023]
rank = {}
for s in ["equal", "critic", "pca"]:
    rows += m2(df, f"T_{s}", f"C_{s}", label=f"R1 {s} weights")
    rank[s] = stats.spearmanr(d23["D"], d23[f"D_{s}"]).statistic
rows += m2(df, "T", "Cmod", label="R2 modified coupling degree")
df["neg_sd"] = -df.imbalance_sd
rows += m2(df, "T", "neg_sd", label="R3 balance = -SD of subsystem indices")
excl = ["ERI", "SSD", "BDI", "GNQ", "LSO", "SYC"]
rows += m2(df[~df.iso3.isin(excl)], label="R4 excluding countries with imputed series")
for L in (0, 2, 3):
    rows += m2(df, lag=L, label=f"R5 lag {L}")
for r in sorted(df.region.unique()):
    rows += m2(df[df.region != r], label=f"R6 without {r} Africa")
R = pd.DataFrame(rows)
R.to_csv(RES / "robustness.csv", index=False)

# R7 network under the denser CV-optimal penalty
from sehs.methods import normal_scores, glasso_net, bridge_strength  # noqa: E402
from sklearn.impute import KNNImputer  # noqa: E402
summ = json.load(open(RES / "rq1_summary.json"))
cent = pd.read_csv(RES / "rq1_centrality.csv", index_col=0)
cs = pd.read_csv(RAW / "cross_section_48.csv").set_index("iso3")
cs["lnHEPC"] = np.log(cs.pop("HEPC"))
nodes = list(cent.index)
sign = [s for lev in LEVELS.values() for _, s in lev.values()] + [-1, -1, 1]
X = cs[nodes].copy()
X["U5MR"], X["MMR"] = np.log(X["U5MR"]), np.log(X["MMR"])
X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X), columns=nodes) * np.array(sign)
W2 = glasso_net(normal_scores(X.values), summ["penalty_best"])[0]
net = dict(penalty=summ["penalty_best"], n_edges=int((np.abs(np.triu(W2, 1)) > 0).sum()),
           rho_strength=float(stats.spearmanr(np.abs(W2).sum(0), cent.strength).statistic),
           rho_bridge=float(stats.spearmanr(bridge_strength(W2, cent.level.values), cent.bridge).statistic),
           rank_agreement_D2023=rank)
json.dump(net, open(RES / "robustness_network.json", "w"), indent=1)
print(json.dumps(net, indent=1))
print(R.round(4).to_string())
