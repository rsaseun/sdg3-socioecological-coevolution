"""Step 6 (RQ4). Is there a health and well-being dividend of balanced (coordinated) development?

* Two-way fixed-effects panels (48 countries, 2001-2023; country and year effects; Driscoll-Kraay standard errors
  robust to cross-sectional dependence and serial correlation) of life expectancy, log under-five mortality,
  log maternal mortality and child stunting on the one-year lags of
    (M1) the coupling coordination degree D;
    (M2) comprehensive development T and coupling C (balance at a given level);
    (M3) M2 plus the T x C interaction (does balance matter more at low development?).
  T and C are z-standardised on the pooled panel, so coefficients are per standard deviation.
* Within-between (Mundlak) hybrid model separating the within-country effect of balance from the
  cross-country (between) association.
* Counterfactual 'coordination dividend': outcome change if each country's 2023 coupling rose to the regional
  upper quartile, holding development constant.
* Obstacle degree model (2023): weighted shortfall of each indicator; subsystem obstacle shares = policy entry
  points.
Outputs: outputs/results/rq4_*.csv|json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import PROC, RES, PANEL, SUBSYSTEMS  # noqa: E402
from sehs.methods import obstacle_degree  # noqa: E402

warnings.filterwarnings("ignore")
P = pd.read_csv(PROC / "panel.csv")
CC = pd.read_csv(PROC / "ccd_panel.csv")
df = CC.merge(P[["iso3", "year", "LE", "U5MR", "MMR", "STUNT", "U5MR_raw", "MMR_raw"]], on=["iso3", "year"])
df = df.sort_values(["iso3", "year"])
mu = {v: df[v].mean() for v in ["T", "C", "D"]}
sd = {v: df[v].std() for v in ["T", "C", "D"]}
for v in ["T", "C", "D"]:
    df[v + "z"] = (df[v] - mu[v]) / sd[v]
    df[v + "z_l1"] = df.groupby("iso3")[v + "z"].shift(1)
df["TCz_l1"] = df["Tz_l1"] * df["Cz_l1"]
OUTS = {"LE": "Life expectancy (years)", "U5MR": "ln under-five mortality", "MMR": "ln maternal mortality",
        "STUNT": "Child stunting (%)"}


def fe(y, xs, data):
    d = data.dropna(subset=[y] + xs).set_index(["iso3", "year"])
    m = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True).fit(cov_type="kernel", kernel="bartlett")
    return m


rows, models = [], {}
for y in OUTS:
    for name, xs in [("M1", ["Dz_l1"]), ("M2", ["Tz_l1", "Cz_l1"]), ("M3", ["Tz_l1", "Cz_l1", "TCz_l1"])]:
        m = fe(y, xs, df)
        models[(y, name)] = m
        for x in xs:
            rows.append(dict(outcome=y, model=name, term=x, coef=m.params[x], se=m.std_errors[x], p=m.pvalues[x],
                             ci_lo=m.conf_int().loc[x, "lower"], ci_hi=m.conf_int().loc[x, "upper"],
                             n=int(m.nobs), countries=int(m.entity_info["total"]), r2_within=m.rsquared_within))
R = pd.DataFrame(rows)
R.to_csv(RES / "rq4_fe_models.csv", index=False)

# marginal effect of balance (C) across development (T) from M3
me = []
for y in OUTS:
    m = models[(y, "M3")]
    V = m.cov
    for t in np.linspace(-2, 2, 41):
        g = np.array([0, 1, t])  # d/dC of b1 T + b2 C + b3 T C
        b = m.params[["Tz_l1", "Cz_l1", "TCz_l1"]].values
        est = g @ b
        se_ = np.sqrt(g @ V.loc[["Tz_l1", "Cz_l1", "TCz_l1"], ["Tz_l1", "Cz_l1", "TCz_l1"]].values @ g)
        me.append(dict(outcome=y, Tz=t, T=mu["T"] + t * sd["T"], effect=est, lo=est - 1.96 * se_, hi=est + 1.96 * se_))
pd.DataFrame(me).to_csv(RES / "rq4_marginal_effects.csv", index=False)

# Mundlak within-between hybrid (random effects with country means), clustered by country
hyb = []
d0 = df.dropna(subset=["Tz_l1", "Cz_l1"]).copy()
for v in ["Tz_l1", "Cz_l1"]:
    d0[v + "_mean"] = d0.groupby("iso3")[v].transform("mean")
    d0[v + "_dev"] = d0[v] - d0[v + "_mean"]
years = pd.get_dummies(d0.year, prefix="y", drop_first=True).astype(float)
d0 = pd.concat([d0, years], axis=1)
for y in OUTS:
    xs = ["Tz_l1_dev", "Cz_l1_dev", "Tz_l1_mean", "Cz_l1_mean"] + list(years.columns)
    d = d0.set_index(["iso3", "year"])
    import statsmodels.api as sm  # noqa: E402
    m = RandomEffects(d[y], sm.add_constant(d[xs])).fit(cov_type="clustered", cluster_entity=True)
    for x in xs[:4]:
        hyb.append(dict(outcome=y, term=x, coef=m.params[x], se=m.std_errors[x], p=m.pvalues[x]))
H = pd.DataFrame(hyb)
H.to_csv(RES / "rq4_mundlak.csv", index=False)

# counterfactual coordination dividend in 2023 (M2 within-country coefficients)
last = df[df.year == 2023].copy()
q75 = last.C.quantile(0.75)
gapz = np.maximum(0, q75 - last.C) / sd["C"]
cf = last[["iso3", "country", "region", "POP", "C", "T", "D", "LE", "U5MR_raw", "MMR_raw"]].copy()
cf["C_gap_sd"] = gapz.values
bLE = models[("LE", "M2")].params["Cz_l1"]
bU = models[("U5MR", "M2")].params["Cz_l1"]
bM = models[("MMR", "M2")].params["Cz_l1"]
cf["dLE"] = bLE * cf.C_gap_sd
cf["dU5MR_pct"] = 100 * (np.exp(bU * cf.C_gap_sd) - 1)
cf["dMMR_pct"] = 100 * (np.exp(bM * cf.C_gap_sd) - 1)
cf.to_csv(RES / "rq4_counterfactual.csv", index=False)

# obstacle degree model (2023)
W = pd.read_csv(RES / "weights.csv").query("scheme=='entropy'")
W["gw"] = W.weight / len(SUBSYSTEMS)
Xn = pd.read_csv(PROC / "panel_normalised.csv", index_col=0)
Xn = Xn.join(P[["iso3", "year"]])
x23 = Xn[Xn.year == 2023].set_index("iso3")[list(W.indicator)]
O = pd.DataFrame([obstacle_degree(r.values, W.gw.values) for _, r in x23.iterrows()], index=x23.index,
                 columns=W.indicator.values)
Osub = pd.DataFrame({s: O[W.indicator[W.subsystem == s]].sum(1) for s in SUBSYSTEMS})
O.to_csv(RES / "rq4_obstacle_indicator_2023.csv")
Osub.to_csv(RES / "rq4_obstacle_subsystem_2023.csv")
lab = dict(zip(W.indicator, W.label))
top3 = O.apply(lambda r: "; ".join(lab[c] for c in r.sort_values(ascending=False).index[:3]), axis=1)
top3.rename("top3").to_csv(RES / "rq4_obstacle_top3.csv")

pick = lambda y, mdl, t: R.query("outcome==@y and model==@mdl and term==@t").iloc[0][["coef", "se", "p"]].round(4).to_dict()
summary = dict(
    sd=sd, mean=mu,
    M1_D={y: pick(y, "M1", "Dz_l1") for y in OUTS},
    M2_T={y: pick(y, "M2", "Tz_l1") for y in OUTS}, M2_C={y: pick(y, "M2", "Cz_l1") for y in OUTS},
    M3_TC={y: pick(y, "M3", "TCz_l1") for y in OUTS},
    mundlak=H.round(4).to_dict("records"),
    C_q75_2023=float(q75), countries_below_q75=int((gapz > 0).sum()),
    dividend_LE_mean=float(cf.dLE.mean()), dividend_LE_max=float(cf.dLE.max()),
    dividend_LE_popw=float(np.average(cf.dLE, weights=cf.POP)),
    dividend_U5MR_pct_mean=float(cf.dU5MR_pct.mean()), dividend_MMR_pct_mean=float(cf.dMMR_pct.mean()),
    top_dividend=cf.sort_values("dLE", ascending=False).head(8)[["country", "dLE", "dU5MR_pct"]].round(2).values.tolist(),
    obstacle_mean_share=Osub.mean().round(4).to_dict(),
    dominant_obstacle_counts=Osub.idxmax(axis=1).value_counts().to_dict(),
    top_obstacle_indicators=O.mean().sort_values(ascending=False).head(6).round(4).rename(lab).to_dict(),
)
json.dump(summary, open(RES / "rq4_summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:5000])
print(R.round(4).to_string())
