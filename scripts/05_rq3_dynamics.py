"""Step 5 (RQ3). Has coordination improved and converged, and how persistent are low-coordination states?

* Trajectories of the four subsystem indices and of D (country, regional and population-weighted means).
* Sigma-convergence (coefficient of variation of D over time, linear trend test).
* Absolute beta-convergence of D and of each subsystem (2000-2023 cross-section) with speed and half-life;
  conditional beta-convergence by two-way fixed effects on 3-year periods.
* Discrete Markov chain of five coordination classes (annual transitions), ergodic distribution and mean first
  passage times; spatial Markov chain conditioned on the mean D of the 4 nearest neighbours (low < 0.55 <= middle
  < 0.65 <= high), with
  likelihood-ratio and Q tests of spatial homogeneity (Rey, 2001).
Outputs: outputs/results/rq3_*.csv|json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RAW, PROC, RES, CLASSES, SUBSYSTEMS, PERIODS  # noqa: E402

warnings.filterwarnings("ignore")
CC = pd.read_csv(PROC / "ccd_panel.csv")
years = sorted(CC.year.unique())
Dw = CC.pivot(index="iso3", columns="year", values="D")
Usub = {s: CC.pivot(index="iso3", columns="year", values=f"U_{s}") for s in SUBSYSTEMS}

# ------------------------------------------------------------------ trajectories
traj = CC.groupby("year")[[f"U_{s}" for s in SUBSYSTEMS] + ["D", "C", "T"]].mean()
traj_pw = CC.groupby("year").apply(lambda g: pd.Series(np.average(g[["D", "T", "C"]], weights=g.POP, axis=0),
                                                       index=["D_pw", "T_pw", "C_pw"]))
reg = CC.groupby(["region", "year"]).D.mean().unstack(0)
pd.concat([traj, traj_pw, reg.add_prefix("D_")], axis=1).to_csv(RES / "rq3_trajectories.csv")

# ------------------------------------------------------------------ sigma convergence
cv = Dw.std() / Dw.mean()
sl = stats.linregress(years, cv.values)
sigma = dict(cv_2000=float(cv.iloc[0]), cv_2023=float(cv.iloc[-1]), slope_per_year=float(sl.slope),
             p=float(sl.pvalue), sd_2000=float(Dw[2000].std()), sd_2023=float(Dw[2023].std()))
cv_sub = {s: dict(cv_2000=float(u[2000].std() / u[2000].mean()), cv_2023=float(u[2023].std() / u[2023].mean()))
          for s, u in Usub.items()}


# ------------------------------------------------------------------ beta convergence
def beta_abs(W):
    g = (np.log(W[2023]) - np.log(W[2000])) / 23
    x = sm.add_constant(np.log(W[2000]))
    m = sm.OLS(g, x).fit(cov_type="HC3")
    b = m.params.iloc[1]
    lam = -np.log(1 + 23 * b) / 23 if 1 + 23 * b > 0 else np.nan
    return dict(beta=float(b), se=float(m.bse.iloc[1]), p=float(m.pvalues.iloc[1]), r2=float(m.rsquared),
                speed=float(lam), half_life=float(np.log(2) / lam) if lam and lam > 0 else np.nan)


beta = {"D": beta_abs(Dw)}
beta.update({s: beta_abs(u) for s, u in Usub.items()})
# conditional beta: Delta ln D (3-year period means) on lagged ln D with country and period fixed effects
CC["period"] = np.searchsorted([b for a, b in PERIODS], CC.year, side="left")
pm = CC.groupby(["iso3", "period"]).D.mean().reset_index()
pm["lnD"] = np.log(pm.D)
pm["dlnD"] = pm.groupby("iso3").lnD.diff()
pm["lag"] = pm.groupby("iso3").lnD.shift(1)
pm = pm.dropna()
from linearmodels.panel import PanelOLS  # noqa: E402
m = PanelOLS(pm.set_index(["iso3", "period"]).dlnD, pm.set_index(["iso3", "period"])[["lag"]],
             entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
bc = float(m.params["lag"])
beta_cond = dict(beta=bc, se=float(m.std_errors["lag"]), p=float(m.pvalues["lag"]),
                 speed_per_year=float(-np.log(1 + bc) / 3), half_life_years=float(3 * np.log(2) / -np.log(1 + bc)))

# ------------------------------------------------------------------ Markov chains
import giddy  # noqa: E402
from giddy.markov import Markov, Spatial_Markov  # noqa: E402
from libpysal.weights import KNN  # noqa: E402

cut = [c[1] for c in CLASSES[:-1]]
names = [c[2] for c in CLASSES]
Y = Dw.values
cls = np.digitize(Y, cut)          # 0..4
mk = Markov(cls, classes=np.arange(5))
P = pd.DataFrame(mk.p, index=names, columns=names)
T = pd.DataFrame(mk.transitions.astype(int), index=names, columns=names)
steady = pd.Series(np.asarray(mk.steady_state).ravel(), index=names)
from giddy.ergodic import mfpt  # noqa: E402
try:
    fpt = pd.DataFrame(mfpt(mk.p), index=names, columns=names)
except Exception:  # pragma: no cover
    fpt = pd.DataFrame(np.nan, index=names, columns=names)
P.to_csv(RES / "rq3_markov_P.csv"); T.to_csv(RES / "rq3_markov_counts.csv"); fpt.to_csv(RES / "rq3_markov_mfpt.csv")

geo = gpd.read_file(RAW / "ssa48_boundaries.geojson").set_index("iso3").loc[Dw.index].to_crs("ESRI:102022")
pts = np.column_stack([geo.geometry.representative_point().x, geo.geometry.representative_point().y])
w = KNN.from_array(pts, k=4)
w.transform = "r"
LAGCUT = [0.55, 0.65]                     # neighbourhood: low (<0.55), middle, high (>=0.65) spatial-lag D
LAGNAMES = ["Low-coordination neighbours", "Middle", "High-coordination neighbours"]
sm_ = Spatial_Markov(Y, w, cutoffs=np.array(cut), lag_cutoffs=np.array(LAGCUT), fixed=True, k=5, m=3)
cond = {}
for k in range(3):
    cond[LAGNAMES[k]] = pd.DataFrame(sm_.P[k], index=names, columns=names)
    pd.DataFrame(sm_.T[k].astype(int), index=names, columns=names).to_csv(RES / f"rq3_spatial_counts_lag{k}.csv")
pd.concat(cond, names=["neighbour_class"]).to_csv(RES / "rq3_spatial_markov_P.csv")
homog = dict(LR=float(sm_.LR), LR_dof=int(sm_.dof_hom), LR_p=float(sm_.LR_p_value),
             Q=float(sm_.Q), Q_dof=int(sm_.dof_hom), Q_p=float(sm_.Q_p_value))
# key contrast: probability of upgrading from barely coordinated when neighbours are low vs high
up, down = {}, {}
for k, nm in enumerate(LAGNAMES):
    Pk, Tk = sm_.P[k], sm_.T[k]
    up[nm] = {names[i]: (float(Pk[i, i + 1:].sum()) if Tk[i].sum() > 0 else None) for i in range(4)}
    down[nm] = {names[i]: (float(Pk[i, :i].sum()) if Tk[i].sum() > 0 else None) for i in range(1, 5)}
    up[nm]["n_low_start"] = int(Tk[:3].sum())
    up[nm]["upgrade_rate_from_below_primary"] = float(np.triu(Tk[:3], 1).sum() / max(Tk[:3].sum(), 1))

# ------------------------------------------------------------------ country change and kernel densities
chg = pd.DataFrame({"D_2000": Dw[2000], "D_2023": Dw[2023], "dD": Dw[2023] - Dw[2000],
                    "class_2000": [names[i] for i in cls[:, 0]], "class_2023": [names[i] for i in cls[:, -1]]})
chg = chg.join(CC[CC.year == 2023].set_index("iso3")[["country", "lagging", "region"]])
chg.to_csv(RES / "rq3_country_change.csv")
grid = np.linspace(0.3, 1.0, 200)
kde = pd.DataFrame({y: stats.gaussian_kde(Dw[y])(grid) for y in [2000, 2008, 2016, 2023]}, index=grid)
kde.to_csv(RES / "rq3_kde.csv")

summary = dict(mean_D=dict(y2000=float(Dw[2000].mean()), y2023=float(Dw[2023].mean())),
               mean_D_popweighted=dict(y2000=float(traj_pw.loc[2000, "D_pw"]), y2023=float(traj_pw.loc[2023, "D_pw"])),
               mean_U_2000=traj.loc[2000].round(4).to_dict(), mean_U_2023=traj.loc[2023].round(4).to_dict(),
               sigma=sigma, cv_sub=cv_sub, beta_abs=beta, beta_cond=beta_cond,
               markov_diag=dict(zip(names, np.diag(mk.p).round(4).tolist())), steady_state=steady.round(4).to_dict(),
               mfpt_from_imbalanced_or_near_to_primary=float(fpt.loc["Near imbalance", "Primary coordination"]),
               spatial_homogeneity=homog, upgrade_probability_by_neighbourhood=up,
               downgrade_probability_by_neighbourhood=down,
               improved=int((chg.dD > 0).sum()), declined=int((chg.dD < 0).sum()),
               top_gain=chg.sort_values("dD", ascending=False).head(5)[["country", "dD"]].round(3).values.tolist(),
               declines=chg[chg.dD < 0].sort_values("dD")[["country", "dD"]].round(3).values.tolist(),
               lagging_2000=CC[CC.year == 2000].lagging.value_counts().to_dict(),
               lagging_2023=CC[CC.year == 2023].lagging.value_counts().to_dict(),
               class_2000=chg.class_2000.value_counts().to_dict(), class_2023=chg.class_2023.value_counts().to_dict(),
               giddy_version=giddy.__version__)
json.dump(summary, open(RES / "rq3_summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:6000])
print(P.round(3)); print(fpt.round(1))
