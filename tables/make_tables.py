"""Tables 1-6 and Supplementary Tables S1-S7 as CSV files and one Excel workbook (outputs/tables)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import PROC, RES, TAB, PANEL, OUTCOMES, SUBSYSTEMS  # noqa: E402

f2 = lambda v: f"{v:.2f}"
f3 = lambda v: f"{v:.3f}"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


T = {}
# ------------------------------------------------------------------ Table 1 indicator system
W = pd.read_csv(RES / "weights.csv").query("scheme=='entropy'").set_index("indicator")
prov = pd.read_csv(RES / "provenance.csv").set_index("short")
rows = []
for code, (short, label, sub, d, tr) in PANEL.items():
    rows.append({"Subsystem": sub, "Indicator": label, "WDI/WGI code": code, "Direction": "+" if d > 0 else "−",
                 "Entropy weight": f3(W.loc[short, "weight"]), "Observed cells (%)": f"{prov.loc[short, 'observed_pct']:.1f}"})
for code, (short, label, tr) in OUTCOMES.items():
    rows.append({"Subsystem": "Outcome", "Indicator": label, "WDI/WGI code": code, "Direction": "",
                 "Entropy weight": "–", "Observed cells (%)": f"{prov.loc[short, 'observed_pct']:.1f}"})
T["Table_1"] = pd.DataFrame(rows)

# ------------------------------------------------------------------ Table 2 empirical modules (RQ1)
cent = pd.read_csv(RES / "rq1_centrality.csv", index_col=0)
S1 = json.load(open(RES / "rq1_summary.json"))
MODNAME = {}
for m, g in cent.groupby("module"):
    labs = set(g.label)
    if {"Government effectiveness", "Control of corruption"} <= labs:
        MODNAME[m] = "State capacity"
    elif "Physicians" in labs:
        MODNAME[m] = "Human resources and capabilities"
    elif "Child survival" in labs:
        MODNAME[m] = "Survival and financial protection"
    elif "Poverty" in labs:
        MODNAME[m] = "Material deprivation"
    elif "Fertility" in labs:
        MODNAME[m] = "Demographic–gender transition and coverage"
    else:
        MODNAME[m] = "Livelihoods and climate"
json.dump({int(k): v for k, v in MODNAME.items()}, open(RES / "rq1_module_names.json", "w"), indent=1)
rows = []
for m, g in cent.groupby("module"):
    rows.append({"Module": f"M{m}", "Interpretation": MODNAME[m],
                 "Members (theoretical level)": "; ".join(f"{l} ({lv})" for l, lv in zip(g.label, g.level)),
                 "Levels spanned": g.level.nunique(), "Mean bridge strength": f2(g.bridge.mean()),
                 "Mean predictability": f2(g.predictability.mean())})
T["Table_2"] = pd.DataFrame(rows)

# ------------------------------------------------------------------ Table 3 lead-lag (RQ2)
E = pd.read_csv(RES / "rq2_temporal_edges.csv")
st = E[(E.inclusion >= 0.9) & (~E.autoregressive)].sort_values("beta", key=np.abs, ascending=False)
A = pd.DataFrame({"Panel": "A. Stable lead–lag edges",
                  "Leading change (t−1)": st.source_label + " [" + st.source_group + "]",
                  "Following change (t)": st.target_label + " [" + st.target_group + "]",
                  "Estimate": st.beta.map(f3), "95% interval": [f"{a:.3f} to {b:.3f}" for a, b in zip(st.ci_lo, st.ci_hi)],
                  "Inclusion": st.inclusion.map(f2)})
DH = pd.read_csv(RES / "rq2_dumitrescu_hurlin.csv")
lab = {"U_Social": "Social", "U_Institutional": "Institutional", "U_Environmental": "Environmental",
       "U_Health system": "Health system", "D": "Coordination D", "LE": "Life expectancy",
       "U5MR": "Child survival", "MMR": "Maternal survival"}
b_rows = []
for x in ["U_Social", "U_Institutional", "U_Environmental", "U_Health system", "D"]:
    for y in ["LE", "U5MR", "MMR"]:
        r1 = DH[(DH.K == 1) & (DH.x == x) & (DH.y == y)].iloc[0]
        r2 = DH[(DH.K == 2) & (DH.x == x) & (DH.y == y)].iloc[0]
        rr = DH[(DH.K == 1) & (DH.x == y) & (DH.y == x)].iloc[0]
        b_rows.append({"Subsystem index": lab[x], "Outcome": lab[y],
                       "Z\u0303 lag 1 (subsystem \u2192 outcome)": f"{r1.Ztilde:.2f}{star(r1.p_Ztilde)}",
                       "Z\u0303 lag 2 (subsystem \u2192 outcome)": f"{r2.Ztilde:.2f}{star(r2.p_Ztilde)}",
                       "Z\u0303 lag 1 (outcome \u2192 subsystem)": f"{rr.Ztilde:.2f}{star(rr.p_Ztilde)}"})
T["Table_3"] = A.drop(columns="Panel")
T["Table_3B"] = pd.DataFrame(b_rows)

# ------------------------------------------------------------------ Table 4 convergence (RQ3)
S3 = json.load(open(RES / "rq3_summary.json"))
rows = []
for k, nm in [("D", "Coordination D")] + [(s, s) for s in SUBSYSTEMS]:
    b = S3["beta_abs"][k]
    cv0 = S3["sigma"]["cv_2000"] if k == "D" else S3["cv_sub"][k]["cv_2000"]
    cv1 = S3["sigma"]["cv_2023"] if k == "D" else S3["cv_sub"][k]["cv_2023"]
    rows.append({"Series": nm, "CV 2000": f2(cv0), "CV 2023": f2(cv1),
                 "Absolute β (SE)": f"{b['beta']:.4f} ({b['se']:.4f}){star(b['p'])}",
                 "R²": f2(b["r2"]), "Speed (%/year)": f"{100 * b['speed']:.2f}", "Half-life (years)": f"{b['half_life']:.0f}"})
bc = S3["beta_cond"]
rows.append({"Series": "Coordination D, conditional (two-way FE, 3-year periods)", "CV 2000": "", "CV 2023": "",
             "Absolute β (SE)": f"{bc['beta']:.3f} ({bc['se']:.3f}){star(bc['p'])}", "R²": "",
             "Speed (%/year)": f"{100 * bc['speed_per_year']:.1f}", "Half-life (years)": f"{bc['half_life_years']:.1f}"})
T["Table_4"] = pd.DataFrame(rows)

# ------------------------------------------------------------------ Table 5 FE models (RQ4)
R = pd.read_csv(RES / "rq4_fe_models.csv")
H = pd.read_csv(RES / "rq4_mundlak.csv")
OUT = {"LE": "Life expectancy (years)", "U5MR": "ln under-five mortality", "MMR": "ln maternal mortality",
       "STUNT": "Child stunting (%)"}
TERM = {"Dz_l1": "Coordination D", "Tz_l1": "Development T", "Cz_l1": "Balance (coupling C)", "TCz_l1": "T × C"}
rows = []
for mdl, terms in [("M1", ["Dz_l1"]), ("M2", ["Tz_l1", "Cz_l1"]), ("M3", ["Tz_l1", "Cz_l1", "TCz_l1"])]:
    for t in terms:
        row = {"Model": mdl, "Term (per SD, lagged 1 year)": TERM[t]}
        for o, on in OUT.items():
            r = R[(R.outcome == o) & (R.model == mdl) & (R.term == t)].iloc[0]
            row[on] = f"{r.coef:.3f} ({r.se:.3f}){star(r.p)}"
        rows.append(row)
    row = {"Model": mdl, "Term (per SD, lagged 1 year)": "Within R²"}
    for o, on in OUT.items():
        row[on] = f2(R[(R.outcome == o) & (R.model == mdl)].r2_within.iloc[0])
    rows.append(row)
for t, nm in [("Cz_l1_dev", "Balance, within-country"), ("Cz_l1_mean", "Balance, between-country"),
              ("Tz_l1_dev", "Development, within-country"), ("Tz_l1_mean", "Development, between-country")]:
    row = {"Model": "Mundlak", "Term (per SD, lagged 1 year)": nm}
    for o, on in OUT.items():
        r = H[(H.outcome == o) & (H.term == t)].iloc[0]
        row[on] = f"{r.coef:.3f} ({r.se:.3f}){star(r.p)}"
    rows.append(row)
T["Table_5"] = pd.DataFrame(rows)

# ------------------------------------------------------------------ Table 6 / S2 country entry points
CC = pd.read_csv(PROC / "ccd_panel.csv")
d0 = CC[CC.year == 2000].set_index("iso3"); d1 = CC[CC.year == 2023].set_index("iso3")
top3 = pd.read_csv(RES / "rq4_obstacle_top3.csv", index_col=0).top3
cf = pd.read_csv(RES / "rq4_counterfactual.csv").set_index("iso3")
full = pd.DataFrame({"Country": d1.country, "Region": d1.region, "D 2000": d0.D.map(f2), "D 2023": d1.D.map(f2),
                     "Class 2023": d1["class"], "Lagging subsystem": d1.lagging,
                     "Top three obstacles (2023)": top3.str.replace(" (log)", "", regex=False),
                     "Dividend: life expectancy (years)": cf.dLE.map(f2),
                     "Dividend: under-five mortality (%)": cf.dU5MR_pct.map(lambda v: f"{v:.1f}")})
full = full.loc[d1.D.sort_values().index]
T["Table_6"] = full.head(12).reset_index(drop=True)
T["Table_S2"] = full.reset_index(drop=True)

# ------------------------------------------------------------------ supplementary tables
T["Table_S1"] = prov.reset_index().rename(columns={"short": "Variable"}).round(1)
rb = pd.read_csv(RES / "robustness.csv")
T["Table_S3"] = rb.assign(coef_C=rb.coef_C.map(f3), se_C=rb.se_C.map(f3), p_C=rb.p_C.map(pfmt), coef_T=rb.coef_T.map(f3))
P = pd.read_csv(RES / "rq3_markov_P.csv", index_col=0)
C = pd.read_csv(RES / "rq3_markov_counts.csv", index_col=0)
SP = pd.read_csv(RES / "rq3_spatial_markov_P.csv")
mk = P.round(3).astype(str) + " (" + C.astype(str) + ")"
mk.insert(0, "Neighbourhood", "All countries (classical chain)")
SP.columns = ["Neighbourhood", "Class t"] + list(P.columns)
SP = SP.set_index("Class t")
T["Table_S4"] = pd.concat([mk.rename_axis("Class t").reset_index(),
                           SP.round(3).reset_index()], ignore_index=True)
T["Table_S5"] = DH.round(3)
T["Table_S6"] = pd.read_csv(RES / "rq2_unitroot_summary.csv").round(3)
T["Table_S7"] = cent.round(3).reset_index().rename(columns={"index": "node"})

with pd.ExcelWriter(TAB / "All_Tables.xlsx") as xw:
    for k, v in T.items():
        v.to_csv(TAB / f"{k}.csv", index=False)
        v.to_excel(xw, sheet_name=k, index=False)
print({k: v.shape for k, v in T.items()})
print(T["Table_2"].to_string()); print(T["Table_3"].head(12).to_string())
