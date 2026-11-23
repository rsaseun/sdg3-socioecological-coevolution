"""Step 3 (RQ3 core). Subsystem indices and coupling coordination, 48 countries x 2000-2023.

* Pooled goalposts (1st-99th percentile of all country-years) make index values comparable over time.
* Entropy weights inside each subsystem, estimated once on the pooled panel (fixed over time).
* Coupling degree C, comprehensive development T and coupling coordination degree D = sqrt(C x T), plus the
  modified coupling degree of Wang et al. (2021); five coordination classes; the lagging subsystem.
* Alternative weights (equal, CRITIC, PCA) stored for the robustness analysis.
Outputs: data/processed/ccd_panel.csv, outputs/results/weights.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import PROC, RES, PANEL, SUBSYSTEMS, CLASSES, REGION_OF  # noqa: E402
from sehs.methods import (minmax_pooled, entropy_weights, critic_weights, pca_weights, ccd, ccd_modified,  # noqa
                          classify)

P = pd.read_csv(PROC / "panel.csv")
short = {v[0]: v for v in PANEL.values()}
direction = {s: v[3] for s, v in short.items()}
Xn = minmax_pooled(P[list(short)], direction)
Xn.index = P.index
Xn.to_csv(PROC / "panel_normalised.csv")

schemes = {"entropy": entropy_weights, "equal": lambda a: np.ones(a.shape[1]) / a.shape[1],
           "critic": critic_weights, "pca": pca_weights}
rows, out = [], P[["iso3", "country", "year", "POP"]].copy()
out["region"] = out.iso3.map(REGION_OF)
for name, f in schemes.items():
    U = pd.DataFrame(index=P.index)
    for sub in SUBSYSTEMS:
        cols = [s for s, v in short.items() if v[2] == sub]
        w = np.asarray(f(Xn[cols].values), float)
        U[sub] = Xn[cols].values @ w
        rows += [dict(scheme=name, subsystem=sub, indicator=c, label=short[c][1], direction=direction[c], weight=wi)
                 for c, wi in zip(cols, w)]
    C, T, D = ccd(U.values)
    Cm, _, Dm = ccd_modified(U.values)
    sfx = "" if name == "entropy" else f"_{name}"
    if name == "entropy":
        for sub in SUBSYSTEMS:
            out[f"U_{sub}"] = U[sub].values
        out["lagging"] = U.idxmin(axis=1).values
        out["imbalance_sd"] = U.std(axis=1, ddof=0).values
    out["C" + sfx], out["T" + sfx], out["D" + sfx] = C, T, D
    out["Cmod" + sfx], out["Dmod" + sfx] = Cm, Dm
out["class"] = classify(out["D"].values, CLASSES)
out.to_csv(PROC / "ccd_panel.csv", index=False)
pd.DataFrame(rows).to_csv(RES / "weights.csv", index=False)

y0, y1 = out[out.year == 2000], out[out.year == 2023]
print("mean D 2000 -> 2023:", round(y0.D.mean(), 3), "->", round(y1.D.mean(), 3))
print("mean U 2000:", y0[[f"U_{s}" for s in SUBSYSTEMS]].mean().round(3).to_dict())
print("mean U 2023:", y1[[f"U_{s}" for s in SUBSYSTEMS]].mean().round(3).to_dict())
print("classes 2000:", y0["class"].value_counts().to_dict())
print("classes 2023:", y1["class"].value_counts().to_dict())
print("lagging 2023:", y1.lagging.value_counts().to_dict())
print("C range", out.C.min().round(3), out.C.max().round(3), "Cmod range", out.Cmod.min().round(3), out.Cmod.max().round(3))
print(pd.DataFrame(rows).query("scheme=='entropy'")[["subsystem", "indicator", "weight"]].round(3).to_string())
