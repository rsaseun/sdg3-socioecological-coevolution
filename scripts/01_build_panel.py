"""Step 1. Build the analysis panel (48 countries x 2000-2023) from the World Bank API extract.

Gap handling, in order:
  1. linear interpolation inside each country series (e.g. WGI has no 2001 release);
  2. nearest-value carry at series edges (no extrapolated trends);
  3. series missing for a whole country: year-wise k-nearest-neighbour imputation (k = 5) on the other
     standardised indicators of the same year.
Every filled cell is flagged so that provenance can be reported (Table S1) and countries with many filled cells
can be excluded in robustness checks.
Outputs: data/processed/panel.csv, data/processed/panel_flags.csv, outputs/results/provenance.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RAW, PROC, RES, PANEL, OUTCOMES, POP, YEARS  # noqa: E402

long = pd.read_csv(RAW / "wdi_wgi_long_2000_2023.csv")
countries = pd.read_csv(RAW / "countries.csv")
isos = list(countries.iso3)
codes = list(PANEL) + list(OUTCOMES) + [POP]
wide = long.pivot_table(index=["iso3", "year"], columns="indicator", values="value")
idx = pd.MultiIndex.from_product([isos, YEARS], names=["iso3", "year"])
wide = wide.reindex(idx)[codes]
observed = wide.notna()

# 1-2. interpolation and edge carry within each country
filled = wide.groupby(level=0, group_keys=False).apply(
    lambda g: g.interpolate(method="linear", limit_area="inside").ffill().bfill())
step = pd.DataFrame("observed", index=wide.index, columns=codes)
step[~observed & filled.notna()] = "interpolated/carried"

# 3. whole-series gaps: year-wise kNN on the standardised subsystem indicators only (outcomes are never used
#    as donors' matching features, so imputed predictors carry no outcome information)
need = filled.isna()
step[need] = "kNN-imputed"
feat = list(PANEL)
for y in YEARS:
    block = filled.xs(y, level=1)[feat]
    z = (block - block.mean()) / block.std()
    imp = KNNImputer(n_neighbors=5).fit_transform(z)
    imp = pd.DataFrame(imp, index=block.index, columns=feat) * block.std() + block.mean()
    for c in feat:
        miss = block[c].isna()
        if miss.any():
            filled.loc[(block.index[miss], y), c] = imp.loc[miss, c].values
assert filled.notna().all().all()

# transformed analysis variables with short names
out = pd.DataFrame(index=filled.index)
for c, (short, _lab, _sub, _dir, tr) in PANEL.items():
    out[short] = np.log(filled[c]) if tr == "log" else filled[c]
for c, (short, _lab, tr) in OUTCOMES.items():
    out[short] = np.log(filled[c]) if tr == "log" else filled[c]
    out[short + "_raw"] = filled[c]
out["POP"] = filled[POP]
out = out.reset_index().merge(countries, on="iso3")
out.to_csv(PROC / "panel.csv", index=False)
flags = step.rename(columns={c: (PANEL.get(c) or OUTCOMES.get(c) or ("POP",))[0] for c in codes})
flags.reset_index().to_csv(PROC / "panel_flags.csv", index=False)

prov = pd.DataFrame({
    "short": [flags.columns[i] for i in range(len(codes))],
    "code": codes,
    "observed_pct": [100 * (step[c] == "observed").mean() for c in codes],
    "interpolated_pct": [100 * (step[c] == "interpolated/carried").mean() for c in codes],
    "knn_pct": [100 * (step[c] == "kNN-imputed").mean() for c in codes],
    "countries_with_no_data": [", ".join(sorted(set(isos) - set(long.loc[long.indicator == c, "iso3"])))
                               for c in codes],
})
prov.to_csv(RES / "provenance.csv", index=False)
share_filled = (flags.drop(columns=["POP"]) != "observed").groupby(level=0).mean().mean(axis=1)
share_filled.rename("share_filled").to_csv(RES / "country_fill_share.csv")
tot = (step.values == "observed").mean()
print(f"panel {out.shape}, observed cells {100 * tot:.1f}%, kNN cells {100 * (step.values == 'kNN-imputed').mean():.2f}%")
print("countries with >25% filled cells:", list(share_filled[share_filled > 0.25].index))
