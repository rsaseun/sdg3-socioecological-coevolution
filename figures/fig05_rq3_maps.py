"""Figure 5 (RQ3). Coupling coordination in 2000 and 2023, its change, and the lagging subsystem in 2023.

Map lines delineate study areas and do not necessarily depict accepted national boundaries.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RAW, PROC, W15, MM, SUBCOL, SUBSYSTEMS, save_fig  # noqa: E402

geo = gpd.read_file(RAW / "ssa48_boundaries.geojson").to_crs("ESRI:102022")
CC = pd.read_csv(PROC / "ccd_panel.csv")
d0 = CC[CC.year == 2000].set_index("iso3")
d1 = CC[CC.year == 2023].set_index("iso3")
geo["D2000"] = geo.iso3.map(d0.D); geo["D2023"] = geo.iso3.map(d1.D)
geo["dD"] = geo.D2023 - geo.D2000
geo["lag"] = geo.iso3.map(d1.lagging)
small = ["CPV", "COM", "MUS", "STP", "SYC"]
pts = geo.copy(); pts["geometry"] = geo.geometry.representative_point()

fig = plt.figure(figsize=(W15, 150 * MM))
axs = np.array([[fig.add_axes([0.0, 0.58, 0.5, 0.38]), fig.add_axes([0.5, 0.58, 0.5, 0.38])],
                [fig.add_axes([0.0, 0.085, 0.5, 0.365]), fig.add_axes([0.5, 0.085, 0.5, 0.365])]])
cmap = plt.get_cmap("YlGnBu")
norm = Normalize(0.3, 1.0)
for ax, col, title in [(axs[0, 0], "D2000", "a  Coordination D, 2000"), (axs[0, 1], "D2023", "b  Coordination D, 2023")]:
    geo.plot(column=col, ax=ax, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.25)
    pts[pts.iso3.isin(small)].plot(column=col, ax=ax, cmap=cmap, norm=norm, markersize=14, edgecolor="#555555",
                                   linewidth=0.4)
    ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cax = fig.add_axes([0.14, 0.565, 0.72, 0.010])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("Coupling coordination degree D", fontsize=5.8)
cb.ax.tick_params(labelsize=5.6)

dn = TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.26)
ax = axs[1, 0]
geo.plot(column="dD", ax=ax, cmap="RdBu", norm=dn, edgecolor="white", linewidth=0.25)
pts[pts.iso3.isin(small)].plot(column="dD", ax=ax, cmap="RdBu", norm=dn, markersize=14, edgecolor="#555555",
                               linewidth=0.4)
ax.set_title("c  Change in D, 2000–2023", loc="left", fontsize=7.2, fontweight="bold")
cax2 = fig.add_axes([0.07, 0.066, 0.36, 0.010])
cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu", norm=dn), cax=cax2, orientation="horizontal")
cb2.set_ticks([-0.05, 0, 0.1, 0.2]); cb2.ax.tick_params(labelsize=5.6); cb2.set_label("ΔD", fontsize=6)
for k in ["RWA", "SLE", "CAF", "SSD"]:
    p = pts.set_index("iso3").geometry[k]
    nm = {"RWA": "Rwanda +0.26", "SLE": "Sierra Leone +0.20", "CAF": "CAR −0.03", "SSD": "S. Sudan −0.02"}[k]
    off = {"RWA": (24, -22), "SLE": (-4, -30), "CAF": (-38, 24), "SSD": (26, 24)}[k]
    ax.annotate(nm, (p.x, p.y), xytext=off, textcoords="offset points", ha="center",
                fontsize=5.3, arrowprops=dict(arrowstyle="-", lw=0.4, color="#333333"))

ax = axs[1, 1]
geo.plot(color=geo.lag.map(SUBCOL), ax=ax, edgecolor="white", linewidth=0.25)
pts[pts.iso3.isin(small)].plot(color=pts[pts.iso3.isin(small)].lag.map(SUBCOL), ax=ax, markersize=14,
                               edgecolor="#555555", linewidth=0.4)
ax.set_title("d  Lagging subsystem, 2023", loc="left", fontsize=7.2, fontweight="bold")
cnt = d1.lagging.value_counts()
ax.legend(handles=[Patch(fc=SUBCOL[s], label=f"{s} ({cnt.get(s, 0)})") for s in SUBSYSTEMS], loc="lower left",
          frameon=False, fontsize=5.6, bbox_to_anchor=(0.05, -0.16), ncol=2)
for a in axs.ravel():
    a.set_axis_off()
    a.set_xlim(geo.total_bounds[0] - 2e5, geo.total_bounds[2] + 2e5)
    a.set_ylim(geo.total_bounds[1] - 1e5, geo.total_bounds[3] + 1e5)
save_fig(fig, "Figure_5")
