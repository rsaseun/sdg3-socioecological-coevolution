"""Graphical abstract (Elsevier: 13.3 x 5.3 cm aspect, >= 1328 x 531 px; exported at 750 dpi)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RAW, PROC, RES, FIG, SUBCOL  # noqa: E402

S1 = json.load(open(RES / "rq1_summary.json")); S3 = json.load(open(RES / "rq3_summary.json"))
S4 = json.load(open(RES / "rq4_summary.json"))
R = pd.read_csv(RES / "rq4_fe_models.csv").query("model=='M2' and term=='Cz_l1'").set_index("outcome")
cm = 1 / 2.54
fig = plt.figure(figsize=(26.6 * cm, 10.6 * cm))   # 2x the 13.3 x 5.3 cm target; text sized accordingly
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 266); ax.set_ylim(0, 106); ax.axis("off")
ax.text(133, 99, "Balanced development as a health determinant: socioecological co-evolution in 48 "
        "Sub-Saharan African countries, 2000–2023", ha="center", va="center", fontsize=11.5, fontweight="bold")


def box(x, y, w, h, ec, head, body, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.5", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 5.5, head, ha="center", va="center", fontsize=10, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + (h - 9) / 2 + 0.5, body, ha="center", va="center", fontsize=8.6, linespacing=1.35)


# inputs
subs = ["Social", "Institutional", "Environmental", "Health system"]
for i, s in enumerate(subs):
    y = 72 - i * 16.5
    ax.add_patch(FancyBboxPatch((4, y), 34, 12, boxstyle="round,pad=0,rounding_size=2", fc=SUBCOL[s], ec="none"))
    ax.text(21, y + 6, s, ha="center", va="center", fontsize=9.2, color="white", fontweight="bold")
    ax.add_patch(FancyArrowPatch((38, y + 6), (47, 50), arrowstyle="-|>,head_length=4,head_width=2", color="#555555",
                                 lw=1.0))
ax.text(21, 14, "20 indicators × 24 years\n+ 24-node cross-section", ha="center", va="center", fontsize=8.2,
        color="#444444")
ax.add_patch(FancyBboxPatch((47, 41), 20, 18, boxstyle="round,pad=0,rounding_size=2", fc="#1d1d1f", ec="none"))
ax.text(57, 50, "C · T · D\ncoupling\ncoordination", ha="center", va="center", fontsize=8.4, color="white",
        fontweight="bold", linespacing=1.2)
ax.add_patch(FancyArrowPatch((67, 50), (73, 50), arrowstyle="-|>,head_length=4,head_width=2", color="#1d1d1f", lw=1.4))

# findings
w, h = 44, 38
box(74, 47, w, h, "#0072B2", "1  Structure",
    f"{100 * S1['share_cross']:.0f}% of links cross levels;\nlevels are not modules\n(ARI = {S1['ari_levels_vs_modules']:.2f});"
    "\nout-of-pocket costs, poverty,\nschooling and water bridge")
box(121, 47, w, h, "#CC79A7", "2  Lead–lag",
    "health spending leads\ngovernance and income;\nenvironment leads social\nchange; social gains\nprecede longer lives")
box(74, 5, w, h, "#D55E00", "3  Coordination",
    f"D {S3['mean_D']['y2000']:.2f} → {S3['mean_D']['y2023']:.2f}; σ- and β-converging;\n"
    f"environment lags in {S3['lagging_2023']['Environmental']}/48;\nweak neighbourhoods raise\ndowngrade risk (14% vs ≤3%)")
le, u5 = R.loc["LE", "coef"], 100 * (np.exp(R.loc["U5MR", "coef"]) - 1)
st = R.loc["STUNT", "coef"]
box(121, 5, w, h, "#009E73", "4  Dividend",
    f"+1 SD balance at the same\ndevelopment level:\n+{le:.1f} years life expectancy,\n\u2212{abs(u5):.1f}% under-five deaths,\n"
    f"\u2212{abs(st):.1f} pp stunting")

# map
geo = gpd.read_file(RAW / "ssa48_boundaries.geojson").to_crs("ESRI:102022")
CC = pd.read_csv(PROC / "ccd_panel.csv")
geo["D"] = geo.iso3.map(CC[CC.year == 2023].set_index("iso3").D)
m = fig.add_axes([0.635, 0.06, 0.25, 0.8])
geo.plot(column="D", ax=m, cmap="YlGnBu", vmin=0.3, vmax=1.0, edgecolor="white", linewidth=0.3)
m.set_axis_off()
m.set_title("Coordination D, 2023", fontsize=9.5, fontweight="bold")
cax = fig.add_axes([0.66, 0.07, 0.2, 0.025])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(0.3, 1.0)), cax=cax,
                  orientation="horizontal")
cb.ax.tick_params(labelsize=7.5)
ax.add_patch(FancyBboxPatch((238, 8), 25, 78, boxstyle="round,pad=0,rounding_size=2.5", fc="#F4F6FB", ec="#1d1d1f",
                            lw=1.2))
ax.text(250.5, 81, "Policy", ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(250.5, 46, "Invest in\nbridges\n(water, fuel,\nschooling,\nprotection)\n\nTarget each\ncountry's\nlagging\nsubsystem"
        "\n\nCoordinate\nacross\nneighbours", ha="center", va="center", fontsize=8.3, linespacing=1.25)
fig.savefig(FIG / "Graphical_Abstract.tif", dpi=375, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
fig.savefig(FIG / "Graphical_Abstract.png", dpi=375, facecolor="white")
fig.savefig(FIG / "Graphical_Abstract.pdf", facecolor="white")
plt.close(fig)
