"""Shared configuration: paths, indicator system, colours and figure export.

Every script imports from here so that indicator definitions, seeds and figure settings are identical
throughout the pipeline.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for p in (PROC, RES, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20251120
YEARS = list(range(2000, 2024))
PERIODS = [(2000, 2002), (2003, 2005), (2006, 2008), (2009, 2011), (2012, 2014), (2015, 2017), (2018, 2020),
           (2021, 2023)]

# ---------------------------------------------------------------------------------------------------------------
# Panel indicator system (World Bank WDI, source 2; Worldwide Governance Indicators, source 3).
# code: (short name, label, subsystem, direction (+1 = higher is better), transform)
# ---------------------------------------------------------------------------------------------------------------
PANEL = {
    # Social subsystem: household and community living conditions
    "NY.GDP.PCAP.PP.KD": ("GDP", "GDP per capita, PPP (log)", "Social", +1, "log"),
    "SP.DYN.TFRT.IN": ("TFR", "Total fertility rate", "Social", -1, None),
    "SP.ADO.TFRT": ("ABR", "Adolescent birth rate", "Social", -1, None),
    "SN.ITK.DEFC.ZS": ("PoU", "Prevalence of undernourishment", "Social", -1, None),
    "SL.TLF.CACT.FE.ZS": ("FLFP", "Female labour-force participation", "Social", +1, None),
    # Institutional subsystem: governance (WGI estimates)
    "GOV_WGI_GE.EST": ("GE", "Government effectiveness", "Institutional", +1, None),
    "GOV_WGI_CC.EST": ("CC", "Control of corruption", "Institutional", +1, None),
    "GOV_WGI_RL.EST": ("RL", "Rule of law", "Institutional", +1, None),
    "GOV_WGI_RQ.EST": ("RQ", "Regulatory quality", "Institutional", +1, None),
    "GOV_WGI_VA.EST": ("VA", "Voice and accountability", "Institutional", +1, None),
    "GOV_WGI_PV.EST": ("PV", "Political stability", "Institutional", +1, None),
    # Environmental subsystem: living environment (WASH and energy)
    "SH.H2O.BASW.ZS": ("WAT", "Basic drinking water", "Environmental", +1, None),
    "SH.STA.BASS.ZS": ("SAN", "Basic sanitation", "Environmental", +1, None),
    "EG.CFT.ACCS.ZS": ("FUEL", "Clean cooking fuel", "Environmental", +1, None),
    "EG.ELC.ACCS.ZS": ("ELEC", "Access to electricity", "Environmental", +1, None),
    # Health-system subsystem
    "SH.XPD.CHEX.PP.CD": ("CHE", "Health spending per capita, PPP (log)", "Health system", +1, "log"),
    "SH.XPD.GHED.GE.ZS": ("GHP", "Government health priority", "Health system", +1, None),
    "SH.XPD.OOPC.CH.ZS": ("OOP", "Out-of-pocket share", "Health system", -1, None),
    "SH.IMM.IDPT": ("DTP3", "DTP3 immunisation", "Health system", +1, None),
    "SH.IMM.MEAS": ("MCV1", "Measles immunisation", "Health system", +1, None),
}
OUTCOMES = {  # health and well-being outcomes (not part of any subsystem index)
    "SP.DYN.LE00.IN": ("LE", "Life expectancy at birth (years)", None),
    "SH.DYN.MORT": ("U5MR", "Under-five mortality (log)", "log"),
    "SH.STA.MMRT": ("MMR", "Maternal mortality ratio (log)", "log"),
    "SH.DYN.NMRT": ("NMR", "Neonatal mortality (log)", "log"),
    "SH.STA.STNT.ME.ZS": ("STUNT", "Child stunting (%)", None),
}
POP = "SP.POP.TOTL"
SUBSYSTEMS = ["Social", "Institutional", "Environmental", "Health system"]
SUBCOL = {"Social": "#0072B2", "Institutional": "#CC79A7", "Environmental": "#D55E00", "Health system": "#009E73",
          "Outcome": "#222222"}

# Cross-sectional five-level socioecological system (latest available year; see data/raw/cross_section_48.csv)
LEVELS = {
    "Household": {"POV": ("Poverty", -1), "FSCH": ("Female schooling", +1), "FIES": ("Food insecurity", -1),
                  "TFR": ("Fertility", -1)},
    "Community": {"CHILDMAR": ("Child marriage", -1), "TRUST": ("Social trust", +1),
                  "WATERCOLL": ("Water collection by women", -1), "FLFP": ("Female labour", +1)},
    "Health system": {"NURSE": ("Nurses and midwives", +1), "PHYS": ("Physicians", +1),
                      "lnHEPC": ("Health spending", +1), "OOP": ("Out-of-pocket share", -1)},
    "Governance": {"GOVEFF": ("Government effectiveness", +1), "CORRUPT": ("Control of corruption", +1),
                   "POLSTAB": ("Political stability", +1), "EXCL_EPR": ("Political exclusion", -1)},
    "Ecological": {"WATER_B": ("Basic water", +1), "SAN_SM": ("Safe sanitation", +1), "CLEANFUEL": ("Clean fuel", +1),
                   "PM25": ("PM2.5", -1), "ND_E": ("Climate exposure", -1)},
}
XOUT = {"U5MR": "Under-five mortality", "MMR": "Maternal mortality", "SCI": "UHC service coverage"}
LEVCOL = {"Household": "#0072B2", "Community": "#56B4E9", "Health system": "#009E73", "Governance": "#CC79A7",
          "Ecological": "#D55E00", "Outcome": "#222222"}

# Sub-regions (African Union regions) used for descriptive contrasts
AU_REGION = {
    "Central": ["CMR", "CAF", "TCD", "COG", "COD", "GNQ", "GAB", "BDI", "STP"],
    "East": ["COM", "ERI", "ETH", "KEN", "MDG", "MUS", "RWA", "SYC", "SOM", "SSD", "SDN", "TZA", "UGA"],
    "Southern": ["AGO", "BWA", "SWZ", "LSO", "MWI", "MOZ", "NAM", "ZAF", "ZMB", "ZWE"],
    "West": ["BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "SEN", "SLE",
             "TGO"],
}
REGION_OF = {iso: r for r, l in AU_REGION.items() for iso in l}

# Coupling coordination classes (Liao, 1999 style ten-grade scale collapsed to five classes)
CLASSES = [(0.0, 0.4, "Imbalanced"), (0.4, 0.5, "Near imbalance"), (0.5, 0.6, "Barely coordinated"),
           (0.6, 0.7, "Primary coordination"), (0.7, 1.01, "Intermediate or better")]

# ---------------------------------------------------------------------------------------------------------------
# Figure style and export: vector PDF (fonts embedded), 1000-dpi LZW TIFF (Elsevier line-art minimum) and a
# 600-dpi PNG used for the Word files.
# ---------------------------------------------------------------------------------------------------------------
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
})


def save_fig(fig, name, tiff_dpi=1000, png_dpi=600):
    """Export one figure in the three formats used for submission."""
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{name}.tif", dpi=tiff_dpi, bbox_inches="tight", pad_inches=0.03,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIG / f"{name}.png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
