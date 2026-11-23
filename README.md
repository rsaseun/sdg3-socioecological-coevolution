# sdg3-socioecological-coevolution

Code and data for the article:

> **Balanced development as a determinant of health: socioecological co-evolution, coordination dynamics and the well-being dividend in Sub-Saharan Africa, 2000–2023.**
> Raji, S. A., Oladimeji, A., Demehin, M., & Wunude, E. Submitted to *SSHO Health & Well-being* (Elsevier).

The study covers all 48 countries of the World Bank Sub-Saharan Africa region and answers four research questions:

| RQ | Question | Main methods | Script |
|----|----------|--------------|--------|
| RQ1 | How are household, community, health-system, governance and ecological conditions connected across levels, and do the theoretical levels form empirical modules? | Nonparanormal graphical lasso (repeated-CV, one-SE penalty), bridge strength, predictability, Louvain modules, adjusted Rand index, modularity permutation test, bootstrap and case-dropping stability | `scripts/02_rq1_network.py` |
| RQ2 | Which conditions led and which followed in their co-evolution with health, 2000–2023? | Regularised first-difference panel VAR (lead–lag network) with country-block bootstrap; Dumitrescu–Hurlin panel Granger tests | `scripts/04_rq2_leadlag.py` |
| RQ3 | Has the coupling coordination of the social, institutional, environmental and health-system subsystems improved and converged, and is low coordination persistent and spatially contagious? | Entropy-weighted coupling coordination on pooled goalposts; σ- and β-convergence; Markov and spatial Markov chains (Rey homogeneity tests) | `scripts/03_indices_ccd.py`, `scripts/05_rq3_dynamics.py` |
| RQ4 | Is there a health dividend of balanced development beyond its level, and what obstructs coordination? | Two-way fixed effects with Driscoll–Kraay errors; within–between (Mundlak) model; counterfactual dividend; obstacle degree model | `scripts/06_rq4_dividend.py` |

Robustness checks are in `scripts/07_robustness.py`.

## Quick start

```bash
git clone https://github.com/<account>/sdg3-socioecological-coevolution.git
cd sdg3-socioecological-coevolution
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py          # about 10 minutes; add --skip-network to reuse the RQ1 bootstrap
pytest -q                  # unit tests for the core methods
```

Python 3.11 or later is required (the published results were produced with Python 3.13). A conda environment is
provided in `environment.yml`.

## Repository structure

```
data/
  raw/wdi_wgi_long_2000_2023.csv   World Bank API extract (20 Nov 2026): 20 indicators, 5 outcomes, population
  raw/cross_section_48.csv         latest-year cross-section (21 determinants, 3 SDG 3 outcomes)
  raw/ssa48_boundaries.geojson     country boundaries for maps and spatial weights
  raw/countries.csv                country names and ISO3 codes
  codebook.csv                     every variable: label, direction, transformation, source
  processed/                       panel, flags, normalised indicators, coordination panel (generated)
src/sehs/
  config.py                        paths, indicator system, colours, figure export (PDF, 1000-dpi TIFF, PNG)
  methods.py                       all statistical methods as tested functions
scripts/                           00 fetch → 01 panel → 02 RQ1 → 03 indices → 04 RQ2 → 05 RQ3 → 06 RQ4 → 07 robustness
figures/                           one script per figure (Figures 1–8, S1–S4, graphical abstract)
tables/make_tables.py              Tables 1–6 and S1–S7 (CSV and one Excel workbook)
outputs/                           results/ (CSV, JSON), figures/, tables/ (generated)
tests/                             pytest unit tests
run_all.py                         runs every step in order
```

## Figures and tables

| Output | Script | Content |
|--------|--------|---------|
| Figure 1 | `figures/fig01_framework.py` | Conceptual framework and study design |
| Figure 2 | `figures/fig02_rq1_network.py` | Cross-level network, bridge strength, module recovery |
| Figure 3 | `figures/fig03_rq2_leadlag.py` | Lead–lag network, drivers and responders, panel Granger tests |
| Figure 4 | `figures/fig04_rq3_trajectories.py` | Trajectories, densities, β- and σ-convergence |
| Figure 5 | `figures/fig05_rq3_maps.py` | Maps of D (2000, 2023), change and lagging subsystem |
| Figure 6 | `figures/fig06_rq3_markov.py` | Markov and spatial Markov transitions |
| Figure 7 | `figures/fig07_rq4_dividend.py` | Coordination dividend, robustness, heterogeneity, counterfactual |
| Figure 8 | `figures/fig08_rq4_obstacles.py` | Obstacle degrees by country, 2023 |
| Figures S1–S4 | `figures/figS_supplementary.py` | Provenance, network stability, subsystem maps, trajectories |
| Graphical abstract | `figures/graphical_abstract.py` | Summary graphic |
| Tables 1–6, S1–S7 | `tables/make_tables.py` | All tables (`outputs/tables/*.csv`, `All_Tables.xlsx`) |

Figures are written as vector PDF (fonts embedded), 1000-dpi LZW-compressed TIFF (Elsevier line-art minimum) and
600-dpi PNG.

## Data

* **Panel (RQ2–RQ4).** World Bank World Development Indicators (source 2) and Worldwide Governance Indicators
  (source 3), 48 countries × 2000–2023, retrieved through the World Bank API on 20 November 2026. The shipped extract
  is used by default; `python scripts/00_fetch_wdi.py --refresh` downloads a new one (values may differ after
  World Bank revisions). Gaps: 97.3% of cells observed; 2.1% interpolated within series or carried from the nearest
  year; 0.56% (seven whole series in six countries) imputed by year-wise k-nearest neighbours using subsystem
  indicators only. Every filled cell is flagged in `data/processed/panel_flags.csv`.
* **Cross-section (RQ1).** Latest available values (mostly 2019–2023) from WHO, UN IGME, WHO/UNICEF JMP, the
  Worldwide Governance Indicators, Ethnic Power Relations, FAO, UNICEF, ILO, UNDP, Afrobarometer and ND-GAIN; see
  `data/codebook.csv`.
* **Boundaries.** Natural Earth 1:110m country polygons (public domain), with Somaliland merged into Somalia and
  the five small island states (Cabo Verde, Comoros, Mauritius, São Tomé and Príncipe, Seychelles) represented by
  schematic circles. Map lines delineate study areas and do not necessarily depict accepted national boundaries.

World Bank data are licensed under CC BY 4.0. All other inputs are public aggregate statistics; please cite the
original providers listed in the codebook.

## Reproducibility

* Fixed random seed (`SEED = 20251120` in `src/sehs/config.py`) for all bootstraps, cross-validation folds and
  community detection.
* `requirements.txt` pins the package versions used for the published results.
* A clean run of `python run_all.py` regenerates every number, figure and table reported in the article.

## Citation

Please cite the article and this repository (see `CITATION.cff`). A Zenodo DOI will be minted on release.

## Licence

Code: MIT licence (`LICENSE`). Derived data in `data/` and `outputs/`: CC BY 4.0, with attribution to the
original data providers.

## Contact

Saheed A. Raji (corresponding author), Fire/EMS Department, Prince George's County, Largo, Maryland, United States —
SARaji@co.pg.md.us · ORCID [0000-0002-5890-5414](https://orcid.org/0000-0002-5890-5414)
