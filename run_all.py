"""Reproduce every result, figure and table of the paper in one command:  python run_all.py

Steps run in order; each writes to data/processed or outputs/. Total run time is about 10 minutes on a laptop
(the network bootstrap in step 02 takes most of it). Use --skip-network to reuse an existing RQ1 result.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "scripts/00_fetch_wdi.py",
    "scripts/01_build_panel.py",
    "scripts/02_rq1_network.py",
    "scripts/03_indices_ccd.py",
    "scripts/04_rq2_leadlag.py",
    "scripts/05_rq3_dynamics.py",
    "scripts/06_rq4_dividend.py",
    "scripts/07_robustness.py",
    "figures/fig01_framework.py",
    "figures/fig02_rq1_network.py",
    "figures/fig03_rq2_leadlag.py",
    "figures/fig04_rq3_trajectories.py",
    "figures/fig05_rq3_maps.py",
    "figures/fig06_rq3_markov.py",
    "figures/fig07_rq4_dividend.py",
    "figures/fig08_rq4_obstacles.py",
    "figures/figS_supplementary.py",
    "figures/graphical_abstract.py",
    "tables/make_tables.py",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-network", action="store_true", help="do not rerun scripts/02_rq1_network.py")
    a = ap.parse_args()
    t0 = time.time()
    for s in STEPS:
        if a.skip_network and s.endswith("02_rq1_network.py") and (ROOT / "outputs/results/rq1_summary.json").exists():
            print(f"-- skipping {s}")
            continue
        t = time.time()
        print(f"== {s}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT)
        if r.returncode:
            sys.exit(f"step failed: {s}")
        print(f"   done in {time.time() - t:.0f} s", flush=True)
    print(f"all steps finished in {(time.time() - t0) / 60:.1f} min")
