"""Step 0. (Optional) Re-download the panel from the World Bank API.

The repository ships the extract used in the paper (data/raw/wdi_wgi_long_2000_2023.csv, retrieved on
20 November 2026). Running this script with --refresh downloads the same series again; values can change when the
World Bank revises its databases, so results may differ slightly from the published ones.

Usage:  python scripts/00_fetch_wdi.py            # checks the shipped extract
        python scripts/00_fetch_wdi.py --refresh  # downloads a fresh extract (needs internet access)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.config import RAW, PANEL, OUTCOMES, POP  # noqa: E402

API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicators}"
OUTFILE = RAW / "wdi_wgi_long_2000_2023.csv"


def fetch(indicators, source, isos, retries=3):
    import requests
    url = API.format(countries=";".join(isos), indicators=";".join(indicators))
    params = {"source": source, "format": "json", "date": "2000:2023", "per_page": 5000}
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.ok:
            meta, recs = r.json()
            assert meta["pages"] == 1, meta
            return [(x["countryiso3code"], x["date"], x["indicator"]["id"], x["value"]) for x in recs
                    if x["value"] is not None]
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    isos = [row["iso3"] for row in csv.DictReader(open(RAW / "countries.csv"))]
    wdi = [c for c in list(PANEL) + list(OUTCOMES) + [POP] if not c.startswith("GOV_WGI")]
    wgi = [c for c in PANEL if c.startswith("GOV_WGI")]
    if not a.refresh:
        n = sum(1 for _ in open(OUTFILE)) - 1
        print(f"shipped extract: {OUTFILE.name}, {n:,} rows; use --refresh to download again")
        return
    rows = []
    for group, src in [(wdi[i:i + 3], 2) for i in range(0, len(wdi), 3)] + [(wgi[:3], 3), (wgi[3:], 3)]:
        rows += fetch(group, src, isos)
        print("fetched", group, len(rows))
    with open(OUTFILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso3", "year", "indicator", "value"])
        w.writerows(rows)
    print("wrote", OUTFILE, len(rows))


if __name__ == "__main__":
    main()
