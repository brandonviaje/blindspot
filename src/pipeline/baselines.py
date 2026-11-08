"""
The scores to beat.

Ranks 2024 and 2025 the "standard" ways (highest CVSS, highest EPSS,
exploit code then EPSS) and counts how many exploited CVEs land in the top 500.
Just for comparison, nothing else uses it.
(The README's numbers for 2026: CVSS 7, EPSS 6, exploit code then EPSS 8.)

Run:  python src/pipeline/baselines.py > output/baselines.log
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lets us import common.py
from common import DATASET_PATH, hits_at

# name of the score column → how it shows up in the printout
BASELINES = {"b_cvss3": "Highest CVSS 3 first",
             "b_epss": "EPSS first",
             "b_code_epss": "Exploit code first, then EPSS"}


def baseline_scores(d):
    """Add one score column per standard ranking (missing values go to the bottom)."""
    d = d.copy()
    d["b_cvss3"] = d.cvss3_score.fillna(-1)
    d["b_epss"] = d.epss.fillna(-1)
    d["b_code_epss"] = d.any_exploit_code.fillna(0) * 10 + d.epss.fillna(0)  # code first, EPSS breaks ties
    return d


def main():
    df = pd.read_parquet(DATASET_PATH)
    rows = []
    for y in (2024, 2025):
        d = baseline_scores(df[df.cohort_year == y])
        r = {"year": y, "rows": len(d), "positives": int(d.label.sum())}
        for col, name in BASELINES.items():
            r[name] = hits_at(d, col)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    # For scale: what picking 500 at random would get you
    for y in (2024, 2025):
        d = df[df.cohort_year == y]
        print(f"random top-500 expectation {y}: {500 * d.label.mean():.2f}")


if __name__ == "__main__":
    main()
