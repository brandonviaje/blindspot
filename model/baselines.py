"""
baselines.py — the README's standard rankings, replayed on past years.

README numbers on the 2026 scoreboard: CVSS 7, EPSS 6, exploit code then EPSS 8.

Run:  python model/baselines.py
"""

import numpy as np
import pandas as pd

from common import DATASET_PATH, hits_at


def baseline_scores(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["b_cvss3"] = d.cvss3_score.fillna(-1)
    d["b_epss"] = d.epss.fillna(-1)
    d["b_code_epss"] = d.any_exploit_code.fillna(0) * 10 + d.epss.fillna(0)
    return d


BASELINES = {"b_cvss3": "Highest CVSS 3 first",
             "b_epss": "EPSS first",
             "b_code_epss": "Exploit code first, then EPSS"}


def main():
    df = pd.read_parquet(DATASET_PATH)
    rows = []
    for y in (2024, 2025):
        d = baseline_scores(df[df.cohort_year == y])
        r = {"year": y, "rows": len(d), "positives": int(d.label.sum())}
        for col, name in BASELINES.items():
            r[name] = hits_at(d, col)
        rows.append(r)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    for y in (2024, 2025):
        d = df[df.cohort_year == y]
        print(f"random top-500 expectation {y}: {500 * d.label.mean():.2f}")


if __name__ == "__main__":
    main()
