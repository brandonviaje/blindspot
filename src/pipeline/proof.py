"""
Makes the proof: the numbers that show Blindspot works.

It replays 2024 and 2025 (training only on earlier years, so the model never sees the answers) and measures:
  1. Blindspot vs CVSS / EPSS / exploit code: exploited CVEs caught in the top 500
  2. Blindspot with publish-day info only (no EPSS, no exploit code, nothing from NIST)
  3. The NIST leak: every exploited CVE got reviewed, many others didn't
  4. Our 2026 top 500: how many NIST hasn't reviewed, how many have no CVSS score

Saves:
  output/results.csv                    → the website's "Does it work?" chart (via prepare_demo.py)
  output/backtest_predictions.parquet   → every 2024/2025 score, reused by the next tests
  output/proof_report.log               → the printout, for slides (if you redirect it, see below)

Runs after train.py --final, before prepare_demo.py. ~2 min.

Run:  python src/pipeline/proof.py > output/proof_report.log
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lets us import common.py
from common import OUT_DIR, REPO_ROOT, hits_at

import train as T                                   # reuse the model code, so this tests the real thing
from baselines import BASELINES, baseline_scores    # and the same standard rankings

REVIEWED = ["Analyzed", "Modified"]  # NIST statuses that mean "has been reviewed"


def main():
    df = T.load()
    lab = df[df.cohort_year.isin(T.TRAIN_YEARS)]
    models = {
        "Blindspot": T.features(T.SAFE_GROUPS),
        "Blindspot, publish-day info only": T.features([g for g in T.SAFE_GROUPS if g != "signals"]),
    }

    # 1 + 2. Forward test: train on years before Y, test on Y
    rows, preds = [], []
    for y in (2024, 2025):
        test = baseline_scores(lab[lab.cohort_year == y])
        train = lab[lab.cohort_year < y]
        for col, name in BASELINES.items():
            rows.append({"year": y, "ranking": name, "hits": hits_at(test, col)})
        p = test[["cve_id", "cohort_year", "published", "label", "nvd_status"]].copy()
        for name, cols in models.items():
            p[name] = T.fit_predict(train, test, cols)
            rows.append({"year": y, "ranking": name, "hits": hits_at(p, name)})
        preds.append(p)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    pd.concat(preds).to_parquet(os.path.join(OUT_DIR, "backtest_predictions.parquet"), index=False)

    print("== Exploited CVEs in the top 500 (trained only on earlier years)")
    print(res.pivot(index="ranking", columns="year", values="hits").to_string())

    # 3. The leak: does NIST review status give the answer away?
    print("\n== NIST review status today, by whether the CVE was exploited")
    for y in (2024, 2025):
        d = lab[lab.cohort_year == y]
        unrev = ~d.nvd_status.isin(REVIEWED)
        print(f"  {y}: unreviewed {unrev.sum():>6,} of {len(d):,} CVEs ({unrev.mean():.0%})   "
              f"exploited & unreviewed {int((unrev & (d.label == 1)).sum())} of {int(d.label.sum())}")
    print("  → every exploited CVE got reviewed, so review status leaks the answer. Blindspot never uses it.")

    # 4. What our 2026 top 500 looks like
    top = pd.read_csv(os.path.join(REPO_ROOT, "top500.csv"))
    s26 = df[df.cohort_year == T.SCORE_YEAR].set_index("cve_id")
    t = s26.loc[top.cve_id]
    print("\n== Our 2026 top 500")
    print(f"  not yet reviewed by NIST: {int((~t.nvd_status.isin(REVIEWED)).sum())}")
    print(f"  no CVSS 3 score at all:   {int(t.cvss3_score.isna().sum())}  (invisible to 'highest CVSS first')")
    print(f"  (all 2026 scoreboard: {int((~s26.nvd_status.isin(REVIEWED)).sum()):,} of {len(s26):,} unreviewed)")


if __name__ == "__main__":
    main()
