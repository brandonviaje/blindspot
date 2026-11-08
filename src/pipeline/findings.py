"""
The extra numbers for the write-up, taken from the backtest proof.py already ran.

  1. Same strictness: 2026 has about twice as many CVEs as earlier years, so 500 picks is a
     tighter cut. Every ranking is also counted at the same share of its year (500 / 72,670).
  2. Same catches or different ones? Which exploited CVEs Blindspot found vs the best standard ranking.
  3. Warning time: days from when the score existed to when CISA added the CVE to KEV.
  4. Review order: if a team worked through the year in this order, how soon would it reach
     the exploited CVEs?
  5. Descriptions are today's text: how many hits mention in-the-wild exploitation?

Saves:
  output/findings.json        the key numbers (the demo header reads them, via prepare_demo.py)
  output/findings_report.log  the printout, for slides (if you redirect it, see below)

Runs after proof.py, before prepare_demo.py. Takes a few seconds.

Run:  python src/pipeline/findings.py > output/findings_report.log
"""

import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]  # lets us import common.py and baselines.py
from common import DATASET_PATH, OUT_DIR, hits_at, load_kev
from baselines import BASELINES, baseline_scores

SB_FRAC = 500 / 72670  # 2026: 500 picks out of 72,670
BLINDSPOT = "Blindspot"
DAY_ONE = "Blindspot, publish-day info only"
BEST = "Exploit code first, then EPSS"  # the strongest standard ranking in both test years
CVSS = "Highest CVSS 3 first"
ORDER = [BLINDSPOT, DAY_ONE, BEST, "EPSS first", CVSS]


def ranked(d, col):
    """The year's CVEs in this ranking's order (ties broken the same way as hits_at)."""
    return d.sort_values([col, "cve_id"], ascending=[False, True])


def sign_test(a, b):
    """Chance of a split at least this lopsided if both rankings were equally good (two-sided)."""
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def main():
    bt = pd.read_parquet(os.path.join(OUT_DIR, "backtest_predictions.parquet"))
    ds = pd.read_parquet(DATASET_PATH, columns=["cve_id", "cohort_year", "cvss3_score", "epss",
                                                "any_exploit_code", "snapshot_end", "kw_in_wild"])
    df = baseline_scores(bt.merge(ds, on=["cve_id", "cohort_year"], how="left"))
    col = {BLINDSPOT: BLINDSPOT, DAY_ONE: DAY_ONE, **{name: c for c, name in BASELINES.items()}}
    kev_date = load_kev().set_index("cve_id").date_added
    years = sorted(int(y) for y in df.cohort_year.unique())
    out = {"test_years": years}

    print("== 1. Exploited CVEs caught: top 500 / same strictness as 500 picks in 2026")
    rows = []
    for y in years:
        d = df[df.cohort_year == y]
        k = round(len(d) * SB_FRAC)
        for name in ORDER:
            rows.append({"year": y, "ranking": name, "top500": hits_at(d, col[name], 500),
                         "strict": hits_at(d, col[name], k), "strict_k": k,
                         "positives": int(d.label.sum())})
    hits = pd.DataFrame(rows)
    print(hits.pivot(index="ranking", columns="year", values=["top500", "strict"])
          .reindex(ORDER).to_string())
    per_year = hits.drop_duplicates("year")
    print("  strict cut = top " + ", ".join(f"{r.strict_k} in {r.year}" for r in per_year.itertuples())
          + "   exploited CVEs: " + ", ".join(f"{r.positives} in {r.year}" for r in per_year.itertuples()))
    out["hits"] = rows

    print(f"\n== 2. Which exploited CVEs each ranking caught in its top 500, vs '{BEST}'")
    out["overlap"] = []
    for y in years:
        d = df[df.cohort_year == y]
        pos = set(d.cve_id[d.label == 1])
        base = set(ranked(d, col[BEST]).cve_id.head(500)) & pos
        for name in (BLINDSPOT, DAY_ONE):
            ours = set(ranked(d, col[name]).cve_id.head(500)) & pos
            a, b = len(ours - base), len(base - ours)
            p = sign_test(a, b)
            print(f"  {y} {name:33s} both {len(ours & base):>2}   only ours {a:>2}   "
                  f"only baseline {b:>2}   p={p:.2f}")
            out["overlap"].append({"year": y, "ranking": name, "both": len(ours & base),
                                   "only_ours": a, "only_baseline": b, "p": round(p, 3)})

    print("\n== 3. Warning time for exploited CVEs in the top 500: days until CISA added them to KEV")
    print("  counted from the end of the publish month, when every input to the score exists")
    out["warning"] = []
    for y in years:
        d = df[df.cohort_year == y]
        for name in (BLINDSPOT, DAY_ONE):
            t = ranked(d, col[name]).head(500)
            t = t[t.label == 1]
            kd = t.cve_id.map(kev_date)
            days = (kd - pd.to_datetime(t.snapshot_end)).dt.days
            from_pub = (kd - pd.to_datetime(t.published)).dt.days
            print(f"  {y} {name:33s} {len(t):>2} hits   median {days.median():.0f} days "
                  f"(range {days.min()} to {days.max()})   {from_pub.median():.0f} days after publication")
            out["warning"].append({"year": y, "ranking": name, "hits": len(t),
                                   "median_days": int(round(days.median())),
                                   "min_days": int(days.min()), "max_days": int(days.max()),
                                   "median_days_from_publication": int(round(from_pub.median()))})

    print("\n== 4. Review order: share of the year's exploited CVEs reached early, working in this order")
    out["review_order"] = []
    for y in years:
        d = df[df.cohort_year == y]
        for name in (BLINDSPOT, DAY_ONE, CVSS):
            r = ranked(d, col[name])
            position = pd.Series(np.arange(1, len(r) + 1) / len(r), index=r.cve_id.values)
            p = position[r.cve_id[r.label == 1].values]
            row = {"year": y, "ranking": name,
                   "in_first_1pct": round(float((p <= .01).mean()), 3),
                   "in_first_5pct": round(float((p <= .05).mean()), 3),
                   "in_first_10pct": round(float((p <= .10).mean()), 3),
                   "median_position": round(float(p.median()), 3)}
            print(f"  {y} {name:33s} first 1%: {row['in_first_1pct']:>4.0%}   first 5%: "
                  f"{row['in_first_5pct']:>4.0%}   first 10%: {row['in_first_10pct']:>4.0%}   "
                  f"median exploited CVE at {row['median_position']:.1%}")
            out["review_order"].append(row)

    print("\n== 5. Descriptions are today's text. Could later edits ('exploited in the wild') explain the hits?")
    out["in_the_wild_wording"] = []
    for y in years:
        d = df[df.cohort_year == y]
        t = ranked(d, BLINDSPOT).head(500)
        t = t[t.label == 1]
        n = int(t.kw_in_wild.sum())
        print(f"  {y}: {n} of Blindspot's {len(t)} hits mention it")
        out["in_the_wild_wording"].append({"year": y, "hits": len(t), "with_wording": n})

    # Header numbers for the demo: the last test year is the one full test of the final model
    # (the 2022-23 training years have no EPSS or exploit-code data)
    last = years[-1]
    w = next(x for x in out["warning"] if x["year"] == last and x["ranking"] == BLINDSPOT)
    positives = next(h["positives"] for h in rows if h["year"] == last)
    out["header"] = {"year": last, "positives": positives, "warning_days_median": w["median_days"]}

    with open(os.path.join(OUT_DIR, "findings.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote output/findings.json")


if __name__ == "__main__":
    main()
