"""
train.py — Blindspot exploit-likelihood model.

  python model/train.py            backtests + feature-removal table + leakage audit
  python model/train.py --final    also trains on 2022-2025 and scores the 2026 scoreboard:
                                   writes top500.csv, output/risk_feed.csv, and (if
                                   src/blindspot.db exists) cve_data.predicted_risk_score

Evaluation
  forward   train on cohorts before Y, test on Y (honest: what we could have done then)
  loyo      leave-one-year-out over 2022-2025 (more positives -> steadier comparison)
Metrics
  hits@500  positives in the top 500 (the judges' metric)
  hits@top0.69%  positives in the top 500/72,670 of the cohort — the 2026 scoreboard is
                 2-4x larger than past cohorts, so this matches its selectivity
"""

import argparse
import os
import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (CACHE_DIR, DATASET_PATH, DB_PATH, OUT_DIR, REPO_ROOT,
                    SCORE_YEAR, TRAIN_YEARS, hits_at)

SB_FRAC = 500 / 72670

KW = ["kw_unauth", "kw_rce", "kw_auth_bypass", "kw_privesc", "kw_in_wild", "kw_memory",
      "kw_deserial", "kw_path_trav", "kw_file_upload", "kw_sqli", "kw_xss", "kw_csrf",
      "kw_dos", "kw_edge", "kw_mail", "kw_mgmt", "kw_browser_os", "kw_wordpress",
      "kw_mobile_app", "kw_requires_auth", "kw_local"]

FEATURE_GROUPS = {
    "horizon":     ["months_to_cutoff"],
    "assigner":    ["assigner_kev_all", "assigner_kev_12m", "assigner_cves_12m",
                    "assigner_kev_rate_12m"],
    "vendor":      ["vendor_kev_all", "vendor_kev_12m", "product_kev_all", "product_kev_12m"],
    "citations":   ["cites_cve", "cites_kev_cve"],
    "keywords":    KW + ["desc_len"],
    "text_model":  ["text_score"],
    "cvss4_cna":   ["cvss4_score", "cvss4_av_network", "cvss4_pr_none", "cvss4_ui_none"],
    "signals":     ["epss", "epss_pct_cohort", "exploitdb", "metasploit", "nuclei",
                    "poc_github", "any_exploit_code"],
    # never used in the submitted model — leakage audit only
    "LEAKY":       ["leak_nvd_reviewed", "leak_cvss3_missing", "leak_cvss3_by_nist",
                    "leak_cwe_missing", "n_references", "n_affected_cpes", "cvss3_score"],
}
SAFE_GROUPS = [g for g in FEATURE_GROUPS if g != "LEAKY"]

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_child_samples=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              scale_pos_weight=10.0, verbose=-1, n_jobs=8)
N_ROUNDS = 300
SEEDS = [0, 1, 2, 3, 4]


def features(groups):
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def load():
    df = pd.read_parquet(DATASET_PATH)
    ts = pd.read_parquet(os.path.join(CACHE_DIR, "text_scores.parquet"))
    return df.merge(ts, on=["cve_id", "cohort_year"], how="left")


def fit_predict(train, test, cols):
    pred = np.zeros(len(test))
    for s in SEEDS:
        m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(train[cols], train.label),
                      num_boost_round=N_ROUNDS)
        pred += m.predict(test[cols]) / len(SEEDS)
    return pred


def fit_final(train, cols):
    return [lgb.train({**PARAMS, "seed": s}, lgb.Dataset(train[cols], train.label),
                      num_boost_round=N_ROUNDS) for s in SEEDS]


def score(test, pred):
    t = test.assign(score=pred)
    k = max(1, round(len(t) * SB_FRAC))
    return hits_at(t, "score", 500), hits_at(t, "score", k)


def evaluate(df, cols, mode):
    """Returns {year: (hits@500, hits@frac)}."""
    lab = df[df.cohort_year.isin(TRAIN_YEARS)]
    res = {}
    years = [2024, 2025] if mode == "forward" else TRAIN_YEARS
    for y in years:
        test = lab[lab.cohort_year == y]
        train = lab[lab.cohort_year < y] if mode == "forward" else lab[lab.cohort_year != y]
        res[y] = score(test, fit_predict(train, test, cols))
    return res


def fmt(res):
    return "  ".join(f"{y}: {a:>2}/{b:<2}" for y, (a, b) in res.items())


def total(res):
    return sum(a for a, _ in res.values()), sum(b for _, b in res.values())


def report(df):
    lab = df[df.cohort_year.isin(TRAIN_YEARS)]
    print("positives per cohort:", lab.groupby("cohort_year").label.sum().astype(int).to_dict())
    print("format per year = hits@500 / hits@top0.69%\n")

    full = features(SAFE_GROUPS)
    print("== Full safe model")
    for mode in ("forward", "loyo"):
        r = evaluate(df, full, mode)
        print(f"  {mode:8s} {fmt(r)}   total {total(r)}")

    print("\n== Feature removal (loyo, drop one group)")
    base = total(evaluate(df, full, "loyo"))
    print(f"  {'(none)':12s} total {base}")
    for g in SAFE_GROUPS:
        if g == "horizon":
            continue
        r = total(evaluate(df, features([x for x in SAFE_GROUPS if x != g]), "loyo"))
        print(f"  -{g:11s} total {r}   change {r[0]-base[0]:+d} / {r[1]-base[1]:+d}")

    print("\n== Publish-day only (no EPSS / exploit code) — the 'unreviewed, day one' claim")
    r = evaluate(df, features([g for g in SAFE_GROUPS if g != "signals"]), "loyo")
    print(f"  loyo     {fmt(r)}   total {total(r)}")

    print("\n== Leakage audit: adding the columns the README warns about")
    r = evaluate(df, full + FEATURE_GROUPS["LEAKY"], "loyo")
    print(f"  loyo     {fmt(r)}   total {total(r)}   (vs {base} without)")


def final(df):
    cols = features(SAFE_GROUPS)
    train = df[df.cohort_year.isin(TRAIN_YEARS)]
    test = df[df.cohort_year == SCORE_YEAR].copy()
    models = fit_final(train, cols)
    test["predicted_risk_score"] = np.mean([m.predict(test[cols]) for m in models], axis=0)
    test = test.sort_values(["predicted_risk_score", "cve_id"], ascending=[False, True])
    test["rank"] = np.arange(1, len(test) + 1)

    top500 = os.path.join(REPO_ROOT, "top500.csv")
    test[["cve_id"]].head(500).to_csv(top500, index=False)
    feed_cols = ["rank", "cve_id", "published", "snapshot_month", "predicted_risk_score",
                 "desc_vendor", "desc_product", "epss", "any_exploit_code", "text_score",
                 "vendor_kev_12m", "assigner_kev_12m", "kw_unauth", "kw_rce", "kw_in_wild",
                 "kw_edge", "description"]
    test[feed_cols].to_csv(os.path.join(OUT_DIR, "risk_feed.csv"), index=False)

    imp = pd.DataFrame({"feature": cols,
                        "gain": np.mean([m.feature_importance("gain") for m in models], axis=0)})
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    print("\n== Final model (trained 2022-2025) -> 2026")
    print(f"  wrote {top500} and output/risk_feed.csv")
    print("  top features:", ", ".join(imp.feature.head(10)))
    print("  top 15:")
    print(test[["rank", "cve_id", "predicted_risk_score", "desc_vendor", "epss",
                "any_exploit_code"]].head(15).to_string(index=False))

    if os.path.isfile(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        existing = {r[1] for r in cur.execute("PRAGMA table_info(cve_data)")}
        if "predicted_risk_score" not in existing:
            cur.execute("ALTER TABLE cve_data ADD COLUMN predicted_risk_score REAL")
        cur.executemany("UPDATE cve_data SET predicted_risk_score=? WHERE cve_id=?",
                        list(zip(test.predicted_risk_score.astype(float), test.cve_id)))
        conn.commit()
        conn.close()
        print(f"  wrote predicted_risk_score to {DB_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    a = ap.parse_args()
    df = load()
    if not a.skip_report:
        report(df)
    if a.final:
        final(df)


if __name__ == "__main__":
    main()
