"""
text_model.py — "does this description read like past exploited vulnerabilities?"

For each cohort year Y, fits TF-IDF + logistic regression on descriptions
published 2015..Y-1, labelled 1 if on KEV before Jan 1 of Y. Scores the
cohort-Y rows with it. Strictly backward-looking, so the score is safe to use
as a feature for every cohort (and for 2026).

Run:  python model/text_model.py      (writes cache/text_scores.parquet)
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import CACHE_DIR, DATASET_PATH, SCORE_YEAR, TRAIN_YEARS, load_kev, load_vulns

OUT = os.path.join(CACHE_DIR, "text_scores.parquet")


def main():
    t0 = time.time()
    v = load_vulns()
    kev = load_kev()
    ds = pd.read_parquet(DATASET_PATH, columns=["cve_id", "cohort_year", "description"])
    kev_date = kev.set_index("cve_id")["date_added"]
    v["kev_date"] = v.cve_id.map(kev_date)
    out = []
    for y in TRAIN_YEARS + [SCORE_YEAR]:
        hist = v[(v.published >= "2015-01-01") & (v.published < f"{y}-01-01")]
        yl = (hist.kev_date < f"{y}-01-01").astype(int).values
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000,
                              sublinear_tf=True, stop_words="english", dtype=np.float32)
        X = vec.fit_transform(hist.description)
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, solver="liblinear")
        clf.fit(X, yl)
        tgt = ds[ds.cohort_year == y]
        s = clf.decision_function(vec.transform(tgt.description))
        out.append(pd.DataFrame({"cve_id": tgt.cve_id.values, "cohort_year": y, "text_score": s}))
        print(f"[{time.time()-t0:6.1f}s] {y}: trained on {len(hist)} docs / {yl.sum()} KEV", flush=True)
        if y == SCORE_YEAR:
            terms = np.array(vec.get_feature_names_out())
            order = np.argsort(clf.coef_[0])
            print("top KEV-like terms:", ", ".join(terms[order[-30:]][::-1]))
    pd.concat(out).to_parquet(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
