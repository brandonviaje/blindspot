"""
"Does this description sound like past attacked bugs?"

A simple word-matching model. For each year it learns which words showed up in
descriptions of already-exploited CVEs (using only EARLIER years), then gives every
CVE in that year a text_score. Higher = sounds more like past attacks.

Saves cache/text_scores.parquet → used as a feature by train.py.
Takes ~2 min.

Run:  python src/pipeline/text_model.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lets us import common.py
from common import CACHE_DIR, DATASET_PATH, SCORE_YEAR, TRAIN_YEARS, load_kev, load_vulns

OUT = os.path.join(CACHE_DIR, "text_scores.parquet")


def main():
    t0 = time.time()
    v = load_vulns()
    kev = load_kev()
    ds = pd.read_parquet(DATASET_PATH, columns=["cve_id", "cohort_year", "description"])
    v["kev_date"] = v.cve_id.map(kev.set_index("cve_id")["date_added"])

    out = []
    for y in TRAIN_YEARS + [SCORE_YEAR]:
        # Learn from 2015 up to the year before, with answers as known on Jan 1 of year y
        hist = v[(v.published >= "2015-01-01") & (v.published < f"{y}-01-01")]
        answer = (hist.kev_date < f"{y}-01-01").astype(int).values

        # Words and word pairs → weights, then a simple classifier on top
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000,
                              sublinear_tf=True, stop_words="english", dtype=np.float32)
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, solver="liblinear")
        clf.fit(vec.fit_transform(hist.description), answer)

        # Score this year's CVEs
        tgt = ds[ds.cohort_year == y]
        s = clf.decision_function(vec.transform(tgt.description))
        out.append(pd.DataFrame({"cve_id": tgt.cve_id.values, "cohort_year": y, "text_score": s}))
        print(f"[{time.time()-t0:6.1f}s] {y}: trained on {len(hist)} docs / {answer.sum()} KEV", flush=True)

        # For curiosity: the words that most scream "exploited"
        if y == SCORE_YEAR:
            terms = np.array(vec.get_feature_names_out())
            print("top KEV-like terms:", ", ".join(terms[np.argsort(clf.coef_[0])[-30:]][::-1]))

    pd.concat(out).to_parquet(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
