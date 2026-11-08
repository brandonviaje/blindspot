"""
prepare_demo.py — package everything the Streamlit app needs into blindspot/demo_data/.

Run on the machine that has the model cache (after train.py --final and explain.py).
Copy (or commit) demo_data/ to the presentation laptops; the app reads only that folder.

  feed.parquet           every 2026 scoreboard vulnerability, ranked, with key fields
  explanations.csv       top-N "why" text (template + Granite, if generated)
  neighbors.parquet      top-10 similar past vulnerabilities for the explained rows
  kev_emb.f16.npy        Granite embeddings of already-exploited vulnerabilities (live scoring)
  kev_meta.parquet       cve_id, vendor, product, name, date added, ransomware use for those rows
  results.csv            backtest numbers for the "Does it work?" tab (from proof.py)

Run:  python src/web/prepare_demo.py
"""

import os
import shutil

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

from common import CACHE_DIR, OUT_DIR, REPO_ROOT, load_kev, load_vulns

DEMO = os.path.join(REPO_ROOT, "demo_data")


def main():
    os.makedirs(DEMO, exist_ok=True)

    feed = pd.read_csv(os.path.join(OUT_DIR, "risk_feed.csv"))
    v = load_vulns()[["cve_id", "nvd_status"]]
    feed = feed.merge(v, on="cve_id", how="left")
    feed["nist_reviewed"] = feed.nvd_status.isin(["Analyzed", "Modified"])  # display only
    feed = feed.drop(columns=["nvd_status"])
    g = pd.read_parquet(os.path.join(CACHE_DIR, "granite_features.parquet"))
    g = g[g.cohort_year == 2026][["cve_id", "g_knn10_kev", "g_kev_top5_sim"]]
    feed = feed.merge(g, on="cve_id", how="left")
    feed.to_parquet(os.path.join(DEMO, "feed.parquet"), index=False)

    shutil.copy(os.path.join(OUT_DIR, "explanations.csv"), DEMO)

    nb = pd.read_csv(os.path.join(OUT_DIR, "explanations_neighbors.csv"))
    desc = load_vulns().set_index("cve_id").description
    nb["neighbor_description"] = nb.neighbor_id.map(desc).str.slice(0, 400)
    nb.to_parquet(os.path.join(DEMO, "neighbors.parquet"), index=False)

    ids = pd.read_parquet(os.path.join(CACHE_DIR, "granite_ids.parquet"))
    emb = np.load(os.path.join(CACHE_DIR, "granite_emb.f16.npy"), mmap_mode="r")
    kev = load_kev()
    pos = pd.Series(np.arange(len(ids)), index=ids.cve_id)
    kev = kev[kev.cve_id.isin(pos.index)].copy()
    np.save(os.path.join(DEMO, "kev_emb.f16.npy"), np.asarray(emb[pos[kev.cve_id].values]))
    keep = [c for c in ["cve_id", "vendor", "product", "name", "date_added",
                        "known_ransomware_use"] if c in kev.columns]
    kev = kev[keep].merge(ids, on="cve_id", how="left")
    kev["description"] = kev.cve_id.map(desc).str.slice(0, 400)
    kev.to_parquet(os.path.join(DEMO, "kev_meta.parquet"), index=False)

    shutil.copy(os.path.join(OUT_DIR, "results.csv"), DEMO)  # made by src/pipeline/proof.py

    for f in sorted(os.listdir(DEMO)):
        print(f"{os.path.getsize(os.path.join(DEMO, f)) / 1e6:7.1f} MB  {f}")


if __name__ == "__main__":
    main()
