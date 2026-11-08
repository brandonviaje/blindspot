"""
which past bugs look like this one, and did they get attacked?

Uses the numbers from embed_granite.py. For every CVE we find its most similar
*older* CVEs and count how many were already on the KEV list at that time.
(No peeking at the future: only bugs published before it, only KEV entries known by then.)

Saves to cache/:
  granite_features.parquet        features for the model. The two it uses:
                                    g_knn10_kev     how many of the 10 closest were exploited
                                    g_kev_top5_sim  how close it is to the 5 nearest exploited ones
                                  (the other g_* columns were tried and didn't help)
  granite_neighbors_2026.parquet  the 10 closest matches for every 2026 CVE → evidence on the website

Run:  python src/granite/granite_neighbors.py   (GPU makes it fast; CPU works, just slow)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lets us import common.py
from common import CACHE_DIR, DATASET_PATH, SCORE_YEAR, load_kev

EMB_PATH = os.path.join(CACHE_DIR, "granite_emb.f16.npy")
IDS_PATH = os.path.join(CACHE_DIR, "granite_ids.parquet")
OUT = os.path.join(CACHE_DIR, "granite_features.parquet")
OUT_NB = os.path.join(CACHE_DIR, "granite_neighbors_2026.parquet")
K = 50         # neighbours to look at per CVE
CHUNK = 1024   # CVEs per GPU batch

def days(s):
    """Dates → plain day numbers, so the GPU can compare them."""
    return pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int64)

def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32

    # Every embedded CVE ("the pool"), with its publish date and KEV date (if any)
    ids = pd.read_parquet(IDS_PATH)
    emb = torch.from_numpy(np.load(EMB_PATH).astype(np.float16 if dev == "cuda" else np.float32)).to(dev)
    pool_pub = torch.from_numpy(days(ids.published)).to(dev)

    kev = load_kev()
    kev_date = ids.cve_id.map(kev.set_index("cve_id")["date_added"])
    never = np.iinfo(np.int64).max // 2  # "never added to KEV"
    kd = np.where(kev_date.notna(), days(kev_date.fillna(pd.Timestamp("2100-01-01"))), never)
    pool_kev = torch.from_numpy(kd).to(dev)
    kev_idx = torch.nonzero(torch.from_numpy(kev_date.notna().values).to(dev)).squeeze(1)
    kev_emb, kev_pub, kev_kd = emb[kev_idx], pool_pub[kev_idx], pool_kev[kev_idx]

    # The CVEs we need features for (2022-2026), and the date each one is "as of"
    ds = pd.read_parquet(DATASET_PATH, columns=["cve_id", "cohort_year", "published", "hist_date"])
    row_pos = ds.cve_id.map(pd.Series(np.arange(len(ids)), index=ids.cve_id))
    if row_pos.isna().any():
        print(f"warning: {row_pos.isna().sum()} rows have no embedding; their features stay empty")
    have = row_pos.notna().values
    rp = row_pos[have].astype(int).values
    r_pub = days(ds.published[have])
    r_hist = days(ds.hist_date[have])

    feats = np.full((have.sum(), 6), np.nan, dtype=np.float32)
    nb_rows = []
    is2026 = (ds.cohort_year[have] == SCORE_YEAR).values
    hide = torch.tensor(-10.0, dtype=dt, device=dev)  # similarity for "not allowed" matches

    for s in range(0, len(rp), CHUNK):
        e = min(s + CHUNK, len(rp))
        q = emb[torch.from_numpy(rp[s:e]).to(dev)]
        qp = torch.from_numpy(r_pub[s:e]).to(dev)[:, None]
        qh = torch.from_numpy(r_hist[s:e]).to(dev)[:, None]

        # Closest past CVEs (anything published later is hidden)
        sim = torch.where(pool_pub[None, :] < qp, q @ emb.T, hide)
        top_s, top_i = sim.topk(K, dim=1)
        valid = top_s > -5
        exploited = (pool_kev[top_i] <= qh) & valid  # on KEV by the "as of" date?
        w = top_s.float().clamp(min=0) * valid
        feats[s:e, 0] = exploited[:, :10].sum(1).cpu().numpy()
        feats[s:e, 1] = exploited.sum(1).cpu().numpy()
        feats[s:e, 2] = ((w * exploited).sum(1) / w.sum(1).clamp(min=1e-6)).cpu().numpy()
        feats[s:e, 5] = top_s[:, 0].float().cpu().numpy()

        # Closest already-exploited CVEs only
        ks = torch.where((kev_kd[None, :] <= qh) & (kev_pub[None, :] < qp), q @ kev_emb.T, hide)
        kt = ks.topk(5, dim=1).values.float()
        kt = torch.where(kt > -5, kt, torch.full_like(kt, float("nan")))
        feats[s:e, 3] = kt[:, 0].cpu().numpy()
        feats[s:e, 4] = torch.nanmean(kt, dim=1).cpu().numpy()

        # Keep the top 10 matches for 2026 CVEs (shown as evidence on the website)
        m = is2026[s:e]
        if m.any():
            ti, ts, ex = (top_i[:, :10].cpu().numpy()[m], top_s[:, :10].float().cpu().numpy()[m],
                          exploited[:, :10].cpu().numpy()[m])
            for row, (ii, ss, xx) in zip(np.nonzero(m)[0] + s, zip(ti, ts, ex)):
                for rank, (i, sv, x) in enumerate(zip(ii, ss, xx), 1):
                    nb_rows.append((rp[row], rank, i, float(sv), bool(x)))
        if (s // CHUNK) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] {e}/{len(rp)}", flush=True)

    # Save the features
    cols = ["g_knn10_kev", "g_knn50_kev", "g_knn50_wkev", "g_kev_max_sim", "g_kev_top5_sim", "g_nn1_sim"]
    out = ds.loc[have, ["cve_id", "cohort_year"]].reset_index(drop=True)
    out[cols] = feats
    out.to_parquet(OUT, index=False)
    print("wrote", OUT, out.shape)

    # Save the 2026 evidence, with how many days each exploited match took to hit KEV
    nb = pd.DataFrame(nb_rows, columns=["q", "rank", "n", "similarity", "exploited"])
    nb["cve_id"] = ids.cve_id.values[nb.q]
    nb["neighbor_id"] = ids.cve_id.values[nb.n]
    nb["neighbor_published"] = ids.published.values[nb.n]
    nb["neighbor_kev_date"] = kev_date.values[nb.n]
    nb["days_to_kev"] = (pd.to_datetime(nb.neighbor_kev_date) - pd.to_datetime(nb.neighbor_published)).dt.days
    nb.loc[~nb.exploited, ["neighbor_kev_date", "days_to_kev"]] = None
    nb = nb[["cve_id", "rank", "neighbor_id", "similarity", "exploited", "neighbor_published",
             "neighbor_kev_date", "days_to_kev"]]
    nb.to_parquet(OUT_NB, index=False)
    print("wrote", OUT_NB, nb.shape)

    # Quick sanity check: exploited CVEs should score higher on these
    lab = pd.read_parquet(DATASET_PATH, columns=["cve_id", "cohort_year", "label"])
    chk = out.merge(lab, on=["cve_id", "cohort_year"])
    chk = chk[chk.cohort_year < SCORE_YEAR]
    print("\naverage feature value, exploited (1) vs not (0):")
    print(chk.groupby("label")[cols].mean().T.to_string())


if __name__ == "__main__":
    main()
