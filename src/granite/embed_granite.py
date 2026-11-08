"""
turn every description into numbers.

IBM Granite Embedding reads each CVE description (2015 onward, ~334k) and gives
back 768 numbers that capture what it *means*. Similar attacks → similar numbers.

Saves to cache/: granite_emb.f16.npy (the numbers) + granite_ids.parquet (which CVE is which row).
Takes ~15 min on a GPU. If it stops halfway, just run it again and it picks up where it left off.

Run:  python src/granite/embed_granite.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lets us import common.py
from common import CACHE_DIR, load_vulns

MODEL_ID = "ibm-granite/granite-embedding-english-r2"
EMB_PATH = os.path.join(CACHE_DIR, "granite_emb.f16.npy")
IDS_PATH = os.path.join(CACHE_DIR, "granite_ids.parquet")
PROGRESS = os.path.join(CACHE_DIR, "granite_emb.progress.json")  # how far we got, for resuming
CHUNK = 8192  # save to disk every 8k descriptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)  # longer descriptions get cut off here
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    # Load Granite (downloads the first time). Half precision on GPU = faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    model = SentenceTransformer(MODEL_ID, device=device)
    model.max_seq_length = a.max_len
    if device == "cuda":
        model.half()
    dim = model.get_sentence_embedding_dimension()

    v = load_vulns()
    v = v[v.published >= "2015-01-01"].sort_values(["published", "cve_id"]).reset_index(drop=True)
    n = len(v)

    # Resume if a previous run got partway
    done = 0
    if os.path.exists(PROGRESS) and os.path.exists(EMB_PATH):
        p = json.load(open(PROGRESS))
        if p.get("n") == n and p.get("dim") == dim:
            done = p["done"]
    if done == 0:
        v[["cve_id", "published"]].to_parquet(IDS_PATH, index=False)
        emb = np.lib.format.open_memmap(EMB_PATH, mode="w+", dtype=np.float16, shape=(n, dim))
    else:
        emb = np.load(EMB_PATH, mmap_mode="r+")
        print(f"resuming at {done}/{n}", flush=True)

    # Embed chunk by chunk, writing straight to disk so RAM stays low
    t0 = time.time()
    texts = v.description.tolist()
    for start in range(done, n, CHUNK):
        end = min(start + CHUNK, n)
        vec = model.encode(texts[start:end], batch_size=a.batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        emb[start:end] = vec.astype(np.float16)
        emb.flush()
        json.dump({"n": n, "dim": dim, "done": end}, open(PROGRESS, "w"))
        rate = (end - done) / (time.time() - t0)
        print(f"{end:>7}/{n}  {rate:6.0f} desc/s  eta {(n - end) / rate / 60:5.1f} min", flush=True)
    print(f"wrote {EMB_PATH} {emb.shape}")


if __name__ == "__main__":
    main()
