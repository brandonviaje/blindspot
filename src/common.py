"""
Stuff every script shares: where files live, the key dates, and how to load the data.

Every script in src/pipeline, src/granite and src/web imports from here, so if a
folder or date ever changes, you change it once, here.
"""

import os

import pandas as pd

# --- Where things live (all relative to the repo, so it works on any machine) ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "data")   # raw track CSVs (not in git)
CACHE_DIR = os.path.join(REPO_ROOT, "cache")         # big in-between files (not in git)
OUT_DIR = os.path.join(REPO_ROOT, "output")          # results, logs, risk feed

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(CACHE_DIR, "dataset.parquet")  # the big table from build_dataset.py

# --- Key dates ---
# Our KEV file stops here, so "how often was this vendor exploited before?" for 2026
# CVEs is counted up to this date.
KEV_END = pd.Timestamp("2025-12-31")

# Years we learn from (answers known) and the year we predict (answers hidden).
# Each year copies the scoreboard's setup: published Jan-Sep, judged on KEV by Nov 4.
TRAIN_YEARS = [2022, 2023, 2024, 2025]
SCORE_YEAR = 2026


def cutoff_for(year):
    """The judging date for a year: added to KEV by Nov 4 counts as exploited."""
    return pd.Timestamp(f"{year}-11-04")


def path(name):
    """Full path to one of the raw CSVs, e.g. path("known_exploited.csv")."""
    return os.path.join(DATA_DIR, name)


def load_vulns(extra_cols=()):
    """vulnerabilities.csv, only the columns we need (the full file is huge)."""
    cols = [
        "cve_id", "published", "assigner", "description",
        "cvss4_score", "cvss4_vector",
        # NIST columns: only for the leakage audit, the CVSS baseline and display, never the model
        "nvd_status", "cvss3_score", "cvss3_source", "attack_vector",
        "privileges_required", "user_interaction", "cwe_ids",
        "n_references", "n_affected_cpes",
        *extra_cols,
    ]
    v = pd.read_csv(path("vulnerabilities.csv"), usecols=cols, low_memory=False)
    v["published"] = pd.to_datetime(v["published"])
    v["description"] = v["description"].fillna("")
    return v


def load_kev():
    """The KEV list (the answer key) with dates parsed."""
    k = pd.read_csv(path("known_exploited.csv"))
    k["date_added"] = pd.to_datetime(k["date_added"])
    return k


def hits_at(df, score_col, k=500, label_col="label"):
    """The judges' metric: how many exploited CVEs are in the top k by this score."""
    top = df.sort_values([score_col, "cve_id"], ascending=[False, True]).head(k)
    return int(top[label_col].sum())
