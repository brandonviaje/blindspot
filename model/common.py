"""Shared paths, constants and loaders for the Blindspot model."""

import os

import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MODEL_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "data")
CACHE_DIR = os.path.join(MODEL_DIR, "cache")
OUT_DIR = os.path.join(MODEL_DIR, "output")
DB_PATH = os.path.join(REPO_ROOT, "src", "blindspot.db")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(CACHE_DIR, "dataset.parquet")

# KEV file stops here; vendor/assigner history for 2026 is frozen at this date
# so 2026 features are computed the same way as training features.
KEV_END = pd.Timestamp("2025-12-31")

# Cohort years. Each mirrors the scoreboard: published Jan-Sep, predicted from
# the end of the publish month, counted if added to KEV by Nov 4 of that year.
TRAIN_YEARS = [2022, 2023, 2024, 2025]
SCORE_YEAR = 2026


def cutoff_for(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-11-04")


def path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def load_vulns(extra_cols=()) -> pd.DataFrame:
    cols = [
        "cve_id", "published", "assigner", "description",
        "cvss4_score", "cvss4_vector",
        # used only by the leakage audit / extended feature sets
        "nvd_status", "cvss3_score", "cvss3_source", "attack_vector",
        "privileges_required", "user_interaction", "cwe_ids",
        "n_references", "n_affected_cpes",
        *extra_cols,
    ]
    v = pd.read_csv(path("vulnerabilities.csv"), usecols=cols, low_memory=False)
    v["published"] = pd.to_datetime(v["published"])
    v["description"] = v["description"].fillna("")
    return v


def load_kev() -> pd.DataFrame:
    k = pd.read_csv(path("known_exploited.csv"))
    k["date_added"] = pd.to_datetime(k["date_added"])
    return k


def hits_at(df: pd.DataFrame, score_col: str, k: int = 500, label_col: str = "label") -> int:
    """Number of positives in the top-k rows by score (ties broken by cve_id)."""
    top = df.sort_values([score_col, "cve_id"], ascending=[False, True]).head(k)
    return int(top[label_col].sum())
