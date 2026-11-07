"""
db_builder.py — Blindspot Database Builder
===========================================
Reads vulnerabilities.csv and scoreboard_2026.csv from data/data/,
drops leaky columns, merges on cve_id, and writes a clean SQLite DB.

Leaky columns (MUST be excluded — they reveal the answer):
  • nvd_status          — tells us NVD review state
  • n_affected_cpes     — available only after NVD analysis
  • n_references        — grows over time; unavailable on publish day
  • All CVSS columns    — assigned after review; blank = not yet scored
"""

import os
import sys
import sqlite3
import logging
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "data")
DB_PATH  = os.path.join(BASE_DIR, "blindspot.db")

VULN_CSV       = os.path.join(DATA_DIR, "vulnerabilities.csv")
SCOREBOARD_CSV = os.path.join(DATA_DIR, "scoreboard_2026.csv")

# Columns that leak future knowledge — drop before any modelling
LEAKY_EXACT = {"nvd_status", "n_affected_cpes", "n_references"}
LEAKY_SUBSTRINGS = ["cvss"]  # catches cvss3_*, cvss4_*, cvss2_*

# Granite-enriched columns we pre-create (populated later by granite_router.py)
GRANITE_COLUMNS = {
    "vendor":                 "TEXT",
    "product_type":           "TEXT",
    "needs_login":            "INTEGER",   # 0/1 boolean
    "impact":                 "TEXT",
    "cites_other_cves":       "TEXT",      # JSON list stored as text
    "granite_baseline_score": "REAL",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("blindspot.db_builder")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Return the list of column names that must be removed."""
    drops = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in LEAKY_EXACT:
            drops.append(col)
            continue
        if any(sub in col_lower for sub in LEAKY_SUBSTRINGS):
            drops.append(col)
    return drops


def load_and_clean(csv_path: str, label: str) -> pd.DataFrame:
    """Read a CSV, drop leaky columns, and return a cleaned DataFrame."""
    log.info("Reading %s  (%s)", label, csv_path)
    if not os.path.isfile(csv_path):
        log.error("File not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path, low_memory=False)
    log.info("  → %d rows × %d columns", len(df), len(df.columns))

    drops = _columns_to_drop(df)
    if drops:
        log.info("  → Dropping leaky columns: %s", drops)
        df = df.drop(columns=drops)
    else:
        log.info("  → No leaky columns found")

    log.info("  → Remaining columns: %s", list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_database() -> None:
    """End-to-end: load CSVs → clean → merge → write SQLite."""

    # ── 1. Load & clean both CSVs ─────────────────────────────────────────
    df_vuln = load_and_clean(VULN_CSV, "vulnerabilities")
    df_score = load_and_clean(SCOREBOARD_CSV, "scoreboard_2026")

    # ── 2. Merge on cve_id (left join keeps every vulnerability) ──────────
    #    scoreboard_2026 may have a 'published' column too — suffix it
    df = df_vuln.merge(
        df_score,
        on="cve_id",
        how="left",
        suffixes=("", "_score"),
    )
    log.info("Merged dataset: %d rows × %d columns", len(df), len(df.columns))

    # ── 3. Ensure mandatory columns exist ─────────────────────────────────
    for required in ("cve_id", "description"):
        if required not in df.columns:
            log.error("FATAL — required column '%s' missing after merge!", required)
            sys.exit(1)

    # ── 4. Write to SQLite ────────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        log.info("Removed existing %s", DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        # Write the main table
        df.to_sql("cve_data", conn, if_exists="replace", index=False)
        log.info("Wrote %d rows to table 'cve_data'", len(df))

        # Add Granite enrichment columns (nullable, filled by granite_router)
        cur = conn.cursor()
        existing = {
            row[1].lower()
            for row in cur.execute("PRAGMA table_info(cve_data)").fetchall()
        }
        for col_name, col_type in GRANITE_COLUMNS.items():
            if col_name.lower() not in existing:
                cur.execute(
                    f"ALTER TABLE cve_data ADD COLUMN {col_name} {col_type}"
                )
                log.info("  → Added column %s (%s)", col_name, col_type)

        # Create indexes for fast lookups
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cve_id ON cve_data (cve_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_published ON cve_data (published)"
        )
        log.info("Created indexes on cve_id and published")

        conn.commit()
    finally:
        conn.close()

    log.info("✅  Database ready at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_database()

