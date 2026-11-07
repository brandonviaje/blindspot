"""
app.py — Blindspot Enterprise Risk Feed API (Flask)
====================================================
Lightweight REST API that serves CVE threat intelligence from blindspot.db.
Designed to simulate an automated SOC / Jira webhook payload.

Endpoints:
    GET /api/top-threats   - JSON feed of highest-risk CVEs
    GET /api/health        - Simple health-check

=======================================================================
 ██  ML MODELER HOOK — RANKING UPGRADE  ██
-----------------------------------------------------------------------
 The /api/top-threats endpoint currently orders by granite_baseline_score
 (the LLM-only heuristic). Once the ML modeler writes a final
 `predicted_risk_score` column to the cve_data table, update the
 ORDER BY clause below to use that column instead.
=======================================================================
"""

import os
import sqlite3
import json
import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, request, g

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "blindspot.db")

app = Flask(__name__)
app.json.sort_keys = False  # preserve field order in responses

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("blindspot.api")

# Default number of CVEs returned (overridable via ?limit=N query param)
DEFAULT_LIMIT = 500
MAX_LIMIT     = 2000


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Open (or reuse) a per-request SQLite connection."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc) -> None:
    """Close the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------------------------
# Helper: format a row as a Jira-style ticket payload
# ---------------------------------------------------------------------------

def _row_to_ticket(row: sqlite3.Row, rank: int) -> dict:
    """
    Transform a DB row into a structured JSON object that looks like
    an automated Jira / ServiceNow ticket payload for a SOC analyst.
    """
    # Parse cites_other_cves from JSON string back to list
    cites_raw = row["cites_other_cves"]
    try:
        cites = json.loads(cites_raw) if cites_raw else []
    except (json.JSONDecodeError, TypeError):
        cites = []

    return {
        # ── Ticket metadata ───────────────────────────────────────────
        "ticket_id":    f"BLINDSPOT-{rank:05d}",
        "severity":     _score_to_severity(row["granite_baseline_score"]),
        "created_at":   datetime.now(timezone.utc).isoformat(),

        # ── CVE core fields ───────────────────────────────────────────
        "cve_id":       row["cve_id"],
        "published":    row["published"],
        "description":  row["description"],

        # ── Granite-extracted context ─────────────────────────────────
        "vendor":                  row["vendor"],
        "product_type":            row["product_type"],
        "needs_login":             bool(row["needs_login"]) if row["needs_login"] is not None else None,
        "impact":                  row["impact"],
        "cites_other_cves":        cites,
        "granite_baseline_score":  row["granite_baseline_score"],

        # ── Placeholder for the ML modeler's final score ──────────────
        # "predicted_risk_score": row.get("predicted_risk_score"),
    }


def _score_to_severity(score) -> str:
    """Map a 0-10 score to a SOC severity label."""
    if score is None:
        return "UNKNOWN"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if s >= 8.0:
        return "CRITICAL"
    if s >= 6.0:
        return "HIGH"
    if s >= 4.0:
        return "MEDIUM"
    if s >= 2.0:
        return "LOW"
    return "INFO"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/top-threats", methods=["GET"])
def top_threats():
    """
    GET /api/top-threats?limit=500

    Returns the top-N CVEs ranked by granite_baseline_score (descending),
    formatted as simulated SOC ticket payloads.

    ──────────────────────────────────────────────────────────────────────
    ML MODELER: Replace `granite_baseline_score` in the ORDER BY clause
    with your final `predicted_risk_score` once it is available.
    ──────────────────────────────────────────────────────────────────────
    """
    # Parse optional limit query parameter
    try:
        limit = int(request.args.get("limit", DEFAULT_LIMIT))
        limit = max(1, min(limit, MAX_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT

    db = get_db()
    try:
        rows = db.execute(
            """
            SELECT cve_id,
                   published,
                   description,
                   vendor,
                   product_type,
                   needs_login,
                   impact,
                   cites_other_cves,
                   granite_baseline_score
            FROM   cve_data
            ORDER  BY granite_baseline_score DESC NULLS LAST
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.error("Database query failed: %s", exc)
        return jsonify({
            "error": "database_error",
            "detail": str(exc),
        }), 500

    tickets = [_row_to_ticket(row, rank=i + 1) for i, row in enumerate(rows)]

    return jsonify({
        "feed":       "blindspot-top-threats",
        "version":    "1.0.0",
        "generated":  datetime.now(timezone.utc).isoformat(),
        "count":      len(tickets),
        "threats":    tickets,
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    db = get_db()
    try:
        total = db.execute("SELECT COUNT(*) FROM cve_data").fetchone()[0]
        enriched = db.execute(
            "SELECT COUNT(*) FROM cve_data WHERE vendor IS NOT NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        total, enriched = 0, 0

    return jsonify({
        "status":   "ok",
        "database": DB_PATH,
        "total_cves":    total,
        "enriched_cves": enriched,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.isfile(DB_PATH):
        log.error(
            "blindspot.db not found — run db_builder.py first.\n"
            "  python db_builder.py"
        )
    else:
        log.info("Starting Blindspot API on http://127.0.0.1:5000")
        app.run(host="127.0.0.1", port=5000, debug=True)

