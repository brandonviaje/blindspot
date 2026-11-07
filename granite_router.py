"""
granite_router.py — IBM Granite LLM Context Extraction
=======================================================
Fetches CVE rows that have not yet been enriched (vendor/product_type IS NULL),
sends each description to the IBM Granite API, parses the structured JSON
response, and writes the extracted fields back to blindspot.db.

Environment Variables:
    IBM_API_KEY — Bearer token for the Granite endpoint

=======================================================================
 ██  ML MODELER HOOK — INSERT TIME-TO-EVENT SCORING HERE  ██
-----------------------------------------------------------------------
 After this module enriches each CVE with Granite context fields
 (vendor, product_type, needs_login, impact, cites_other_cves),
 the ML modeler should:
   1. Read from the `cve_data` table (all Granite fields + raw features).
   2. Run the survival / time-to-event model to produce a final
      `predicted_risk_score` column.
   3. Write that score back to `cve_data` so that app.py can serve it.
 The granite_baseline_score produced here is an LLM-only heuristic
 baseline for comparison — it is NOT the final production score.
=======================================================================
"""

import json
import os
import re
import sqlite3
import logging
import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "blindspot.db")

# IBM Granite API endpoint (swap for your vLLM / watsonx URL if needed)
GRANITE_ENDPOINT = os.environ.get(
    "GRANITE_ENDPOINT",
    "https://api.ibm.com/granite/v1/generate",
)
IBM_API_KEY = os.environ.get("IBM_API_KEY", "")

# How many CVEs to process per run (prevents quota exhaustion)
BATCH_SIZE = int(os.environ.get("GRANITE_BATCH_SIZE", "10"))

# Retry / back-off settings
MAX_RETRIES    = 3
RETRY_DELAY_S  = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("blindspot.granite_router")


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a cybersecurity analyst. Given the following CVE vulnerability \
description, extract the requested fields and return ONLY a valid JSON \
object — no markdown fences, no commentary.

### CVE Description
{description}

### Required JSON fields
{{
  "vendor": "<company or project name>",
  "product_type": "<one of: VPN, firewall, edge device, mail server, \
web app, OS, library, router, CI/CD, CMS, browser, other>",
  "needs_login": <true if exploitation requires authentication, else false>,
  "impact": "<one of: code execution, auth bypass, privilege escalation, \
info leak, denial of service, XSS, SQL injection, file upload, other>",
  "cites_other_cves": ["CVE-XXXX-YYYY", ...],
  "granite_baseline_score": <integer 0-10 rating of how likely this \
vulnerability is to be exploited in the wild>
}}

Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# Granite API caller
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    """Construct authorization headers for the Granite API."""
    if not IBM_API_KEY:
        log.warning(
            "IBM_API_KEY is not set — requests will likely fail. "
            "Set it via: export IBM_API_KEY=<your-token>"
        )
    return {
        "Authorization": f"Bearer {IBM_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


def query_granite_extract(description_text: str) -> dict[str, Any]:
    """
    Send a CVE description to IBM Granite and return parsed JSON fields.

    Returns a dict with keys:
        vendor, product_type, needs_login, impact,
        cites_other_cves, granite_baseline_score

    On failure returns a dict with all keys set to None.
    """
    empty = {
        "vendor": None,
        "product_type": None,
        "needs_login": None,
        "impact": None,
        "cites_other_cves": None,
        "granite_baseline_score": None,
    }

    if not description_text or not description_text.strip():
        log.warning("Empty description — skipping Granite call")
        return empty

    prompt = EXTRACTION_PROMPT.format(description=description_text)

    payload = {
        "model_id": "ibm/granite-3.1-8b-instruct",
        "input":    prompt,
        "parameters": {
            "max_new_tokens":  512,
            "temperature":     0.1,   # near-deterministic for extraction
            "decoding_method": "greedy",
        },
    }

    headers = _build_headers()

    # Retry loop with exponential back-off
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GRANITE_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()

            body = resp.json()

            # Extract generated text — handles both watsonx and vLLM shapes
            generated = (
                body.get("results", [{}])[0].get("generated_text")
                or body.get("choices", [{}])[0].get("text")
                or body.get("generated_text")
                or ""
            )

            return _parse_granite_response(generated)

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            log.error(
                "Granite HTTP %s on attempt %d/%d: %s",
                status, attempt, MAX_RETRIES, exc,
            )
            if status == 429 or (isinstance(status, int) and status >= 500):
                time.sleep(RETRY_DELAY_S * attempt)
                continue
            return empty

        except requests.exceptions.RequestException as exc:
            log.error(
                "Granite request failed (attempt %d/%d): %s",
                attempt, MAX_RETRIES, exc,
            )
            time.sleep(RETRY_DELAY_S * attempt)

    log.error("All %d Granite retries exhausted", MAX_RETRIES)
    return empty


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_granite_response(raw_text: str) -> dict[str, Any]:
    """
    Parse the raw LLM output into a clean dict.
    Handles markdown fences, trailing commas, and partial JSON.
    """
    empty = {
        "vendor": None,
        "product_type": None,
        "needs_login": None,
        "impact": None,
        "cites_other_cves": None,
        "granite_baseline_score": None,
    }

    if not raw_text:
        return empty

    # Strip markdown code fences if the model wrapped its answer
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()

    # Try to find a JSON object in the text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        log.warning("No JSON object found in Granite response: %.200s", raw_text)
        return empty

    json_str = match.group(0)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt to fix trailing commas
        json_str_fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            parsed = json.loads(json_str_fixed)
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse Granite JSON: %s — %.200s", exc, json_str)
            return empty

    # Normalize fields
    result = {}
    result["vendor"]       = parsed.get("vendor")
    result["product_type"] = parsed.get("product_type")
    result["impact"]       = parsed.get("impact")

    # needs_login → integer 0/1
    nl = parsed.get("needs_login")
    if isinstance(nl, bool):
        result["needs_login"] = int(nl)
    elif isinstance(nl, (int, float)):
        result["needs_login"] = int(bool(nl))
    else:
        result["needs_login"] = None

    # cites_other_cves → JSON string
    cites = parsed.get("cites_other_cves", [])
    if isinstance(cites, list):
        result["cites_other_cves"] = json.dumps(cites)
    else:
        result["cites_other_cves"] = json.dumps([])

    # granite_baseline_score → float 0–10
    score = parsed.get("granite_baseline_score")
    try:
        score = float(score)
        result["granite_baseline_score"] = max(0.0, min(10.0, score))
    except (TypeError, ValueError):
        result["granite_baseline_score"] = None

    return result


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_cves_with_granite() -> int:
    """
    Main entry point: fetch un-enriched CVEs, call Granite, update the DB.

    Returns the number of CVEs successfully processed.
    """
    if not os.path.isfile(DB_PATH):
        log.error("Database not found at %s — run db_builder.py first", DB_PATH)
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    processed = 0

    try:
        cur = conn.cursor()

        # Fetch CVEs that have NOT been enriched yet
        rows = cur.execute(
            """
            SELECT cve_id, description
            FROM   cve_data
            WHERE  vendor IS NULL
               OR  product_type IS NULL
            ORDER  BY published DESC
            LIMIT  ?
            """,
            (BATCH_SIZE,),
        ).fetchall()

        log.info("Found %d un-enriched CVEs (batch limit %d)", len(rows), BATCH_SIZE)

        for row in rows:
            cve_id      = row["cve_id"]
            description = row["description"] or ""

            log.info("Processing %s …", cve_id)
            fields = query_granite_extract(description)

            # Write extracted fields back to the database
            cur.execute(
                """
                UPDATE cve_data
                SET    vendor                 = ?,
                       product_type           = ?,
                       needs_login            = ?,
                       impact                 = ?,
                       cites_other_cves       = ?,
                       granite_baseline_score = ?
                WHERE  cve_id = ?
                """,
                (
                    fields["vendor"],
                    fields["product_type"],
                    fields["needs_login"],
                    fields["impact"],
                    fields["cites_other_cves"],
                    fields["granite_baseline_score"],
                    cve_id,
                ),
            )
            conn.commit()
            processed += 1
            log.info("  ✓ %s → vendor=%s  type=%s  score=%s",
                     cve_id, fields["vendor"], fields["product_type"],
                     fields["granite_baseline_score"])

    finally:
        conn.close()

    log.info("✅  Enriched %d / %d CVEs", processed, len(rows))
    return processed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not IBM_API_KEY:
        log.warning(
            "⚠  IBM_API_KEY not set. Export it first:\n"
            "    export IBM_API_KEY=<your-token>\n"
            "Proceeding anyway (calls will fail gracefully)."
        )
    process_cves_with_granite()

