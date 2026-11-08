"""
server.py — Blindspot demo UI (Flask + one HTML page). No GPU needed.

Reads only blindspot/demo_data/ (built by src/web/prepare_demo.py).

  python src/web/server.py        then open http://localhost:5000

Endpoints
  GET  /api/summary                       counts for the header
  GET  /api/vendors                       vendor suggestions for the patch list
  GET  /api/patch-plan?q=fortinet,ivanti&per_week=10
  GET  /api/feed?limit=50&offset=0&q=...&unreviewed=1
  GET  /api/cve/<id>                      details, explanation, similar past vulnerabilities
  POST /api/score   {"description": ...}  live Granite embedding vs already-exploited CVEs
  GET  /api/results                       backtest numbers
  GET  /api/feed.csv                      download the full risk feed

Live scoring needs `pip install sentence-transformers` (CPU torch is fine); the rest
of the app works without it.
"""

import json
import os
import threading

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "demo_data")  # repo root
EMB_MODEL = "ibm-granite/granite-embedding-english-r2"

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
app.json.sort_keys = False

feed = pd.read_parquet(os.path.join(DEMO, "feed.parquet"))
feed["desc_lc"] = feed.description.fillna("").str.lower()
expl = pd.read_csv(os.path.join(DEMO, "explanations.csv")).set_index("cve_id")
nbrs = pd.read_parquet(os.path.join(DEMO, "neighbors.parquet"))
kev_meta = pd.read_parquet(os.path.join(DEMO, "kev_meta.parquet"))
kev_emb = np.load(os.path.join(DEMO, "kev_emb.f16.npy")).astype(np.float32)
results = pd.read_csv(os.path.join(DEMO, "results.csv"))
_findings_path = os.path.join(DEMO, "findings.json")  # from src/pipeline/findings.py (optional)
findings = json.load(open(_findings_path)) if os.path.exists(_findings_path) else {}

_model, _model_lock = None, threading.Lock()


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def row_json(r, full=False):
    e = expl.loc[r.cve_id] if r.cve_id in expl.index else None
    why = evidence = None
    if e is not None:
        evidence = e["why"]  # built from the data only
        why = e.get("why_granite") if isinstance(e.get("why_granite"), str) else None
    out = {
        "rank": clean(r["rank"]), "cve_id": r.cve_id,
        "published": str(r.published)[:10],
        "score": clean(r.predicted_risk_score),
        "vendor": r.desc_vendor if isinstance(r.desc_vendor, str) else None,
        "product": r.desc_product if isinstance(r.desc_product, str) else None,
        "epss": clean(r.epss), "exploit_code": bool(r.any_exploit_code == 1),
        "no_login": bool(r.kw_unauth == 1), "code_exec": bool(r.kw_rce == 1),
        "nist_reviewed": bool(r.nist_reviewed),
        "why": why, "evidence": evidence,
        "description": r.description if full else (r.description or "")[:260],
    }
    return out


def match(q):
    terms = [t.strip().lower() for t in q.split(",") if t.strip()]
    if not terms:
        return feed.iloc[:0]
    m = np.zeros(len(feed), dtype=bool)
    for t in terms:
        m |= feed.desc_lc.str.contains(t, regex=False).values
    return feed[m]


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/summary")
def summary():
    # The last backtest year is the one full test of the final model
    yr = int(results.year.max())
    r = results[results.year == yr].set_index("ranking").hits
    standard = r[[n for n in r.index if not n.startswith("Blindspot")]]
    return jsonify({
        "scored": len(feed),
        "explained": len(expl),
        "test_year": yr,
        "hits": int(r["Blindspot"]),
        "best_standard_hits": int(standard.max()),
        "positives": findings.get("header", {}).get("positives"),
        "warning_days": findings.get("header", {}).get("warning_days_median"),
    })


@app.get("/api/vendors")
def vendors():
    top = feed.head(5000).desc_vendor.dropna().str.lower().value_counts().head(40)
    return jsonify([v for v in top.index if len(v) > 2])


@app.get("/api/patch-plan")
def patch_plan():
    q = request.args.get("q", "")
    per_week = max(1, min(100, int(request.args.get("per_week", 10))))
    m = match(q)
    total = float(m.predicted_risk_score.sum()) or 1.0
    week = m.head(per_week)
    curve = (m.predicted_risk_score.cumsum() / total).head(100).round(4).tolist()
    return jsonify({
        "matching": len(m), "per_week": per_week,
        "covered": round(float(week.predicted_risk_score.sum()) / total, 4),
        "coverage_curve": curve,
        "items": [row_json(r) for _, r in week.iterrows()],
    })


@app.get("/api/feed")
def feed_page():
    limit = max(1, min(200, int(request.args.get("limit", 50))))
    offset = max(0, int(request.args.get("offset", 0)))
    d = match(request.args["q"]) if request.args.get("q") else feed
    if request.args.get("unreviewed") == "1":
        d = d[~d.nist_reviewed]
    return jsonify({"total": len(d),
                    "items": [row_json(r) for _, r in d.iloc[offset:offset + limit].iterrows()]})


@app.get("/api/feed.csv")
def feed_csv():
    cols = ["rank", "cve_id", "published", "predicted_risk_score", "desc_vendor", "desc_product",
            "epss", "any_exploit_code", "nist_reviewed"]
    return Response(feed[cols].to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=blindspot_risk_feed.csv"})


@app.get("/api/cve/<cve_id>")
def cve(cve_id):
    hit = feed[feed.cve_id == cve_id.upper()]
    if hit.empty:
        return jsonify({"error": "not in the 2026 scoreboard"}), 404
    out = row_json(hit.iloc[0], full=True)
    if cve_id in expl.index:
        e = expl.loc[cve_id]
        out["facts"] = e["facts"]
        out["why_template"] = e["why"]
    nb = nbrs[nbrs.cve_id == cve_id].sort_values("rank")
    out["neighbors"] = [{
        "cve_id": n.neighbor_id, "similarity": clean(n.similarity),
        "exploited": bool(n.exploited), "days_to_kev": clean(n.days_to_kev),
        "published": str(n.neighbor_published)[:10],
        "description": n.neighbor_description,
    } for n in nb.itertuples()]
    return jsonify(out)


@app.post("/api/score")
def score():
    global _model
    text = (request.get_json(silent=True) or {}).get("description", "").strip()
    if len(text) < 20:
        return jsonify({"error": "Paste a full vulnerability description."}), 400
    try:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMB_MODEL, device="cpu")
        vec = _model.encode([text], normalize_embeddings=True)[0].astype(np.float32)
    except ImportError:
        return jsonify({"error": "Live scoring needs: pip install sentence-transformers"}), 503
    sims = kev_emb @ vec
    order = np.argsort(-sims)[:5]
    top5 = float(sims[order].mean())
    pct = float((feed.g_kev_top5_sim.dropna().values < top5).mean())
    return jsonify({
        "top5_similarity": round(top5, 4),
        "percentile_2026": round(pct, 4),
        "matches": [{
            "cve_id": kev_meta.cve_id.iloc[i], "similarity": round(float(sims[i]), 4),
            "vendor": kev_meta.vendor.iloc[i], "product": kev_meta["product"].iloc[i],
            "name": kev_meta.name.iloc[i] if "name" in kev_meta else None,
            "date_added": str(kev_meta.date_added.iloc[i])[:10],
            "ransomware": str(kev_meta.known_ransomware_use.iloc[i]) == "Known",
        } for i in order],
    })


@app.get("/api/results")
def get_results():
    return jsonify(results.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
