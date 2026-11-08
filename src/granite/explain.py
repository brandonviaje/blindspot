"""
write the "why is this risky?" text for the top CVEs.

For each of the top 100 in the risk feed we gather facts (how many similar past bugs got exploited, how fast, vendor history, exploit code) and write:
  why a plain sentence built straight from the data (always)
  why_granite  Granite 4.0 Micro rewording those facts in plain English (with --granite)

Granite only ever sees the facts, so it can't invent the numbers.
Saves output/explanations.csv + output/explanations_neighbors.csv (the evidence rows).

Run:  python src/granite/explain.py --granite     (~10 min on GPU; leave off --granite for just the data sentence)
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lets us import common.py
from common import CACHE_DIR, OUT_DIR

# What we tell Granite to do
SYSTEM = """You write short risk notes for an IT team with no security analyst.
Using ONLY the facts given, write at most two plain sentences: what the vulnerability
lets an attacker do, and why it is high risk. Keep every number exactly as given.
Do not add facts, product names or numbers that are not in the input."""


# Small wording helpers: "PHP" not "Php", "1 vulnerability" not "1 vulnerabilities"
def vname(v):
    return v.upper() if len(v) <= 4 else v.title()

def plural(n, stem):
    return f"{n} {stem}" + ("y" if n == 1 else "ies")

def days_phrase(d):
    d = max(0, int(d))
    return "on or near publish day" if d <= 1 else f"after a median of {d} days"

def facts_for(r, nb):
    """The fact list we hand to Granite."""
    f = [f"{int(nb.exploited.sum())} of its 10 most similar past vulnerabilities were later exploited"]
    d = nb.days_to_kev.dropna()
    if len(d):
        f.append(f"exploited {days_phrase(d.median())} (median)")
    if r.vendor_kev_12m > 0 and isinstance(r.desc_vendor, str):
        f.append(f"{vname(r.desc_vendor)} had {plural(int(r.vendor_kev_12m), 'exploited vulnerabilit')}"
                 f" in the 12 months before")
    if r.any_exploit_code == 1:
        f.append("public exploit code exists")
    if r.kw_unauth == 1:
        f.append("no login needed")
    if r.kw_rce == 1:
        f.append("allows code or command execution")
    if pd.notna(r.epss):
        f.append(f"EPSS {r.epss:.1%}")
    return f

def template(r, nb):
    """The data-only sentence (shown as 'evidence' on the website)."""
    k = int(nb.exploited.sum())
    if k:
        d = nb.days_to_kev.dropna()
        when = f", {days_phrase(d.median())}" if len(d) else ""
        bits = [f"{k} of its 10 closest past matches were exploited{when}."]
    else:
        bits = ["None of its 10 closest past matches were exploited."]
    if r.vendor_kev_12m > 0 and isinstance(r.desc_vendor, str):
        bits.append(f"{vname(r.desc_vendor)} had "
                    f"{plural(int(r.vendor_kev_12m), 'exploited vulnerabilit')} in the prior 12 months.")
    if r.any_exploit_code == 1:
        bits.append("Public exploit code exists.")
    return " ".join(bits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--granite", action="store_true")
    a = ap.parse_args()

    # Top CVEs + their 10 closest past matches (from granite_neighbors.py)
    feed = pd.read_csv(os.path.join(OUT_DIR, "risk_feed.csv")).head(a.top)
    nbs = pd.read_parquet(os.path.join(CACHE_DIR, "granite_neighbors_2026.parquet"))
    nbs = nbs[nbs.cve_id.isin(feed.cve_id)]
    nbs.to_csv(os.path.join(OUT_DIR, "explanations_neighbors.csv"), index=False)
    by = dict(tuple(nbs.groupby("cve_id")))

    rows = []
    for r in feed.itertuples():
        nb = by.get(r.cve_id, nbs.iloc[:0])
        rows.append({"rank": r.rank, "cve_id": r.cve_id,
                     "risk_score": round(r.predicted_risk_score, 4),
                     "similar_exploited_of_10": int(nb.exploited.sum()),
                     "median_days_to_kev": nb.days_to_kev.median(),
                     "facts": "; ".join(facts_for(r, nb)),
                     "why": template(r, nb),
                     "description": r.description})
    out = pd.DataFrame(rows)

    # have Granite reword each one
    if a.granite:
        import local_granite as G
        texts = []
        for i, r in enumerate(out.itertuples(), 1):
            msg = f"Description: {r.description[:1200]}\n\nFacts: {r.facts}"
            texts.append(G.chat([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": msg}], max_tokens=120))
            print(f"{i}/{len(out)} {r.cve_id}: {texts[-1]}", flush=True)
        out["why_granite"] = texts

    path = os.path.join(OUT_DIR, "explanations.csv")
    out.to_csv(path, index=False)
    print("wrote", path)
    print(out[["rank", "cve_id", "why"]].head(10).to_string(index=False, max_colwidth=110))


if __name__ == "__main__":
    main()
