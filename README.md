# Blindspot

**What to patch first, weeks before an attack is confirmed.**

Blindspot ranks new vulnerabilities by how likely attackers are to exploit them, without waiting for NIST's review. IBM Granite compares each CVE's description with vulnerabilities already known to be exploited. A gradient-boosted model combines that with a word-based reading of the description, who filed the CVE, that vendor's attack history, and EPSS and public exploit code once they exist. The demo turns the ranking into a weekly patch list for an IT team of one or two people.

Built for the IBM track, Before the Breach, at Sheridan Datathon 2026.

## Why

NIST reviews attacked vulnerabilities first, so its review tends to arrive after the attack. All 75 exploited CVEs in our 2024 and 2025 test years have a NIST review today, while 15% and 35% of all CVEs from those years still have none. A ranking that waits for NIST's scores sees the danger late, and a model that uses review status is reading the answer key. Blindspot uses none of it.

## Results

Each model was trained only on years before the one it was tested on. Counts are CVEs that CISA added to its Known Exploited Vulnerabilities (KEV) list by November 4 of the test year.

| Exploited CVEs caught | 2024, top 500 | 2025, top 500 | 2024, strict | 2025, strict |
|---|---:|---:|---:|---:|
| **Blindspot** | 8 | **21** | 4 | **18** |
| Blindspot, publish-day info only | 8 | 14 | 4 | 11 |
| Exploit code first, then EPSS | 8 | 17 | **6** | 13 |
| EPSS first | 4 | 8 | 3 | 8 |
| Highest CVSS 3 first | 0 | 5 | 0 | 5 |
| Exploited CVEs in the test year | 27 | 48 | 27 | 48 |

- **Strict** picks the same share of each year that 500 is of 2026 (198 and 243 CVEs). 2026 has about twice as many CVEs as earlier years, so the scoreboard's top 500 is a tighter cut.
- **Publish-day info only** leaves out EPSS, public exploit code and every NIST field.
- **2025 is the one full test of the final model.** In 2024 the two Blindspot rows are effectively the same model, because the 2022–23 years it trained on have no EPSS or exploit-code data.
- **The 2025 lead is suggestive, not proven.** 6 exploited CVEs were caught only by Blindspot and 2 only by the best standard ranking (p = 0.29, two-sided sign test).

`top500.csv` comes from the same model trained on 2022–2025 and run on the 72,670 scoreboard CVEs.

## What we found

1. **The description finds different exploited CVEs than exploit code does.** In 2024 the publish-day model and the exploit-code ranking each caught 8 exploited CVEs, with only 1 in common. In 2025 the publish-day model caught 8 that the exploit-code ranking missed.
2. **About a month of warning.** Blindspot's 21 hits in 2025 reached KEV a median of 30 days after their score existed, and 45 days after publication.
3. **Review order matters.** Working through 2025 in the publish-day model's order reaches 52% of the year's exploited CVEs within the first 5% of its CVEs. CVSS order reaches 29%.
4. **The columns the track warns about would inflate results by a third.** Adding them (NIST review status, blank or NIST-given scores, reference and product counts) raises hits from 50 to 67 across four leave-one-year-out tests. We left them out.

The numbers come from `output/proof_report.log`, `output/findings_report.log` and `output/train_report.log`.

## How it works

1. `build_dataset.py` builds one row per CVE with only what was known at the end of its publish month. Each training year copies the scoreboard: CVEs published January to September, exploited if added to KEV by November 4, dropped if already on KEV by the end of their publish month.
2. `text_model.py` learns which words in descriptions mark past KEV entries (TF-IDF and logistic regression, trained only on earlier years).
3. `embed_granite.py` and `granite_neighbors.py` embed every description with IBM Granite and find each CVE's most similar earlier CVEs, and which of them were already exploited.
4. `train.py` combines everything in LightGBM (small trees, averaged over 5 seeds) and writes `top500.csv` and the risk feed.
5. `proof.py` and `findings.py` replay 2024 and 2025 against the standard rankings.
6. `explain.py` has Granite 4.0 Micro write a plain-English note for each of the top 100.
7. `src/web/` serves the demo: patch list, risk feed, a page per CVE with its similar past vulnerabilities, live scoring of a pasted description, and the results.

## IBM Granite

- **Granite Embedding English R2** (`ibm-granite/granite-embedding-english-r2`) embeds all 334,068 descriptions published since 2015. For each CVE we find its most similar earlier CVEs, and its most similar CVEs already on KEV by its snapshot date. Two features made the model: how many of its 10 closest earlier CVEs were exploited, and how similar its 5 closest exploited CVEs are. The second is the model's fourth-largest signal by gain.
- The same neighbours are the evidence on each top-100 CVE's page in the demo. The demo's *Score a new CVE* tab embeds a pasted description live, on a CPU.
- **Granite 4.0 Micro** (`ibm-granite/granite-4.0-micro`, 4-bit on one GPU) rewrites each top-100 CVE's facts as a short note for an IT team. It sees only the description and a list of facts computed from the data, and the demo shows that data-only sentence under every note.
- Removing the Granite features costs 5 exploited CVEs at the top-500 cut across four leave-one-year-out tests, and gains 1 at the strict cut.

The risk feed (every 2026 CVE ranked, downloadable from the demo's *Risk feed* tab) is the kind of early signal a prioritization tool such as IBM Concert Protect could take in. We didn't build that integration.

## Data used

| File | Columns | Used for |
|---|---|---|
| `vulnerabilities.csv` | `cve_id`, `published`, `assigner`, `description`, `cvss4_score`, `cvss4_vector` | Model features |
| `vulnerabilities.csv` | `cvss3_score`, `cvss3_source`, `nvd_status`, `cwe_ids`, `n_references`, `n_affected_cpes` | Never in the model: the CVSS baseline, the leakage audit, and the "reviewed by NIST" label in the demo |
| `known_exploited.csv` | `cve_id`, `date_added`, `vendor`, `product` | Answers, and each assigner's, vendor's and product's KEV history before the snapshot |
| `known_exploited.csv` | `name`, `known_ransomware_use` | Demo display only |
| `exploit_signals_2024.csv`, `exploit_signals_2025.csv` | `month`, `cve_id`, `epss`, `exploitdb`, `metasploit`, `nuclei`, `poc_github` | EPSS and exploit code at the end of each CVE's publish month |
| `scoreboard_2026.csv` | `cve_id`, `published`, `snapshot_month`, `epss`, `exploitdb`, `metasploit`, `nuclei`, `poc_github` | The CVEs we rank, with their signals |

Not used: `affected_products.csv`, `references.csv` (including its news and threat-report links), `weakness_types.csv`, `in_kev` from the signals files, and every other column of `vulnerabilities.csv`. No outside data.

No look-ahead:

- History features count KEV additions only up to the end of the CVE's publish month. For 2026 CVEs that means up to December 31, 2025, where the KEV file ends.
- The text model for each year trains on CVEs published before January 1 of that year, with KEV answers as known then.
- A Granite neighbour must be published before the CVE, and counts as exploited only if it was on KEV by the CVE's snapshot date.

## Limitations

- Small numbers: 27 and 48 exploited CVEs in the two test years.
- KEV is incomplete. A CVE that isn't on it may still have been attacked.
- Feature groups were chosen on the same backtest, so the results are slightly optimistic.
- The CVSS baseline uses today's NIST scores, which flatters it. On publish day many CVEs have no NIST score yet.
- Vendor names come from matching KEV vendor names in the description. A vendor that has never been on KEV isn't recognized, and a description can match the wrong one.
- Granite's notes sometimes describe the similar CVEs' history as the CVE's own. The data-only evidence line under each note is the source of truth.
- Descriptions are today's text and could have been edited after an attack. Only 2 of Blindspot's 21 hits in 2025 mention in-the-wild exploitation, so edits can't explain the result.

## Run it

Python 3.12. Unzip the track's `data.zip` so the CSVs are in `data/data/`. Steps marked GPU also run on a CPU, just slower.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/pipeline/build_dataset.py                             # ~1.5 min
python src/pipeline/text_model.py                                # ~2 min
python src/granite/embed_granite.py                              # ~15 min, GPU
python src/granite/granite_neighbors.py                          # GPU
python src/pipeline/baselines.py > output/baselines.log
python src/pipeline/train.py --final > output/train_report.log   # writes top500.csv
python src/pipeline/proof.py > output/proof_report.log           # ~2 min
python src/pipeline/findings.py > output/findings_report.log
python src/granite/explain.py --granite                          # ~10 min, GPU
python src/web/prepare_demo.py                                   # packs demo_data/
```

Demo only. It needs no GPU and no raw data, because `demo_data/` is committed:

```bash
pip install -r src/web/requirements.txt
python src/web/server.py    # then open http://localhost:5000
```

## Repo layout

```
src/common.py      paths, key dates, the hits metric
src/pipeline/      dataset, text model, baselines, model, proof, findings
src/granite/       Granite embeddings, neighbours, explanations
src/web/           the demo: server.py, prepare_demo.py, and static/ (index.html, styles/main.css, scripts/app.js)
demo_data/         everything the demo reads
output/            logs, results, explanations
top500.csv         our submission
```
