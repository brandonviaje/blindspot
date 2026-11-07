"""
build_dataset.py — one row per vulnerability, as known at the end of its snapshot month.

Cohorts (each mirrors scoreboard_2026.csv):
  2022-2025  published Jan-Sep, snapshot = publish month,
             dropped if already on KEV by the end of the snapshot month,
             label = added to KEV after the snapshot and by Nov 4 of that year.
  2026       exactly the rows of scoreboard_2026.csv (label unknown).

Feature groups (see FEATURE_GROUPS in train.py):
  publish-day safe : assigner/vendor/product KEV history, description keywords,
                     cited CVEs, CNA-supplied CVSS v4, months to cutoff
  signals          : EPSS + public exploit-code flags at the snapshot
  leaky (audit only): nvd_status, n_references, n_affected_cpes, CVSS v3 (NIST)

Run:  python model/build_dataset.py
"""

import re
import time

import numpy as np
import pandas as pd

from common import (DATASET_PATH, KEV_END, SCORE_YEAR, TRAIN_YEARS, cutoff_for,
                    load_kev, load_vulns, path)

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------

def build_cohorts(v: pd.DataFrame, kev: pd.DataFrame) -> pd.DataFrame:
    kev_date = kev.set_index("cve_id")["date_added"]
    parts = []
    for y in TRAIN_YEARS:
        d = v[(v.published >= f"{y}-01-01") & (v.published < f"{y}-10-01")].copy()
        d["snapshot_month"] = d.published.dt.to_period("M").astype(str)
        d["cohort_year"] = y
        parts.append(d)

    sb = pd.read_csv(path("scoreboard_2026.csv"))
    sig_cols = ["epss", "exploitdb", "metasploit", "nuclei", "poc_github"]
    d = v.drop(columns=["published"]).merge(
        sb[["cve_id", "published", "snapshot_month"] + sig_cols], on="cve_id", how="right")
    d["published"] = pd.to_datetime(d["published"])
    d["cohort_year"] = SCORE_YEAR
    d["description"] = d["description"].fillna("")
    parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    df["snapshot_end"] = (pd.PeriodIndex(df.snapshot_month, freq="M").end_time.normalize())
    df["cutoff"] = df.cohort_year.map(cutoff_for)
    df["kev_date"] = df.cve_id.map(kev_date)

    # Zero-day filter: already on KEV by the end of the snapshot month.
    zero_day = df.kev_date.notna() & (df.kev_date <= df.snapshot_end)
    log(f"dropping {int(zero_day[df.cohort_year < SCORE_YEAR].sum())} already-on-KEV rows "
        f"from training cohorts")
    df = df[~zero_day | (df.cohort_year == SCORE_YEAR)].copy()

    df["label"] = np.where(
        df.cohort_year == SCORE_YEAR, np.nan,
        (df.kev_date.notna() & (df.kev_date <= df.cutoff)).astype(float))
    df["months_to_cutoff"] = (df.cutoff - df.snapshot_end).dt.days / 30.44
    # History features are computed as of this date (KEV data ends 2025-12-31).
    df["hist_date"] = df.snapshot_end.clip(upper=KEV_END)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exploit signals (EPSS + public exploit code) at the snapshot
# ---------------------------------------------------------------------------

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    sig_cols = ["epss", "exploitdb", "metasploit", "nuclei", "poc_github"]
    frames = []
    for y in (2024, 2025):
        s = pd.read_csv(path(f"exploit_signals_{y}.csv"),
                        usecols=["month", "cve_id"] + sig_cols)
        sub = df.loc[df.cohort_year == y, ["cve_id", "snapshot_month"]]
        s = s.merge(sub, left_on=["cve_id", "month"],
                    right_on=["cve_id", "snapshot_month"], how="inner")
        frames.append(s.drop(columns=["month"]))
    s = pd.concat(frames, ignore_index=True)
    s = s.rename(columns={c: c + "_s" for c in sig_cols})
    df = df.merge(s, on=["cve_id", "snapshot_month"], how="left")
    has_file = df.cohort_year.isin([2024, 2025])
    for c in sig_cols:
        df.loc[has_file, c] = df.loc[has_file, c + "_s"]
        df = df.drop(columns=c + "_s")
    flags = ["exploitdb", "metasploit", "nuclei", "poc_github"]
    # Same convention as the scoreboard: no data -> blank EPSS and 0 flags.
    # Years without any signals file (2022-23) stay NaN so the model can tell.
    with_signals = df.cohort_year >= 2024
    df.loc[with_signals, flags] = df.loc[with_signals, flags].fillna(0)
    df["signals_available"] = with_signals.astype(int)
    df["any_exploit_code"] = df[flags].max(axis=1)
    df["epss_pct_cohort"] = df.groupby("snapshot_month")["epss"].rank(pct=True)
    df["epss_v4"] = (df.snapshot_end >= "2025-03-17").astype(int)
    return df


# ---------------------------------------------------------------------------
# History counts (time-safe): events of the same key up to hist_date
# ---------------------------------------------------------------------------

def count_prior(row_keys, row_dates, ev_keys, ev_dates, window_days=None):
    """For each row, count events with the same key and date <= row date
    (and > row date - window_days, if given)."""
    out = np.zeros(len(row_keys), dtype=np.int32)
    ev = pd.DataFrame({"k": ev_keys, "d": ev_dates}).dropna()
    groups = {k: np.sort(g.d.values.astype("datetime64[D]"))
              for k, g in ev.groupby("k")}
    rows = pd.DataFrame({"k": row_keys, "d": row_dates.values.astype("datetime64[D]")})
    for k, idx in rows.groupby("k").groups.items():
        arr = groups.get(k)
        if arr is None:
            continue
        d = rows.loc[idx, "d"].values
        hi = np.searchsorted(arr, d, side="right")
        if window_days is None:
            out[idx] = hi
        else:
            lo = np.searchsorted(arr, d - np.timedelta64(window_days, "D"), side="right")
            out[idx] = hi - lo
    return out


def build_vocab_regex(names):
    names = sorted({n for n in names if n}, key=len, reverse=True)
    pat = r"(?<![\w-])(" + "|".join(re.escape(n) for n in names) + r")(?![\w-])"
    return re.compile(pat, re.IGNORECASE)


def add_history(df: pd.DataFrame, v: pd.DataFrame, kev: pd.DataFrame) -> pd.DataFrame:
    kev = kev.merge(v[["cve_id", "assigner"]], on="cve_id", how="left")

    # -- assigner (who issued the CVE; known on publish day)
    log("assigner history")
    df["assigner_kev_all"] = count_prior(df.assigner, df.hist_date, kev.assigner, kev.date_added)
    df["assigner_kev_12m"] = count_prior(df.assigner, df.hist_date, kev.assigner, kev.date_added, 365)
    df["assigner_cves_12m"] = count_prior(df.assigner, df.hist_date, v.assigner, v.published, 365)
    df["assigner_kev_rate_12m"] = df.assigner_kev_12m / (df.assigner_cves_12m + 1)

    # -- vendor / product named in the description, matched to CISA's KEV names
    log("vendor / product matching")
    vendor_names = kev.vendor.str.strip().str.lower()
    vendor_re = build_vocab_regex(vendor_names.unique())
    product_names = kev["product"].str.strip().str.lower()
    ok = (product_names.str.len() >= 4) & ~product_names.str.contains("multiple|unspecified")
    product_re = build_vocab_regex(product_names[ok].unique())

    def first(regex, text):
        m = regex.search(text)
        return m.group(1).lower() if m else None

    df["desc_vendor"] = [first(vendor_re, t) for t in df.description]
    df["desc_product"] = [first(product_re, t) for t in df.description]
    kev_vendor = vendor_names
    kev_product = product_names
    df["vendor_kev_all"] = count_prior(df.desc_vendor.fillna("~"), df.hist_date, kev_vendor, kev.date_added)
    df["vendor_kev_12m"] = count_prior(df.desc_vendor.fillna("~"), df.hist_date, kev_vendor, kev.date_added, 365)
    df["product_kev_all"] = count_prior(df.desc_product.fillna("~"), df.hist_date, kev_product, kev.date_added)
    df["product_kev_12m"] = count_prior(df.desc_product.fillna("~"), df.hist_date, kev_product, kev.date_added, 365)

    # -- cited CVEs (e.g. "incomplete fix for CVE-2024-1234")
    log("cited CVEs")
    kev_dates = kev.set_index("cve_id")["date_added"]
    cve_re = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
    cites_any, cites_kev = [], []
    for cid, text, hd in zip(df.cve_id, df.description, df.hist_date):
        cited = {c.upper() for c in cve_re.findall(text)} - {cid}
        cites_any.append(int(bool(cited)))
        cites_kev.append(int(any(c in kev_dates.index and kev_dates[c] <= hd for c in cited)))
    df["cites_cve"] = cites_any
    df["cites_kev_cve"] = cites_kev
    return df


# ---------------------------------------------------------------------------
# Description keywords + CNA-supplied CVSS v4
# ---------------------------------------------------------------------------

KEYWORDS = {
    "kw_unauth": r"unauthenticated|without authentication|no authentication|pre-?auth|"
                 r"does not require authentication|remote,? unauthenticated",
    "kw_rce": r"remote code execution|execute arbitrary (?:code|commands)|arbitrary code execution|"
              r"code execution|command injection|os command",
    "kw_auth_bypass": r"authentication bypass|bypass (?:the )?authentication|auth(?:entication)? bypass|"
                      r"bypass (?:the )?login",
    "kw_privesc": r"privilege escalation|elevation of privilege|elevate privileges|"
                  r"escalate privileges|gain root|root privileges|system privileges",
    "kw_in_wild": r"exploited in the wild|may have been exploited|actively exploited|being exploited|"
                  r"limited,? targeted attack|aware of (?:a )?report|in-the-wild",
    "kw_memory": r"use[- ]after[- ]free|type confusion|out[- ]of[- ]bounds write|heap (?:buffer )?overflow|"
                 r"memory corruption|integer overflow",
    "kw_deserial": r"deseriali[sz]ation",
    "kw_path_trav": r"path traversal|directory traversal",
    "kw_file_upload": r"arbitrary file upload|unrestricted (?:file )?upload|upload arbitrary",
    "kw_sqli": r"sql injection",
    "kw_xss": r"cross[- ]site scripting|\bxss\b",
    "kw_csrf": r"cross[- ]site request forgery|\bcsrf\b",
    "kw_dos": r"denial of service|\bdos\b|crash",
    "kw_edge": r"\bvpn\b|firewall|gateway|\brouter\b|appliance|load balancer|ssl[- ]vpn|edge|"
               r"remote access|secure access|netscaler|fortigate|pan-os",
    "kw_mail": r"exchange server|mail server|email|\bsmtp\b|\bimap\b|webmail|zimbra|roundcube",
    "kw_mgmt": r"management (?:console|interface|server|center)|admin(?:istration)? (?:console|panel)|"
               r"\bmdm\b|endpoint manager|file transfer",
    "kw_browser_os": r"\bchrome\b|chromium|webkit|safari|\bios\b|ipados|macos|windows|android|kernel",
    "kw_wordpress": r"wordpress|plugin for wordpress|\bwp\b",
    "kw_mobile_app": r"android application|ios application|mobile app",
    "kw_requires_auth": r"authenticated (?:attacker|user)|requires? authentication|"
                        r"with (?:admin|administrator|high) privileges|low[- ]privileged",
    "kw_local": r"local attacker|locally|physical access",
}


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    log("keyword flags")
    text = df.description.str.lower()
    for name, pat in KEYWORDS.items():
        df[name] = text.str.contains(pat, regex=True).astype(np.int8)
    df["desc_len"] = df.description.str.len()

    vec = df.cvss4_vector.fillna("")
    df["cvss4_av_network"] = np.where(vec == "", np.nan, vec.str.contains("AV:N").astype(float))
    df["cvss4_pr_none"] = np.where(vec == "", np.nan, vec.str.contains("PR:N").astype(float))
    df["cvss4_ui_none"] = np.where(vec == "", np.nan, vec.str.contains("UI:N").astype(float))

    # Leaky columns — kept ONLY for the leakage audit, never in the default model.
    df["leak_nvd_reviewed"] = df.nvd_status.isin(["Analyzed", "Modified"]).astype(int)
    df["leak_cvss3_missing"] = df.cvss3_score.isna().astype(int)
    df["leak_cvss3_by_nist"] = (df.cvss3_source == "nvd@nist.gov").astype(int)
    df["leak_cwe_missing"] = (df.cwe_ids.isna() | df.cwe_ids.fillna("").str.startswith("NVD-CWE")).astype(int)
    return df


def main():
    log("loading")
    v = load_vulns()
    kev = load_kev()
    df = build_cohorts(v, kev)
    log(f"cohort rows: {df.groupby('cohort_year').size().to_dict()}")
    log(f"positives:   {df.groupby('cohort_year').label.sum().to_dict()}")
    df = add_signals(df)
    df = add_history(df, v, kev)
    df = add_text_features(df)
    drop = ["cvss4_vector", "kev_date", "cutoff"]
    df = df.drop(columns=drop)
    df.to_parquet(DATASET_PATH, index=False)
    log(f"wrote {DATASET_PATH}  {df.shape}")


if __name__ == "__main__":
    main()
