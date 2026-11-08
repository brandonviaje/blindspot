const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = x => (x == null ? "–" : (x * 100).toFixed(x < 0.01 ? 2 : 1) + "%");
const api = async (u, o) => { const r = await fetch(u, o); const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j; };
const debounce = (f, ms = 250) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };

// tabs
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll("section").forEach(s => s.hidden = s.id !== "tab-" + b.dataset.tab);
  if (b.dataset.tab === "feed" && !feedLoaded) loadFeed();
  if (b.dataset.tab === "proof") drawChart();
});

function tags(it) {
  const t = [];
  if (!it.nist_reviewed) t.push('<span class="tag warn">Not reviewed by NIST</span>');
  if (it.exploit_code) t.push('<span class="tag crit">Exploit code public</span>');
  if (it.no_login) t.push('<span class="tag">No login needed</span>');
  if (it.code_exec) t.push('<span class="tag">Code execution</span>');
  return t.join(" ");
}

// header
api("/api/summary").then(s => {
  // lead: exploited CVEs caught replaying the last test year; the bar shows the share of that year's exploited CVEs
  const share = s.positives ? s.hits / s.positives : null;
  $("#kpis").innerHTML = `
    <div class="kpi lead"><b>${s.hits}</b><span>exploited CVEs in our top 500, replaying ${s.test_year}</span>${share != null ? `<div class="gap"><i style="width:${(share * 100).toFixed(1)}%"></i></div><small>of ${s.positives} that year · best standard ranking: ${s.best_standard_hits}</small>` : `<small>best standard ranking: ${s.best_standard_hits}</small>`}</div>
    ${s.warning_days != null ? `<div class="kpi"><b>${s.warning_days} days</b><span>median warning before CISA listed them</span></div>` : ""}`;
});

// patch plan
api("/api/vendors").then(vs => {
  // terms already in the box get their own chip first, so each one shows as ticked
  const shown = vs.slice(0, 18);
  const typed = $("#q").value.toLowerCase().split(",").map(s => s.trim()).filter(t => t && !shown.includes(t));
  $("#vendorChips").innerHTML = [...typed, ...shown].map(v => `<button class="chip" data-v="${esc(v)}">${esc(v)}</button>`).join("");
  syncChips();
  document.querySelectorAll(".chip").forEach(c => c.onclick = () => {
    let terms = $("#q").value.split(",").map(s => s.trim()).filter(Boolean);
    const v = c.dataset.v;
    terms = terms.map(t => t.toLowerCase()).includes(v) ? terms.filter(t => t.toLowerCase() !== v) : [...terms, v];
    $("#q").value = terms.join(", "); syncChips(); loadPlan();
  });
});


function syncChips() {
  const terms = $("#q").value.toLowerCase().split(",").map(s => s.trim());
  document.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", terms.includes(c.dataset.v)));
}

async function loadPlan() {
  const q = $("#q").value, n = $("#perWeek").value;
  $("#perWeekVal").textContent = n;
  const d = await api(`/api/patch-plan?q=${encodeURIComponent(q)}&per_week=${n}`);
  if (!d.matching) {
    $("#planSummary").innerHTML = "";
    $("#planList").innerHTML = `<div class="empty">No 2026 vulnerabilities mention that software. Try a vendor name like "cisco" or "wordpress".</div>`;
    return;
  }
  $("#planSummary").innerHTML = `
    <div><div class="big">${d.items.length}</div><span class="muted">fixes this week, of ${d.matching.toLocaleString()} that affect you</span></div>
    <div><div class="big">${pct(d.covered)}</div><span class="muted">of your total risk score covered</span></div>
    <div class="meter" title="share of risk covered"><i style="width:${(d.covered * 100).toFixed(1)}%"></i></div>`;
  const mx = Math.max(...d.items.map(i => i.score)) || 1;
  $("#planList").innerHTML = d.items.map((it, i) => `
    <div class="item" data-id="${it.cve_id}" tabindex="0" role="button" style="--r:${(it.score / mx).toFixed(3)}">
      <div class="rk">${i + 1}</div>
      <div>
        <div class="top"><span class="id">${it.cve_id}</span>${it.vendor ? `<span class="muted">${esc(it.vendor)}${it.product ? " · " + esc(it.product) : ""}</span>` : ""} ${tags(it)}</div>
        ${it.why ? `<div class="why">${esc(it.why)} <span class="tag" title="Written by IBM Granite 4.0 from the evidence below">Granite</span></div>` : ""}
        ${it.evidence ? `<div class="evidence">${esc(it.evidence)}</div>` : ""}
        <div class="desc">${esc(it.description)}${it.description.length >= 260 ? "…" : ""}</div>
      </div>
      <div class="score"><b>${it.score.toFixed(2)}</b><small>risk · #${it.rank.toLocaleString()} overall</small></div>
    </div>`).join("");
  bindOpen("#planList .item");
}
$("#q").addEventListener("input", debounce(() => { syncChips(); loadPlan(); }));
$("#perWeek").addEventListener("input", debounce(loadPlan, 80));
loadPlan();

// feed
let feedLoaded = false, offset = 0; const LIMIT = 50;
async function loadFeed() {
  feedLoaded = true;
  const q = $("#feedQ").value, u = $("#unrev").checked ? 1 : 0;
  const d = await api(`/api/feed?limit=${LIMIT}&offset=${offset}&unreviewed=${u}${q ? "&q=" + encodeURIComponent(q) : ""}`);
  $("#feedCount").textContent = `${d.total.toLocaleString()} vulnerabilities, highest risk first.`;
  const fm = Math.max(...d.items.map(i => i.score)) || 1;
  $("#feedBody").innerHTML = d.items.map(it => `
    <tr data-id="${it.cve_id}" tabindex="0" class="${it.nist_reviewed ? "" : "gap"}">
      <td class="n">${it.rank.toLocaleString()}</td><td class="mono">${it.cve_id}</td>
      <td>${esc(it.vendor || "–")}</td><td class="mono">${it.published}</td>
      <td>${it.nist_reviewed ? '<span class="tag ok">Reviewed</span>' : '<span class="tag warn">Not yet</span>'}</td>
      <td class="n"><span class="rcell"><i style="--r:${(it.score / fm).toFixed(3)}"></i>${it.score.toFixed(3)}</span></td><td class="n">${pct(it.epss)}</td>
    </tr>`).join("");
  $("#pageInfo").textContent = `${(offset + 1).toLocaleString()}–${Math.min(offset + LIMIT, d.total).toLocaleString()}`;
  $("#prev").disabled = offset === 0; $("#next").disabled = offset + LIMIT >= d.total;
  bindOpen("#feedBody tr");
}
$("#feedQ").addEventListener("input", debounce(() => { offset = 0; loadFeed(); }));
$("#unrev").onchange = () => { offset = 0; loadFeed(); };
$("#prev").onclick = () => { offset = Math.max(0, offset - LIMIT); loadFeed(); };
$("#next").onclick = () => { offset += LIMIT; loadFeed(); };

// detail dialog
function bindOpen(sel) {
  document.querySelectorAll(sel).forEach(el => {
    el.onclick = () => openCve(el.dataset.id);
    el.onkeydown = e => { if (e.key === "Enter") openCve(el.dataset.id); };
  });
}
async function openCve(id) {
  const d = await api(`/api/cve/${id}`);
  const nb = (d.neighbors || []).map(n => `
    <div class="nb">
      <div class="mono">${n.cve_id}<br>${n.exploited ? `<span class="tag crit">Exploited${n.days_to_kev != null ? " · " + Math.max(0, n.days_to_kev) + "d" : ""}</span>` : '<span class="tag">Not exploited</span>'}</div>
      <div class="mono">${(n.similarity * 100).toFixed(0)}%<br><span class="muted" style="font-size:11px">similar</span></div>
      <div class="nd muted">${esc(n.description)}</div>
    </div>`).join("");
  $("#dlgBody").innerHTML = `
    <button class="btn ghost close" onclick="dlg.close()">Close</button>
    <div><div class="mono muted">Rank #${d.rank.toLocaleString()} of 72,670 · published ${d.published}</div><h2 class="mono">${d.cve_id}</h2></div>
    <div>${tags(d)}</div>
    ${d.why || d.evidence ? `<div class="panel"><b>Why it's high risk</b>${d.why ? `<br>${esc(d.why)} <span class="tag">Granite</span>` : ""}${d.evidence ? `<div class="evidence">Evidence: ${esc(d.evidence)}</div>` : ""}</div>` : ""}
    <div><b>Description</b><p class="muted" style="margin:4px 0 0">${esc(d.description)}</p></div>
    ${nb ? `<div><b>10 most similar past vulnerabilities (IBM Granite embeddings)</b>${nb}</div>` : `<p class="note">Similar-vulnerability evidence is precomputed for the top ${100} only.</p>`}`;
  dlg.showModal();
}

// live scoring
$("#scoreBtn").onclick = async () => {
  const out = $("#scoreOut"), btn = $("#scoreBtn");
  btn.disabled = true; btn.textContent = "Granite is reading…";
  out.innerHTML = '<div class="empty">Embedding with IBM Granite on this machine. The first run loads the model, which takes a few seconds.</div>';
  try {
    const d = await api("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ description: $("#newDesc").value }) });
    const p = d.percentile_2026;
    const level = p >= 0.95 ? ["crit", "Very close to known attacks"] : p >= 0.8 ? ["warn", "Close to known attacks"] : ["ok", "Not especially close to known attacks"];
    out.innerHTML = `
      <div style="display:grid; gap:10px">
        <div class="verdict ${level[0]}">${level[1]}</div>
        <div><span class="big">${(p * 100).toFixed(0)}th</span> <span class="muted">percentile: closer to already-exploited vulnerabilities than ${(p * 100).toFixed(0)}% of 2026 CVEs</span></div>
        <div><div class="gauge" role="img" aria-label="${(p * 100).toFixed(0)}th percentile"><i style="left:${(p * 100).toFixed(1)}%"></i></div><div class="gscale"><span style="left:80%">80th</span><span style="left:95%">95th</span></div></div>
        <b>Closest already-exploited vulnerabilities</b>
        ${d.matches.map(m => `
          <div class="nb" style="grid-template-columns:130px 1fr 56px">
            <div class="mono">${m.cve_id}</div>
            <div>${esc(m.vendor)} ${esc(m.product)}${m.ransomware ? ' <span class="tag crit">Ransomware</span>' : ""}<br><span class="muted" style="font-size:12.5px">${esc(m.name || "")} · on KEV ${m.date_added}</span></div>
            <div class="mono simc">${(m.similarity * 100).toFixed(0)}%<i style="--r:${m.similarity}"></i></div>
          </div>`).join("")}
        <p class="note">Similarity alone, not the full Blindspot score. The full score also uses assigner and vendor history, exploit code and EPSS.</p>
      </div>`;
  } catch (e) {
    out.innerHTML = `<div class="empty">${esc(e.message)}</div>`;
  } finally { btn.disabled = false; btn.textContent = "Score with Granite"; }
};

// chart
let drawn = false;
async function drawChart() {
  if (drawn) return; drawn = true;
  const rows = await api("/api/results");
  const order = ["Highest CVSS 3 first", "EPSS first", "Exploit code first, then EPSS", "Blindspot, publish-day info only", "Blindspot"];
  const years = [...new Set(rows.map(r => r.year))];
  const max = Math.max(...rows.map(r => r.hits));
  const W = 960, rowH = 30, labelW = 250, groupGap = 34, top = 6;
  const H = top + years.length * (order.length * rowH + groupGap);
  const stats = '<div class="proofstats">' + years.map(yr => {
    const rs = rows.filter(r => r.year === yr), b = rs.find(r => r.ranking === "Blindspot");
    const o = rs.filter(r => !r.ranking.startsWith("Blindspot")).sort((a, c) => c.hits - a.hits)[0];
    return b ? `<div class="stat"><span>Tested on ${yr}</span><b>${b.hits}</b><small>caught by Blindspot${o ? ` · best other method: ${o.hits} (${esc(o.ranking)})` : ""}</small></div>` : "";
  }).join("") + '</div><div class="legend"><span><i style="background:var(--accent)"></i>Blindspot</span><span><i style="background:var(--accent);opacity:.5"></i>Blindspot, publish-day info only</span><span><i style="background:var(--bar-base)"></i>Other methods</span></div>';
  let y = top, svg = "";
  years.forEach(yr => {
    svg += `<text x="0" y="${y + 12}" style="font-weight:600;fill:var(--ink)">Tested on ${yr}</text><line x1="0" x2="${W}" y1="${y + 20}" y2="${y + 20}" stroke="var(--line)"/>`;
    y += 8;
    y += 22;
    order.forEach(name => {
      const r = rows.find(x => x.year === yr && x.ranking === name); if (!r) return;
      const w = (W - labelW - 50) * r.hits / max, isB = name.startsWith("Blindspot");
      svg += `<text x="${labelW - 10}" y="${y + 15}" text-anchor="end" ${isB ? 'style="fill:var(--ink);font-weight:600"' : ""}>${esc(name)}</text>
              <rect x="${labelW}" y="${y + 3}" width="${Math.max(w, 2)}" height="${rowH - 10}" rx="3" class="${name === "Blindspot" ? "hl" : isB ? "mid" : "base"}"></rect>
              <text class="val" x="${labelW + w + 6}" y="${y + 15}">${r.hits}</text>`;
      y += rowH;
    });
    y += groupGap - 22;
  });
  $("#chart").innerHTML = stats + `<svg class="chart" viewBox="0 0 ${W} ${y}" width="100%" role="img" aria-label="Exploited vulnerabilities in the top 500 by ranking method and test year">${svg}</svg>`;
}

// theme toggle
(() => {
  const root = document.documentElement, btn = $("#theme");
  const dark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const sync = () => btn.textContent = dark() ? "Light mode" : "Dark mode";
  btn.onclick = () => { root.dataset.theme = dark() ? "light" : "dark"; sync(); };
  sync();
})();
