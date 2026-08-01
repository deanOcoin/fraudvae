// training data tab
import { get, post, pollTask, fmt, el, esc, palette } from "./api.js";

let data = null;
let schema = null;
let charts = [];

export async function initData(state) {
  schema = state.schema || (await get("/api/schema"));
  await load();
  document.getElementById("btn-fetch").addEventListener("click", () => runPipeline("/api/data/fetch", {}));
  document.getElementById("btn-build").addEventListener("click", () => runPipeline("/api/data/build", {}));
  document.getElementById("btn-prepare").addEventListener("click", () => {
    const name = document.getElementById("prepare-name").value || "default";
    const cap = parseInt(document.getElementById("prepare-cap").value, 10);
    runPipeline("/api/data/prepare", { name, max_rows_per_source: Number.isFinite(cap) ? cap : null });
  });
}

async function load() {
  data = await get("/api/datasets");
  renderSummary();
  renderList();
  renderCoverage();
  const first = data.sources.find((s) => s.built);
  if (first) showSource(first.key);
}

function renderSummary() {
  const built = data.sources.filter((s) => s.built);
  const real = built.filter((s) => /real/.test(s.context)).length;
  const span = built.map((s) => s.stats).filter((st) => st.absolute_time);
  const lo = Math.min(...span.map((s) => s.ts_min)), hi = Math.max(...span.map((s) => s.ts_max));
  document.getElementById("corpus-summary").innerHTML = `
    <div class="stat"><b>${fmt.int(data.total_rows)}</b><span>transactions</span></div>
    <div class="stat"><b>${fmt.int(data.total_fraud)}</b><span>labelled fraud (${fmt.pct(data.total_fraud / Math.max(data.total_rows, 1), 3)})</span></div>
    <div class="stat"><b>${built.length} / ${data.sources.length}</b><span>sources built, ${real} with real transactions</span></div>
    <div class="stat"><b>${span.length ? fmt.date(lo).slice(0, 4) + "–" + fmt.date(hi).slice(0, 4) : "–"}</b><span>calendar span of dated sources</span></div>
    <div class="stat"><b>${data.prepared.length}</b><span>prepared set${data.prepared.length === 1 ? "" : "s"}${data.prepared.length ? ": " + data.prepared.map((p) => p.name).join(", ") : ""}</span></div>`;
  document.getElementById("data-lede").textContent = `${data.sources.length} public transaction datasets mapped onto one schema of ${schema.numeric.length} numeric and ${schema.categorical.length} categorical fields. Where a source has no notion of a field, the field is marked unobserved and the model's likelihood skips it.`;
}

function renderList() {
  const host = document.getElementById("source-list");
  host.innerHTML = "";
  for (const s of data.sources) {
    const st = s.stats;
    const row = el("div", { class: "source-row" + (s.built ? "" : " missing"), "data-key": s.key, onclick: () => showSource(s.key) },
      el("span", { class: "name" }, s.name),
      el("span", { class: "rows" }, st ? fmt.int(st.rows) : fmt.int(s.approx_rows) + " expected"),
      el("span", { class: "meta" },
        el("span", { class: "pill " + (s.built ? "ok" : s.raw_present ? "" : "no") }, s.built ? "built" : s.raw_present ? "raw only" : "missing"),
        " ", s.access === "kaggle" ? "kaggle" : "github", " ",
        st ? `fraud ${fmt.pct(st.fraud_rate, 2)}` : `fraud ≈${fmt.pct(s.label_rate_hint, 2)}`));
    host.append(row);
  }
}

function destroyCharts() { charts.forEach((c) => c.destroy()); charts = []; }

function showSource(key) {
  document.querySelectorAll(".source-row").forEach((r) => r.classList.toggle("active", r.dataset.key === key));
  const s = data.sources.find((x) => x.key === key);
  const box = document.getElementById("source-detail");
  destroyCharts();
  const st = s.stats;
  const parts = [];
  parts.push(`<h2>${esc(s.name)}</h2><p class="dim" style="margin:0">${esc(s.context)}</p>`);
  parts.push(`<dl class="kv">
    <dt>source</dt><dd><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.url)}</a></dd>
    <dt>license</dt><dd>${esc(s.license)}</dd>
    <dt>cite</dt><dd>${esc(s.citation)}</dd>
    <dt>files</dt><dd><code>${s.files.map(esc).join("</code>, <code>")}</code>${s.lock ? " – sha256 " + s.lock.files.map((f) => `<code title="${esc(f.path)}">${f.sha256.slice(0, 12)}…</code>`).join(", ") : ""}</dd>
    ${s.notes ? `<dt>notes</dt><dd>${esc(s.notes)}</dd>` : ""}
  </dl>`);
  if (!st) {
    parts.push(`<div class="notice warn">Not built. ${s.access === "kaggle" ? `Download it from Kaggle and drop the zip into <code>${esc(data.raw_dir)}/${esc(s.key)}/</code>, then press <b>Build unified parquet</b>.` : "Press <b>Fetch raw data</b> then <b>Build</b>."}</div>`);
    box.innerHTML = parts.join("");
    return;
  }
  const sp = st.splits;
  parts.push(`<table class="metric-table"><thead><tr><th>split</th><th>rows</th><th>fraud</th><th>rate</th></tr></thead><tbody>
    ${["train", "val", "test"].map((k) => `<tr><td>${k}</td><td>${fmt.int(sp[k].rows)}</td><td>${fmt.int(sp[k].fraud)}</td><td>${fmt.pct(sp[k].fraud / Math.max(sp[k].rows, 1), 2)}</td></tr>`).join("")}
    <tr><td><b>all</b></td><td><b>${fmt.int(st.rows)}</b></td><td><b>${fmt.int(st.fraud)}</b></td><td><b>${fmt.pct(st.fraud_rate, 3)}</b></td></tr></tbody></table>
    <p class="dim small">${st.absolute_time ? `dated ${fmt.date(st.ts_min)} to ${fmt.date(st.ts_max)}` : "relative time only (simulation steps)"}; split is by time, oldest 70% train, next 10% validation, newest 20% test. built in ${st.build_seconds}s.</p>`);
  const aq = st.amount;
  parts.push(`<p class="small">Median legit amount ${fmt.num(aq.legit_quantiles["0.5"])}, fraud ${aq.fraud_quantiles["0.5"] != null ? fmt.num(aq.fraud_quantiles["0.5"]) : "–"} (99th pct legit ${fmt.num(aq.legit_quantiles["0.99"])}).</p>`);
  parts.push(`<div class="chart-pair">
    <div class="chart-box"><h4>amount distribution, log scale (legit and fraud each sum to 100%)</h4><div class="chart-wrap" style="height:190px"><canvas id="c-amount"></canvas></div></div>
    <div class="chart-box"><h4>${st.by_hour ? "fraud rate by hour of day" : st.by_txn_type ? "fraud rate by transaction type" : "fraud rate by category"}</h4><div class="chart-wrap" style="height:190px"><canvas id="c-by"></canvas></div></div>
  </div>`);
  if (st.by_category && Object.keys(st.by_category).length > 1) {
    parts.push(`<div class="chart-box" style="margin-top:1rem"><h4>fraud rate by merchant category</h4><div class="chart-wrap" style="height:180px"><canvas id="c-cat"></canvas></div></div>`);
  }
  if (st.by_scenario) {
    parts.push(`<p class="small dim">fraud scenarios: ${Object.entries(st.by_scenario).filter(([k]) => k !== "0").map(([k, v]) => `scenario ${k}: ${fmt.int(v)}`).join(", ")} (1 = amount above 220, 2 = compromised terminal, 3 = compromised card)</p>`);
  }
  box.innerHTML = parts.join("");
  if (!window.Chart) return;
  charts.push(histChart(document.getElementById("c-amount"), aq.legit_hist_log10, aq.fraud_hist_log10));
  const by = st.by_hour || st.by_txn_type || st.by_category;
  charts.push(rateChart(document.getElementById("c-by"), by, st.by_hour ? (k) => k : (k) => k.replace(/_/g, " ")));
  if (st.by_category && document.getElementById("c-cat")) charts.push(rateChart(document.getElementById("c-cat"), st.by_category, (k) => k.replace(/_/g, " "), true));
}

function histChart(canvas, legit, fraud) {
  const norm = (c) => { const s = c.reduce((a, b) => a + b, 0) || 1; return c.map((x) => x / s); };
  const labels = legit.edges.slice(0, -1).map((e, i) => ((e + legit.edges[i + 1]) / 2).toFixed(2));
  const fraudCounts = fraud.counts.length ? rebin(fraud, legit.edges) : [];
  return new Chart(canvas, {
    type: "line",
    data: { labels, datasets: [
      { label: "legit", data: norm(legit.counts), borderColor: palette.ink2, backgroundColor: "rgba(86,104,121,0.18)", fill: true, pointRadius: 0, tension: 0.3, borderWidth: 1.5 },
      { label: "fraud", data: fraudCounts.length ? norm(fraudCounts) : [], borderColor: palette.alarm, backgroundColor: "rgba(200,16,46,0.18)", fill: true, pointRadius: 0, tension: 0.3, borderWidth: 1.5 },
    ] },
    options: { animation: false, responsive: true, maintainAspectRatio: false,
      scales: { x: { title: { display: true, text: "log10(amount + 1)" }, ticks: { maxTicksLimit: 8 } }, y: { ticks: { display: false }, grid: { color: "#eef1f4" }, title: { display: true, text: "share of rows" } } },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${(100 * c.parsed.y).toFixed(2)}% of rows` } } } },
  });
}

// fraud histogram has its own edges; put it on the legit edges so the bars line up
function rebin(h, edges) {
  const out = new Array(edges.length - 1).fill(0);
  for (let i = 0; i < h.counts.length; i++) {
    const mid = (h.edges[i] + h.edges[i + 1]) / 2;
    let j = edges.findIndex((e, k) => k < edges.length - 1 && mid >= e && mid < edges[k + 1]);
    if (j < 0) j = mid >= edges[edges.length - 1] ? out.length - 1 : 0;
    out[j] += h.counts[i];
  }
  return out;
}

function rateChart(canvas, by, labelFn, sortByRate = false) {
  let entries = Object.entries(by);
  if (sortByRate) entries.sort((a, b) => b[1].fraud_rate - a[1].fraud_rate);
  return new Chart(canvas, {
    type: "bar",
    data: { labels: entries.map(([k]) => labelFn(k)), datasets: [{ label: "fraud rate", data: entries.map(([, v]) => 100 * v.fraud_rate), backgroundColor: palette.accent, borderWidth: 0 }] },
    options: { animation: false, responsive: true, maintainAspectRatio: false,
      scales: { y: { title: { display: true, text: "% fraud" }, grid: { color: "#eef1f4" } }, x: { ticks: { autoSkip: false, maxRotation: 60, font: { size: 10 } } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.parsed.y.toFixed(3)}% of ${fmt.int(entries[c.dataIndex][1].count)} rows` } } } },
  });
}

function renderCoverage() {
  const host = document.getElementById("coverage-grid");
  const built = data.sources.filter((s) => s.built);
  if (!built.length) { host.innerHTML = "<p class='dim'>nothing built yet</p>"; return; }
  const groups = [["core", "Transaction"], ["velocity", "Customer history"], ["account", "Account & customer"], ["ext", "Source specific"], ["ulb_pca", "ULB PCA block"]];
  const rows = [];
  for (const [g, title] of groups) {
    const feats = [...schema.categorical.filter((c) => c.group === g), ...schema.numeric.filter((f) => f.group === g)];
    if (!feats.length) continue;
    rows.push(`<tr><td class="grp" colspan="${built.length + 1}">${esc(title)}</td></tr>`);
    const shown = g === "ulb_pca" ? feats.slice(0, 1) : feats;
    for (const f of shown) {
      const label = g === "ulb_pca" ? "V1 … V28" : (f.label || f.name);
      rows.push(`<tr><td class="lbl">${esc(label)}</td>${built.map((s) => {
        const c = s.stats.coverage[f.name] ?? 0;
        const col = c <= 0.005 ? "#f1f3f5" : `rgba(11,110,121,${0.15 + 0.85 * c})`;
        return `<td class="cell" style="background:${col}" title="${esc(s.key)}: ${(100 * c).toFixed(1)}%"></td>`;
      }).join("")}</tr>`);
    }
  }
  host.innerHTML = `<table><thead><tr><th></th>${built.map((s) => `<th class="rot">${esc(s.key)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

async function runPipeline(path, body) {
  const log = document.getElementById("pipeline-log");
  log.textContent = "starting…";
  try {
    const { task } = await post(path, body);
    const t = await pollTask(task, (t) => { log.textContent = t.lines.join("\n"); log.scrollTop = log.scrollHeight; });
    if (t.state === "finished") { await load(); }
  } catch (e) {
    log.textContent += "\n" + e.message;
  }
}
