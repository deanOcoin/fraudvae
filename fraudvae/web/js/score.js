// score tab: the form, the receipt, the latent map
import { get, post, fmt, el, esc, palette } from "./api.js";

let schema, state;
let latentChart = null;
let latentBackground = null;   // {xy, y, src} from the active run
let lastResult = null;
let showAll = false;

const GROUP_ORDER = ["core", "velocity", "merchant", "account", "ext", "ulb_pca"];
const GROUP_NOTE = {
  core: "what the transaction itself says",
  velocity: "the customer's own history up to this moment, computed causally",
  merchant: "the merchant / terminal side, where the source has an id",
  account: "known about the account or customer",
  ext: "fields only some sources have",
  ulb_pca: "anonymised PCA components, ULB only",
};

export async function initScore(appState) {
  state = appState;
  schema = state.schema || (await get("/api/schema"));
  buildSourceSelect();
  buildFields();
  wireForm();
  await loadLatentBackground();
  window.addEventListener("metachanged", () => { loadLatentBackground(); updateLede(); });
  updateLede();
}

function updateLede() {
  const m = state.meta?.active_run;
  const lede = document.getElementById("score-lede");
  if (!m) {
    lede.textContent = "No trained model is loaded. Train one in Model & training and come back.";
  } else {
    lede.textContent = `The VAE reconstructs the transaction from a low-dimensional summary of normal traffic across ${m.sources.length} datasets; what it cannot reconstruct is novelty.`;
  }
}

function buildSourceSelect() {
  const sel = document.getElementById("source-select");
  sel.innerHTML = "";
  sel.append(el("option", { value: "" }, "Generic (no specific bank)"));
  for (const s of schema.sources) {
    if (!s.built) continue;
    sel.append(el("option", { value: s.key }, `${s.key}: ${s.name}`));
  }
  sel.addEventListener("change", () => { markUnused(); showSourceHint(); });
  showSourceHint();
}

function showSourceHint() {
  const key = document.getElementById("source-select").value;
  const s = schema.sources.find((x) => x.key === key);
  document.getElementById("source-hint").textContent = s ? s.context : "Which dataset's \"normal\" to compare against. Generic uses the shared representation only.";
}

function buildFields() {
  const host = document.getElementById("field-groups");
  host.innerHTML = "";
  for (const g of GROUP_ORDER) {
    const nums = schema.numeric.filter((f) => f.group === g);
    const cats = schema.categorical.filter((c) => c.group === g);
    if (!nums.length && !cats.length) continue;
    const grid = el("div", { class: "fgrid" });
    // categoricals first in core so channel/type/category lead, numbers after
    for (const c of cats) grid.append(catField(c));
    for (const f of nums) grid.append(numField(f));
    const body = g === "ulb_pca"
      ? el("details", {}, el("summary", { class: "dim" }, "28 PCA components (only meaningful for the ULB context)"), grid)
      : grid;
    const group = el("div", { class: "fgroup", "data-group": g },
      el("div", { class: "fgroup-head" }, el("h3", {}, schema.groups[g] || g), el("span", { class: "dim" }, GROUP_NOTE[g] || "")),
      body);
    host.append(group);
  }
  markUnused();
}

function numField(f) {
  const unit = f.unit ? ` (${f.unit})` : "";
  return el("label", { "data-feature": f.name, "data-kind": "numeric" },
    el("span", {}, f.label + unit),
    el("input", { type: "number", step: "any", name: f.name, placeholder: "unobserved" }));
}

function catField(c) {
  const sel = el("select", { name: c.name }, el("option", { value: "" }, "unobserved"));
  for (let i = 1; i < c.values.length; i++) {
    let label = c.values[i];
    if (c.name === "dow") label = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i - 1];
    if (c.name === "hour") label = `${String(i - 1).padStart(2, "0")}:00`;
    sel.append(el("option", { value: c.values[i] }, label.replace(/_/g, " ")));
  }
  return el("label", { "data-feature": c.name, "data-kind": "categorical" }, el("span", {}, c.label || c.name), sel);
}

// grey out fields the chosen source never fills, they'd be pure extrapolation
function markUnused() {
  const key = document.getElementById("source-select").value;
  document.querySelectorAll("#field-groups label[data-feature]").forEach((lab) => {
    const name = lab.dataset.feature;
    const f = schema.numeric.find((x) => x.name === name) || schema.categorical.find((x) => x.name === name);
    const cov = key ? (f?.coverage?.[key] ?? 0) : 1;
    lab.classList.toggle("unused", key && cov < 0.01);
    lab.title = key ? `observed in ${(100 * cov).toFixed(0)}% of ${key} rows` : "";
  });
}

function readForm() {
  const form = document.getElementById("score-form");
  const fd = new FormData(form);
  const features = {};
  for (const [k, v] of fd.entries()) {
    if (k === "source" || k === "fpr_target") continue;
    if (v === "" || v == null) continue;
    features[k] = v;
  }
  return { source: fd.get("source") || null, fpr_target: fd.get("fpr_target"), features };
}

function fillForm(features, source) {
  const form = document.getElementById("score-form");
  form.reset();
  if (source != null) document.getElementById("source-select").value = source;
  for (const [k, v] of Object.entries(features)) {
    const inp = form.elements[k];
    if (!inp) continue;
    if (inp.tagName === "SELECT") {
      inp.value = k === "hour" || k === "dow" ? String(Math.round(v)) : String(v);
    } else {
      inp.value = typeof v === "number" ? +v.toFixed(4) : v;
    }
  }
  markUnused();
  showSourceHint();
}

function wireForm() {
  const form = document.getElementById("score-form");
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await score();
  });
  document.getElementById("clear-form").addEventListener("click", () => {
    form.reset(); markUnused(); showSourceHint();
    document.getElementById("example-note").textContent = "";
  });
  document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", async () => {
    const want = b.dataset.example;
    const src = document.getElementById("source-select").value;
    const q = new URLSearchParams({ fraud: want, n: 1 });
    if (src) q.set("source", src);
    try {
      const [ex] = await get(`/api/examples?${q}`);
      fillForm(ex.features, ex.source);
      const scen = ex.fraud_scenario ? `, scenario ${ex.fraud_scenario}` : "";
      document.getElementById("example-note").textContent = `${ex.txn_id} (${ex.split} split, labelled ${ex.is_fraud ? "fraud" : "legitimate"}${scen})`;
      await score(ex.is_fraud);
    } catch (e) {
      document.getElementById("example-note").textContent = e.message;
    }
  }));
  document.getElementById("fpr-select").addEventListener("change", () => { if (lastResult) score(lastResult._label); });
  document.getElementById("batch-form").addEventListener("submit", batchScore);
}

async function score(label = null) {
  const btn = document.getElementById("score-btn");
  const status = document.getElementById("score-status");
  const req = readForm();
  if (!Object.keys(req.features).length) { status.textContent = "fill in at least one field"; return; }
  btn.disabled = true; status.textContent = "scoring"; status.classList.add("spin");
  try {
    const r = await post("/api/score", req);
    r._label = label;
    lastResult = r;
    showAll = false;
    renderReceipt(r, label);
    plotLatent(r);
    status.textContent = `${r.observed_features} fields observed, ${r.n_samples} posterior samples`;
  } catch (e) {
    status.textContent = e.message;
  } finally {
    btn.disabled = false; status.classList.remove("spin");
  }
}

function zClass(z) { return z >= 3 ? "hot" : z >= 1.5 ? "warm" : z < 0.5 ? "cold" : ""; }

function fmtVal(c) {
  if (c.kind === "categorical") {
    if (c.feature === "dow") return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][+c.observed] || c.observed;
    if (c.feature === "hour") return String(c.observed).padStart(2, "0") + ":00";
    return c.observed.replace(/_/g, " ");
  }
  const v = c.observed;
  if (Math.abs(v) >= 1000) return fmt.int(v);
  return fmt.num(v, Math.abs(v) < 10 ? 2 : 1);
}
function fmtExp(c) {
  if (c.kind === "categorical") return `p=${(c.prob_observed * 100).toFixed(c.prob_observed < 0.1 ? 1 : 0)}%`;
  const v = c.expected;
  return "exp " + (Math.abs(v) >= 1000 ? fmt.int(v) : fmt.num(v, Math.abs(v) < 10 ? 2 : 1));
}

function renderReceipt(r, label) {
  const box = document.getElementById("receipt");
  box.classList.remove("empty");
  const flagged = r.flag;
  const verdictClass = r.flag == null ? "na" : flagged ? "flag" : "ok";
  const verdictText = r.flag == null ? "NO CALIBRATION" : flagged ? "FLAG FOR REVIEW" : "LOOKS NORMAL";
  const op = r.operating_point ? r.operating_point.replace("fpr_", "") : "";
  const opPct = op ? (parseFloat(op) * 100).toFixed(op === "0.001" || op === "0.005" ? 1 : 0) + "%" : "";
  const items = r.contributions || [];
  const shown = showAll ? items : items.slice(0, 9);
  const maxZ = Math.max(3, ...items.map((c) => c.z));
  const ctxName = r.calibration === "generic" ? "generic context" : `${r.calibration} context`;
  const pct = r.legit_percentile;
  const pctStr = pct == null ? fmt.num(r.score, 2) : (pct >= 99.99 ? ">99.99" : pct.toFixed(pct > 99 ? 2 : 1)) + "th";

  const lines = [];
  lines.push(`<div class="rc-head"><div class="rc-title">FRAUDVAE SCORE</div><div class="dim">${esc(new Date().toISOString().replace("T", " ").slice(0, 19))}  ${esc(ctxName)}</div></div>`);
  lines.push(`<div class="rc-verdict ${verdictClass}"><span class="big">${pctStr}</span>${pct == null ? "raw score" : "percentile of legit traffic"}<br>${verdictText}</div>`);
  if (label != null) lines.push(`<div class="rc-line"><span class="k">ground truth</span><span>${label ? "FRAUD" : "legitimate"}</span></div>`);
  lines.push(`<div class="rc-sep"></div>`);
  lines.push(`<div class="rc-line"><span class="k">score (${esc(r.score_type)})</span><span>${fmt.num(r.score, 3)}</span></div>`);
  if (r.score_type === "norm") lines.push(`<div class="rc-line"><span class="k">raw -ELBO</span><span>${fmt.num(r.score_raw, 2)}</span></div>`);
  if (r.z_score != null) lines.push(`<div class="rc-line"><span class="k">z vs legit</span><span>${r.z_score >= 0 ? "+" : ""}${fmt.num(r.z_score, 2)}</span></div>`);
  lines.push(`<div class="rc-line"><span class="k">KL(q‖p)</span><span>${fmt.num(r.kl, 2)}</span></div>`);
  if (r.threshold != null) {
    lines.push(`<div class="rc-line"><span class="k">alert budget</span><span>${esc(opPct)} FPR</span></div>`);
    lines.push(`<div class="rc-line"><span class="k">threshold</span><span>${fmt.num(r.threshold, 3)}</span></div>`);
    if (r.expected_recall != null) lines.push(`<div class="rc-line"><span class="k">recall at budget</span><span>${fmt.pct(r.expected_recall, 1)}</span></div>`);
  }
  lines.push(`<div class="rc-line"><span class="k">field</span><span class="k">surprise (z)</span></div>`);
  lines.push(`<div class="rc-items">`);
  for (const c of shown) {
    const zc = zClass(c.z);
    const w = Math.max(2, Math.min(100, (Math.max(c.z, 0) / maxZ) * 100));
    lines.push(`<div class="rc-item"><div class="top"><span>${esc(c.label)}</span><span class="z ${zc}">${c.z >= 0 ? "+" : ""}${c.z.toFixed(1)}</span></div>
      <div class="sub"><span>${esc(fmtVal(c))}${c.unit ? " " + esc(c.unit) : ""}</span><span>${esc(fmtExp(c))}</span></div>
      <div class="rc-bar"><i class="${zc}" style="width:${w}%"></i></div></div>`);
  }
  lines.push(`</div>`);
  if (items.length > 9) {
    lines.push(`<div class="rc-more"><button type="button" id="rc-toggle">${showAll ? "show fewer" : "show all fields"}</button></div>`);
  }
  lines.push(`<div class="rc-sep"></div>`);
  lines.push(`<div class="rc-foot">surprise = how much worse the VAE predicted a field than it does for normal ${esc(r.source === "generic" ? "traffic" : r.source + " traffic")}, in standard deviations; "exp" is its expectation.</div>`);
  box.innerHTML = lines.join("");
  const tog = document.getElementById("rc-toggle");
  if (tog) tog.addEventListener("click", () => { showAll = !showAll; renderReceipt(r, label); });
}

// ---- latent map -----------------------------------------------------------------

async function loadLatentBackground() {
  const run = state.meta?.active_run?.name;
  latentBackground = null;
  if (!run) { drawLatent(); return; }
  try {
    latentBackground = await get(`/api/runs/${encodeURIComponent(run)}/latent`);
    document.getElementById("latent-note").textContent = `${latentBackground.y.length.toLocaleString()} validation rows, PCA of posterior means`;
  } catch (e) {
    document.getElementById("latent-note").textContent = "no latent map for this run";
  }
  drawLatent();
}

function drawLatent(point = null) {
  const canvas = document.getElementById("latent-chart");
  if (!window.Chart) { canvas.replaceWith(el("p", { class: "dim" }, "charts need the Chart.js CDN")); return; }
  const datasets = [];
  if (latentBackground) {
    const bg = latentBackground;
    const legit = [], fraud = [];
    for (let i = 0; i < bg.y.length; i++) {
      (bg.y[i] ? fraud : legit).push({ x: bg.xy[i][0], y: bg.xy[i][1] });
    }
    datasets.push({ label: "legitimate", data: legit, backgroundColor: "rgba(86,104,121,0.22)", pointRadius: 1.6, pointHoverRadius: 1.6 });
    datasets.push({ label: "fraud", data: fraud, backgroundColor: "rgba(200,16,46,0.45)", pointRadius: 1.8, pointHoverRadius: 1.8 });
  }
  if (point) {
    datasets.push({ label: "this transaction", data: [{ x: point[0], y: point[1] }], backgroundColor: "#fff", borderColor: palette.ink, borderWidth: 2.5, pointRadius: 9, pointStyle: "crossRot", pointHoverRadius: 9 });
  }
  if (latentChart) latentChart.destroy();
  latentChart = new Chart(canvas, {
    type: "scatter",
    data: { datasets },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, usePointStyle: true } }, tooltip: { enabled: false } },
      scales: { x: { title: { display: true, text: "latent PC 1" }, grid: { color: "#eef1f4" } }, y: { title: { display: true, text: "latent PC 2" }, grid: { color: "#eef1f4" } } },
    },
  });
}

function plotLatent(r) {
  if (r.latent_xy) drawLatent(r.latent_xy);
}

// ---- batch ------------------------------------------------------------------------

async function batchScore(ev) {
  ev.preventDefault();
  const form = ev.target;
  const status = document.getElementById("batch-status");
  const fd = new FormData(form);
  const src = document.getElementById("source-select").value;
  const fpr = document.getElementById("fpr-select").value;
  const q = new URLSearchParams({ fpr_target: fpr });
  if (src) q.set("source", src);
  status.textContent = "scoring file"; status.classList.add("spin");
  try {
    const res = await fetch(`/api/score/batch?${q}`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const summary = JSON.parse(res.headers.get("X-Summary") || "{}");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "scored.csv"; a.click();
    let msg = `${fmt.int(summary.rows)} rows, ${fmt.int(summary.flagged)} flagged`;
    if (summary.auroc != null) msg += `; AUROC ${fmt.auc(summary.auroc)}, recall ${fmt.pct(summary.recall, 1)}, precision ${fmt.pct(summary.precision, 1)}`;
    status.textContent = msg + " (downloaded scored.csv)";
  } catch (e) {
    status.textContent = e.message;
  } finally { status.classList.remove("spin"); }
}
