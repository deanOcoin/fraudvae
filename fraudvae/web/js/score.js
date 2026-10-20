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

let models = [];   // the generation timeline, for the picker and the comparison strip

export async function initScore(appState) {
  state = appState;
  schema = state.schema || (await get("/api/schema"));
  buildSourceSelect();
  await buildRunSelect();
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
    lede.textContent = "No trained model is loaded. Train one in Model & training (a few minutes on the smoke config) and come back.";
  } else {
    const dim = m.params ? `${(m.params / 1e6).toFixed(1)}M-parameter` : "";
    const hybrid = m.has_booster
      ? ` A ${m.booster_trees ? m.booster_trees + "-tree" : ""} gradient-boosted model trained on the labelled fraud adds what known fraud looks like; the two are blended per bank into the p(fraud) the verdict uses.`
      : " (No booster in this run: the verdict is the novelty score alone.)";
    lede.textContent = `The ${dim} VAE reconstructs the transaction from a low-dimensional summary of normal traffic across ${m.sources.length} datasets; what it cannot reconstruct is novelty.${hybrid}`;
  }
}

async function buildRunSelect() {
  const sel = document.getElementById("run-select");
  if (!sel) return;
  try { models = (await get("/api/models")).models; } catch (e) { models = []; }
  const active = state.meta?.active_run?.name;
  sel.innerHTML = "";
  const shown = models.filter((m) => m.milestone || m.name === active);
  const best = shown.reduce((a, b) => (b.mean_test_auroc ?? -1) > (a?.mean_test_auroc ?? -1) ? b : a, null);
  for (const m of [...shown].reverse()) {
    const tag = m.name === best?.name ? "  — newest, best" : (m.schema_version < 5 ? `  — schema v${m.schema_version}` : "");
    sel.append(el("option", { value: m.name }, `${m.version} · ${m.label}${tag}`));
  }
  let want = null;
  try { want = localStorage.getItem("fraudvae.run"); localStorage.removeItem("fraudvae.run"); } catch (_) {}
  sel.value = (want && shown.some((m) => m.name === want)) ? want : (active && shown.some((m) => m.name === active) ? active : (best?.name || sel.value));
  sel.addEventListener("change", () => { if (lastResult) score(lastResult._label); });
  document.getElementById("compare-all")?.addEventListener("change", () => { if (lastResult) score(lastResult._label); });
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

// v5: the loaded example's raw source columns (the "wide" block). the form has no fields for
// them - there are 431 on ieee - so they ride along invisibly until the form is cleared
let loadedWide = null;
let loadedEntities = null;   // v5: the example row's entity codes, for the graph module

function readForm() {
  const form = document.getElementById("score-form");
  const fd = new FormData(form);
  const features = {};
  for (const [k, v] of fd.entries()) {
    if (k === "source" || k === "fpr_target" || k === "label_delay_days" || k === "run" || k === "compare_all") continue;
    if (v === "" || v == null) continue;
    features[k] = v;
  }
  const req = { source: fd.get("source") || null, fpr_target: fd.get("fpr_target"), features, wide: loadedWide, entities: loadedEntities,
                label_delay_days: Number(fd.get("label_delay_days") || 7) };
  // which generation scores it, and whether every generation should
  const runSel = document.getElementById("run-select");
  if (runSel && runSel.value) req.run = runSel.value;
  if (document.getElementById("compare-all")?.checked) req.compare = models.filter((m) => m.milestone).map((m) => m.name);
  // v5: history as json in the textarea (a list of {features, ts|secs_before}), and/or a customer id
  const histEl = document.getElementById("history-json");
  if (histEl && histEl.value.trim()) {
    try { req.history = JSON.parse(histEl.value); } catch (e) { throw new Error("history is not valid json: " + e.message); }
  }
  const cid = document.getElementById("customer-id");
  if (cid && cid.value.trim()) req.customer_id = cid.value.trim();
  // v5: the device telemetry payload, mapped server side onto the device block
  const devEl = document.getElementById("device-json");
  if (devEl && devEl.value.trim()) {
    try { req.device = JSON.parse(devEl.value); } catch (e) { throw new Error("device payload is not valid json: " + e.message); }
  }
  // v5: free text for the text heads
  const txtEl = document.getElementById("text-json");
  if (txtEl && txtEl.value.trim()) {
    try { req.text = JSON.parse(txtEl.value); } catch (e) { throw new Error("text payload is not valid json: " + e.message); }
  }
  // v5: the session's raw interaction events for the behaviour verifiers
  const sesEl = document.getElementById("session-json");
  if (sesEl && sesEl.value.trim()) {
    try { req.session = JSON.parse(sesEl.value); } catch (e) { throw new Error("session payload is not valid json: " + e.message); }
  }
  return req;
}

// a form reset must not forget which generation the visitor picked, the compare tick, the
// alert budget or the label latency: those are settings, not the transaction
function resetKeepingSettings(form) {
  const keep = ["run-select", "compare-all", "fpr-select", "delay-select"].map((id) => {
    const e = document.getElementById(id);
    return e ? [e, e.type === "checkbox" ? e.checked : e.value] : null;
  }).filter(Boolean);
  form.reset();
  for (const [e, v] of keep) { if (e.type === "checkbox") e.checked = v; else e.value = v; }
}

function fillForm(features, source) {
  const form = document.getElementById("score-form");
  resetKeepingSettings(form);
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
    resetKeepingSettings(form); markUnused(); showSourceHint();
    loadedWide = null; loadedEntities = null;
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
      loadedWide = ex.wide || null;
      loadedEntities = ex.entities && Object.keys(ex.entities).length ? ex.entities : null;
      const scen = ex.fraud_scenario ? `, scenario ${ex.fraud_scenario}` : "";
      const wideNote = (ex.wide ? `, + ${Object.keys(ex.wide).length} raw source columns` : "")
        + (loadedEntities ? `, entities: ${Object.keys(loadedEntities).join("/")}` : "");
      document.getElementById("example-note").textContent = `${ex.txn_id} (${ex.split} split, labelled ${ex.is_fraud ? "fraud" : "legitimate"}${scen}${wideNote})`;
      await score(ex.is_fraud);
    } catch (e) {
      document.getElementById("example-note").textContent = e.message;
    }
  }));
  document.getElementById("fpr-select").addEventListener("change", () => { if (lastResult) score(lastResult._label); });
  document.getElementById("delay-select").addEventListener("change", () => { if (lastResult) score(lastResult._label); });
  document.getElementById("batch-form").addEventListener("submit", batchScore);
}

async function score(label = null) {
  const btn = document.getElementById("score-btn");
  const status = document.getElementById("score-status");
  let req;
  try { req = readForm(); } catch (e) { status.textContent = e.message; return; }
  if (!Object.keys(req.features).length) { status.textContent = "fill in at least one field"; return; }
  btn.disabled = true; status.textContent = "scoring"; status.classList.add("spin");
  try {
    const r = await post("/api/score", req);
    r._label = label;
    lastResult = r;
    showAll = false;
    renderReceipt(r, label);
    plotLatent(r);
    status.textContent = `${r.observed_features} fields observed, ${r.n_samples} posterior samples${r.supervised ? ", booster + blend" : ""}${r.sequence ? `, ${r.sequence.n_history} past txns` : ""}`;
  } catch (e) {
    status.textContent = e.message;
  } finally {
    btn.disabled = false; status.classList.remove("spin");
  }
}

function zClass(z) { return z >= 3 ? "hot" : z >= 1.5 ? "warm" : z < 0.5 ? "cold" : ""; }

// right or wrong, in words: a labelled row scored against the model's flag
function verdictCall(label, flag) {
  if (flag == null) return null;
  if (label && flag) return { ok: true, cls: "call ok", text: "caught" };
  if (label && !flag) return { ok: false, cls: "call bad", text: "missed" };
  if (!label && flag) return { ok: false, cls: "call bad", text: "false alarm" };
  return { ok: true, cls: "call ok", text: "correctly passed" };
}

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

function pctText(pct, fallback) {
  if (pct == null) return fallback;
  return (pct >= 99.99 ? ">99.99" : pct.toFixed(pct > 99 ? 2 : 1)) + "th";
}
function opLabel(op) {
  const v = op ? op.replace("fpr_", "") : "";
  return v ? (parseFloat(v) * 100).toFixed(v === "0.001" || v === "0.005" ? 1 : 0) + "%" : "";
}
function pushVal(c) {
  if (!c.is_observed) return "unobserved";
  if (c.observed == null) return "";
  if (c.kind === "categorical") {
    if (c.feature === "dow") return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][+c.observed] || String(c.observed);
    if (c.feature === "hour") return String(c.observed).padStart(2, "0") + ":00";
    return String(c.observed).replace(/_/g, " ");
  }
  const v = c.observed;
  return (Math.abs(v) >= 1000 ? fmt.int(v) : fmt.num(v, Math.abs(v) < 10 ? 2 : 1)) + (c.unit ? " " + c.unit : "");
}

function renderReceipt(r, label) {
  const box = document.getElementById("receipt");
  box.classList.remove("empty");
  const bl = r.blend, sup = r.supervised, nov = r.novelty || {};
  const flagged = r.flag;
  const verdictClass = r.flag == null ? "na" : flagged ? "flag" : "ok";
  const verdictText = r.flag == null ? "NO CALIBRATION" : flagged ? "FLAG FOR REVIEW" : "LOOKS NORMAL";
  const opPct = opLabel(r.operating_point);
  const items = r.contributions || [];
  const shown = showAll ? items : items.slice(0, 9);
  const maxZ = Math.max(3, ...items.map((c) => c.z));
  const ctxName = r.calibration === "generic" ? "generic context" : `${r.calibration} context`;
  const pct = r.legit_percentile;

  const lines = [];
  lines.push(`<div class="rc-head"><div class="rc-title">FRAUDVAE SCORE</div><div class="dim">${esc(new Date().toISOString().replace("T", " ").slice(0, 19))}  ${esc(ctxName)}</div></div>`);
  // ---- headline: the blend ----
  if (bl) {
    lines.push(`<div class="rc-verdict ${verdictClass}"><span class="big">${(100 * bl.prob).toFixed(bl.prob < 0.1 ? 2 : 1)}%</span>p(fraud)  ·  ${pctText(pct, "")} percentile of legit<br>${verdictText}</div>`);
    if (bl.threshold != null) {
      lines.push(`<div class="rc-line"><span class="k">alert budget</span><span>${esc(opPct)} FPR</span></div>`);
      lines.push(`<div class="rc-line"><span class="k">flag above p(fraud)</span><span>${(100 * bl.threshold).toFixed(bl.threshold < 0.1 ? 2 : 1)}%</span></div>`);
      if (bl.expected_recall != null) lines.push(`<div class="rc-line"><span class="k">recall at budget</span><span>${fmt.pct(bl.expected_recall, 1)}</span></div>`);
      if (bl.expected_precision != null) lines.push(`<div class="rc-line"><span class="k">precision at budget</span><span>${fmt.pct(bl.expected_precision, 1)}</span></div>`);
    }
  } else {
    lines.push(`<div class="rc-verdict ${verdictClass}"><span class="big">${pct == null ? fmt.num(nov.score, 2) : pctText(pct, "")}</span>${pct == null ? "raw score" : "percentile of legit traffic"}<br>${verdictText}</div>`);
  }
  if (label != null) {
    // an example row carries its label: say in words whether this verdict was right
    const call = verdictCall(label, flagged);
    lines.push(`<div class="rc-line"><span class="k">ground truth</span><span>${label ? "FRAUD" : "legitimate"}${call ? ` <span class="${call.cls}">· ${esc(call.text)}</span>` : ""}</span></div>`);
  }
  if (r.model) {
    const tl = models.find((m) => m.name === r.model.run);
    lines.push(`<div class="rc-line"><span class="k">scored by</span><span>${esc(tl ? `${tl.version} · ${tl.label}` : r.model.run)}</span></div>`);
    if (r.model.note) lines.push(`<div class="rc-line"><span class="k"></span><span class="dim">${esc(r.model.note)}${r.model.bank_known ? "" : "; this bank is unknown to it, scored as generic"}</span></div>`);
  }
  // ---- every generation on the same transaction ----
  if (r.compare && r.compare.length > 1) {
    lines.push(`<div class="rc-compare">`);
    lines.push(`<div class="rc-sub">GENERATIONS  <span class="dim">the same transaction on each</span></div>`);
    lines.push(`<div class="rc-line head"><span class="k">model</span><span>p(fraud) or novelty pct · verdict</span></div>`);
    const byOrder = [...r.compare].sort((a, b) => (models.find((m) => m.name === a.run)?.order ?? 99) - (models.find((m) => m.name === b.run)?.order ?? 99));
    for (const c of byOrder) {
      const tl = models.find((m) => m.name === c.run);
      const name = tl ? `${tl.version} ${tl.label}` : c.run;
      const val = c.p_fraud != null ? `${(100 * c.p_fraud).toFixed(c.p_fraud < 0.1 ? 2 : 1)}%` : (c.novelty_percentile != null ? `${pctText(c.novelty_percentile, "")} pct` : "–");
      const fl = c.flag == null ? (c.novelty_flag == null ? "" : (c.novelty_flag ? "  FLAG" : "  ok")) : (c.flag ? "  FLAG" : "  ok");
      const hot = (c.flag ?? c.novelty_flag) ? "z hot" : "";
      const extra = c.bank_known === false ? " (bank unknown)" : "";
      const call = label != null ? verdictCall(label, c.flag ?? c.novelty_flag) : null;
      const mark = call ? ` <span class="${call.cls}" title="${esc(call.text)}">${call.ok ? "✓" : "✗"}</span>` : "";
      lines.push(`<div class="rc-line ${c.run === r.model?.run ? "me" : ""}"><span>${esc(name)}${esc(extra)}</span><span class="${hot}">${val}${fl}${mark}</span></div>`);
    }
    lines.push(`</div>`);
  }

  // ---- supervised evidence ----
  if (sup) {
    lines.push(`<div class="rc-sep"></div>`);
    lines.push(`<div class="rc-sub">SUPERVISED  <span class="dim">booster, trained on labelled fraud</span></div>`);
    const sf = sup.flag == null ? "" : (sup.flag ? "  FLAG" : "  ok");
    lines.push(`<div class="rc-line"><span class="k">p(fraud) on its own</span><span class="${sup.flag ? "z hot" : ""}">${(100 * sup.prob).toFixed(sup.prob < 0.1 ? 2 : 1)}%${sf}</span></div>`);
    if (sup.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">percentile of legit</span><span>${pctText(sup.legit_percentile, "")}</span></div>`);
    if (r.labels) {
      const lb = r.labels;
      const txt = lb.fast_block === "blanked" ? `blanked (labels take ${lb.latency_days}d)` : lb.fast_block === "used" ? "one-week history used" : "not in this request";
      lines.push(`<div class="rc-line"><span class="k">fast fraud history</span><span>${esc(txt)}</span></div>`);
    }
    const pushes = (sup.pushes || []).slice(0, showAll ? 12 : 6);
    const maxC = Math.max(0.5, ...pushes.map((c) => Math.abs(c.contribution)));
    lines.push(`<div class="rc-line"><span class="k">feature</span><span class="k">push (log-odds)</span></div>`);
    lines.push(`<div class="rc-items">`);
    for (const c of pushes) {
      const up = c.contribution > 0;
      const w = Math.max(2, Math.min(100, (Math.abs(c.contribution) / maxC) * 100));
      lines.push(`<div class="rc-item"><div class="top"><span>${esc(c.label)}</span><span class="z ${up ? (Math.abs(c.contribution) > 1 ? "hot" : "warm") : "cold"}">${up ? "+" : "−"}${Math.abs(c.contribution).toFixed(2)}</span></div>
        <div class="sub"><span>${esc(pushVal(c))}</span><span>${up ? "toward fraud" : "toward legit"}</span></div>
        <div class="rc-bar"><i class="${up ? (Math.abs(c.contribution) > 1 ? "hot" : "warm") : "calm"}" style="width:${w}%"></i></div></div>`);
    }
    lines.push(`</div>`);
  }

  // ---- v5 sequence evidence: the customer's history through the sequence head ----
  const sq = r.sequence;
  if (sq) {
    lines.push(`<div class="rc-sep"></div>`);
    lines.push(`<div class="rc-sub">SEQUENCE  <span class="dim">history transformer, ${sq.n_history} of ${sq.window} past txns${sq.history_from ? ` (${sq.history_from})` : ""}</span></div>`);
    if (sq.prob != null) {
      const qf = sq.flag == null ? "" : (sq.flag ? "  FLAG" : "  ok");
      lines.push(`<div class="rc-line"><span class="k">p(fraud) from history</span><span class="${sq.flag ? "z hot" : ""}">${(100 * sq.prob).toFixed(sq.prob < 0.1 ? 2 : 1)}%${qf}</span></div>`);
      if (sq.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">percentile of legit</span><span>${pctText(sq.legit_percentile, "")}</span></div>`);
    }
    if (!sq.n_history) lines.push(`<div class="rc-line dim"><span>no history given: scored as a first transaction. add past transactions below, or a customer id so the server remembers them</span></div>`);
  }

  // ---- v5 text: free text through the text heads, fused as likelihood ratios ----
  if (r.text) {
    lines.push(`<div class="rc-sep"></div>`);
    const tf = r.text.fields || {};
    const keys = Object.keys(tf);
    lines.push(`<div class="rc-sub">TEXT  <span class="dim">${keys.length ? `${keys.length} field${keys.length === 1 ? "" : "s"} read` : "no text sent"}${r.fusion ? ", fused into the verdict" : ""}</span></div>`);
    for (const k of keys) {
      const t = tf[k];
      lines.push(`<div class="rc-line"><span class="k">${esc(k)} → ${esc(t.label)}</span><span class="${t.llr > 1 ? "z hot" : ""}">p ${(100 * t.prob).toFixed(1)}%, LR ${t.llr >= 0 ? "+" : ""}${t.llr.toFixed(2)}</span></div>`);
    }
    if (!keys.length && r.text.sent && r.text.sent.length) lines.push(`<div class="rc-line dim"><span>sent ${r.text.sent.join(", ")}: no head reads those keys (url, description, memo, message, profile, bio)</span></div>`);
  }
  if (r.fusion) {
    lines.push(`<div class="rc-sep"></div>`);
    lines.push(`<div class="rc-line"><span class="k">blend ⊕ ${Object.keys(r.fusion.terms).join(" ⊕ ")}</span><span class="${r.flag ? "z hot" : ""}">${(100 * r.fusion.prob).toFixed(r.fusion.prob < 0.1 ? 2 : 1)}% (blend alone ${(100 * r.blend.prob).toFixed(r.blend.prob < 0.1 ? 2 : 1)}%)${r.flag == null ? "" : (r.flag ? "  FLAG" : "  ok")}</span></div>`);
  }

  // ---- v5 behaviour: the session against the customer's enrolled profile ----
  if (r.behaviour) {
    lines.push(`<div class="rc-sep"></div>`);
    const bf = r.behaviour.fields || {};
    const mods = Object.keys(bf);
    lines.push(`<div class="rc-sub">BEHAVIOUR  <span class="dim">${mods.length ? `${mods.join(", ")} session` : "no session sent"}${r.behaviour.enrolled_sessions ? `, profile from ${r.behaviour.enrolled_sessions} earlier session${r.behaviour.enrolled_sessions === 1 ? "" : "s"}` : ""}</span></div>`);
    for (const m of mods) {
      const v = bf[m];
      if (v.llr == null) lines.push(`<div class="rc-line dim"><span>${esc(v.label)}: ${v.windows} window${v.windows === 1 ? "" : "s"} read, no enrolment for this customer yet${r.behaviour.enrolled_now ? " — enrolled now" : ""}</span></div>`);
      else lines.push(`<div class="rc-line"><span class="k">${esc(v.label)}</span><span class="${v.llr < -1 ? "z hot" : ""}">cosine ${v.cosine.toFixed(2)}, LR ${v.llr >= 0 ? "+" : ""}${v.llr.toFixed(2)}${v.llr < 0 ? " (not like the account holder)" : ""}</span></div>`);
    }
    if (!mods.length && r.behaviour.sent && r.behaviour.sent.length) lines.push(`<div class="rc-line dim"><span>sent ${r.behaviour.sent.join(", ")}: too few events for a window, or no verifier for that modality</span></div>`);
    if (!r.behaviour.sent || !r.behaviour.sent.length) lines.push(`<div class="rc-line dim"><span>add session events below and a customer id; the second session is compared with the first</span></div>`);
  }

  // ---- v5 field heads: the device block (and whatever lands next) through its own head ----
  for (const name of ["local", "device", "graph"]) {
    const hb = r[name];
    if (!hb) continue;
    lines.push(`<div class="rc-sep"></div>`);
    lines.push(`<div class="rc-sub">${name.toUpperCase()}  <span class="dim">${hb.label || name}${name === "local" ? "" : " head"}${hb.present && hb.fields.length ? `, ${hb.fields.length} field${hb.fields.length === 1 ? "" : "s"}` : ""}</span></div>`);
    if (hb.present && hb.prob != null) {
      const hf = hb.flag == null ? "" : (hb.flag ? "  FLAG" : "  ok");
      lines.push(`<div class="rc-line"><span class="k">p(fraud) from ${name}</span><span class="${hb.flag ? "z hot" : ""}">${(100 * hb.prob).toFixed(hb.prob < 0.1 ? 2 : 1)}%${hf}</span></div>`);
      if (hb.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">percentile of legit</span><span>${pctText(hb.legit_percentile, "")}</span></div>`);
      if (hb.fields.length) lines.push(`<div class="rc-line dim"><span>${hb.fields.join(", ")}</span></div>`);
    } else if (name === "local") {
      lines.push(`<div class="rc-line dim"><span>this bank has no specialist booster (too little labelled fraud, or the generic context): the shared booster carries it</span></div>`);
    } else if (name === "graph") {
      lines.push(`<div class="rc-line dim"><span>${hb.available === false ? "this bank has no entity graph" : (hb.sent && hb.sent.length ? `none of the entities sent (${hb.sent.join(", ")}) is known to the graph` : "no entity ids sent (load an example row, or pass entities in the API)")}: signal absent, the blend goes without it</span></div>`);
    } else {
      lines.push(`<div class="rc-line dim"><span>no ${name} fields on this transaction: signal absent, the blend goes without it</span></div>`);
    }
  }

  // ---- novelty evidence ----
  lines.push(`<div class="rc-sep"></div>`);
  lines.push(`<div class="rc-sub">NOVELTY  <span class="dim">VAE, never saw a fraud label${sq ? ", conditioned on the history" : ""}</span></div>`);
  const nf = nov.flag == null ? "" : (nov.flag ? "  FLAG" : "  ok");
  if (nov.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">percentile of legit</span><span class="${nov.flag ? "z hot" : ""}">${pctText(nov.legit_percentile, "")}${nf}</span></div>`);
  lines.push(`<div class="rc-line"><span class="k">score (${esc(nov.score_type)})</span><span>${fmt.num(nov.score, 3)}</span></div>`);
  if (nov.score_type === "norm") lines.push(`<div class="rc-line"><span class="k">raw -ELBO</span><span>${fmt.num(nov.score_raw, 2)}</span></div>`);
  if (nov.z_score != null) lines.push(`<div class="rc-line"><span class="k">z vs legit</span><span>${nov.z_score >= 0 ? "+" : ""}${fmt.num(nov.z_score, 2)}</span></div>`);
  lines.push(`<div class="rc-line"><span class="k">KL(q‖p)</span><span>${fmt.num(nov.kl, 2)}</span></div>`);
  if (nov.threshold != null) lines.push(`<div class="rc-line"><span class="k">threshold at budget</span><span>${fmt.num(nov.threshold, 3)}</span></div>`);
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
  if (items.length > 9 || (sup && (sup.pushes || []).length > 6)) {
    lines.push(`<div class="rc-more"><button type="button" id="rc-toggle">${showAll ? "show fewer" : "show everything"}</button></div>`);
  }
  lines.push(`<div class="rc-sep"></div>`);
  lines.push(`<div class="rc-foot">${bl ? "the verdict is the blend: booster + novelty, weighted per bank on labelled validation rows.<br>" : "the verdict is the novelty score alone (no blender in this run).<br>"}surprise = how much worse the VAE predicted a field than it does for normal ${esc(r.source === "generic" ? "traffic" : r.source + " traffic")}, in standard deviations; "exp" is its expectation.${sup ? "<br>pushes are the booster's SHAP contributions in log-odds; unobserved fields can still push (missingness is information to a tree)." : ""}</div>`);
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
  const q = new URLSearchParams({ fpr_target: fpr, label_delay_days: document.getElementById("delay-select").value });
  if (src) q.set("source", src);
  status.textContent = "scoring file"; status.classList.add("spin");
  try {
    const res = await fetch(`/api/score/batch?${q}`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const summary = JSON.parse(res.headers.get("X-Summary") || "{}");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "scored.csv"; a.click();
    let msg = `${fmt.int(summary.rows)} rows, ${fmt.int(summary.flagged)} flagged (${summary.verdict_from})`;
    if (summary.auroc != null) msg += `; AUROC ${fmt.auc(summary.auroc)}${summary.booster_auroc != null ? ` (booster ${fmt.auc(summary.booster_auroc)}, novelty ${fmt.auc(summary.novelty_auroc)})` : ""}, recall ${fmt.pct(summary.recall, 1)}, precision ${fmt.pct(summary.precision, 1)}`;
    status.textContent = msg + " (downloaded scored.csv)";
  } catch (e) {
    status.textContent = e.message;
  } finally { status.classList.remove("spin"); }
}
