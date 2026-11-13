// score tab: the form, the receipt, the latent map
import { get, post, fmt, el, esc, palette, glide } from "./api.js";
import { getSelectedRun, setSelectedRun, selectedModel, refreshModels, rank } from "./app.js";

let schema, state;
let latentChart = null;
let latentBackground = null;   // {xy, y, src} from the active run
let latentPoint = null;        // where the shown payment landed on it
let latentZoom = false;        // the map framed around that payment, not the whole sample
let lastResult = null;
let showAll = false;
// the label of the loaded example, while the form has not been edited since: the receipt
// then says whether the verdict was right
let pendingLabel = null;
let lastRequest = null;     // what the shown verdict was computed from ("copy as API call")
let loadedExample = null;   // the labelled row in the form, if one is
let mystery = false;        // its label stays hidden until it is scored
let userSource = false;     // the visitor picked the bank: examples then come from it
// this visit's record on labelled rows, each row counted once however often it is rescored
const tally = { right: 0, total: 0, counted: new Set() };
// the stylesheet's breakpoint: at or under it the receipt sits below the form, not beside it
const stacked = () => !!(window.matchMedia && window.matchMedia("(max-width: 1100px)").matches);

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
  restoreSettings();
  wireForm();
  await loadLatentBackground();
  loadTestRates();
  window.addEventListener("metachanged", () => { loadLatentBackground(); updateLede(); });
  // the version chosen elsewhere (the model tab on a one-host workbench) reaches the picker here
  window.addEventListener("runselected", (e) => {
    const sel = document.getElementById("run-select");
    if (sel && e.detail && sel.value !== e.detail && [...sel.options].some((o) => o.value === e.detail)) { sel.value = e.detail; onRunChanged(); }
  });
  updateLede();
}

// the alert budget and the label timing a visitor last chose, as the model version already is:
// this browser only, and a page that cannot store anything simply starts from the defaults
const KEEP = { "fpr-select": "fraudvae.budget", "delay-select": "fraudvae.delay" };
function restoreSettings() {
  for (const [id, key] of Object.entries(KEEP)) {
    const sel = document.getElementById(id);
    let v = null;
    try { v = localStorage.getItem(key); } catch (_) { /* storage blocked */ }
    if (sel && v && [...sel.options].some((o) => o.value === v)) sel.value = v;
    sel?.addEventListener("change", () => { try { localStorage.setItem(key, sel.value); } catch (_) { /* storage blocked */ } });
  }
}

// the introduction is said once in plain words; the line under it names the version that will
// score (not the server's default) and what it is made of, for anyone who wants that
let ledeText = null;
function updateLede() {
  const m = selectedModel();
  const lede = document.getElementById("score-lede");
  const line = document.getElementById("score-model-line");
  if (ledeText == null) ledeText = lede.textContent;
  if (!m) {
    lede.textContent = state.meta?.active_run ? ledeText : "No trained model is loaded. Train one in Model & training (a few minutes on the smoke config) and come back.";
    if (line) line.textContent = "";
    return;
  }
  lede.textContent = ledeText;
  const n = (m.sources || []).length;
  const vae = `${m.params ? `a ${(m.params / 1e6).toFixed(1)}-million-parameter` : "a"} VAE that learnt honest payments from ${n} dataset${n === 1 ? "" : "s"}`;
  // v6 on: the trees also read a pretrained model of payment histories; v7: the blend leaves the VAE out
  const enc = m.encoder ? ` and read a pretrained model of payment histories${m.encoder.sources ? ` for ${m.encoder.sources.join(" and ")}` : ""}` : "";
  const blend = m.blend_without_vae ? "and a per-bank blend of the fraud-trained models that the verdict is read from (the VAE explains a payment, it does not vote)"
    : "and a per-bank blend of the two that the verdict is read from";
  const parts = m.has_booster
    ? `${vae}, ${m.booster_trees ? m.booster_trees.toLocaleString("en-US") + " decision trees" : "boosted decision trees"} that learnt the labelled fraud${enc}, ${blend}`
    : `${vae}, on its own: this version has no fraud-trained model, so the verdict is the novelty score, and it knows an older, smaller set of fields`;
  if (line) line.textContent = `Judging with ${m.version && m.label ? `${m.version} · ${m.label}` : m.name}: ${parts}.`;
}

async function buildRunSelect() {
  const sel = document.getElementById("run-select");
  if (!sel) return;
  models = state.models?.length ? state.models : await refreshModels();
  const active = state.meta?.active_run?.name;
  sel.innerHTML = "";
  const shown = models.filter((m) => m.milestone || m.name === active);
  // newest by the timeline's order; best on the nine shared datasets (the model tab's ranking)
  const newest = shown[shown.length - 1];
  const best = shown.reduce((a, b) => (a == null || rank(b) > rank(a) ? b : a), null);
  for (const m of [...shown].reverse()) {
    const tag = m === newest ? (m === best ? "  — newest, best" : "  — newest") : m === best ? "  — best" : (m.schema_version < 5 ? `  — older field set (v${m.schema_version})` : "");
    sel.append(el("option", { value: m.name }, `${m.version} · ${m.label}${tag}`));
  }
  // the version selected on the model tab (kept across tabs and hosts), else the default
  const want = getSelectedRun();
  sel.value = (want && shown.some((m) => m.name === want)) ? want : (active && shown.some((m) => m.name === active) ? active : (best?.name || sel.value));
  if (sel.value !== want) setSelectedRun(sel.value);
  sel.addEventListener("change", () => { setSelectedRun(sel.value); onRunChanged(); });
  document.getElementById("compare-all")?.addEventListener("change", () => markDirty("settings changed"));
}

// a different version will score next: say so, show its map, do not score until asked
function onRunChanged() {
  updateLede();
  loadLatentBackground();
  markDirty("the model version changed");
}

// a verdict belongs to the inputs it was computed from: once they change, it is stale until
// the button is pressed again (nothing scores on its own)
let dirtyReason = null;
function markDirty(reason) {
  if (!lastResult) return;
  dirtyReason = reason || "inputs changed";
  document.getElementById("score-btn")?.classList.add("attention");
  const status = document.getElementById("score-status");
  if (status) status.textContent = `${dirtyReason} — the verdict shown is for the previous inputs`;
  const r = document.getElementById("receipt");
  let bar = document.getElementById("rc-stale");
  if (!bar) { bar = el("div", { id: "rc-stale", class: "rc-stale" }); r.prepend(bar); }
  bar.textContent = `${dirtyReason} — press Score transaction for a new verdict`;
}
function clearDirty() {
  dirtyReason = null;
  document.getElementById("score-btn")?.classList.remove("attention");
  document.getElementById("rc-stale")?.remove();
}

function buildSourceSelect() {
  const sel = document.getElementById("source-select");
  sel.innerHTML = "";
  sel.append(el("option", { value: "" }, "Generic: a bank the model has never seen"));
  for (const s of schema.sources) {
    if (!s.built) continue;
    sel.append(el("option", { value: s.key }, `${s.key}: ${s.name}`));
  }
  // a bank picked by hand narrows the examples to it; one set by loading an example does not
  sel.addEventListener("change", () => { userSource = !!sel.value; markUnused(); showSourceHint(); });
  showSourceHint();
}

function showSourceHint() {
  const key = document.getElementById("source-select").value;
  const s = schema.sources.find((x) => x.key === key);
  document.getElementById("source-hint").textContent = s ? s.context : "Each training dataset counts as a bank with its own habits. Generic judges the payment as one from a bank the model has never seen, using only what all the datasets share.";
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
    const n = nums.length + cats.length;
    // the transaction itself and the customer's history stay open; the rest folds, with a
    // count of what is filled in so a loaded example is not hidden from view
    if (g === "core" || g === "velocity") {
      host.append(el("div", { class: "fgroup", "data-group": g },
        el("div", { class: "fgroup-head" }, el("h3", {}, schema.groups[g] || g), el("span", { class: "dim" }, GROUP_NOTE[g] || "")),
        grid));
    } else {
      host.append(el("details", { class: "fgroup fold", "data-group": g },
        el("summary", {}, el("h3", {}, schema.groups[g] || g), el("span", { class: "dim" }, GROUP_NOTE[g] || ""), el("span", { class: "fcount", "data-total": n }, `0 of ${n} filled`)),
        grid));
    }
  }
  markUnused();
  updateFoldCounts();
}

function updateFoldCounts() {
  document.querySelectorAll("#field-groups details.fold").forEach((d) => {
    const total = d.querySelectorAll("label[data-feature]").length;
    const filled = [...d.querySelectorAll("label[data-feature] input, label[data-feature] select")].filter((e) => e.value !== "").length;
    const c = d.querySelector(".fcount");
    if (c) { c.textContent = `${filled} of ${total} filled`; c.classList.toggle("some", filled > 0); }
  });
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
  updateFoldCounts();
}

function wireForm() {
  const form = document.getElementById("score-form");
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await score();
  });
  // typing or picking anything makes the current verdict stale; nothing scores on its own
  const SETTINGS = new Set(["run-select", "compare-all", "fpr-select", "delay-select", "source-select"]);
  const touched = (e) => {
    if (e.target.id === "run-select") return;
    if (!SETTINGS.has(e.target.id) && pendingLabel != null) {
      // an edited example is no longer the labelled row (nor the mystery one)
      pendingLabel = null; mystery = false;
      const note = document.getElementById("example-note");
      if (note && loadedExample) note.innerHTML = `<b>${esc(loadedExample.txn_id)}</b>, edited: now a payment of your own, so the verdict will not be checked against a label.`;
      loadedExample = null;
    }
    updateFoldCounts();
    markDirty(SETTINGS.has(e.target.id) ? "settings changed" : "inputs changed");
  };
  form.addEventListener("input", touched);
  form.addEventListener("change", touched);
  // Ctrl+Enter (⌘+Enter on a Mac) scores from any field, the way a message is sent
  form.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" || !(ev.metaKey || ev.ctrlKey)) return;
    ev.preventDefault();
    if (!document.getElementById("score-btn")?.disabled) score();
  });
  if (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "")) {
    const k = document.querySelector("#kbd-hint kbd");
    if (k) k.textContent = "⌘";
  }
  document.getElementById("clear-form").addEventListener("click", () => {
    resetKeepingSettings(form); markUnused(); showSourceHint(); updateFoldCounts();
    loadedWide = null; loadedEntities = null; pendingLabel = null; loadedExample = null; mystery = false; userSource = false;
    document.getElementById("example-note").textContent = "";
    markDirty("the form was cleared");
  });
  document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", async () => {
    const hidden = b.dataset.example === "mystery";
    const want = hidden ? (Math.random() < 0.5 ? 1 : 0) : Number(b.dataset.example);
    const src = document.getElementById("source-select").value;
    const q = new URLSearchParams({ fraud: want, n: 1 });
    if (src && userSource) q.set("source", src);
    const note = document.getElementById("example-note");
    note.textContent = "fetching a payment…";
    try {
      const [ex] = await get(`/api/examples?${q}`);
      fillForm(ex.features, ex.source);
      loadedWide = ex.wide || null;
      loadedEntities = ex.entities && Object.keys(ex.entities).length ? ex.entities : null;
      loadedExample = ex;
      // the payments this customer made before it go into the history box, where they can be read and edited:
      // every version since v5 was measured with them, and without them v6 takes an old customer for a new one
      const histBox = document.getElementById("history-json");
      if (histBox) histBox.value = ex.history && ex.history.length ? JSON.stringify(ex.history) : "";
      pendingLabel = ex.is_fraud;
      mystery = hidden;
      note.innerHTML = exampleNote(ex, hidden);
      markDirty("an example was loaded");
      document.getElementById("score-btn")?.classList.add("attention");
    } catch (e) {
      note.textContent = e.message;
    }
  }));
  document.getElementById("batch-form").addEventListener("submit", batchScore);
  document.getElementById("latent-zoom")?.addEventListener("click", () => {
    if (!latentPoint) return;
    latentZoom = !latentZoom;
    drawLatent(latentPoint);
  });
}

// what was loaded, in words; a mystery row keeps its label to itself until it is scored
function exampleNote(ex, hidden) {
  const label = hidden ? "what it really was stays hidden until it is scored"
    : `labelled <b>${ex.is_fraud ? "fraud" : "honest"}</b>${ex.fraud_scenario ? ` (fraud scenario ${esc(ex.fraud_scenario)})` : ""}`;
  const extra = [ex.history && ex.history.length ? `the customer's ${ex.history.length === 1 ? "previous payment rides" : `last ${ex.history.length} payments ride`} along` : null,
                 ex.wide ? `${Object.keys(ex.wide).length} of the bank's own columns ${ex.history && ex.history.length ? "too" : "ride along"}` : null,
                 loadedEntities ? `entities known: ${Object.keys(loadedEntities).join(", ")}` : null].filter(Boolean).join(" · ");
  return `<b>${esc(ex.txn_id)}</b> · ${esc(sourceName(ex.source))} · a real payment from the test data the model never saw: ${label}.`
    + (extra ? ` <span class="dim">${esc(cap(extra))}.</span>` : "");
}


async function score() {
  const btn = document.getElementById("score-btn");
  const status = document.getElementById("score-status");
  const label = pendingLabel;
  let req;
  try { req = readForm(); } catch (e) { status.textContent = e.message; return; }
  if (!Object.keys(req.features).length) { status.textContent = "fill in at least one field"; return; }
  btn.disabled = true; status.textContent = "scoring"; status.classList.add("spin");
  try {
    const r = await post("/api/score", req);
    r._label = label;
    lastResult = r;
    lastRequest = req;
    showAll = false;
    clearDirty();
    const ex = label != null ? loadedExample : null;
    if (ex && r.flag != null && !tally.counted.has(ex.txn_id)) {
      tally.counted.add(ex.txn_id);
      tally.total += 1;
      if (!!label === !!r.flag) tally.right += 1;
    }
    const reveal = mystery && ex != null;
    renderReceipt(r, label, { fresh: true, reveal });
    if (reveal) {
      mystery = false;
      document.getElementById("example-note").innerHTML = exampleNote(ex, false);
    }
    plotLatent(r);
    // where the receipt sits below the form (the stylesheet's own breakpoint), bring it up
    if (stacked()) document.getElementById("receipt").scrollIntoView({ behavior: glide(), block: "start" });
    const nPast = Math.max(r.sequence ? r.sequence.n_history : 0, r.sequence_model && r.sequence_model.used ? r.sequence_model.n_history : 0);
    status.textContent = `scored from ${r.observed_features} known fields · ${r.n_samples} posterior samples${r.supervised ? " · booster + blend" : " · VAE alone"}${r.sequence || r.sequence_model ? ` · ${nPast} past payments` : ""}`;
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

// a value that rounds to zero is 0, not "-0"
const tidy = (v, d) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v);
function fmtVal(c) {
  if (c.kind === "categorical") {
    if (c.feature === "dow") return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][+c.observed] || c.observed;
    if (c.feature === "hour") return String(c.observed).padStart(2, "0") + ":00";
    return c.observed.replace(/_/g, " ");
  }
  const v = c.observed;
  if (Math.abs(v) >= 1000) return fmt.int(v);
  const d = Math.abs(v) < 10 ? 2 : 1;
  return fmt.num(tidy(v, d), d);
}
function fmtExp(c) {
  if (c.kind === "categorical") return `p=${(c.prob_observed * 100).toFixed(c.prob_observed < 0.1 ? 1 : 0)}%`;
  const v = c.expected;
  const d = Math.abs(v) < 10 ? 2 : 1;
  return "exp " + (Math.abs(v) >= 1000 ? fmt.int(v) : fmt.num(tidy(v, d), d));
}

// 99.92 -> "99.92nd", 41 -> "41st", 12 -> "12th"
function ordinal(s) {
  const digits = s.replace(/\D/g, "");
  const two = +digits.slice(-2), one = +digits.slice(-1);
  return s + (two >= 11 && two <= 13 ? "th" : one === 1 ? "st" : one === 2 ? "nd" : one === 3 ? "rd" : "th");
}
function pctText(pct, fallback) {
  if (pct == null) return fallback;
  return pct >= 99.99 ? ">99.99th" : ordinal(pct.toFixed(pct > 99 ? 2 : 1));
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
  const d = Math.abs(v) < 10 ? 2 : 1;
  return (Math.abs(v) >= 1000 ? fmt.int(v) : fmt.num(tidy(v, d), d)) + (c.unit ? " " + c.unit : "");
}

// ---- the verdict in plain numbers -------------------------------------------------------
// Every alert budget is a line drawn among a bank's honest payments: "1% of legitimate traffic
// flagged" is the score that 1 in 100 of them exceed. A payment's percentile among the same
// payments reads the same way round, so the verdict, each model's own reading and the line
// can all be said in one unit, "1 in N honest payments score this high", and drawn on one
// ruler. Nothing here asks the server for anything new: thresholds for every budget, the
// percentiles and the per-model flags come with every score, from this server and older ones.
const BUDGETS = ["fpr_0.001", "fpr_0.005", "fpr_0.01", "fpr_0.02", "fpr_0.05"];   // strictest first
const shareOf = (op) => parseFloat(String(op || "").replace("fpr_", "")) || null;
function nice(n) {            // 1,284 -> "1,300", 47.3 -> "47", 3.4 -> "3"
  if (n >= 100) return Number(n.toPrecision(2)).toLocaleString("en-US");
  return String(Math.max(1, Math.round(n)));
}
const xOf = (n) => Math.max(0, Math.min(1, Math.log10(Math.max(1, n)) / 4));   // 1 .. 10,000 on a log scale

// a percentile among honest payments -> how many of them it takes to meet one scoring this high
function rarity(pct) {
  if (pct == null || Number.isNaN(pct)) return null;
  if (pct >= 99.99) return { n: 10000, x: 1, short: "< 1 in 10k", say: "fewer than 1 in 10,000", rare: true };
  const n = 1 / (1 - pct / 100);
  if (pct < 50) return { n, x: xOf(n), short: "common", say: "more than half of", common: true };
  return { n, x: xOf(n), short: `1 in ${nice(n)}`, say: `about 1 in ${nice(n)}` };
}
function lineOf(op) {
  const s = shareOf(op);
  return s ? { op, s, n: 1 / s, x: xOf(1 / s), say: `1 in ${nice(1 / s)}` } : null;
}

// keep a dot on the side of each line its score is on (the text and behaviour models move the
// final score after the percentile is taken, and a percentile is interpolated between stored
// quantiles): the quantile a threshold sits at bounds where the payment can be
function placePct(pct, thr, value) {
  if (!thr || value == null) return pct;
  let lo = 0, hi = 100;
  for (const b of BUDGETS) {
    const t = thr[b]?.threshold;
    if (t == null) continue;
    const q = 100 * (1 - shareOf(b));
    if (value > t) lo = Math.max(lo, q); else hi = Math.min(hi, q);
  }
  if (pct == null) return null;
  return Math.min(Math.max(pct, lo + 1e-4), hi);
}
// the same for a model whose thresholds are not sent: its own flag at the chosen line decides
function placeByFlag(pct, flag, line) {
  if (pct == null || flag == null || !line) return pct;
  const q = 100 * (1 - line.s);
  return flag ? Math.max(pct, q + 1e-4) : Math.min(pct, q);
}

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
// "Merchant category" -> "merchant category", "CVV match" stays as it is
const inSentence = (s) => (s && s.length > 1 && s[1] === s[1].toLowerCase() ? s[0].toLowerCase() + s.slice(1) : s);
function sourceName(key) {
  const s = (schema?.sources || []).find((x) => x.key === key);
  return s ? s.name : key;
}
const frac10 = (x) => Math.max(0, Math.min(10, Math.round(10 * x)));
function inTen(x) {
  if (x == null) return null;
  if (x >= 0.995) return "almost every";
  if (x >= 0.95) return "more than 9 in 10";
  if (x < 0.05) return "fewer than 1 in 20";
  return `about ${frac10(x)} in 10`;
}
const one = (x) => x >= 0.995;   // "almost every fraud", not "frauds"
const caught = (x) => `${one(x) ? "fraud" : "frauds"} caught`;
const real = (x) => `${one(x) ? "alert was" : "alerts were"} real fraud`;

// how common fraud was in the test data a quoted precision was measured on, per bank (all the
// version's banks together for an unknown one): the data tab's statistics, fetched once
let testRates = null;
function loadTestRates() {
  if (testRates) return;
  testRates = {};
  get("/api/datasets").then((d) => {
    for (const s of d.sources || []) {
      const t = s.stats?.splits?.test;
      if (t && t.rows) testRates[s.key] = { rows: t.rows, fraud: t.fraud };
    }
  }).catch(() => { /* the clause is left out */ });
}
function fraudShare(calibration, run) {
  if (!testRates) return null;
  const keys = calibration && calibration !== "generic" ? [calibration] : ((models.find((m) => m.name === run) || {}).sources || []);
  let rows = 0, fraud = 0;
  for (const k of keys) { const t = testRates[k]; if (t) { rows += t.rows; fraud += t.fraud; } }
  return rows && fraud ? fraud / rows : null;
}
const dots = (x, cls) => `<span class="dots ${cls}" aria-hidden="true">${Array.from({ length: 10 }, (_, i) => `<i class="${i < frac10(x) ? "on" : ""}"></i>`).join("")}</span>`;

function readVerdict(r, label) {
  const bl = r.blend, nov = r.novelty || {}, sup = r.supervised;
  const op = r.operating_point || r.fpr_target;
  const line = lineOf(op);
  const thr = (bl && bl.thresholds) || nov.thresholds || null;
  const value = bl ? (r.p_fraud ?? bl.prob) : nov.score;
  // the lines this payment clears, strictest first
  const clears = thr ? BUDGETS.filter((b) => thr[b]?.threshold != null && value > thr[b].threshold) : [];
  const looser = line ? clears.filter((b) => shareOf(b) > line.s).sort((a, b) => shareOf(a) - shareOf(b))[0] : null;
  // a close call is one the next budget up would flag; anything further off is simply not flagged
  const next = line ? BUDGETS[BUDGETS.indexOf(op) + 1] : null;
  const state = r.flag == null ? "na" : r.flag ? "flag" : looser && looser === next ? "close" : "clear";
  const verb = bl ? "score this high" : "look this unusual";
  const overallPct = placePct(r.legit_percentile, thr, value);
  const ov = rarity(overallPct);
  const v = { state, line, ov, value, isBlend: !!bl };

  v.word = { flag: "Flag for review", close: "Not flagged · close call", clear: "Not flagged", na: "No alert line" }[state];
  const B = line ? line.say : "";
  if (state === "na") {
    v.say = "This bank has no alert line in this version, so the payment is scored but not judged.";
  } else if (!ov) {
    v.say = state === "flag" ? `Over your alert line of ${B}.` : `Under your alert line of ${B}.`;
  } else if (state === "flag") {
    v.say = `${ov.rare ? "Fewer than 1 in 10,000" : `Only ${ov.say}`} honest payments ${verb}; your alert line is ${B}.`;
    if (clears[0] === "fpr_0.001" && op !== "fpr_0.001") v.say += " It would be flagged even at the strictest line, 1 in 1,000.";
  } else if (state === "close") {
    v.say = `${cap(ov.say)} honest payments ${verb}: short of your alert line of ${B}, but it would be flagged at ${lineOf(looser).say}.`;
  } else {
    v.say = ov.common ? `More than half of honest payments ${verb}; your alert line is ${B}.` : `${cap(ov.say)} honest payments ${verb}, short of your alert line of ${B}.`;
    if (looser) v.say += ` It would take a budget of ${lineOf(looser).say} to flag it.`;
  }
  // the same sentence for a data scientist
  const tParts = [];
  if (overallPct != null) tParts.push(`${pctText(overallPct, "")} percentile among legitimate validation payments`);
  if (bl && bl.threshold != null) tParts.push(`flagged above ${(100 * bl.threshold).toFixed(bl.threshold < 0.1 ? 2 : 1)} at ${opLabel(op)} FPR`);
  else if (!bl && nov.threshold != null) tParts.push(`novelty flagged above ${fmt.num(nov.threshold, 3)} at ${opLabel(op)} FPR`);
  v.tech = tParts.join(" · ");
  v.score = bl ? `${(100 * value).toFixed(value < 0.1 ? 2 : 1)}` : null;

  // the ruler: the verdict, then each of the two models on its own
  v.rows = [];
  // a v7 blend is fit without the novelty model: it weighs the models that learnt from labelled fraud
  if (bl) v.rows.push({ key: "Overall", note: bl.without_vae ? "the models that learnt fraud, blended for this bank" : "every model, blended for this bank", pct: overallPct, flag: r.flag, main: true });
  if (nov.legit_percentile != null || !bl) v.rows.push({ key: "How unusual", note: "the model that learnt honest payments", pct: bl ? placeByFlag(nov.legit_percentile, nov.flag, line) : overallPct, flag: bl ? nov.flag : r.flag, main: !bl });
  if (sup && sup.legit_percentile != null) v.rows.push({ key: "How fraud-like", note: "the model that learnt known fraud", pct: placeByFlag(sup.legit_percentile, sup.flag, line), flag: sup.flag });

  // two models, two questions: what their answers mean together
  if (sup && nov.flag != null && sup.flag != null) {
    v.opinion = nov.flag && sup.flag ? "Both models see it: unusual for this bank, and shaped like known fraud."
      : nov.flag ? "Unusual, but not shaped like the fraud the models were trained on: a rare honest payment, or a new kind of fraud."
      : sup.flag ? "Ordinary-looking, yet shaped like known fraud: the kind that hides in plain sight."
      : "Neither model is alarmed: ordinary for this bank, and unlike known fraud.";
    // the verdict is the blend's, and the blend weighs more than these two
    const also = [r.sequence?.prob != null ? "the customer's history" : null, r.sequence_supervised?.present ? "the pretrained history model" : null,
                  r.local?.present ? "this bank's own model" : null,
                  r.device?.present ? "the device" : null, r.graph?.present ? "the network around the payment" : null].filter(Boolean);
    const weighs = also.length ? `the blend, which also weighs ${also.length > 1 ? also.slice(0, -1).join(", ") + " and " + also[also.length - 1] : also[0]},` : "the blend, weighting the two for this bank,";
    if (bl && r.flag === false && sup.flag) v.opinion += ` Even so, ${weighs} keeps it under your line.`;
    else if (bl && r.flag === true && !nov.flag && !sup.flag) v.opinion += ` Even so, ${weighs} puts it over your line.`;
  }

  // the reasons, in words: the booster's largest pushes either way, the VAE's largest surprises
  v.why = [];
  const pushes = ((sup && sup.pushes) || []).filter((c) => c.is_observed && c.observed != null && c.kind !== "history" && !/\(raw\)$/.test(c.label));
  const phrase = (c) => `${inSentence(c.label)} (${pushVal(c)})`;
  const toward = pushes.filter((c) => c.contribution > 0.15).slice(0, 2);
  const away = pushes.filter((c) => c.contribution < -0.15).slice(0, 2);
  if (toward.length) v.why.push(["Pointed to fraud", toward.map(phrase).join(", ")]);
  if (away.length) v.why.push(["Pointed away", away.map(phrase).join(", ")]);
  const odd = (r.contributions || []).filter((c) => c.z >= 2).sort((a, b) => b.z - a.z).slice(0, 2);
  const chance = (q) => (q < 0.001 ? "less than a 0.1% chance" : `a ${(100 * q).toFixed(q < 0.1 ? 1 : 0)}% chance`);
  // "about" takes a round figure: about 646, not about 646.1 (the working keeps the decimals)
  const about = (c) => (Math.abs(c.expected) >= 10 ? fmt.int(c.expected) : fmtExp(c).replace(/^exp /, ""));
  const oddPhrase = (c) => c.kind === "categorical"
    ? `${inSentence(c.label)} (${fmtVal(c)}, which it gave ${chance(c.prob_observed)})`
    : `${inSentence(c.label)} (${fmtVal(c)}${c.unit ? " " + c.unit : ""}; it expected about ${about(c)})`;
  if (odd.length) v.why.push(["Most out of place", odd.map(oddPhrase).join(", ")]);

  // what the line means, as counts: of 10 frauds, of 10 alerts
  const vb = bl || nov;
  const rec = vb.test_recall ?? vb.expected_recall, prec = vb.test_precision ?? vb.expected_precision;
  if (line && rec != null) {
    // precision depends on how common fraud is: say how common it was where it was measured.
    // held-out figures come with the test split's rate; in-sample ones with the validation
    // rate, recovered from what the line flagged there (share flagged = fpr·(1−π) + recall·π)
    const held = vb.test_recall != null;
    const t = thr?.[op];
    let base = null;
    if (held) base = fraudShare(r.calibration, r.model?.run);
    else if (t && t.flagged_share != null && t.recall != null && t.recall > line.s) base = (t.flagged_share - line.s) / (t.recall - line.s);
    v.budget = { rec, prec, held, base: base > 0 && base < 0.5 ? base : null };
  }

  if (label != null) {
    const call = verdictCall(label, r.flag);
    if (call) v.truth = { ok: call.ok, label, text: call.text };
  }
  v.verb = verb;
  return v;
}

const TRUTH_SAID = { caught: "caught", missed: "missed", "false alarm": "a false alarm", "correctly passed": "correctly let through" };

function rulerHtml(v) {
  const rows = v.rows.filter((row) => row.pct != null);
  if (!rows.length || !v.line) return "";
  const pos = (x) => `${(100 * x).toFixed(2)}%`;
  const N = rows.length;
  const cells = rows.map((row, i) => {
    const rr = rarity(row.pct);
    const hot = row.flag ? " hot" : "";
    return `<span class="ru-lab${row.main ? " main" : ""}" style="grid-area:${i + 1} / 1" title="${esc(row.note)}">${esc(row.key)}</span>
      <span class="ru-track${row.main ? " main" : ""}" style="grid-area:${i + 1} / 2"><i class="ru-dot${hot}" style="left:${pos(rr.x)}"></i></span>
      <span class="ru-val${hot}${row.main ? " main" : ""}" style="grid-area:${i + 1} / 3">${esc(rr.short)}</span>`;
  }).join("");
  const ticks = BUDGETS.map((b) => lineOf(b)).map((l) => `<i class="ru-tick${l.op === v.line.op ? " on" : ""}" style="left:${pos(l.x)}" title="${esc(l.say)}"></i>`).join("");
  const axis = [[0, "common", "edge-l"], [0.25, "1 in 10", "minor"], [0.5, "1 in 100", "mid"], [0.75, "1 in 1k", "minor"], [1, "1 in 10k", "edge-r"]]
    .map(([x, t, c]) => `<span class="${c}" style="left:${pos(x)}">${t}</span>`).join("");
  const aria = rows.map((row) => `${row.key}: ${rarity(row.pct).short}`).join("; ") + `; your alert line: ${v.line.say}`;
  return `<div class="ruler no-terms" role="img" aria-label="${esc(aria)}">
    <div class="ru-head"><span>How rare among honest payments</span><span class="ru-key"><i></i>your line: ${esc(v.line.say)}</span></div>
    <div class="ru-grid">${cells}
      <span class="ru-over" style="grid-area:1 / 2 / ${N + 1} / 3"><i class="ru-zone" style="left:${pos(v.line.x)}"></i><i class="ru-line" style="left:${pos(v.line.x)}"></i></span>
      <span class="ru-axis" style="grid-area:${N + 1} / 2">${ticks}${axis}</span>
    </div>
    <p class="ru-legend">${rows.map((row) => `<b>${esc(row.key)}</b>: ${esc(row.note)}`).join(". ")}.</p>
  </div>`;
}

function summaryHtml(v, reveal) {
  const p = [];
  p.push(`<section class="rs" aria-label="The verdict in plain words">`);
  p.push(`<div class="rs-verdict ${v.state}"><span class="rs-word">${esc(v.word)}</span>${v.score != null ? `<span class="rs-score">fraud score <b>${v.score}</b><small>/100</small></span>` : ""}</div>`);
  p.push(`<p class="rs-say">${esc(v.say)}</p>`);
  if (v.tech) p.push(`<p class="rs-tech">${esc(v.tech)}</p>`);
  p.push(rulerHtml(v));
  if (v.opinion) p.push(`<p class="rs-opinion">${esc(v.opinion)}</p>`);
  if (v.why.length) p.push(`<dl class="rs-why">${v.why.map(([k, t]) => `<dt>${esc(k)}</dt><dd>${esc(t)}</dd>`).join("")}</dl>`);
  if (v.budget) {
    const b = v.budget;
    p.push(`<div class="rs-budget"><p class="rs-bhead">At your line, ${esc(v.line.say)} honest payments get flagged. ${b.held ? "On this bank's held-out test data" : "On the data the line was set on"}, that meant:</p>`);
    p.push(`<div class="rs-brow">${dots(b.rec, "catch")}<span><b>${esc(inTen(b.rec))}</b> ${caught(b.rec)}</span></div>`);
    if (b.prec != null) p.push(`<div class="rs-brow">${dots(b.prec, "real")}<span><b>${esc(inTen(b.prec))}</b> ${real(b.prec)}</span></div>`);
    p.push(`<p class="rs-tech">recall ${fmt.pct(b.rec, 1)}${b.prec != null ? ` · precision ${fmt.pct(b.prec, 1)}` : ""} (${b.held ? "held-out test" : "validation, in-sample"})${b.base ? ` · there, 1 in ${nice(1 / b.base)} payments was fraud; where fraud is rarer, fewer alerts are` : ""}</p></div>`);
  }
  if (v.truth) {
    const t = v.truth;
    const what = t.label ? "fraud" : "honest";
    const said = TRUTH_SAID[t.text] || t.text;
    const rec = tally.total >= 2 ? `<span class="rs-tally">This visit: right on ${tally.right} of ${tally.total} real payments.</span>` : "";
    p.push(`<div class="rs-truth ${t.ok ? "ok" : "bad"}${reveal ? " reveal" : ""}"><span>${reveal ? "The mystery payment was" : "It really was"} <b>${what}</b>: ${esc(said)} <span aria-hidden="true">${t.ok ? "✓" : "✗"}</span></span>${rec}</div>`);
  }
  p.push(`<div class="rs-actions"><button type="button" class="rs-copy" data-copy="summary">Copy summary</button><button type="button" class="rs-copy" data-copy="api">Copy as API call</button><button type="button" class="rs-copy" data-print>Print</button><button type="button" class="rs-copy rs-back" data-back>Back to the form <span aria-hidden="true">↑</span></button></div>`);
  p.push(`</section>`);
  return p.join("");
}

// the summary as plain text, for a case note or an e-mail
function summaryText(v, r) {
  const L = [];
  const tl = models.find((m) => m.name === r.model?.run);
  L.push(`FinFraud: ${v.word.toUpperCase()}`);
  L.push(v.say);
  if (v.rows.length) L.push(v.rows.filter((row) => row.pct != null).map((row) => `${row.key}: ${rarity(row.pct).short}`).join(" · ") + (v.line ? ` · alert line: ${v.line.say}` : ""));
  if (v.opinion) L.push(v.opinion);
  for (const [k, t] of v.why) L.push(`${k}: ${t}`);
  if (v.budget) L.push(`At this line: ${inTen(v.budget.rec)} ${caught(v.budget.rec)}${v.budget.prec != null ? `; ${inTen(v.budget.prec)} ${real(v.budget.prec)}` : ""} (${v.budget.held ? "held-out test" : "validation"}${v.budget.base ? `, where 1 in ${nice(1 / v.budget.base)} payments was fraud` : ""}).`);
  if (v.truth) L.push(`It really was ${v.truth.label ? "fraud" : "honest"}: ${TRUTH_SAID[v.truth.text] || v.truth.text}.`);
  L.push(`Judged as ${r.calibration === "generic" ? "a bank the model has never seen" : sourceName(r.calibration)}; model ${tl ? `${tl.version} · ${tl.label} (${tl.name})` : r.model?.run || "?"}.`);
  if (v.tech) L.push(`(${v.tech}${v.score != null ? `; fraud score ${v.score}/100` : ""})`);
  L.push(`${location.origin}  ${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC`);
  return L.join("\n");
}

// the request behind the verdict, as a command anyone can run against the same API
function apiCall(req) {
  const body = {};
  for (const [k, val] of Object.entries(req || {})) if (val != null && !(typeof val === "object" && !Array.isArray(val) && !Object.keys(val).length)) body[k] = val;
  const json = JSON.stringify(body).replace(/'/g, "'\\''");
  return `curl -s '${location.origin}/api/score' -H 'Content-Type: application/json' -d '${json}'`;
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (_) { /* not allowed here: fall back */ }
  const ta = el("textarea", { class: "copy-buffer", "aria-hidden": "true" });
  ta.value = text;
  document.body.append(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
  ta.remove();
  return ok;
}

function wireCopy(box, v, r) {
  box.querySelector(".rs-copy[data-print]")?.addEventListener("click", () => window.print());
  // where the receipt sits below a long form: straight back up to the examples
  box.querySelector(".rs-copy[data-back]")?.addEventListener("click", () => {
    const form = document.getElementById("score-form");
    form.scrollIntoView({ behavior: glide(), block: "start" });
    form.querySelector(".chip")?.focus({ preventScroll: true });
  });
  box.querySelectorAll(".rs-copy[data-copy]").forEach((b) => b.addEventListener("click", async () => {
    const what = b.dataset.copy;
    const ok = await copyText(what === "api" ? apiCall(lastRequest) : summaryText(v, r));
    const was = b.textContent;
    b.textContent = ok ? "Copied ✓" : "Copy failed";
    b.classList.toggle("done", ok);
    setTimeout(() => { b.textContent = was; b.classList.remove("done"); }, 1600);
  }));
}

function renderReceipt(r, label, { fresh = false, reveal = false } = {}) {
  const box = document.getElementById("receipt");
  box.classList.remove("empty");
  const bl = r.blend, sup = r.supervised, nov = r.novelty || {};
  const opPct = opLabel(r.operating_point);
  const items = r.contributions || [];
  const shown = showAll ? items : items.slice(0, 9);
  const maxZ = Math.max(3, ...items.map((c) => c.z));
  const ctxName = r.calibration === "generic" ? "judged as an unknown bank" : `judged as ${r.calibration}`;
  const v = readVerdict(r, label);

  const lines = [];
  lines.push(`<div class="rc-head"><div class="rc-title">FINFRAUD SCORE</div><div class="dim">${esc(new Date().toISOString().replace("T", " ").slice(0, 19))}  ${esc(ctxName)}</div></div>`);
  lines.push(summaryHtml(v, reveal));
  // ---- the working: every number behind the verdict ----
  lines.push(`<div class="rc-sep"></div>`);
  lines.push(`<div class="rc-sub">THE WORKING  <span class="dim">every number behind the verdict</span></div>`);
  if (bl) {
    lines.push(`<div class="rc-line"><span class="k">fraud score (the blend)</span><span class="${r.flag ? "z hot" : ""}">${(100 * bl.prob).toFixed(bl.prob < 0.1 ? 2 : 1)} / 100</span></div>`);
    if (bl.threshold != null) {
      lines.push(`<div class="rc-line"><span class="k">flagged above</span><span>${(100 * bl.threshold).toFixed(bl.threshold < 0.1 ? 2 : 1)}</span></div>`);
      lines.push(`<div class="rc-line"><span class="k">alert budget</span><span>${esc(opPct)} of legit flagged</span></div>`);
      // what this threshold did on the held-out test split when the run recorded it; the
      // validation figure it was set on is in-sample and says so
      const rec = bl.test_recall != null ? [bl.test_recall, "held-out test"] : bl.expected_recall != null ? [bl.expected_recall, "validation, in-sample"] : null;
      const prec = bl.test_precision != null ? [bl.test_precision, "held-out test"] : bl.expected_precision != null ? [bl.expected_precision, "validation, in-sample"] : null;
      if (rec) lines.push(`<div class="rc-line"><span class="k">fraud caught at this budget (${rec[1]})</span><span>${fmt.pct(rec[0], 1)}</span></div>`);
      if (prec) lines.push(`<div class="rc-line"><span class="k">flags that are fraud (${prec[1]})</span><span>${fmt.pct(prec[0], 1)}</span></div>`);
    }
  } else {
    lines.push(`<div class="rc-line"><span class="k">verdict from</span><span>the novelty score alone (no blend in this version)</span></div>`);
    if (nov.threshold != null) lines.push(`<div class="rc-line"><span class="k">alert budget</span><span>${esc(opPct)} of legit flagged</span></div>`);
  }
  if (r.model) {
    const tl = models.find((m) => m.name === r.model.run);
    lines.push(`<div class="rc-line"><span class="k">version</span><span>${esc(tl ? `${tl.version} · ${tl.label}` : r.model.run)}</span></div>`);
    if (r.model.note) lines.push(`<div class="rc-line"><span class="k"></span><span class="dim">${esc(r.model.note)}${r.model.bank_known ? "" : "; this bank is unknown to it, judged as an unknown bank"}</span></div>`);
  }
  // ---- every generation on the same transaction ----
  if (r.compare && r.compare.length > 1) {
    lines.push(`<div class="rc-compare">`);
    lines.push(`<div class="rc-sub">EVERY VERSION  <span class="dim">the same payment judged by each</span></div>`);
    lines.push(`<div class="rc-line head"><span class="k">version</span><span>fraud score (v3: how unusual) · verdict</span></div>`);
    const byOrder = [...r.compare].sort((a, b) => (models.find((m) => m.name === a.run)?.order ?? 99) - (models.find((m) => m.name === b.run)?.order ?? 99));
    for (const c of byOrder) {
      const tl = models.find((m) => m.name === c.run);
      const name = tl ? `${tl.version} ${tl.label}` : c.run;
      const val = c.p_fraud != null ? `${(100 * c.p_fraud).toFixed(c.p_fraud < 0.1 ? 2 : 1)}` : (c.novelty_percentile != null ? `${pctText(c.novelty_percentile, "")} pct` : "–");
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
    lines.push(`<div class="rc-sub">KNOWN FRAUD  <span class="dim">the booster: decision trees trained on labelled fraud</span></div>`);
    const sf = sup.flag == null ? "" : (sup.flag ? "  FLAG" : "  ok");
    lines.push(`<div class="rc-line"><span class="k">fraud score, this model alone</span><span class="${sup.flag ? "z hot" : ""}">${(100 * sup.prob).toFixed(sup.prob < 0.1 ? 2 : 1)}${sf}</span></div>`);
    if (sup.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">above this share of normal</span><span>${pctText(sup.legit_percentile, "")}</span></div>`);
    if (r.labels) {
      const lb = r.labels;
      const txt = lb.fast_block === "blanked" ? `blanked (labels take ${lb.latency_days}d)` : lb.fast_block === "used" ? "one-week history used" : "not in this request";
      lines.push(`<div class="rc-line"><span class="k">last week's fraud history</span><span>${esc(txt)}</span></div>`);
    }
    const pushes = (sup.pushes || []).slice(0, showAll ? 12 : 6);
    const maxC = Math.max(0.5, ...pushes.map((c) => Math.abs(c.contribution)));
    lines.push(`<div class="rc-line"><span class="k">what it looked at</span><span class="k">push toward / away from fraud</span></div>`);
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
    lines.push(`<div class="rc-sub">CUSTOMER HISTORY  <span class="dim">${sq.n_history} of up to ${sq.window} previous payments read${sq.history_from ? ` (${esc(sq.history_from)})` : ""}</span></div>`);
    if (sq.prob != null) {
      const qf = sq.flag == null ? "" : (sq.flag ? "  FLAG" : "  ok");
      lines.push(`<div class="rc-line"><span class="k">fraud score, from the history alone</span><span class="${sq.flag ? "z hot" : ""}">${(100 * sq.prob).toFixed(sq.prob < 0.1 ? 2 : 1)}${qf}</span></div>`);
      if (sq.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">above this share of normal</span><span>${pctText(sq.legit_percentile, "")}</span></div>`);
    }
    if (!sq.n_history) lines.push(`<div class="rc-line dim"><span>no history given: scored as a first transaction. add past transactions below, or a customer id so the server remembers them</span></div>`);
  }

  // ---- v6 history model: the pretrained transaction encoder's reading, and (v7) its own fraud score ----
  const sm = r.sequence_model;
  if (sm) {
    const banks = sm.sources ? sm.sources.join(" and ") : null;
    lines.push(`<div class="rc-sep"></div>`);
    lines.push(`<div class="rc-sub">HISTORY MODEL  <span class="dim">a transformer pretrained on unlabelled payments${sm.used ? `, ${sm.n_history} of up to ${sm.window - 1} previous payments read` : ""}</span></div>`);
    if (!sm.used) {
      // left out: a bank its block is not for, or a payment that came without the past its own fields speak of
      const why = sm.skipped === "bank" ? `not read for this bank: this version uses it only for ${esc(banks || "some banks")}, the banks where it improved the results`
        : sm.past_in_fields ? "left out: this payment's own fields say the customer has earlier payments, and none came with it. read alone it would pass for a new customer's and raise false alarms, so this version judges it without the history model's columns, a case its decision trees were trained on. add the earlier payments below to bring it in"
        : "left out: no earlier payments came with this one";
      lines.push(`<div class="rc-line dim"><span>${why}</span></div>`);
    } else {
      if (sm.surprise != null) lines.push(`<div class="rc-line"><span class="k">how unexpected this payment is, after the earlier ones</span><span>${fmt.num(sm.surprise, 2)} nats a field</span></div>`);
      for (const f of (sm.most_surprising || []).slice(0, showAll ? 5 : 3)) {
        lines.push(`<div class="rc-line"><span class="k">· ${esc(f.label || f.field)}</span><span>${fmt.num(f.nll, 2)}</span></div>`);
      }
      const ts = r.sequence_supervised;
      if (ts && ts.present && ts.prob != null) {
        const tf = ts.flag == null ? "" : (ts.flag ? "  FLAG" : "  ok");
        lines.push(`<div class="rc-line"><span class="k">fraud score, from its reading alone</span><span class="${ts.flag ? "z hot" : ""}">${(100 * ts.prob).toFixed(ts.prob < 0.1 ? 2 : 1)}${tf}</span></div>`);
        if (ts.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">above this share of normal</span><span>${pctText(ts.legit_percentile, "")}</span></div>`);
      }
      lines.push(`<div class="rc-line dim"><span>its ${sm.columns ?? ""} columns go to the decision trees above${ts && ts.present ? "; its own score is one more signal in the blend" : ""}</span></div>`);
      // read with nothing before it: fine for a first payment, a known weakness when the fields speak of a past
      if (!sm.n_history) lines.push(`<div class="rc-line dim"><span>${sm.past_in_fields ? "no earlier payments came with this one, though its own fields say the customer has some. this version was trained with every customer's history in place and misjudges a payment that arrives alone more often, mostly as false alarms: add the earlier payments below" : "no earlier payments came with this one: read as a customer's first"}</span></div>`);
    }
  }

  // ---- v5 text: free text through the text heads, fused as likelihood ratios ----
  if (r.text) {
    lines.push(`<div class="rc-sep"></div>`);
    const tf = r.text.fields || {};
    const keys = Object.keys(tf);
    lines.push(`<div class="rc-sub">TEXT  <span class="dim">${keys.length ? `${keys.length} field${keys.length === 1 ? "" : "s"} read` : "no text sent"}${r.fusion ? ", added to the verdict" : ""}</span></div>`);
    for (const k of keys) {
      const t = tf[k];
      lines.push(`<div class="rc-line"><span class="k">${esc(k)} → ${esc(t.label)}</span><span class="${t.llr > 1 ? "z hot" : ""}">p ${(100 * t.prob).toFixed(1)}%, LR ${t.llr >= 0 ? "+" : ""}${t.llr.toFixed(2)}</span></div>`);
    }
    if (!keys.length && r.text.sent && r.text.sent.length) lines.push(`<div class="rc-line dim"><span>sent ${esc(r.text.sent.join(", "))}: no head reads those keys (url, description, memo, message, profile, bio)</span></div>`);
  }
  if (r.fusion) {
    lines.push(`<div class="rc-sep"></div>`);
    lines.push(`<div class="rc-line"><span class="k">blend ⊕ ${esc(Object.keys(r.fusion.terms).join(" ⊕ "))}</span><span class="${r.flag ? "z hot" : ""}">${(100 * r.fusion.prob).toFixed(r.fusion.prob < 0.1 ? 2 : 1)}% (blend alone ${(100 * r.blend.prob).toFixed(r.blend.prob < 0.1 ? 2 : 1)}%)${r.flag == null ? "" : (r.flag ? "  FLAG" : "  ok")}</span></div>`);
  }

  // ---- v5 behaviour: the session against the customer's enrolled profile ----
  if (r.behaviour) {
    lines.push(`<div class="rc-sep"></div>`);
    const bf = r.behaviour.fields || {};
    const mods = Object.keys(bf);
    lines.push(`<div class="rc-sub">HOW THE DEVICE WAS USED  <span class="dim">${mods.length ? `${esc(mods.join(", "))} session` : "no session sent"}${r.behaviour.enrolled_sessions ? `, profile from ${r.behaviour.enrolled_sessions} earlier session${r.behaviour.enrolled_sessions === 1 ? "" : "s"}` : ""}</span></div>`);
    for (const m of mods) {
      const v = bf[m];
      if (v.llr == null) lines.push(`<div class="rc-line dim"><span>${esc(v.label)}: ${v.windows} window${v.windows === 1 ? "" : "s"} read, no enrolment for this customer yet${r.behaviour.enrolled_now ? " — enrolled now" : ""}</span></div>`);
      else lines.push(`<div class="rc-line"><span class="k">${esc(v.label)}</span><span class="${v.llr < -1 ? "z hot" : ""}">cosine ${v.cosine.toFixed(2)}, LR ${v.llr >= 0 ? "+" : ""}${v.llr.toFixed(2)}${v.llr < 0 ? " (not like the account holder)" : ""}</span></div>`);
    }
    if (!mods.length && r.behaviour.sent && r.behaviour.sent.length) lines.push(`<div class="rc-line dim"><span>sent ${esc(r.behaviour.sent.join(", "))}: too few events for a window, or no verifier for that modality</span></div>`);
    if (!r.behaviour.sent || !r.behaviour.sent.length) lines.push(`<div class="rc-line dim"><span>add session events below and a customer id; the second session is compared with the first</span></div>`);
  }

  // ---- v5 field heads: the device block (and whatever lands next) through its own head ----
  for (const name of ["local", "device", "graph"]) {
    const hb = r[name];
    if (!hb) continue;
    lines.push(`<div class="rc-sep"></div>`);
    lines.push(`<div class="rc-sub">${name.toUpperCase()}  <span class="dim">${esc(hb.label || name)}${name === "local" ? "" : " head"}${hb.present && hb.fields.length ? `, ${hb.fields.length} field${hb.fields.length === 1 ? "" : "s"}` : ""}</span></div>`);
    if (hb.present && hb.prob != null) {
      const hf = hb.flag == null ? "" : (hb.flag ? "  FLAG" : "  ok");
      lines.push(`<div class="rc-line"><span class="k">fraud score, ${name === "local" ? "this bank's own model" : name + " model"} alone</span><span class="${hb.flag ? "z hot" : ""}">${(100 * hb.prob).toFixed(hb.prob < 0.1 ? 2 : 1)}${hf}</span></div>`);
      if (hb.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">above this share of normal</span><span>${pctText(hb.legit_percentile, "")}</span></div>`);
      if (hb.fields.length) lines.push(`<div class="rc-line dim"><span>${esc(hb.fields.join(", "))}</span></div>`);
    } else if (name === "local") {
      lines.push(`<div class="rc-line dim"><span>this bank has no specialist booster (too little labelled fraud, or the generic context): the shared booster carries it</span></div>`);
    } else if (name === "graph") {
      lines.push(`<div class="rc-line dim"><span>${hb.available === false ? "this bank has no entity graph" : (hb.sent && hb.sent.length ? `none of the entities sent (${esc(hb.sent.join(", "))}) is known to the graph` : "no entity ids sent (load an example row, or pass entities in the API)")}: signal absent, the blend goes without it</span></div>`);
    } else {
      lines.push(`<div class="rc-line dim"><span>no ${name} fields on this transaction: signal absent, the blend goes without it</span></div>`);
    }
  }

  // ---- novelty evidence ----
  lines.push(`<div class="rc-sep"></div>`);
  lines.push(`<div class="rc-sub">HOW UNUSUAL  <span class="dim">the VAE, trained on legitimate payments only${sq ? ", with this customer's history in view" : ""}${bl && bl.without_vae ? "; shown beside the verdict, not weighed in it" : ""}</span></div>`);
  const nf = nov.flag == null ? "" : (nov.flag ? "  FLAG" : "  ok");
  if (nov.legit_percentile != null) lines.push(`<div class="rc-line"><span class="k">above this share of normal</span><span class="${nov.flag ? "z hot" : ""}">${pctText(nov.legit_percentile, "")}${nf}</span></div>`);
  lines.push(`<div class="rc-line"><span class="k">novelty score${nov.score_type === "norm" ? " (per-field, normalised)" : ""}</span><span>${fmt.num(nov.score, 3)}</span></div>`);
  if (nov.score_type === "norm") lines.push(`<div class="rc-line"><span class="k">raw score (negative ELBO)</span><span>${fmt.num(nov.score_raw, 2)}</span></div>`);
  if (nov.z_score != null) lines.push(`<div class="rc-line"><span class="k">z-score against normal</span><span>${nov.z_score >= 0 ? "+" : ""}${fmt.num(nov.z_score, 2)}</span></div>`);
  lines.push(`<div class="rc-line"><span class="k">KL term</span><span>${fmt.num(nov.kl, 2)}</span></div>`);
  if (nov.threshold != null) lines.push(`<div class="rc-line"><span class="k">flagged above</span><span>${fmt.num(nov.threshold, 3)}</span></div>`);
  lines.push(`<div class="rc-line"><span class="k">field</span><span class="k">surprise, in standard deviations</span></div>`);
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
  // what the verdict is made of: a v7 blend leaves the novelty model out, so the foot says who votes
  const footVerdict = !bl ? "the verdict is the novelty score alone (no blend in this version).<br>"
    : (bl.without_vae
      ? "the verdict comes from the blend: the scores of the models that learnt from labelled fraud, weighted per bank. this version's blend leaves the novelty model out (measured as fraud caught at a 1% false-positive rate, it cost slightly more than it added once the others were in), so that model explains the payment and has no weight in the verdict. "
      : "the verdict comes from the blend: every model's score, weighted per bank. ")
      + "a fraud score is the blend's output, and the blend was fit with fraud and honest payments weighted equally, so read the score against the line, not against 50.<br>";
  lines.push(`<div class="rc-foot">${footVerdict}surprise = how much worse than usual the VAE predicted a field, compared with normal ${esc(r.source === "generic" ? "traffic" : r.source + " traffic")}, in standard deviations; "exp" is what it expected.${sup ? "<br>pushes are the booster's SHAP contributions, in log-odds; a field that is unknown can still push, because a missing value tells a decision tree something too." : ""}</div>`);
  box.innerHTML = lines.join("");
  wireCopy(box, v, r);
  const tog = document.getElementById("rc-toggle");
  if (tog) tog.addEventListener("click", () => { showAll = !showAll; renderReceipt(r, label); });
  if (fresh) {
    // the paper feeds out of the printer once per new verdict (not on "show everything")
    box.classList.remove("printing");
    void box.offsetWidth;
    box.classList.add("printing");
    const live = document.getElementById("verdict-live");
    if (live) live.textContent = `${v.word}. ${v.say}`;
  }
}

// ---- latent map -----------------------------------------------------------------

async function loadLatentBackground() {
  const run = selectedModel()?.name || state.meta?.active_run?.name;
  latentBackground = null;
  if (!run) { drawLatent(); return; }
  try {
    latentBackground = await get(`/api/runs/${encodeURIComponent(run)}/latent`);
    const n = latentBackground.y.length;
    const share = latentBackground.y.reduce((a, b) => a + b, 0) / Math.max(n, 1);
    // the map draws every bank's fraud at up to 1,200 rows beside 2,500 honest ones, so fraud
    // shows; a reader should not take the red for how common fraud is
    document.getElementById("latent-note").textContent = `${n.toLocaleString("en-US")} payments from the validation data, laid out so that payments the VAE finds alike sit close together (PCA of its posterior means).`
      + (share > 0.05 ? ` Fraud is over-sampled so that it shows: about 1 in ${Math.round(1 / share)} points here, far more than in real traffic.` : "");
  } catch (e) {
    document.getElementById("latent-note").textContent = "no latent map for this run";
  }
  latentPoint = null;
  latentZoom = false;
  drawLatent();
}

// the map's frame: every point, the payment included, on whole-number bounds; or, zoomed, a
// window reaching an eighth of the map either side of the payment. Fixed bounds also mean that hiding a
// cloud from the legend no longer zooms the axes in on whatever is left
function latentBounds(point, zoom) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const take = (x, y) => { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; };
  if (latentBackground) for (const [x, y] of latentBackground.xy) take(x, y);
  if (point) take(point[0], point[1]);
  if (!Number.isFinite(x0)) return null;
  if (zoom && point) {
    // on half units, so the axes' own ticks land on the edges
    const lo = (v) => Math.floor(v * 2) / 2, hi = (v) => Math.ceil(v * 2) / 2;
    const dx = Math.max(0.5, (x1 - x0) / 8), dy = Math.max(0.5, (y1 - y0) / 8);
    return { x: [lo(point[0] - dx), hi(point[0] + dx)], y: [lo(point[1] - dy), hi(point[1] + dy)] };
  }
  return { x: [Math.floor(x0), Math.ceil(x1)], y: [Math.floor(y0), Math.ceil(y1)] };
}

function drawLatent(point = null) {
  const canvas = document.getElementById("latent-chart");
  if (!window.Chart) { canvas.replaceWith(el("p", { class: "dim" }, "charts need the Chart.js CDN")); return; }
  latentPoint = point;
  const zoomBtn = document.getElementById("latent-zoom");
  if (zoomBtn) { zoomBtn.hidden = !point; zoomBtn.textContent = latentZoom ? "Show the whole map" : "Zoom in on this payment"; zoomBtn.setAttribute("aria-pressed", latentZoom ? "true" : "false"); }
  // Chart.js draws the lowest `order` last, on top. The clouds keep the order they always had
  // (the honest payments over the fraud, so the fraud shows where it leaves normal behind); the
  // payment's ring, white outline and cross go over everything, where no cloud can hide them
  const datasets = [];
  let legitN = 0;
  if (latentBackground) {
    const bg = latentBackground;
    const legit = [], fraud = [];
    for (let i = 0; i < bg.y.length; i++) {
      (bg.y[i] ? fraud : legit).push({ x: bg.xy[i][0], y: bg.xy[i][1] });
    }
    legitN = legit.length;
    datasets.push({ label: "legitimate", data: legit, backgroundColor: "rgba(86,104,121,0.22)", pointRadius: 1.6, pointHoverRadius: 1.6, order: 3, parsing: false });
    datasets.push({ label: "fraud", data: fraud, backgroundColor: "rgba(200,16,46,0.45)", pointRadius: 1.8, pointHoverRadius: 1.8, order: 4, parsing: false });
  }
  const mine = [];
  if (point) {
    const at = [{ x: point[0], y: point[1] }];
    mine.push(datasets.length);
    datasets.push({ label: "", data: at, pointStyle: "circle", pointRadius: 16, pointHoverRadius: 16, backgroundColor: "rgba(255,255,255,0.5)", borderColor: "rgba(21,33,46,0.55)", borderWidth: 1.5, order: 2, parsing: false });
    mine.push(datasets.length);
    datasets.push({ label: "", data: at, pointStyle: "crossRot", pointRadius: 9, pointHoverRadius: 9, borderColor: "#fff", borderWidth: 7, order: 1, parsing: false });
    mine.push(datasets.length);
    datasets.push({ label: "this transaction", data: at, pointStyle: "crossRot", pointRadius: 9, pointHoverRadius: 9, backgroundColor: "#fff", borderColor: palette.ink, borderWidth: 2.75, order: 0, parsing: false });
  }
  const b = latentBounds(point, latentZoom);
  const axis = (title, lim) => ({ title: { display: true, text: title }, grid: { color: "#eef1f4" }, ...(lim ? { min: lim[0], max: lim[1] } : {}) });
  if (latentChart) latentChart.destroy();
  latentChart = new Chart(canvas, {
    type: "scatter",
    data: { datasets },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      events: ["click"],   // no hover work over 36,000 points; the legend still answers a click or a tap
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, usePointStyle: true, filter: (item) => !!item.text, sort: (p, q) => p.datasetIndex - q.datasetIndex },
          // the payment's three layers hide and show together
          onClick: (e, item, legend) => {
            const ch = legend.chart;
            const group = mine.includes(item.datasetIndex) ? mine : [item.datasetIndex];
            const show = !ch.isDatasetVisible(item.datasetIndex);
            for (const i of group) ch.setDatasetVisibility(i, show);
            ch.update();
          },
        },
        tooltip: { enabled: false },
      },
      scales: { x: axis("latent PC 1", b && b.x), y: axis("latent PC 2", b && b.y) },
    },
  });
  // the same picture in words, for a screen reader and for anyone who wants the coordinates
  const n = latentBackground ? latentBackground.y.length : 0;
  const where = point ? `this payment sits at PC 1 ${fmt.num(point[0], 2)}, PC 2 ${fmt.num(point[1], 2)}` : "";
  canvas.setAttribute("aria-label", n
    ? `A map of ${n.toLocaleString("en-US")} validation payments, ${legitN.toLocaleString("en-US")} honest in grey and ${(n - legitN).toLocaleString("en-US")} fraud in red${point ? `; ${where}, marked with a cross` : ""}.`
    : point ? `No map for this version; ${where}.` : "No map for this version.");
  const w = document.getElementById("latent-where");
  if (w) w.textContent = point ? `${where}${latentZoom ? " (zoomed)" : ""}` : "";
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
