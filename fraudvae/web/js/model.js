// model & training tab: the versions (newest first, one selected), the selected run's record, local training
import { get, post, pollTask, fmt, el, esc, palette } from "./api.js";
import { refreshMeta, refreshModels, scoreHref, rememberRun, getSelectedRun, setSelectedRun, state as appState } from "./app.js";

let runs = [];
let timelineItems = [];
let active = null;
let schemaVersion = null;
function state_schema_version() { return schemaVersion ?? "?"; }
let selected = null;
let charts = [];
let refreshTimer = null;

let isPublic = false;

export async function initModel(state) {
  schemaVersion = state.meta?.schema_version ?? null;
  isPublic = !!state.meta?.public?.public;
  if (isPublic) document.getElementById("train-section")?.classList.add("hidden");
  await loadTimeline();
  await loadRuns();
  const cfgs = await get("/api/configs");
  const sel = document.getElementById("train-config");
  for (const c of cfgs.configs) sel.append(el("option", { value: c.name, title: c.text }, c.name));
  sel.value = cfgs.configs.find((c) => c.name === "cpu_small") ? "cpu_small" : cfgs.configs[0]?.name;
  const psel = document.getElementById("train-prepared");
  for (const p of cfgs.prepared) psel.append(el("option", { value: p }, p));
  if (!cfgs.prepared.length) psel.append(el("option", { value: "default" }, "default (not prepared yet)"));
  document.getElementById("train-device").textContent = `this machine: ${state.meta?.device || "?"}, torch ${state.meta?.torch || ""}`;
  document.getElementById("train-form").addEventListener("submit", startTraining);
  window.addEventListener("tabshown", (e) => { if (e.detail === "model") loadRuns(); });
  // the picker on the score tab changed the selection (one-host workbench): follow it here
  window.addEventListener("runselected", (e) => { if (e.detail && e.detail !== selected && runs.some((r) => r.name === e.detail)) selectRun(e.detail); });
}

async function loadTimeline() {
  const all = appState.models?.length ? appState.models : await refreshModels();
  timelineItems = all.filter((m) => m.milestone);
}

const best = () => timelineItems.reduce((a, b) => (b.mean_test_auroc ?? -1) > (a?.mean_test_auroc ?? -1) ? b : a, null);

// the link that opens the score tab with this version chosen (the url carries it across hosts)
function scoreLink(name) {
  return el("a", { href: scoreHref(name), class: "btn ghost small", onclick: (e) => { e.stopPropagation(); rememberRun(name); } }, "score a payment with it →");
}

function card(m, r, { compact = false } = {}) {
  const sel = m.name === selected;
  const b = best();
  const legacy = m.schema_version != null && schemaVersion != null && m.schema_version < schemaVersion;
  const st = r?.status?.state;
  const running = st && st !== "finished";
  const tags = [];
  if (b && m.name === b.name) tags.push(el("span", { class: "tl-tag" }, "newest · best"));
  if (legacy) tags.push(el("span", { class: "tl-tag legacy", title: "scores today's payments with the fields it knows; the receipt says what it could not see" }, `older field set (v${m.schema_version})`));
  if (m.name === active && !isPublic) tags.push(el("span", { class: "tl-tag outline" }, "default scorer"));
  if (running) tags.push(el("span", { class: "tl-tag running" }, st));
  const n9 = m.mean_shared9 != null ? `AUROC ${fmt.auc(m.mean_shared9)} on the 9 shared datasets` : (m.mean_test_auroc != null ? `AUROC ${fmt.auc(m.mean_test_auroc)} on its datasets` : "");
  const mins = m.train_minutes != null ? `${m.train_minutes < 90 ? Math.round(m.train_minutes) + " min" : (m.train_minutes / 60).toFixed(1) + " h"} of training` : "";
  const post = m.post_minutes != null ? ` + ${Math.round(m.post_minutes)} min for the booster side` : "";
  const pick = () => selectRun(m.name, { scroll: true });
  return el("div", { class: "gen-card" + (sel ? " selected" : ""), role: "button", tabindex: 0, "aria-pressed": sel ? "true" : "false",
                     onclick: pick, onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } } },
    el("div", { class: "gen-head" },
      m.version ? el("div", { class: "tl-v" }, m.version) : null,
      el("div", { class: "tl-name" }, m.label || m.name),
      el("div", { class: "tl-run" }, m.name),
      tags.length ? el("div", { class: "gen-tags" }, ...tags) : null),
    el("div", { class: "gen-body" },
      el("div", { class: "tl-num" }, n9 + (m.generic_test_auroc != null ? ` · generic ${fmt.auc(m.generic_test_auroc)}` : "")),
      el("div", { class: "tl-meta" }, [m.hardware, m.corpus, (mins + post).trim()].filter(Boolean).join(" · ")),
      compact ? null : el("div", { class: "tl-changed" }, m.changed || "")),
    el("div", { class: "gen-actions" },
      el("span", { class: "gen-state" }, sel ? "✓ selected · its record is below" : "select to see its record ↓"),
      r?.has_model && r?.compatible !== false ? scoreLink(m.name) : null));
}

function renderTimeline() {
  const box = document.getElementById("timeline");
  if (!box) return;
  box.innerHTML = "";
  if (!timelineItems.length) { box.innerHTML = "<p class='dim'>No trained versions on this server yet.</p>"; return; }
  // newest first
  for (const m of [...timelineItems].reverse()) box.append(card(m, runs.find((r) => r.name === m.name)));
}

async function loadRuns() {
  const r = await get("/api/runs");
  runs = r.runs; active = r.active;
  if (!selected) {
    const want = getSelectedRun();
    selected = (want && runs.some((x) => x.name === want)) ? want : (active || best()?.name || runs[0]?.name || null);
    if (selected && selected !== want) setSelectedRun(selected);
  }
  renderTimeline();
  renderRuns();
  if (selected) showRun(selected);
}

// runs that are not one of the six: a workbench's experiments, partial runs, something still training
function renderRuns() {
  const section = document.getElementById("other-runs");
  const host = document.getElementById("runs-list");
  if (!section || !host) return;
  const known = new Set(timelineItems.map((m) => m.name));
  const others = runs.filter((r) => !known.has(r.name));
  section.classList.toggle("hidden", !others.length);
  host.innerHTML = "";
  for (const r of others) {
    const dev = r.run?.device_name || r.run?.device || "";
    const m = { name: r.name, label: r.config_name ? `${r.config_name} run` : r.name, version: r.schema_version ? `v${r.schema_version}` : "",
                mean_test_auroc: r.mean_test_auroc, generic_test_auroc: r.generic_test_auroc ?? r.generic_val_auroc, schema_version: r.schema_version,
                hardware: dev, corpus: r.run?.prepared ? `set ${r.run.prepared}` : "", changed: "" };
    host.append(card(m, r, { compact: true }));
  }
}

function selectRun(name, { scroll = false } = {}) {
  selected = name;
  setSelectedRun(name);
  renderTimeline();
  renderRuns();
  showRun(name, scroll);
}

function destroyCharts() { charts.forEach((c) => c.destroy()); charts = []; }

async function showRun(name, scroll = false) {
  const box = document.getElementById("run-detail");
  if (scroll) box.scrollIntoView({ behavior: "smooth", block: "start" });
  let d;
  try { d = await get(`/api/runs/${encodeURIComponent(name)}/metrics`); } catch (e) { box.innerHTML = `<p class="dim">${esc(e.message)}</p>`; return; }
  if (name !== selected) return;   // a later selection overtook this one
  destroyCharts();
  const r = runs.find((x) => x.name === name) || {};
  const tl = timelineItems.find((m) => m.name === name);
  const run = d.run || {};
  const cfg = run.config || {};
  const parts = [];
  const stale = d.compatible === false;
  const legacy = !stale && d.schema_version != null && schemaVersion != null && d.schema_version < schemaVersion;
  parts.push(`<div class="card-head"><h2>The record of ${esc(name)}${tl ? ` <span class="dim">· ${esc(tl.version)}, ${esc(tl.label)}</span>` : ""}</h2><div class="row">
    ${r.has_model && !stale ? `<a class="btn ghost small" id="btn-score-with" href="${esc(scoreHref(name))}">score a payment with it →</a>` : ""}
    ${r.has_model && !stale && !isPublic ? `<button class="btn ${name === active ? "ghost" : "primary"} small" id="btn-load-run" ${name === active ? "disabled" : ""}>${name === active ? "the default scorer" : "make it the default scorer"}</button>` : ""}
  </div></div>`);
  if (stale) parts.push(`<div class="notice warn">This version was trained on a set of fields (v${d.schema_version ?? "?"}) the current code (v${state_schema_version()}) cannot read. It stays here for the record.</div>`);
  else if (legacy) parts.push(`<div class="notice info">Trained on an older set of fields (v${d.schema_version}; the current set is v${state_schema_version()}). It scores today's payments with the fields it knows and ignores the rest; the receipt on the Score tab says what it could not see.</div>`);
  const st = d.status || {};
  if (st.state && st.state !== "finished") {
    const prog = st.progress != null ? ` ${(100 * st.progress).toFixed(0)}% of epoch` : "";
    parts.push(`<div class="notice warn">${esc(st.state)} – epoch ${st.epoch ?? "?"} / ${st.epochs ?? "?"}${prog}${st.rows_per_sec ? `, ${fmt.int(st.rows_per_sec)} rows/s` : ""}${st.loss ? `, loss ${fmt.num(st.loss, 3)}` : ""}</div>`);
  }
  parts.push(`<dl class="kv">
    <dt>started</dt><dd>${esc(run.started || "–")}${run.host ? ` on ${esc(run.host)}` : ""}</dd>
    <dt>device</dt><dd>${esc(run.device_name || run.device || "–")}</dd>
    <dt>config</dt><dd>${esc(cfg.name || "–")}: latent ${cfg.model?.latent_dim}, hidden ${(cfg.model?.hidden || []).join("-")}, batch ${cfg.train?.batch_size}, lr ${cfg.train?.lr}, β ${cfg.train?.beta} (warmup ${cfg.train?.beta_warmup_epochs} ep), source dropout ${cfg.train?.source_dropout}</dd>
    <dt>size</dt><dd>${run.params ? (run.params / 1e6).toFixed(2) + "M parameters in the VAE" : "–"}${d.booster ? `, plus a booster of ${d.booster.best_iteration} trees (${d.booster.params?.num_leaves} leaves each, ${d.booster.features?.length} inputs, trained on ${fmt.int(d.booster.train_rows)} labelled rows in ${d.booster.seconds}s)` : ""}</dd>
    <dt>data</dt><dd>${run.prepared_meta ? `set <code>${esc(run.prepared)}</code>: ${fmt.int(run.prepared_meta.counts?.train)} training / ${fmt.int(run.prepared_meta.counts?.val)} validation / ${fmt.int(run.prepared_meta.counts?.test)} test rows from ${(run.prepared_meta.sources || []).join(", ")}${run.prepared_meta.max_rows_per_source ? `, training rows capped at ${fmt.int(run.prepared_meta.max_rows_per_source)} per dataset` : ""}` : "–"}</dd>
    ${d.remote ? `<dt>remote</dt><dd>fetched from ${esc(d.remote.user)}@${esc(d.remote.host)}:${d.remote.port} ${fmt.ago(d.remote.fetched)}</dd>` : ""}
  </dl>`);

  if (d.metrics.length) {
    parts.push(`<div class="chart-pair">
      <div class="chart-box"><h4>training loss per pass (lower is better) and its parts</h4><div class="chart-wrap" style="height:200px"><canvas id="m-loss"></canvas></div></div>
      <div class="chart-box"><h4>validation AUROC per dataset during training (raw score, a sample)</h4><div class="chart-wrap" style="height:200px"><canvas id="m-auc"></canvas></div></div></div>`);
  }
  if (d.eval) {
    const ev = d.eval;
    const evd = d.eval_delay;   // v5: the test rows with the one-week fraud history blanked (labels that take a month)
    const slowCell = (k) => evd ? (() => { const sv = k === "__generic__" ? evd.generic : evd.sources?.[k]; return sv ? `<td class="dim">${fmt.auc(sv.blend.auroc)} / ${fmt.pct(sv.blend.recall_at_fpr["0.01"], 1)}</td>` : "<td>–</td>"; })() : "";
    const hybrid = !!ev.mean;   // v4 eval.json: per-source vae / booster / blend blocks
    const hasSeq = hybrid && Object.values(ev.sources).some((v) => v.seq);   // v5 runs carry the sequence head
    // v5 field heads (device ...): a column each, "–" on the sources without the block
    const heads = hybrid ? ["local", "device", "graph"].filter((h) => Object.values(ev.sources).some((v) => v[h])) : [];
    const headCells = (v) => heads.map((h) => `<td>${v && v[h] ? fmt.auc(v[h].auroc) : "–"}</td>`).join("");
    const row = (k, v) => {
      if (!hybrid) {
        return `<tr><td>${esc(k)}</td><td>${fmt.int(v.n)}</td><td>${fmt.int(v.fraud)}</td><td><b>${fmt.auc(v.auroc)}</b></td><td colspan="8" class="dim">v3 run</td></tr>`;
      }
      const b = v.blend, s_ = v.booster, n_ = v.vae;
      const t = b.at_val_thresholds?.["fpr_0.01"];
      return `<tr><td>${esc(k)}</td><td>${fmt.int(v.n)}</td><td>${fmt.int(v.fraud)}</td>
        <td>${fmt.auc(n_.auroc)}</td><td>${s_ ? fmt.auc(s_.auroc) : "–"}</td>${hasSeq ? `<td>${v.seq ? fmt.auc(v.seq.auroc) : "–"}</td>` : ""}${headCells(v)}<td><b>${fmt.auc(b.auroc)}</b></td><td>${fmt.auc(b.auprc)}</td>
        <td>${fmt.pct(b.recall_at_fpr["0.001"], 1)}</td><td><b>${fmt.pct(b.recall_at_fpr["0.01"], 1)}</b></td><td>${fmt.pct(b.recall_at_fpr["0.05"], 1)}</td>
        <td>${t ? `${fmt.pct(t.fpr, 2)} / ${fmt.pct(t.recall, 1)} / ${fmt.pct(t.precision, 1)}` : "–"}</td>
        <td class="dim">${fmt.pct(n_.recall_at_fpr["0.01"], 1)}${s_ ? ` / ${fmt.pct(s_.recall_at_fpr["0.01"], 1)}` : ""}</td>${slowCell(k)}</tr>`;
    };
    const g = ev.generic;
    parts.push(`<h3 style="margin-top:1.4rem">Results on data the model never saw</h3>
      <p class="dim small">The newest 20% of every dataset, kept out of training and tuning. One column per model: the <b>VAE</b> (how unusual a payment is; trained without labels), the <b>booster</b> (decision trees trained on the labelled fraud${hasSeq ? ", plus the dataset's own raw columns" : ""}),${hasSeq ? " the <b>sequence</b> model (reads the customer's previous payments)," : ""}${heads.length ? ` the ${heads.map((h) => `<b>${h === "local" ? "specialist" : h}</b>`).join(" / ")} model${heads.length > 1 ? "s" : ""} (one booster per bank, a model on the device fields, the entity graph; each only where it applies),` : ""} and the <b>blend</b> the verdict is read from (their scores weighted per bank, fit on the validation split).${ev.pooled ? ` All datasets pooled: blend AUROC ${fmt.auc(ev.pooled.blend_auroc)}, VAE ${fmt.auc(ev.pooled.auroc)}.` : ""}</p>
      <table class="metric-table"><thead><tr><th>dataset</th><th>rows</th><th>fraud</th><th>VAE AUROC</th><th>booster AUROC</th>${hasSeq ? "<th>sequence AUROC</th>" : ""}${heads.map((h) => `<th>${h === "local" ? "specialist" : h} AUROC</th>`).join("")}<th>blend AUROC</th><th>blend AUPRC</th><th>blend recall @0.1% FPR</th><th>@1%</th><th>@5%</th><th>blend at the 1% budget: FPR / recall / precision</th><th>recall @1%: VAE / booster</th>${evd ? `<th title="the same rows scored without last week's fraud history: what a bank whose fraud labels take ${evd.label_delay_days} days gets">blend with ${evd.label_delay_days}-day labels: AUROC / recall @1%</th>` : ""}</tr></thead><tbody>
      ${Object.entries(ev.sources).map(([k, v]) => row(k, v)).join("")}
      ${hybrid ? `<tr><td><b>mean</b></td><td></td><td></td><td><b>${fmt.auc(ev.mean.vae)}</b></td><td><b>${fmt.auc(ev.mean.booster)}</b></td>${hasSeq ? `<td><b>${fmt.auc(ev.mean.seq)}</b></td>` : ""}${heads.map((h) => `<td><b>${fmt.auc(ev.mean[h])}</b></td>`).join("")}<td><b>${fmt.auc(ev.mean.blend)}</b></td><td colspan="6"></td>${evd ? `<td class="dim"><b>${fmt.auc(evd.mean.blend)}</b></td>` : ""}</tr>` : ""}
      ${hybrid && g ? `<tr class="generic-row"><td><b>generic</b> <span class="dim">(bank unknown)</span></td><td>${fmt.int(g.n)}</td><td>${fmt.int(g.fraud)}</td><td>${fmt.auc(g.vae.auroc)}</td><td>${g.booster ? fmt.auc(g.booster.auroc) : "–"}</td>${hasSeq ? `<td>${g.seq ? fmt.auc(g.seq.auroc) : "–"}</td>` : ""}${headCells(g)}<td><b>${fmt.auc(g.blend.auroc)}</b></td><td>${fmt.auc(g.blend.auprc)}</td><td>${fmt.pct(g.blend.recall_at_fpr["0.001"], 1)}</td><td><b>${fmt.pct(g.blend.recall_at_fpr["0.01"], 1)}</b></td><td>${fmt.pct(g.blend.recall_at_fpr["0.05"], 1)}</td><td>${g.blend.at_val_thresholds?.["fpr_0.01"] ? `${fmt.pct(g.blend.at_val_thresholds["fpr_0.01"].fpr, 2)} / ${fmt.pct(g.blend.at_val_thresholds["fpr_0.01"].recall, 1)} / ${fmt.pct(g.blend.at_val_thresholds["fpr_0.01"].precision, 1)}` : "–"}</td><td class="dim">${fmt.pct(g.vae.recall_at_fpr["0.01"], 1)}${g.booster ? ` / ${fmt.pct(g.booster.recall_at_fpr["0.01"], 1)}` : ""}</td>${slowCell("__generic__")}</tr>` : ""}
      </tbody></table>
      <p class="dim small">Each row scores that dataset's own payments <i>with the bank named</i>, against its own idea of normal and its own thresholds, which is how a payment from a known bank is scored in the app. The <b>generic</b> row is the same test rows with the bank blanked: what a bank the model has never seen would get on day one, with no habits of its own to compare against. A real new bank would normally get a few weeks of its own traffic fitted first.</p>`);
    if (d.booster?.importance_gain) {
      parts.push(`<details><summary class="dim">which inputs the booster relies on most (top 20)</summary><div class="imp-list">${d.booster.importance_gain.slice(0, 20).map((x, i) => `<span class="imp"><i style="width:${Math.max(3, 100 * x.gain / d.booster.importance_gain[0].gain).toFixed(0)}%"></i>${esc(x.feature)}</span>`).join("")}</div></details>`);
    }
  }
  // v5: the modules' own summaries (what each learned, on what, how well)
  const mods = [];
  if (d.seqhead?.sources) mods.push(`<tr><td>customer-history model</td><td>the summary of the customer's last ${d.seqhead.window} payments, ${fmt.int(d.seqhead.train_rows)} labelled rows</td><td>val AUROC ${Object.entries(d.seqhead.sources).map(([k, v]) => `${esc(k)} ${fmt.auc(v.auroc)}`).join(", ")}</td></tr>`);
  if (d.specialists?.sources) mods.push(`<tr><td>per-bank specialists</td><td>one booster per bank with at least ${fmt.int(300)} labelled fraud rows</td><td>val AUROC ${Object.entries(d.specialists.sources).map(([k, v]) => `${esc(k)} ${fmt.auc(v.val_auroc)} (${v.best_iteration} trees)`).join(", ")}</td></tr>`);
  for (const [h, info] of Object.entries(d.heads || {})) mods.push(`<tr><td>${esc(info.label || h)} head</td><td>${(info.numeric || []).length + (info.categorical || []).length} fields, ${fmt.int(info.train_rows)} rows from ${Object.keys(info.sources || {}).join(", ")}</td><td>val AUROC ${Object.entries(info.sources || {}).map(([k, v]) => `${esc(k)} ${fmt.auc(v.auroc)}`).join(", ")}</td></tr>`);
  if (d.graph?.sources) mods.push(`<tr><td>entity graph</td><td>a graph neural network (GraphSAGE) per bank; ${Object.entries(d.graph.sources).map(([k, v]) => `${esc(k)} ${fmt.int(v.nodes_a)} nodes / ${fmt.int(v.edges_a)} edges`).join(", ")}</td><td>val AUROC ${Object.entries(d.graph.sources).map(([k, v]) => `${esc(k)} ${v.val_auroc == null ? "–" : fmt.auc(v.val_auroc)}`).join(", ")}</td></tr>`);
  if (d.text?.heads) mods.push(`<tr><td>text models</td><td>${Object.values(d.text.heads).map((v) => `${esc(v.label)}: ${fmt.int(v.n)} texts, ${fmt.pct(v.positive_rate, 1)} positive`).join("; ")}</td><td>test AUROC ${Object.entries(d.text.heads).map(([k, v]) => `${esc(k)} ${fmt.auc(v.test_auroc)}`).join(", ")}</td></tr>`);
  if (d.behaviour?.modalities) mods.push(`<tr><td>behaviour models (mouse, keystrokes, touch, motion)</td><td>${Object.entries(d.behaviour.modalities).map(([k, v]) => `${esc(k)}: ${v.train_users} users (${(v.datasets || []).join(", ")})`).join("; ")}</td><td>${Object.entries(d.behaviour.modalities).map(([k, v]) => v.open_set ? `${esc(k)} EER ${fmt.pct(v.open_set.eer, 1)} on ${v.open_set.users} unseen users` : (v.balabit ? `${esc(k)} Balabit AUROC ${fmt.auc(v.balabit.auroc)}, EER ${fmt.pct(v.balabit.eer, 1)}` : esc(k))).join("; ")}</td></tr>`);
  if (d.graph_bench) mods.push(`<tr><td>graph benchmarks</td><td>the same graph model on four public fraud networks</td><td>test AUROC ${Object.entries(d.graph_bench).map(([k, v]) => `${esc(k)} ${fmt.auc(v.auroc)}`).join(", ")}</td></tr>`);
  if (mods.length) {
    parts.push(`<h3>The extra models (v5)</h3><table class="metric-table"><thead><tr><th>model</th><th>trained on</th><th>its own result</th></tr></thead><tbody>${mods.join("")}</tbody></table>
      <p class="dim small">The customer-history model, the specialists, the device model and the entity graph feed the blend (each appears in the table above where it applies). The text and behaviour models never see a payment row: they add their evidence to the verdict only when a request carries text or a session, and their numbers come from their own public datasets.</p>`);
  }
  if (d.calibration) {
    const c = d.calibration;
    const bsrc = c.blend?.sources || c.sources;
    const isBlend = !!c.blend?.sources;
    parts.push(`<h3>Thresholds (set on the validation split)</h3>
      <p class="dim small">The VAE's score type was chosen by validation AUROC: raw ${fmt.auc(c.score_type_candidates?.raw)} against normalised ${fmt.auc(c.score_type_candidates?.norm)} → <b>${esc(c.score_type)}</b>, averaged over ${c.n_samples} samples per row.${c.blend ? ` The blend has ${c.blend.n_features} weights, fit on ${fmt.int(c.blend.fit_rows)} labelled rows (AUROC ${fmt.auc(c.blend.holdout?.auroc)} on the half it did not see). The thresholds below are on the blend's chance of fraud${c.blend.generic ? `; for an unknown bank the blend's AUROC is ${fmt.auc(c.blend.generic.auroc)}` : ""}.` : ` For an unknown bank: AUROC ${fmt.auc(c.generic?.auroc)}.`}</p>
      <table class="metric-table"><thead><tr><th>dataset</th><th>validation AUROC</th><th>typical ${isBlend ? "chance of fraud" : "score"}, legitimate</th><th>typical, fraud</th>${c.fpr_targets.map((f) => `<th>flag above, at ${(100 * f).toFixed(f < 0.01 ? 1 : 0)}% budget</th>`).join("")}<th>best F1</th></tr></thead><tbody>
      ${Object.entries(bsrc).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${fmt.auc(v.auroc)}</td><td>${isBlend ? fmt.pct(v.legit_quantiles["0.5"], 2) : fmt.num(v.legit_quantiles["0.5"], 2)}</td><td>${isBlend ? fmt.pct(v.fraud_median, 1) : fmt.num(v.fraud_median, 2)}</td>
        ${c.fpr_targets.map((f) => { const t = v.thresholds[`fpr_${f}`]; return `<td title="recall ${fmt.pct(t.recall, 1)}">${isBlend ? fmt.pct(t.threshold, 2) : fmt.num(t.threshold, 2)}</td>`; }).join("")}
        <td>${v.thresholds.best_f1 ? fmt.num(v.thresholds.best_f1.f1, 3) : "–"}</td></tr>`).join("")}</tbody></table>`);
  }
  parts.push(`<details><summary class="dim">the training log, as written while it ran</summary><pre class="log" id="run-log">loading…</pre></details>`);
  box.innerHTML = parts.join("");
  document.getElementById("btn-score-with")?.addEventListener("click", () => rememberRun(name));
  document.getElementById("btn-load-run")?.addEventListener("click", async () => {
    await post(`/api/runs/${encodeURIComponent(name)}/load`, {});
    active = name;
    await refreshMeta();
    await loadRuns();
  });
  const logEl = document.getElementById("run-log");
  logEl.parentElement.addEventListener("toggle", async () => {
    if (logEl.parentElement.open) { const l = await get(`/api/runs/${encodeURIComponent(name)}/log?tail=400`); logEl.textContent = l.lines.join("\n"); }
  }, { once: true });

  if (window.Chart && d.metrics.length) drawMetricCharts(d.metrics);
  clearTimeout(refreshTimer);
  if (st.state && st.state !== "finished") refreshTimer = setTimeout(() => showRun(name), 8000);
}

function drawMetricCharts(metrics) {
  const labels = metrics.map((m) => m.epoch);
  charts.push(new Chart(document.getElementById("m-loss"), {
    type: "line",
    data: { labels, datasets: [
      { label: "train loss", data: metrics.map((m) => m.train_loss), borderColor: palette.ink, backgroundColor: palette.ink, tension: 0.2, pointRadius: 2 },
      { label: "reconstruction NLL", data: metrics.map((m) => m.train_nll), borderColor: palette.accent, backgroundColor: palette.accent, tension: 0.2, pointRadius: 2 },
      { label: "KL", data: metrics.map((m) => m.train_kl), borderColor: palette.amber, backgroundColor: palette.amber, tension: 0.2, pointRadius: 2 },
      { label: "val legit −ELBO", data: metrics.map((m) => m.val_legit_score), borderColor: palette.ink3, borderDash: [4, 3], backgroundColor: palette.ink3, tension: 0.2, pointRadius: 2 },
    ] },
    options: { animation: false, responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } },
      scales: { x: { title: { display: true, text: "epoch" } }, y: { grid: { color: "#eef1f4" } } } },
  }));
  const sources = new Set();
  metrics.forEach((m) => Object.keys(m.val_per_source || {}).forEach((k) => sources.add(k)));
  charts.push(new Chart(document.getElementById("m-auc"), {
    type: "line",
    data: { labels, datasets: [...sources].map((s) => ({ label: s, data: metrics.map((m) => m.val_per_source?.[s]?.auroc ?? null), borderColor: palette.sources[s] || palette.ink2, backgroundColor: palette.sources[s] || palette.ink2, tension: 0.2, pointRadius: 2, spanGaps: true })) },
    options: { animation: false, responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } },
      scales: { x: { title: { display: true, text: "epoch" } }, y: { min: 0.4, max: 1, grid: { color: "#eef1f4" } } } },
  }));
}

async function startTraining(ev) {
  ev.preventDefault();
  const fd = new FormData(ev.target);
  const body = { config: fd.get("config"), prepared: fd.get("prepared"), name: fd.get("name") || null, epochs: fd.get("epochs") ? parseInt(fd.get("epochs"), 10) : null };
  const log = document.getElementById("train-log");
  log.textContent = "starting…";
  try {
    const { task, run } = await post("/api/train/local", body);
    selected = run;
    await loadRuns();
    await pollTask(task, (t) => {
      log.textContent = t.lines.slice(-200).join("\n"); log.scrollTop = log.scrollHeight;
    }, 3000);
    await refreshMeta();
    await loadRuns();
  } catch (e) {
    log.textContent += "\n" + e.message;
  }
}
