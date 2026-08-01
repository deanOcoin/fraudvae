// model & training tab: run list, run detail (metrics, calibration, eval), local training
import { get, post, pollTask, fmt, el, esc, palette } from "./api.js";
import { refreshMeta } from "./app.js";

let runs = [];
let active = null;
let schemaVersion = null;
function state_schema_version() { return schemaVersion ?? "?"; }
let selected = null;
let charts = [];
let refreshTimer = null;

export async function initModel(state) {
  schemaVersion = state.meta?.schema_version ?? null;
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
}

async function loadRuns() {
  const r = await get("/api/runs");
  runs = r.runs; active = r.active;
  renderRuns();
  if (!selected && runs.length) showRun(active || runs[0].name);
  else if (selected) showRun(selected, false);
}

function stateOf(r) {
  const st = r.status?.state;
  if (st === "running" || st === "boosting" || st === "calibrating" || st === "evaluating") return st;
  if (r.name === active) return "loaded";
  if (r.compatible === false) return `schema v${r.schema_version ?? "?"}`;
  if (r.has_eval) return "evaluated";
  if (r.has_model) return "checkpoint";
  return "empty";
}
const RUNNING = new Set(["running", "boosting", "calibrating", "evaluating"]);

function renderRuns() {
  const host = document.getElementById("runs-list");
  host.innerHTML = "";
  if (!runs.length) { host.innerHTML = "<p class='dim'>No runs yet. Train one below.</p>"; return; }
  for (const r of runs) {
    const s = stateOf(r);
    const dev = r.run?.device_name || r.run?.device || "";
    const meta = [r.config_name, r.run?.prepared ? `set ${r.run.prepared}` : null, dev, r.remote ? "fetched from remote" : null].filter(Boolean).join(", ");
    const mean = r.mean || {};
    const sub = [];
    if (mean.vae != null && mean.blend != null) sub.push(`vae ${fmt.auc(mean.vae)}${mean.booster != null ? ` · booster ${fmt.auc(mean.booster)}` : ""}`);
    if (r.generic_test_auroc != null || r.generic_val_auroc != null) sub.push(`generic ${fmt.auc(r.generic_test_auroc ?? r.generic_val_auroc)}`);
    host.append(el("div", { class: "run-row" + (r.name === selected ? " active" : "") + (r.compatible === false ? " stale" : ""), onclick: () => showRun(r.name) },
      el("div", { class: "top" }, el("span", { class: "name" }, r.name),
        el("span", { class: "auc" }, r.mean_test_auroc != null ? "AUROC " + fmt.auc(r.mean_test_auroc) : "",
          sub.length ? el("span", { class: "generic", title: "headline = blend (or the VAE alone for runs without a booster). generic = a bank the model never trained on, source token blanked" }, sub.join("  ·  ")) : "")),
      el("div", { class: "meta" }, el("span", { class: "state " + (RUNNING.has(s) ? "running" : s === "loaded" ? "loaded" : "") }, s), " ", meta)));
  }
}

function destroyCharts() { charts.forEach((c) => c.destroy()); charts = []; }

async function showRun(name, rerender = true) {
  selected = name;
  if (rerender) renderRuns();
  const box = document.getElementById("run-detail");
  let d;
  try { d = await get(`/api/runs/${encodeURIComponent(name)}/metrics`); } catch (e) { box.innerHTML = `<p class="dim">${esc(e.message)}</p>`; return; }
  destroyCharts();
  const r = runs.find((x) => x.name === name) || {};
  const run = d.run || {};
  const cfg = run.config || {};
  const parts = [];
  const stale = d.compatible === false;
  parts.push(`<div class="card-head"><h2>${esc(name)}</h2><div class="row">
    ${r.has_model && !stale ? `<button class="btn ${name === active ? "ghost" : "primary"}" id="btn-load-run" ${name === active ? "disabled" : ""}>${name === active ? "Loaded in scorer" : "Use for scoring"}</button>` : ""}
  </div></div>`);
  if (stale) parts.push(`<div class="notice warn">This run was trained on schema v${d.schema_version ?? "?"}; the code is now v${state_schema_version()}. It stays here for the record but cannot be loaded. Rebuild, prepare and train again to get a comparable run.</div>`);
  const st = d.status || {};
  if (st.state && st.state !== "finished") {
    const prog = st.progress != null ? ` ${(100 * st.progress).toFixed(0)}% of epoch` : "";
    parts.push(`<div class="notice warn">${esc(st.state)} – epoch ${st.epoch ?? "?"} / ${st.epochs ?? "?"}${prog}${st.rows_per_sec ? `, ${fmt.int(st.rows_per_sec)} rows/s` : ""}${st.loss ? `, loss ${fmt.num(st.loss, 3)}` : ""}</div>`);
  }
  parts.push(`<dl class="kv">
    <dt>started</dt><dd>${esc(run.started || "–")}${run.host ? ` on ${esc(run.host)}` : ""}</dd>
    <dt>device</dt><dd>${esc(run.device_name || run.device || "–")}</dd>
    <dt>config</dt><dd>${esc(cfg.name || "–")}: latent ${cfg.model?.latent_dim}, hidden ${(cfg.model?.hidden || []).join("-")}, batch ${cfg.train?.batch_size}, lr ${cfg.train?.lr}, β ${cfg.train?.beta} (warmup ${cfg.train?.beta_warmup_epochs} ep), source dropout ${cfg.train?.source_dropout}</dd>
    <dt>params</dt><dd>${run.params ? (run.params / 1e6).toFixed(2) + "M" : "–"}${d.booster ? ` VAE + ${d.booster.best_iteration}-tree booster (${d.booster.params?.num_leaves} leaves, ${d.booster.features?.length} features, ${fmt.int(d.booster.train_rows)} labelled rows, ${d.booster.seconds}s)` : ""}</dd>
    <dt>data</dt><dd>${run.prepared_meta ? `set <code>${esc(run.prepared)}</code>: ${fmt.int(run.prepared_meta.counts?.train)} train / ${fmt.int(run.prepared_meta.counts?.val)} val / ${fmt.int(run.prepared_meta.counts?.test)} test rows, sources ${(run.prepared_meta.sources || []).join(", ")}${run.prepared_meta.max_rows_per_source ? `, train capped at ${fmt.int(run.prepared_meta.max_rows_per_source)} rows/source` : ""}` : "–"}</dd>
    ${d.remote ? `<dt>remote</dt><dd>fetched from ${esc(d.remote.user)}@${esc(d.remote.host)}:${d.remote.port} ${fmt.ago(d.remote.fetched)}</dd>` : ""}
  </dl>`);

  if (d.metrics.length) {
    parts.push(`<div class="chart-pair">
      <div class="chart-box"><h4>training loss (−ELBO per row) and its parts</h4><div class="chart-wrap" style="height:200px"><canvas id="m-loss"></canvas></div></div>
      <div class="chart-box"><h4>validation AUROC per source (raw score, subsample)</h4><div class="chart-wrap" style="height:200px"><canvas id="m-auc"></canvas></div></div></div>`);
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
    parts.push(`<h3 style="margin-top:1.4rem">Held-out test set</h3>
      <p class="dim small">newest 20% of every source, never seen in training or calibration. Three scores per row: the <b>VAE</b> novelty score (${esc(ev.score_type)}, never saw a label), the <b>booster</b> (gradient-boosted trees on the labelled train split${hasSeq ? " plus the source's raw columns" : ""}),${hasSeq ? " the <b>sequence</b> head (a classifier on the history transformer's read-out of the customer's past)," : ""}${heads.length ? ` the ${heads.map((h) => `<b>${h === "local" ? "specialist" : h}</b>`).join(" / ")} signal${heads.length > 1 ? "s" : ""} (a per-bank booster, a head on the device block, the entity graph; each scored only where it applies),` : ""} and the <b>blend</b> the app flags on (per-bank logistic combination of the booster's logit, the VAE score, its KL term${hasSeq ? " and the sequence logit" : ""}, fit on validation).${ev.pooled ? ` Pooled over all sources: blend AUROC ${fmt.auc(ev.pooled.blend_auroc)}, VAE ${fmt.auc(ev.pooled.auroc)}.` : ""}</p>
      <table class="metric-table"><thead><tr><th>source</th><th>rows</th><th>fraud</th><th>VAE AUROC</th><th>booster AUROC</th>${hasSeq ? "<th>sequence AUROC</th>" : ""}${heads.map((h) => `<th>${h === "local" ? "specialist" : h} AUROC</th>`).join("")}<th>blend AUROC</th><th>blend AUPRC</th><th>blend recall @0.1% FPR</th><th>@1%</th><th>@5%</th><th>blend at 1% budget: FPR / recall / precision</th><th>r@1%: VAE / booster</th>${evd ? `<th title="the same rows scored with the one-week fraud-history block blanked: what a deployment whose fraud labels take ${evd.label_delay_days} days gets">blend with ${evd.label_delay_days}-day labels: AUROC / r@1%</th>` : ""}</tr></thead><tbody>
      ${Object.entries(ev.sources).map(([k, v]) => row(k, v)).join("")}
      ${hybrid ? `<tr><td><b>mean</b></td><td></td><td></td><td><b>${fmt.auc(ev.mean.vae)}</b></td><td><b>${fmt.auc(ev.mean.booster)}</b></td>${hasSeq ? `<td><b>${fmt.auc(ev.mean.seq)}</b></td>` : ""}${heads.map((h) => `<td><b>${fmt.auc(ev.mean[h])}</b></td>`).join("")}<td><b>${fmt.auc(ev.mean.blend)}</b></td><td colspan="6"></td>${evd ? `<td class="dim"><b>${fmt.auc(evd.mean.blend)}</b></td>` : ""}</tr>` : ""}
      ${hybrid && g ? `<tr class="generic-row"><td><b>generic</b> <span class="dim">(bank token blanked)</span></td><td>${fmt.int(g.n)}</td><td>${fmt.int(g.fraud)}</td><td>${fmt.auc(g.vae.auroc)}</td><td>${g.booster ? fmt.auc(g.booster.auroc) : "–"}</td>${hasSeq ? `<td>${g.seq ? fmt.auc(g.seq.auroc) : "–"}</td>` : ""}${headCells(g)}<td><b>${fmt.auc(g.blend.auroc)}</b></td><td>${fmt.auc(g.blend.auprc)}</td><td>${fmt.pct(g.blend.recall_at_fpr["0.001"], 1)}</td><td><b>${fmt.pct(g.blend.recall_at_fpr["0.01"], 1)}</b></td><td>${fmt.pct(g.blend.recall_at_fpr["0.05"], 1)}</td><td>${g.blend.at_val_thresholds?.["fpr_0.01"] ? `${fmt.pct(g.blend.at_val_thresholds["fpr_0.01"].fpr, 2)} / ${fmt.pct(g.blend.at_val_thresholds["fpr_0.01"].recall, 1)} / ${fmt.pct(g.blend.at_val_thresholds["fpr_0.01"].precision, 1)}` : "–"}</td><td class="dim">${fmt.pct(g.vae.recall_at_fpr["0.01"], 1)}${g.booster ? ` / ${fmt.pct(g.booster.recall_at_fpr["0.01"], 1)}` : ""}</td>${slowCell("__generic__")}</tr>` : ""}
      </tbody></table>
      <p class="dim small">Each bank row scores that bank's own transactions <i>with the bank named</i>, i.e. against its own baselines and thresholds, which is how a transaction from any known context is scored in the app. The <b>generic</b> row is the same test rows with the bank token blanked and the generic thresholds: a bank the model has never seen, with no baseline of its own. It is the floor rather than a deployment number; a new bank would normally get a few weeks of its own traffic fitted as a new context first.</p>`);
    if (d.booster?.importance_gain) {
      parts.push(`<details><summary class="dim">what the booster leans on (gain importance, top 20)</summary><div class="imp-list">${d.booster.importance_gain.slice(0, 20).map((x, i) => `<span class="imp"><i style="width:${Math.max(3, 100 * x.gain / d.booster.importance_gain[0].gain).toFixed(0)}%"></i>${esc(x.feature)}</span>`).join("")}</div></details>`);
    }
  }
  // v5: the modules' own summaries (what each learned, on what, how well)
  const mods = [];
  if (d.seqhead?.sources) mods.push(`<tr><td>sequence head</td><td>history read-out of the VAE (window ${d.seqhead.window}), ${fmt.int(d.seqhead.train_rows)} labelled rows</td><td>val AUROC ${Object.entries(d.seqhead.sources).map(([k, v]) => `${esc(k)} ${fmt.auc(v.auroc)}`).join(", ")}</td></tr>`);
  if (d.specialists?.sources) mods.push(`<tr><td>bank specialists</td><td>one booster per bank with ≥ ${fmt.int(300)} labelled fraud rows</td><td>val AUROC ${Object.entries(d.specialists.sources).map(([k, v]) => `${esc(k)} ${fmt.auc(v.val_auroc)} (${v.best_iteration} trees)`).join(", ")}</td></tr>`);
  for (const [h, info] of Object.entries(d.heads || {})) mods.push(`<tr><td>${esc(info.label || h)} head</td><td>${(info.numeric || []).length + (info.categorical || []).length} fields, ${fmt.int(info.train_rows)} rows from ${Object.keys(info.sources || {}).join(", ")}</td><td>val AUROC ${Object.entries(info.sources || {}).map(([k, v]) => `${esc(k)} ${fmt.auc(v.auroc)}`).join(", ")}</td></tr>`);
  if (d.graph?.sources) mods.push(`<tr><td>entity graph</td><td>GraphSAGE per bank; ${Object.entries(d.graph.sources).map(([k, v]) => `${esc(k)} ${fmt.int(v.nodes_a)} nodes / ${fmt.int(v.edges_a)} edges`).join(", ")}</td><td>val AUROC ${Object.entries(d.graph.sources).map(([k, v]) => `${esc(k)} ${v.val_auroc == null ? "–" : fmt.auc(v.val_auroc)}`).join(", ")}</td></tr>`);
  if (d.text?.heads) mods.push(`<tr><td>text heads</td><td>${Object.values(d.text.heads).map((v) => `${esc(v.label)}: ${fmt.int(v.n)} texts, ${fmt.pct(v.positive_rate, 1)} positive`).join("; ")}</td><td>test AUROC ${Object.entries(d.text.heads).map(([k, v]) => `${esc(k)} ${fmt.auc(v.test_auroc)}`).join(", ")}</td></tr>`);
  if (d.behaviour?.modalities) mods.push(`<tr><td>behaviour verifiers</td><td>${Object.entries(d.behaviour.modalities).map(([k, v]) => `${esc(k)}: ${v.train_users} users (${(v.datasets || []).join(", ")})`).join("; ")}</td><td>${Object.entries(d.behaviour.modalities).map(([k, v]) => v.open_set ? `${esc(k)} EER ${fmt.pct(v.open_set.eer, 1)} on ${v.open_set.users} unseen users` : (v.balabit ? `${esc(k)} Balabit AUROC ${fmt.auc(v.balabit.auroc)}, EER ${fmt.pct(v.balabit.eer, 1)}` : esc(k))).join("; ")}</td></tr>`);
  if (d.graph_bench) mods.push(`<tr><td>graph benchmarks</td><td>the same encoder on the public fraud graphs</td><td>test AUROC ${Object.entries(d.graph_bench).map(([k, v]) => `${esc(k)} ${fmt.auc(v.auroc)}`).join(", ")}</td></tr>`);
  if (mods.length) {
    parts.push(`<h3>v5 modules</h3><table class="metric-table"><thead><tr><th>module</th><th>trained on</th><th>its own number</th></tr></thead><tbody>${mods.join("")}</tbody></table>
      <p class="dim small">The sequence head, the specialists, the device head and the graph are blend signals with presence bits (each is in the test table above when it fired). The text heads and the behaviour verifiers never see a transaction row: their calibrated likelihood ratios are fused into the verdict when a request carries text or a session, and the numbers here are from their own public datasets.</p>`);
  }
  if (d.calibration) {
    const c = d.calibration;
    const bsrc = c.blend?.sources || c.sources;
    const isBlend = !!c.blend?.sources;
    parts.push(`<h3>Calibration (validation split)</h3>
      <p class="dim small">VAE score type chosen by mean validation AUROC: raw ${fmt.auc(c.score_type_candidates?.raw)} vs normalised ${fmt.auc(c.score_type_candidates?.norm)} → <b>${esc(c.score_type)}</b>, ${c.n_samples} posterior samples per row.${c.blend ? ` Blender: ${c.blend.n_features} weights fit on ${fmt.int(c.blend.fit_rows)} labelled rows (held-out half AUROC ${fmt.auc(c.blend.holdout?.auroc)}). Thresholds below are on the blend's p(fraud)${c.blend.generic ? `; generic context blend AUROC ${fmt.auc(c.blend.generic.auroc)}` : ""}.` : ` Generic context AUROC ${fmt.auc(c.generic?.auroc)}.`}</p>
      <table class="metric-table"><thead><tr><th>source</th><th>val AUROC</th><th>legit ${isBlend ? "p(fraud)" : "score"} median</th><th>fraud median</th>${c.fpr_targets.map((f) => `<th>thr @${(100 * f).toFixed(f < 0.01 ? 1 : 0)}%</th>`).join("")}<th>best F1</th></tr></thead><tbody>
      ${Object.entries(bsrc).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${fmt.auc(v.auroc)}</td><td>${isBlend ? fmt.pct(v.legit_quantiles["0.5"], 2) : fmt.num(v.legit_quantiles["0.5"], 2)}</td><td>${isBlend ? fmt.pct(v.fraud_median, 1) : fmt.num(v.fraud_median, 2)}</td>
        ${c.fpr_targets.map((f) => { const t = v.thresholds[`fpr_${f}`]; return `<td title="recall ${fmt.pct(t.recall, 1)}">${isBlend ? fmt.pct(t.threshold, 2) : fmt.num(t.threshold, 2)}</td>`; }).join("")}
        <td>${v.thresholds.best_f1 ? fmt.num(v.thresholds.best_f1.f1, 3) : "–"}</td></tr>`).join("")}</tbody></table>`);
  }
  parts.push(`<details><summary class="dim">training log</summary><pre class="log" id="run-log">loading…</pre></details>`);
  box.innerHTML = parts.join("");
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
  if (st.state && st.state !== "finished") refreshTimer = setTimeout(() => showRun(name, false), 8000);
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
