// compute tab: advisor, ssh key, remote provisioning / training / fetching
import { get, post, api, pollTask, fmt, el, esc } from "./api.js";
import { refreshMeta } from "./app.js";

let hosts = { remembered: [], ssh_config: [], jobs: [] };
let autoTimer = null;

export async function initCompute(state) {
  loadKey();
  await loadHosts();
  const cfgs = await get("/api/configs");
  const sel = document.getElementById("launch-config");
  for (const c of cfgs.configs) sel.append(el("option", { value: c.name }, c.name));
  sel.value = cfgs.configs.find((c) => c.name === "gpu_full") ? "gpu_full" : cfgs.configs[0]?.name;

  document.getElementById("btn-advise").addEventListener("click", advise);
  document.getElementById("copy-key").addEventListener("click", async () => {
    await navigator.clipboard.writeText(document.getElementById("pubkey").textContent.trim());
    document.getElementById("copy-key").textContent = "Copied";
    setTimeout(() => (document.getElementById("copy-key").textContent = "Copy public key"), 1500);
  });
  document.getElementById("btn-paste-ssh").addEventListener("click", pasteSsh);
  document.getElementById("host-select").addEventListener("change", pickHost);
  document.getElementById("btn-probe").addEventListener("click", probe);
  document.getElementById("btn-provision").addEventListener("click", provision);
  document.getElementById("btn-launch").addEventListener("click", launch);
  document.getElementById("btn-status").addEventListener("click", status);
  document.getElementById("btn-stop").addEventListener("click", stop);
  document.getElementById("btn-fetch-run").addEventListener("click", fetchRun);
  document.getElementById("auto-status").addEventListener("change", (e) => {
    clearInterval(autoTimer);
    if (e.target.checked) { status(); autoTimer = setInterval(status, 20000); }
  });
  // "ssh -p 1234 root@1.2.3.4" pasted anywhere in the host field also works
  document.querySelector("#host-form input[name=host]").addEventListener("blur", (e) => parseSshLine(e.target.value));
}

async function loadKey() {
  try {
    const k = await get("/api/remote/key");
    document.getElementById("pubkey").textContent = k.public_key;
    document.getElementById("key-fp").textContent = `${k.fingerprint}  (${k.path})`;
  } catch (e) {
    document.getElementById("pubkey").textContent = e.message;
  }
}

async function loadHosts() {
  hosts = await get("/api/remote/hosts");
  const sel = document.getElementById("host-select");
  sel.innerHTML = "<option value=''>—</option>";
  for (const h of hosts.remembered) sel.append(el("option", { value: `r:${h.host}|${h.port}|${h.user}` }, `${h.label || h.user + "@" + h.host + ":" + h.port}${h.fingerprint ? "" : " (never connected)"}`));
  for (const h of hosts.ssh_config) sel.append(el("option", { value: `c:${h.alias}` }, `~/.ssh/config: ${h.alias}`));
  const runSel = document.getElementById("remote-run-select");
  runSel.innerHTML = "";
  for (const j of hosts.jobs) runSel.append(el("option", { value: j.run }, `${j.run}  (${j.state}, ${j.host})`));
  if (hosts.remembered.length && !document.querySelector("#host-form input[name=host]").value) {
    fillHost(hosts.remembered[0]);
  }
}

function fillHost(h) {
  const f = document.getElementById("host-form");
  f.elements.host.value = h.host || "";
  f.elements.port.value = h.port || 22;
  f.elements.user.value = h.user || "root";
  if (h.remote_dir) f.elements.remote_dir.value = h.remote_dir;
}

function pickHost(e) {
  const v = e.target.value;
  if (!v) return;
  if (v.startsWith("r:")) {
    const [host, port, user] = v.slice(2).split("|");
    const h = hosts.remembered.find((x) => x.host === host && String(x.port) === port && x.user === user);
    if (h) fillHost(h);
  } else {
    const h = hosts.ssh_config.find((x) => x.alias === v.slice(2));
    if (h) fillHost({ host: h.host, port: h.port, user: h.user || "root" });
  }
}

function parseSshLine(text) {
  const m = text.match(/ssh\s+(?:-p\s*(\d+)\s+)?(?:(\w[\w.-]*)@)?([\w.-]+)(?:\s+-p\s*(\d+))?/);
  if (!m) return false;
  const f = document.getElementById("host-form");
  f.elements.host.value = m[3];
  f.elements.port.value = m[1] || m[4] || 22;
  if (m[2]) f.elements.user.value = m[2];
  return true;
}

async function pasteSsh() {
  try {
    const txt = await navigator.clipboard.readText();
    if (!parseSshLine(txt)) alert("clipboard didn't look like an ssh line (ssh -p PORT user@host)");
  } catch (e) { alert("clipboard not readable: " + e.message); }
}

function hostBody(extra = {}) {
  const f = document.getElementById("host-form");
  return { host: f.elements.host.value.trim(), port: parseInt(f.elements.port.value, 10) || 22, user: f.elements.user.value.trim() || "root",
    remote_dir: f.elements.remote_dir.value.trim() || "~/fraudvae", ...extra };
}

// ---- advisor --------------------------------------------------------------------

async function advise() {
  const btn = document.getElementById("btn-advise");
  const body = document.getElementById("advisor-body");
  btn.disabled = true; btn.classList.add("spin");
  try {
    const cfg = document.getElementById("launch-config").value || "gpu_full";
    const a = await get(`/api/remote/advise?config=${encodeURIComponent(cfg)}`);
    const loc = a.local;
    body.innerHTML = `
      <p class="dim" style="margin:0">Config <b>${esc(a.config)}</b>, ${a.epochs} epochs over ${fmt.int(a.train_rows)} legitimate training rows (${a.corpus.sources ? Object.keys(a.corpus.sources).length : 0} sources built, ${fmt.bytes(a.corpus.parquet_bytes)} of parquet).</p>
      <div class="verdict ${a.recommend_remote ? "remote" : "local"}">${a.recommend_remote ? "Rent a GPU for this one." : "This machine can do it."}</div>
      <p>This machine (${esc(loc.device)}, ${esc(loc.device_name)}) does <b>${fmt.int(loc.rows_per_sec)} rows/s</b> on the real model → roughly <b>${fmt.hours(loc.est_hours)}</b> end to end. Anything over ${a.threshold_hours} h we send away.</p>
      <table class="tier-table"><thead><tr><th>GPU</th><th>rows/s</th><th>time</th><th>typical cost</th></tr></thead><tbody>
        ${a.gpu_tiers.map((t) => `<tr><td>${esc(t.name)}</td><td>${fmt.int(t.rows_per_sec)}</td><td>${fmt.hours(t.est_hours)}</td><td>$${t.est_cost_usd.toFixed(2)} at $${t.typical_usd_hr}/h</td></tr>`).join("")}
      </tbody></table>
      <h3 style="margin-top:1rem">What to rent</h3>
      <ul class="req-list">
        <li>${esc(a.requirements.gpu)}</li>
        <li>RAM ≥ ${a.requirements.ram_gb_min} GB, disk ≥ ${a.requirements.disk_gb_min} GB (you upload ${a.requirements.upload_gb} GB of parquet, the prepared arrays are built on the box)</li>
        <li>${esc(a.requirements.image)}; CUDA ${esc(a.requirements.cuda)}</li>
        <li>${esc(a.requirements.ssh)}</li>
      </ul>
      <h3 style="margin-top:1rem">vast.ai in four steps</h3>
      <ol class="req-list">${a.vast_ai_notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ol>`;
  } catch (e) {
    body.innerHTML = `<div class="notice err">${esc(e.message)}</div>`;
  } finally { btn.disabled = false; btn.classList.remove("spin"); }
}

// ---- remote ---------------------------------------------------------------------

function setLog(lines) {
  const log = document.getElementById("remote-log");
  log.textContent = Array.isArray(lines) ? lines.join("\n") : lines;
  log.scrollTop = log.scrollHeight;
}

async function probe() {
  const out = document.getElementById("probe-result");
  const btn = document.getElementById("btn-probe");
  btn.disabled = true; btn.classList.add("spin");
  out.innerHTML = "";
  try {
    const p = await post("/api/remote/probe", hostBody());
    const gpus = p.gpus.length ? p.gpus.map((g) => `${g.name} (${Math.round(g.memory_total_mb / 1024)} GB, ${g.memory_used_mb} MB in use)`).join("; ") : "none visible";
    out.innerHTML = `
      <p class="okline">Connected. Host key ${esc(p.fingerprint)}${p.first_seen ? " (first time seeing this host, now pinned)" : ""}.</p>
      <div class="grid">
        <div><b>${esc(p.hostname || "?")}</b>${esc(p.os || "")}</div>
        <div><b>${esc(gpus)}</b>GPU</div>
        <div><b>${p.cpus ?? "?"} cores, ${p.ram_gb ?? "?"} GB RAM</b>${p.disk_free_gb ?? "?"} GB free</div>
        <div><b>${esc(p.python || "no python")}</b>${p.python_version ? "python " + esc(p.python_version) : ""}</div>
        <div><b>${p.torch ? `torch ${esc(p.torch.torch)}` : "no torch"}</b>${p.torch ? (p.torch.cuda ? "CUDA " + esc(p.torch.cuda_version || "") + " available" : "no CUDA") : "will be installed"}</div>
        <div><b>${p.fraudvae_installed ? "fraudvae " + esc(p.fraudvae_installed) : "fraudvae not there yet"}</b>remote dir ${esc(p.remote_dir)}</div>
      </div>
      ${p.warnings.map((w) => `<p class="warn">${esc(w)}</p>`).join("")}`;
    await loadHosts();
  } catch (e) {
    out.innerHTML = `<p class="err">${esc(e.message)}</p>`;
  } finally { btn.disabled = false; btn.classList.remove("spin"); }
}

async function provision() {
  const btn = document.getElementById("btn-provision");
  btn.disabled = true; btn.classList.add("spin");
  setLog("provisioning…");
  try {
    const { task } = await post("/api/remote/provision", hostBody({
      prepared_name: document.getElementById("prov-prepared").value || "full",
      // with raw downloads the box builds its own processed data, so the laptop's parquet stays home
      sync_data: document.getElementById("prov-data").checked && !document.getElementById("prov-raw").checked,
      // v5: raw downloads -> the box builds the processed data itself
      sync_raw: document.getElementById("prov-raw").checked,
      fetch_direct: document.getElementById("prov-raw").checked,
      run_build: document.getElementById("prov-raw").checked,
      raw_dir: document.getElementById("prov-raw-dir").value || null,
    }));
    await pollTask(task, (t) => setLog(t.lines), 2500);
    document.getElementById("launch-prepared").value = document.getElementById("prov-prepared").value || "full";
    await loadHosts();
  } catch (e) {
    setLog(document.getElementById("remote-log").textContent + "\n" + e.message);
  } finally { btn.disabled = false; btn.classList.remove("spin"); }
}

async function launch() {
  const btn = document.getElementById("btn-launch");
  btn.disabled = true; btn.classList.add("spin");
  try {
    const epochs = parseInt(document.getElementById("launch-epochs").value, 10);
    const job = await post("/api/remote/launch", hostBody({
      config: document.getElementById("launch-config").value,
      prepared: document.getElementById("launch-prepared").value || "full",
      run_name: document.getElementById("launch-name").value || null,
      epochs: Number.isFinite(epochs) ? epochs : null,
      stage: document.getElementById("launch-stage")?.value || "train",
    }));
    setLog(`launched ${job.run} on ${job.host} (pid ${job.pid}${job.stage === "post" ? ", post-training stage only" : ""}). refresh status in a minute.`);
    await loadHosts();
    document.getElementById("remote-run-select").value = job.run;
    document.getElementById("auto-status").checked = true;
    document.getElementById("auto-status").dispatchEvent(new Event("change"));
  } catch (e) {
    setLog(e.message);
  } finally { btn.disabled = false; btn.classList.remove("spin"); }
}

async function status() {
  const run = document.getElementById("remote-run-select").value;
  if (!run) { document.getElementById("remote-status").innerHTML = "<p class='dim'>no remote run selected</p>"; return; }
  const box = document.getElementById("remote-status");
  try {
    const s = await post("/api/remote/status", hostBody({ run_name: run }));
    const st = s.status || {};
    const last = s.metrics[s.metrics.length - 1];
    const prog = st.epochs ? ((st.epoch || 0) + (st.progress || 0)) / st.epochs : 0;
    box.innerHTML = `
      <p><b>${esc(run)}</b>: ${s.alive ? "process alive" : "process not running"}, state <b>${esc(st.state || "unknown")}</b>, epoch ${st.epoch ?? "?"} / ${st.epochs ?? "?"}${st.rows_per_sec ? `, ${fmt.int(st.rows_per_sec)} rows/s` : ""}${st.elapsed ? `, ${fmt.hours(st.elapsed / 3600)} elapsed` : ""}</p>
      <div class="bar"><i style="width:${(100 * Math.min(prog, 1)).toFixed(1)}%"></i></div>
      ${last ? `<p class="dim small">last epoch: loss ${fmt.num(last.train_loss, 3)}, val AUROC mean ${fmt.auc(last.val_mean_auroc)} (${Object.entries(last.val_per_source || {}).map(([k, v]) => `${k} ${fmt.auc(v.auroc)}`).join(", ")})</p>` : ""}
      ${st.state === "finished" ? `<p class="okline">Finished. Press <b>Fetch results &amp; load model</b>.</p>` : ""}`;
    setLog(s.log || []);
    if (st.state === "finished") { clearInterval(autoTimer); document.getElementById("auto-status").checked = false; }
  } catch (e) {
    box.innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

async function stop() {
  const run = document.getElementById("remote-run-select").value;
  if (!run || !confirm(`stop ${run}?`)) return;
  try { const r = await post("/api/remote/stop", hostBody({ run_name: run })); setLog(r.stopped ? "stop signal sent" : "nothing to stop"); }
  catch (e) { setLog(e.message); }
}

async function fetchRun() {
  const run = document.getElementById("remote-run-select").value;
  if (!run) return;
  const btn = document.getElementById("btn-fetch-run");
  btn.disabled = true; btn.classList.add("spin");
  try {
    const { task } = await post("/api/remote/fetch", hostBody({ run_name: run, load: true,
      processed: document.getElementById("fetch-processed")?.checked || false }));
    await pollTask(task, (t) => setLog(t.lines), 2000);
    await refreshMeta();
  } catch (e) { setLog(e.message); }
  finally { btn.disabled = false; btn.classList.remove("spin"); }
}
