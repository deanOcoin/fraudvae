// compute tab: advisor, ssh key, remote provisioning / training / fetching
import { get, post, pollTask, fmt, el, esc } from "./api.js";
import { refreshMeta } from "./app.js";

let hosts = { remembered: [], jobs: [] };

export async function initCompute(state) {
  loadKey();
  await loadHosts();
  const cfgs = await get("/api/configs");
  const sel = document.getElementById("launch-config");
  for (const c of cfgs.configs) sel.append(el("option", { value: c.name }, c.name));
  sel.value = "gpu_full";

  document.getElementById("btn-advise").addEventListener("click", advise);
  document.getElementById("copy-key").addEventListener("click", async () => {
    await navigator.clipboard.writeText(document.getElementById("pubkey").textContent.trim());
  });
  document.getElementById("host-select").addEventListener("change", pickHost);
  document.getElementById("btn-probe").addEventListener("click", probe);
  document.getElementById("btn-provision").addEventListener("click", provision);
  document.getElementById("btn-launch").addEventListener("click", launch);
  document.getElementById("btn-status").addEventListener("click", status);
  document.getElementById("btn-stop").addEventListener("click", stop);
  document.getElementById("btn-fetch-run").addEventListener("click", fetchRun);
}

async function loadKey() {
  try {
    const k = await get("/api/remote/key");
    document.getElementById("pubkey").textContent = k.public_key;
    document.getElementById("key-fp").textContent = k.fingerprint;
  } catch (e) {
    document.getElementById("pubkey").textContent = e.message;
  }
}

async function loadHosts() {
  hosts = await get("/api/remote/hosts");
  const sel = document.getElementById("host-select");
  sel.innerHTML = "<option value=''>—</option>";
  hosts.remembered.forEach((h, i) => sel.append(el("option", { value: String(i) }, `${h.user}@${h.host}:${h.port}`)));
  const runSel = document.getElementById("remote-run-select");
  runSel.innerHTML = "";
  for (const j of hosts.jobs) runSel.append(el("option", { value: j.run }, `${j.run}  (${j.state})`));
}

function pickHost(e) {
  const h = hosts.remembered[parseInt(e.target.value, 10)];
  if (!h) return;
  const f = document.getElementById("host-form");
  f.elements.host.value = h.host;
  f.elements.port.value = h.port;
  f.elements.user.value = h.user;
}

function hostBody(extra = {}) {
  const f = document.getElementById("host-form");
  return { host: f.elements.host.value.trim(), port: parseInt(f.elements.port.value, 10) || 22, user: f.elements.user.value.trim() || "root", ...extra };
}

// ---- advisor --------------------------------------------------------------------

async function advise() {
  const body = document.getElementById("advisor-body");
  try {
    const a = await get("/api/remote/advise?config=gpu_full");
    body.innerHTML = `
      <div class="verdict ${a.recommend_remote ? "remote" : "local"}">${a.recommend_remote ? "Rent a GPU for this one." : "This machine can do it."}</div>
      <p>This machine does <b>${fmt.int(a.local.rows_per_sec)} rows/s</b> → roughly <b>${fmt.hours(a.local.est_hours)}</b>.</p>
      <ul class="req-list">${a.gpu_tiers.map((t) => `<li>${esc(t.name)}: ${fmt.hours(t.est_hours)}</li>`).join("")}</ul>`;
  } catch (e) {
    body.innerHTML = `<div class="notice err">${esc(e.message)}</div>`;
  }
}

// ---- remote ---------------------------------------------------------------------

function setLog(lines) {
  const log = document.getElementById("remote-log");
  log.textContent = Array.isArray(lines) ? lines.join("\n") : lines;
}

async function probe() {
  const out = document.getElementById("probe-result");
  out.innerHTML = "";
  try {
    const p = await post("/api/remote/probe", hostBody());
    out.innerHTML = `<p class="okline">Connected to ${esc(p.hostname || "?")}. ${p.gpus.length} GPU(s), ${p.ram_gb ?? "?"} GB RAM.</p>`;
    await loadHosts();
  } catch (e) {
    out.innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

async function provision() {
  setLog("provisioning…");
  try {
    const { task } = await post("/api/remote/provision", hostBody({
      prepared_name: document.getElementById("prov-prepared").value || "full",
      sync_data: document.getElementById("prov-data").checked,
    }));
    await pollTask(task, (t) => setLog(t.lines), 2500);
  } catch (e) {
    setLog(e.message);
  }
}

async function launch() {
  try {
    const job = await post("/api/remote/launch", hostBody({
      config: document.getElementById("launch-config").value,
      prepared: document.getElementById("launch-prepared").value || "full",
      run_name: document.getElementById("launch-name").value || null,
    }));
    setLog(`launched ${job.run} on ${job.host} (pid ${job.pid}).`);
    await loadHosts();
  } catch (e) {
    setLog(e.message);
  }
}

async function status() {
  const run = document.getElementById("remote-run-select").value;
  if (!run) return;
  const box = document.getElementById("remote-status");
  try {
    const s = await post("/api/remote/status", hostBody({ run_name: run }));
    const st = s.status || {};
    box.innerHTML = `<p><b>${esc(run)}</b>: ${s.alive ? "alive" : "not running"}, state ${esc(st.state || "unknown")}, epoch ${st.epoch ?? "?"} / ${st.epochs ?? "?"}</p>`;
    setLog(s.log || []);
  } catch (e) {
    box.innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

async function stop() {
  const run = document.getElementById("remote-run-select").value;
  if (!run) return;
  try { await post("/api/remote/stop", hostBody({ run_name: run })); setLog("stop signal sent"); }
  catch (e) { setLog(e.message); }
}

async function fetchRun() {
  const run = document.getElementById("remote-run-select").value;
  if (!run) return;
  try {
    const { task } = await post("/api/remote/fetch", hostBody({ run_name: run, load: true }));
    await pollTask(task, (t) => setLog(t.lines), 2000);
    await refreshMeta();
  } catch (e) { setLog(e.message); }
}
