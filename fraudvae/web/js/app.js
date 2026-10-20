import { get, fmt } from "./api.js";
import { initScore } from "./score.js";
import { initData } from "./data.js";
import { initModel } from "./model.js";
import { initCompute } from "./compute.js";
import { initExplain } from "./explain.js";
import { initStory } from "./story.js";
import "./terms.js";

const TABS = ["score", "data", "model", "compute", "explain", "story"];
const TITLES = { score: "Score a transaction", data: "Training data", model: "Model & training", compute: "Compute & remote GPU", explain: "How it works", story: "Development process" };
// on the public site every tab is also a subdomain (score.example.org ...): the host picks the
// tab, and the links move between hosts instead of hashes. Anywhere else (an IP, localhost,
// a laptop) the hashes do the work as before.
const SUBDOMAIN = { score: "score", data: "data", model: "models", compute: null, explain: "define", story: "story" };
const TAB_OF_SUB = { www: "score", ...Object.fromEntries(Object.entries(SUBDOMAIN).filter(([, v]) => v).map(([t, v]) => [v, t])) };

function siteBase() {
  const h = location.hostname;
  if (!h || /^[\d.]+$/.test(h) || h === "localhost" || !h.includes(".")) return null;
  const labels = h.split(".");
  if (labels.length >= 3 && (TAB_OF_SUB[labels[0]] || labels[0] === "www" || labels[0] === "api")) return labels.slice(1).join(".");
  return h;
}
const BASE = siteBase();
const hostTab = () => { const first = location.hostname.split(".")[0]; return BASE && location.hostname !== BASE ? TAB_OF_SUB[first] || null : null; };

// ---- the selected model version -----------------------------------------------------------
// chosen on the model tab or in the score tab's picker, kept until changed. Browser storage
// holds it per host; on the subdomain site every link to another host carries it (?run=), so
// the choice follows the visitor from models.* to www.* and back
const SEL_KEY = "fraudvae.selected";
let selectedRun = null;
{
  const fromUrl = new URLSearchParams(location.search).get("run");
  if (fromUrl) {
    selectedRun = fromUrl;
    history.replaceState(null, "", location.pathname + location.hash);
  } else {
    try { selectedRun = localStorage.getItem(SEL_KEY); } catch (_) { /* no storage */ }
  }
}
export function getSelectedRun() { return selectedRun; }
export function setSelectedRun(name) {
  if (name === selectedRun) return;
  selectedRun = name || null;
  try { if (name) localStorage.setItem(SEL_KEY, name); else localStorage.removeItem(SEL_KEY); } catch (_) { /* no storage */ }
  wireSubdomainLinks();
  renderRailModel();
  window.dispatchEvent(new CustomEvent("runselected", { detail: selectedRun }));
}
const withRun = (url) => (BASE && selectedRun ? `${url}${url.includes("?") ? "&" : "?"}run=${encodeURIComponent(selectedRun)}` : url);

function tabHref(name) {
  if (!BASE || !SUBDOMAIN[name]) return `#${name}`;
  return withRun(`${location.protocol}//${name === "score" ? "www" : SUBDOMAIN[name]}.${BASE}/`);
}

// "score a payment with it": a link to the score tab that selects the version on the way
export function scoreHref(run) {
  return BASE ? `${location.protocol}//www.${BASE}/?run=${encodeURIComponent(run)}` : "#score";
}
export function rememberRun(run) { setSelectedRun(run); }

function wireSubdomainLinks() {
  if (!BASE) return;
  document.querySelectorAll(".rail-link, .rail-change").forEach((a) => { if (SUBDOMAIN[a.dataset.tab]) a.href = tabHref(a.dataset.tab); });
  const brand = document.querySelector(".brand");
  if (brand) brand.href = withRun(`${location.protocol}//www.${BASE}/`);
  const api = document.querySelector('a[href="/api/docs"], a.rail-api');
  if (api) { api.href = `${location.protocol}//api.${BASE}/`; api.classList.add("rail-api"); }
}

const inited = {};
const initers = { score: initScore, data: initData, model: initModel, compute: initCompute, explain: initExplain, story: initStory };

export const state = { meta: null, schema: null, models: [] };

// which tab is visible: done before anything is fetched, so a page that opens on the model tab
// never shows the score tab first
function revealTab(name) {
  for (const t of TABS) document.getElementById(`tab-${t}`).classList.toggle("hidden", t !== name);
  document.querySelectorAll(".rail-link").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
  document.title = `FinFraud · ${TITLES[name] || name}`;
}

function showTab(name) {
  if (!TABS.includes(name)) name = "score";
  if (name === "compute" && state.meta?.public?.public) name = "score";   // no remote training on the public site
  // on the subdomain site a tab lives at its own host: an in-page link to another tab goes there
  const here = hostTab();
  if (BASE && here && name !== here && SUBDOMAIN[name]) { location.href = tabHref(name); return; }
  revealTab(name);
  if (!inited[name]) {
    inited[name] = true;
    initers[name](state).catch((e) => console.error(`init ${name} failed`, e));
  }
  window.dispatchEvent(new CustomEvent("tabshown", { detail: name }));
}

// the model timeline, fetched once for every tab that needs it (the score tab's picker, the
// model tab's list, the rail's summary of the selected version)
export async function refreshModels() {
  try { state.models = (await get("/api/models")).models || []; } catch (e) { state.models = []; }
  // a stored selection that no longer exists on this server (another machine, a renamed run)
  if (selectedRun && state.models.length && !state.models.some((m) => m.name === selectedRun)) setSelectedRun(null);
  renderRailModel();
  return state.models;
}

export function selectedModel() {
  const name = selectedRun || state.meta?.active_run?.name;
  return state.models.find((m) => m.name === name) || null;
}

// the rail's corner: the version that scores next, as chosen, not the server's default
function renderRailModel() {
  const box = document.getElementById("rail-model");
  if (!box) return;
  if (!state.meta) return;
  const m = selectedModel();
  if (!m) {
    box.textContent = state.meta.active_run ? "" : "no trained model loaded yet\n→ Model & training";
    return;
  }
  const parts = [`selected model: ${m.name}`];
  if (m.version || m.label) parts.push(`${m.version || ""}${m.version && m.label ? " · " : ""}${m.label || ""}`);
  parts.push(`${(m.sources || []).length} datasets, ${m.has_booster ? "VAE + booster" : "VAE alone"}`);
  if (m.mean_shared9 != null) parts.push(`AUROC ${fmt.auc(m.mean_shared9)} on the 9 shared datasets`);
  else if (m.mean_test_auroc != null) parts.push(`AUROC ${fmt.auc(m.mean_test_auroc)} on unseen data`);
  if (m.generic_test_auroc != null) parts.push(`unknown bank: ${fmt.auc(m.generic_test_auroc)}`);
  const a = state.meta.active_run;
  if (a?.booster_error && a.name === m.name) parts.push(`⚠ booster not loaded, novelty only:\n${a.booster_error}`);
  box.textContent = parts.join("\n");
}

export async function refreshMeta() {
  try {
    state.meta = await get("/api/meta");
  } catch (e) {
    document.getElementById("rail-model").textContent = "server unreachable";
    return;
  }
  // the public site has no remote-training tab
  const rc = document.getElementById("rail-compute");
  if (rc) rc.classList.toggle("hidden", !!state.meta.public?.public);
  renderRailModel();
  window.dispatchEvent(new CustomEvent("metachanged", { detail: state.meta }));
}

const first = location.hash.slice(1) || hostTab() || "score";
revealTab(TABS.includes(first) ? first : "score");
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
await refreshMeta();
try { state.schema = await get("/api/schema"); } catch (e) { console.error(e); }
await refreshModels();
wireSubdomainLinks();
showTab(first);
