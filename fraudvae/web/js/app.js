import { get, fmt } from "./api.js";
import { initScore } from "./score.js";
import { initData } from "./data.js";
import { initModel } from "./model.js";
import { initCompute } from "./compute.js";
import { initExplain } from "./explain.js";
import { initStory } from "./story.js";
import "./terms.js";

const TABS = ["score", "data", "model", "compute", "explain", "story"];
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

function tabHref(name) {
  if (!BASE || !SUBDOMAIN[name]) return `#${name}`;
  return `${location.protocol}//${name === "score" ? "www" : SUBDOMAIN[name]}.${BASE}/`;
}

function wireSubdomainLinks() {
  if (!BASE) return;
  document.querySelectorAll(".rail-link").forEach((a) => { if (SUBDOMAIN[a.dataset.tab]) a.href = tabHref(a.dataset.tab); });
  const brand = document.querySelector(".brand");
  if (brand) brand.href = `${location.protocol}//www.${BASE}/`;
  const api = document.querySelector('a[href="/api/docs"]');
  if (api) api.href = `${location.protocol}//api.${BASE}/`;
}
const inited = {};
const initers = { score: initScore, data: initData, model: initModel, compute: initCompute, explain: initExplain, story: initStory };

export const state = { meta: null, schema: null };

function showTab(name) {
  if (!TABS.includes(name)) name = "score";
  if (name === "compute" && state.meta?.public?.public) name = "score";   // no remote training on the public site
  // on the subdomain site a tab lives at its own host: an in-page link to another tab goes there
  const here = hostTab();
  if (BASE && here && name !== here && SUBDOMAIN[name]) { location.href = tabHref(name); return; }
  for (const t of TABS) {
    document.getElementById(`tab-${t}`).classList.toggle("hidden", t !== name);
  }
  document.querySelectorAll(".rail-link").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
  if (!inited[name]) {
    inited[name] = true;
    initers[name](state).catch((e) => console.error(`init ${name} failed`, e));
  }
  window.dispatchEvent(new CustomEvent("tabshown", { detail: name }));
}

export async function refreshMeta() {
  try {
    state.meta = await get("/api/meta");
  } catch (e) {
    document.getElementById("rail-model").textContent = "server unreachable";
    return;
  }
  const m = state.meta.active_run;
  const box = document.getElementById("rail-model");
  // the public site has no remote-training tab
  const rc = document.getElementById("rail-compute");
  if (rc) rc.classList.toggle("hidden", !!state.meta.public?.public);
  if (!m) {
    box.textContent = "no trained model loaded yet\n→ Model & training";
  } else {
    const parts = [`model: ${m.name}`, `${m.sources.length} datasets, ${m.has_booster ? "VAE + booster" : m.score_type + " score"}`];
    parts.push(`AUROC ${fmt.auc(m.mean_test_auroc)} on unseen data${m.mean?.vae != null && m.has_blend ? ` (VAE alone ${fmt.auc(m.mean.vae)})` : ""}`);
    if (m.generic_test_auroc != null) parts.push(`unknown bank: ${fmt.auc(m.generic_test_auroc)}`);
    // a run that has a booster we couldn't load still serves the novelty score; say so
    if (m.booster_error) parts.push(`⚠ booster not loaded, novelty only:\n${m.booster_error}`);
    box.textContent = parts.join("\n");
  }
  window.dispatchEvent(new CustomEvent("metachanged", { detail: state.meta }));
}

window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
await refreshMeta();
try { state.schema = await get("/api/schema"); } catch (e) { console.error(e); }
wireSubdomainLinks();
showTab(location.hash.slice(1) || hostTab() || "score");
