import { get, fmt } from "./api.js";
import { initScore } from "./score.js";
import { initData } from "./data.js";
import { initModel } from "./model.js";
import { initCompute } from "./compute.js";
import { initExplain } from "./explain.js";
import { initStory } from "./story.js";

const TABS = ["score", "data", "model", "compute", "explain", "story"];
const inited = {};
const initers = { score: initScore, data: initData, model: initModel, compute: initCompute, explain: initExplain, story: initStory };

export const state = { meta: null, schema: null };

function showTab(name) {
  if (!TABS.includes(name)) name = "score";
  if (name === "compute" && state.meta?.public?.public) name = "score";   // no remote training on the public site
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
    const parts = [`model: ${m.name}`, `${m.sources.length} sources, ${m.has_booster ? "VAE + booster" : m.score_type + " score"}`];
    parts.push(`test AUROC ${fmt.auc(m.mean_test_auroc)}${m.mean?.vae != null && m.has_blend ? ` (vae ${fmt.auc(m.mean.vae)})` : ""}`);
    if (m.generic_test_auroc != null) parts.push(`generic ${fmt.auc(m.generic_test_auroc)}`);
    // a run that has a booster we couldn't load still serves the novelty score; say so
    if (m.booster_error) parts.push(`⚠ booster not loaded, novelty only:\n${m.booster_error}`);
    box.textContent = parts.join("\n");
  }
  window.dispatchEvent(new CustomEvent("metachanged", { detail: state.meta }));
}

window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
await refreshMeta();
try { state.schema = await get("/api/schema"); } catch (e) { console.error(e); }
showTab(location.hash.slice(1) || "score");
