// "how it works" tab: static html partial + katex auto-render + live numbers where we have them
import { get, fmt } from "./api.js";

export async function initExplain(state) {
  const host = document.getElementById("explain-article");
  const res = await fetch("/static/explain.html");
  host.innerHTML = await res.text();
  fillLiveNumbers(state).catch(() => {});
  renderMath(host);
}

function renderMath(host) {
  const go = () => {
    if (!window.renderMathInElement) return false;
    window.renderMathInElement(host, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }],
      throwOnError: false,
    });
    return true;
  };
  if (!go()) {
    // katex is loaded with defer, it may land after us
    let tries = 0;
    const t = setInterval(() => { if (go() || ++tries > 40) clearInterval(t); }, 250);
  }
}

// the prose quotes a few numbers from the current corpus / model so it doesn't go stale
async function fillLiveNumbers(state) {
  const d = await get("/api/datasets");
  const built = d.sources.filter((s) => s.built);
  const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
  set("x-rows", fmt.int(d.total_rows));
  set("x-fraud", fmt.int(d.total_fraud));
  set("x-nsrc", built.length);
  set("x-rate", fmt.pct(d.total_fraud / Math.max(d.total_rows, 1), 3));
  const m = state.meta?.active_run;
  if (m) {
    set("x-model", m.name + (m.has_booster ? ` (VAE + ${m.booster_trees || ""}-tree booster)` : " (VAE only)"));
    set("x-score-type", m.score_type === "norm" ? "the normalised score" : "the raw −ELBO");
    if (m.mean) {
      set("x-mean-vae", fmt.auc(m.mean.vae));
      set("x-mean-booster", fmt.auc(m.mean.booster));
      set("x-mean-blend", fmt.auc(m.mean.blend));
      set("x-mean-generic", fmt.auc(m.generic_test_auroc));
    }
  }
  const schema = state.schema;
  if (schema) {
    set("x-nnum", schema.numeric.length);
    set("x-ncat", schema.categorical.length);
  }
}
