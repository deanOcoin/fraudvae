// the "development process" tab: how the model got here, in two registers (the full account and a plain
// one), plus the datasets with their licences and how to train it yourself. Static html
// partials with the live model timeline filled in where the text quotes numbers.
import { get, fmt, el } from "./api.js";

const PAGES = { full: "/static/story.html", plain: "/static/story_plain.html" };

export async function initStory(state) {
  const host = document.getElementById("story-article");
  host.innerHTML = `<div class="story-switch"><span>Reading level:</span>
      <button class="btn ghost small active" data-level="full">the full account</button>
      <button class="btn ghost small" data-level="plain">plain language</button></div><div id="story-body"></div>`;
  const body = document.getElementById("story-body");
  const show = async (level) => {
    host.querySelectorAll(".story-switch button").forEach((b) => b.classList.toggle("active", b.dataset.level === level));
    const res = await fetch(PAGES[level]);
    body.innerHTML = await res.text();
    try { localStorage.setItem("fraudvae.story", level); } catch (_) {}
    fillTimeline(body).catch(() => {});
    body.querySelectorAll("a[href^='#']").forEach((a) => {
      // in-page links scroll inside the story; tab links keep working
      const id = a.getAttribute("href").slice(1);
      if (body.querySelector(`#${CSS.escape(id)}`)) a.addEventListener("click", (e) => { e.preventDefault(); body.querySelector(`#${CSS.escape(id)}`).scrollIntoView({ behavior: "smooth", block: "start" }); });
    });
    // the two versions point at each other
    body.querySelectorAll("a.to-plain, a.to-full").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); show(a.classList.contains("to-plain") ? "plain" : "full"); }));
    window.scrollTo(0, 0);
  };
  host.querySelectorAll(".story-switch button").forEach((b) => b.addEventListener("click", () => show(b.dataset.level)));
  let level = "full";
  try { level = localStorage.getItem("fraudvae.story") || "full"; } catch (_) {}
  await show(PAGES[level] ? level : "full");
}

// the generations table in the story is rendered from /api/models so it never disagrees
// with the runs actually present on this server
async function fillTimeline(body) {
  const slot = body.querySelector("#story-timeline");
  if (!slot) return;
  const d = await get("/api/models");
  const items = d.models.filter((m) => m.milestone);
  if (!items.length) { slot.innerHTML = `<p class="dim">No milestone runs on this server yet.</p>`; return; }
  const t = el("table", { class: "metric-table" },
    el("thead", {}, el("tr", {}, ...["generation", "what it is", "trained on", "time", "mean AUROC, 9 shared banks", "generic", "status"].map((h) => el("th", {}, h)))),
    el("tbody", {}, ...items.map((m) => {
      const mins = m.train_minutes != null ? (m.train_minutes < 90 ? `${Math.round(m.train_minutes)} min` : `${(m.train_minutes / 60).toFixed(1)} h`) : "–";
      const post = m.post_minutes != null ? ` + ${Math.round(m.post_minutes)} min post` : "";
      return el("tr", {},
        el("td", {}, el("b", {}, m.version), " · ", m.label),
        el("td", {}, m.changed || ""),
        el("td", {}, m.hardware || "", el("br"), el("span", { class: "dim" }, m.corpus || "")),
        el("td", {}, mins + post),
        el("td", {}, m.mean_shared9 != null ? fmt.auc(m.mean_shared9) : (m.mean_test_auroc != null ? `${fmt.auc(m.mean_test_auroc)} (its own banks)` : "–")),
        el("td", {}, m.generic_test_auroc != null ? fmt.auc(m.generic_test_auroc) : "–"),
        el("td", {}, m.loaded ? "loaded" : "on disk"));
    })));
  slot.innerHTML = "";
  slot.append(t);
}


