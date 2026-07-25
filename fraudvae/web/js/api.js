// tiny fetch wrapper. every api error becomes an Error with the server's detail.

export async function api(path, opts = {}) {
  const init = { headers: {}, ...opts };
  if (init.body && !(init.body instanceof FormData)) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(init.body);
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: "POST", body });

// poll a background task until it finishes, calling onUpdate with the task dict
export async function pollTask(id, onUpdate, interval = 1500) {
  for (;;) {
    const t = await get(`/api/tasks/${id}?tail=400`);
    onUpdate(t);
    if (t.state !== "running") return t;
    await new Promise((r) => setTimeout(r, interval));
  }
}

export const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString()),
  num: (n, d = 2) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d })),
  pct: (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : (100 * x).toFixed(d) + "%"),
  auc: (x) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(3)),
  bytes: (b) => (b == null ? "–" : b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : b > 1e6 ? (b / 1e6).toFixed(0) + " MB" : (b / 1e3).toFixed(0) + " kB"),
  hours: (h) => (h == null ? "–" : h < 1 ? Math.round(h * 60) + " min" : h < 48 ? h.toFixed(1) + " h" : (h / 24).toFixed(1) + " days"),
  date: (ts) => (ts == null ? "–" : new Date(ts * 1000).toISOString().slice(0, 10)),
  ago: (t) => {
    if (!t) return "";
    const s = (Date.now() / 1000 - t);
    if (s < 90) return "just now";
    if (s < 5400) return Math.round(s / 60) + " min ago";
    if (s < 172800) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " days ago";
  },
};

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v != null) e.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export const palette = {
  ink: "#15212e", ink2: "#566879", ink3: "#8a98a8", line: "#d6dde5",
  accent: "#0b6e79", alarm: "#c8102e", calm: "#1f7a4d", amber: "#b8730a",
  sources: {
    fdh: "#0b6e79", capone: "#5b4fcf", tabformer: "#b8730a", ulb: "#1f7a4d",
    paysim: "#c2185b", sparkov: "#2b6cb0", banksim: "#7b5e2a", ieee: "#555f6d",
  },
};
