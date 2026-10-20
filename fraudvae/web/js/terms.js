// The glossary. Every page can use the words the field uses; this module makes each of them
// explain itself on demand. Nothing is underlined or marked: on a computer a word's
// definition appears after the pointer rests on it for a moment (or on a double click), on a
// phone after a tap. The box is plain text in a monospace face, nothing more.
//
// Text nodes under the app's content are scanned once rendered (a MutationObserver keeps up
// with tabs that render later) and known terms are wrapped in <span class="term">. Code,
// inputs, links and the receipt's numbers are left alone.

const GLOSSARY = {
  // --- measuring the model ---
  "AUROC": "AUROC, area under the ROC curve: a score from 0.5 to 1.0 for how well the model ranks fraud above legitimate payments. Pick a random fraud and a random legitimate payment; AUROC is the chance the fraud gets the higher score. 0.5 = guessing, 1.0 = perfect ordering. It ignores where you set the threshold.",
  "AUPRC": "AUPRC, area under the precision-recall curve: like AUROC but focused on the rare class (fraud). Harsher: a model can have AUROC 0.95 and AUPRC 0.3 when fraud is 0.1% of traffic. 1.0 is perfect.",
  "recall": "recall: of all the fraud that happened, the share the model caught. 70% recall at a 1% false-positive budget = it catches 7 in 10 frauds while wrongly flagging 1 in 100 legitimate payments.",
  "precision": "precision: of the payments the model flagged, the share that really were fraud. Low precision = lots of false alarms for the review team.",
  "false positive": "false positive: a legitimate payment the model flagged as fraud. The alert budget is the share of legitimate traffic you are willing to flag by mistake.",
  "false-positive": "false positive: a legitimate payment the model flagged as fraud. The alert budget is the share of legitimate traffic you are willing to flag by mistake.",
  "FPR": "FPR, false-positive rate: the share of legitimate payments wrongly flagged. 1% FPR = 1 in 100 honest payments gets reviewed. Fraud teams pick this budget first and read recall at it.",
  "alert budget": "alert budget: how many legitimate payments you accept flagging by mistake, as a share of all legitimate traffic (0.1% to 5% here). The threshold is set so that exactly this share of validation traffic gets flagged; recall is then whatever the model achieves at that cost.",
  "threshold": "threshold: the score above which a payment is flagged. Chosen on the validation split so that a given share of legitimate payments (the alert budget) lands above it; frozen before the test split is scored.",
  "calibration": "calibration: the step after training that turns raw scores into probabilities you can act on, picks thresholds per bank for each alert budget, and records what recall to expect at them. Done on the validation split, never on test data.",
  "calibrated": "calibrated: a score that behaves like a probability. 'p(fraud) 0.3' from a calibrated model means roughly 3 in 10 payments with that score are fraud.",
  "validation split": "validation split: the 10% of each dataset (by time, after the oldest 70%) used to tune thresholds and the blend. The model does not train on it, and it is not the data the reported numbers come from.",
  "validation": "validation (data/split): the 10% slice of each dataset used to tune thresholds and the blend after training. Not trained on, not the test data either.",
  "test split": "test split: the newest 20% of every dataset, held out from training and tuning entirely. Every number on this site comes from scoring it. Newest-by-time, because fraud changes and the fair test is 'does it work on what came after'.",
  "held-out": "held-out: data the model never saw while training or tuning. The test split is held out; numbers on held-out data are the honest ones.",
  "held out": "held out: data the model never saw while training or tuning. The test split is held out; numbers on held-out data are the honest ones.",
  "overfit": "overfitting: when a model memorises its training data instead of learning the pattern, so it looks great on data it has seen and falls apart on new data. The reason everything here is measured on a held-out split.",
  "overfitting": "overfitting: when a model memorises its training data instead of learning the pattern, so it looks great on data it has seen and falls apart on new data. The reason everything here is measured on a held-out split.",
  "generic mode": "generic mode: scoring a payment from a bank the model never trained on. The bank identity is blanked and the model uses only what the eleven datasets have in common. The 'generic' number is what a new institution would get on day one.",
  "generic": "generic (context): the model's mode for a payment from a bank it never trained on. The bank identity is blanked and only what the eleven datasets share is used.",
  "percentile": "percentile (of legit traffic): where this payment's score sits among legitimate payments of the same bank. 99.5th percentile = scored higher than 99.5% of honest traffic, i.e. in the top 0.5%.",
  "z-score": "z-score: how many standard deviations a value is from what was expected. z = 3 means 'three times the usual spread above normal': rare. The receipt shows one per field.",
  "standard deviation": "standard deviation: the usual spread of a quantity around its average. A value 3 standard deviations out is rare; 1 is unremarkable.",
  "baseline": "baseline: the simplest version of something, kept as the point of comparison for everything added later.",
  "ablation": "ablation: removing one part of a model (a block of features, a module) and measuring again, to learn what that part was worth. Negative results are the point.",
  "mean AUROC": "mean AUROC: the AUROC averaged across datasets, each counting equally regardless of its size. The headline number here, usually over the nine datasets every generation shares.",

  // --- the models ---
  "VAE": "VAE, variational autoencoder: a neural network trained to compress a payment into a few numbers and rebuild it. Trained on legitimate payments only, so what it rebuilds badly is what it has not seen before: a measure of 'unusual' that needs no fraud labels at all.",
  "variational autoencoder": "variational autoencoder (VAE): a neural network trained to compress a payment into a few numbers and rebuild it. Trained on legitimate payments only, so what it rebuilds badly is what it has not seen before: a measure of 'unusual' that needs no fraud labels at all.",
  "autoencoder": "autoencoder: a network that learns to compress its input and reconstruct it. The reconstruction error says how unlike the training data the input is.",
  "novelty": "novelty: how unlike normal traffic a payment looks to the VAE, which was trained on legitimate payments only. High novelty = the model cannot explain this payment from what it knows. It is not the same as fraud; it is 'unusual'.",
  "booster": "booster: the gradient-boosted decision-tree model (LightGBM) trained on the labelled fraud. It learns what known fraud looks like from the payment's fields and the customer's history. In this project it does most of the catching; the VAE does the explaining.",
  "gradient-boosted": "gradient boosting: building a model out of hundreds of small decision trees, each one correcting the mistakes of the ones before. The standard strong method for tabular data like payments. LightGBM is the implementation used here.",
  "gradient boosting": "gradient boosting: building a model out of hundreds of small decision trees, each one correcting the mistakes of the ones before. The standard strong method for tabular data like payments. LightGBM is the implementation used here.",
  "LightGBM": "LightGBM: a fast, widely used library for gradient-boosted decision trees (Microsoft). The 'booster' here is a LightGBM model.",
  "decision tree": "decision tree: a model that asks a sequence of yes/no questions about the fields ('amount > 220?', 'first time at this merchant?') to reach a verdict. Hundreds of them combined make a boosted model.",
  "supervised": "supervised: trained on examples that carry the answer (here, payments labelled fraud or not). The booster is supervised; the VAE is not (it never sees a label).",
  "unsupervised": "unsupervised: trained without labels. The VAE only sees legitimate payments and learns what they look like; it is never told what fraud is.",
  "blend": "blend: the small final model that combines the scores of the others (booster, VAE, the extra heads) into one probability, with a separate weighting per bank. It is what the verdict is read from.",
  "blender": "blender: the small final model that combines the scores of the others (booster, VAE, the extra heads) into one probability, with a separate weighting per bank. It is what the verdict is read from.",
  "logistic regression": "logistic regression: the simplest classifier: a weighted sum of the inputs squashed into a probability. The blend is one, over the other models' scores.",
  "specialist": "specialist (booster): a second boosted-tree model trained on one bank's data only, next to the shared one. A second opinion that knows that bank's recent months better.",
  "sequence head": "sequence head: a small model that reads the summary the history transformer makes of the customer's previous payments and scores the new one from that alone. A third opinion in the blend.",
  "device head": "device head: a small model that looks only at the device fields (operating system, browser, screen, session length...) and says how risky the device looks. Only exists for the datasets that carry device data.",
  "graph head": "graph head: the model that scores a payment from the network around it - which customers, merchants, devices and cards are connected to which - using a graph neural network per dataset. Reads the entity graph.",
  "head": "head (of a model): a small final layer that reads an internal representation and outputs one specific thing, such as a fraud score. Several heads can share one representation.",
  "module": "module: a self-contained part of the system with its own training and its own score, plugged into the blend: the device head, the graph head, the text heads, the behaviour verifiers.",
  "transformer": "transformer: the neural-network design behind modern language models. Here a small one reads the customer's previous payments, in order, and summarises them so the VAE can judge the next payment against that customer's own habits.",
  "history encoder": "history encoder: the small transformer that reads a customer's previous payments (up to 32) and turns them into a summary vector. Both halves of the VAE read it, so 'unusual' means unusual for this customer.",
  "latent": "latent (space / vector): the few numbers the VAE compresses a payment into - its summary. Nearby points in latent space are payments the model sees as similar.",
  "latent space": "latent space: the compressed representation inside the VAE, a few numbers per payment. Nearby points are payments the model sees as similar; the chart on the score tab shows where one lands among validation traffic.",
  "embedding": "embedding: a learned list of numbers that stands in for a category (a merchant type, a bank, a card brand) so a neural network can use it. Similar categories end up with similar numbers.",
  "ELBO": "ELBO (evidence lower bound): the quantity a VAE is trained to maximise: how well it reconstructs the payment minus a penalty for straying from a simple summary. Its negative is used as a raw 'surprise' score.",
  "KL": "KL divergence (KL(q|p)): the penalty term in a VAE that keeps the compressed summary close to a standard reference distribution. On the receipt, a high KL z means the summary itself was unusual.",
  "posterior": "posterior (sample): the VAE's compressed summary is a distribution, not a point; a posterior sample is one draw from it. Scoring averages several draws (16 here) so the score is stable.",
  "Monte Carlo": "Monte Carlo: estimating something by averaging random draws. The VAE's score is averaged over several random samples of its summary.",
  "log-odds": "log-odds (logit): a probability written on a scale where adding is meaningful: 0 means 50/50, +2 roughly 88%, -2 roughly 12%. The booster's per-feature pushes add up on this scale.",
  "logit": "logit: a probability on the log-odds scale, where contributions can be added. 0 = 50/50, +2 roughly 88%, -2 roughly 12%.",
  "SHAP": "SHAP: a method that splits a tree model's prediction into a contribution per input field, so the receipt can say 'amount pushed +0.5 toward fraud, device -0.2 away'. The pushes add up to the model's output.",
  "likelihood ratio": "likelihood ratio: how much more likely the evidence is under 'fraud' than under 'legitimate'. The text and behaviour modules answer with one, which is added to the verdict's log-odds.",
  "GraphSAGE": "GraphSAGE: a graph neural network design: each node (customer, merchant, device...) is described by sampling and summarising its neighbours. Used per dataset to score a payment from its surroundings.",
  "graph neural network": "graph neural network: a model that reads a network of connected things - here customers, merchants, devices, cards joined by payments - and learns from the structure, not just the fields of one row.",
  "GNN": "GNN, graph neural network: a model that reads a network of connected things - here customers, merchants, devices, cards joined by payments - and learns from the structure, not just the fields of one row.",
  "dropout": "dropout: deliberately blanking a random part of the input (or the network) during training so the model learns to cope without it. Used here so the booster works whether or not the fresh fraud labels are available.",
  "early stopping": "early stopping: ending training when the validation score stops improving, instead of running all planned epochs. Prevents overfitting and saves time.",
  "epoch": "epoch: one complete pass over the training data. The shipped model made 30; the laptop-sized one 10.",
  "checkpoint": "checkpoint: a saved copy of the model's weights at a point in training. 'The same checkpoint' means the same trained network, reused.",
  "weights": "weights: the numbers inside a model that training adjusts. A trained model is its weights plus the code that uses them.",
  "parameters": "parameters: the adjustable numbers inside a model (its weights). 3M parameters = three million of them. More is not automatically better, as the v3 runs showed.",
  "learning rate": "learning rate: how big a step training takes each time it adjusts the weights. Too big and training blows up; too small and it crawls.",
  "fp32": "fp32 / bf16: number formats on the GPU. fp32 (32-bit) is precise; bf16 (16-bit) is faster but coarser. The history transformer only trained stably in fp32.",
  "bf16": "bf16: a 16-bit number format that makes GPU training faster but less precise. It made the history transformer's training blow up here; fp32 (32-bit) did not.",
  "autocast": "autocast: letting the GPU use a faster, less precise number format for parts of the computation automatically.",
  "inference": "inference: using a trained model to score new data, as opposed to training it. The public site does inference only.",
  "hyperparameters": "hyperparameters: the settings of training you choose rather than learn: learning rate, number of trees, model size. 'A rounding error next to features' here means tuning them barely mattered.",
  "regularisation": "regularisation: any technique that stops a model from fitting its training data too closely. The shrinkage on the blend's per-bank weights is one.",
  "shrinkage": "shrinkage: pulling a weight towards zero (or towards a shared value) in proportion to how little evidence supports it. Used so a bank with 21 labelled frauds cannot dictate its own blend weights.",

  // --- the data ---
  "corpus": "corpus: the whole body of training data: here the eleven public datasets merged into one table of 42.8 million payments.",
  "schema": "schema: the fixed list of fields every payment is described with (amount, hour, channel, the customer's history...). Each dataset is translated into it; a dataset without a field leaves it 'unobserved'.",
  "unobserved": "unobserved: a field a dataset (or a request) simply does not have. Marked as missing, not filled with zero, and the model skips it rather than guessing.",
  "source": "source (bank / dataset): one of the eleven public datasets. The model treats each as a separate bank with its own idea of normal and its own thresholds.",
  "feature": "feature: one input the model reads: a field of the payment, or something computed from the customer's history (how many payments in the last hour, how this amount compares to their usual).",
  "features": "features: the inputs the model reads: fields of the payment, plus numbers computed from the customer's and merchant's history (payments in the last hour, the customer's usual amount, first time at this merchant...).",
  "label": "label: the recorded truth for a payment, fraud or not. Labels arrive late in real life (chargebacks take weeks), which is why the model is careful about when a label would have been known.",
  "labels": "labels: the recorded truth for each payment, fraud or not. They arrive late in real life (chargebacks take weeks), which is why the model is careful about when a label would have been known.",
  "label delay": "label delay / latency: the time between a payment and the moment its fraud label is known. The model's fraud-history features only count labels old enough to have been known at the time; 7 or 30 days here.",
  "label latency": "label latency: the time between a payment and the moment its fraud label is known. The model's fraud-history features only count labels old enough to have been known at the time; 7 or 30 days here.",
  "delayed labels": "delayed labels: fraud labels counted only once they are at least N days old, so a feature like 'fraud confirmed at this merchant so far' uses only what would have been known at the time.",
  "velocity": "velocity (features): counts and sums over recent time windows - the customer's payments in the last hour, day, week - which is where bursts of fraud show up.",
  "context features": "context features: numbers computed from the customer's and the merchant's past: payments in the last hour and day, the customer's usual amount, how long they have been a customer, first time at this merchant. Where most of the signal lives.",
  "entity": "entity: a thing a payment involves that recurs across payments: the customer, the counterparty (merchant), the card, the device, the e-mail domain, the address. The model keeps a history per entity.",
  "entities": "entities: the recurring things a payment involves: the customer, the counterparty (merchant), the card, the device, the e-mail domain, the address. The model keeps a history per entity and a graph of how they connect.",
  "counterparty": "counterparty: the other side of the payment: the merchant for a card purchase, the receiving account for a transfer.",
  "entity graph": "entity graph: the network of customers, merchants, devices, cards, e-mails and addresses joined by the payments between them. Shared devices and busy merchants are visible there and nowhere else.",
  "wide block": "wide block: a dataset's own raw columns (IEEE-CIS has 431 of them) kept in a side table that only the booster reads. The common schema stays small; nothing bank-specific leaks into the shared model.",
  "device block": "device block: the fields describing the device and session - operating system, browser, screen, session length, keep-alive - for the datasets that have them.",
  "causal": "causal (features): computed only from what was known at the time of the payment, never from anything that came after. The opposite would be cheating with hindsight.",
  "time split": "time split: dividing each dataset by date: the oldest 70% trains, the next 10% tunes, the newest 20% tests. The model is always judged on what came after what it learnt from.",
  "simulator": "simulator: a dataset generated by a program that imitates real payment behaviour, with fraud injected by rules. Realistic in shape, but its fraud is learnable in a way real fraud is not.",
  "synthetic": "synthetic: data produced by a generator rather than recorded from real people. Several datasets here are synthetic but calibrated on real behaviour.",
  "PCA": "PCA, principal component analysis: a way of summarising many columns into fewer. One dataset (ULB) was published as PCA components for privacy; the score tab's chart uses it to draw the latent space in two dimensions.",
  "parquet": "parquet: a compressed column-oriented file format for tables, used for the processed datasets.",
  "chargeback": "chargeback: a customer disputing a card payment with their bank, often weeks after the fact. Usually how card fraud becomes a known label.",
  "card-not-present": "card-not-present: a payment where the physical card is not used - online, by phone. Where most card fraud happens.",
  "MCC": "MCC, merchant category code: the standard code for what kind of business a merchant is (grocery, fuel, travel...).",
  "CVV": "CVV: the three-digit security code on a card. A mismatch between the code entered and the one on file is a rare but strong fraud signal.",
  "Kaggle": "Kaggle: a website hosting public datasets and machine-learning competitions. Eight of the eleven datasets are downloaded from there, with a free account.",
  "licence": "licence (of a dataset): the terms its authors allow it to be used and shared under. They differ; it is why only six of the eleven ship example rows here.",

  // --- the machinery ---
  "GPU": "GPU: a graphics card; the kind of processor that trains neural networks fast. The RTX 4090 is a high-end consumer one, the H200 a data-centre one.",
  "CPU": "CPU: the computer's main processor. Fine for scoring and for the laptop-sized model; days instead of hours for the full one.",
  "vCPU": "vCPU: a virtual processor core on a rented server. This site runs on six.",
  "MPS": "MPS: Apple's GPU interface for PyTorch, so a Mac's graphics chip can run the model.",
  "CUDA": "CUDA: NVIDIA's platform for running computations on their GPUs. 'cu130' in a version string means the build targets CUDA 13.",
  "PyTorch": "PyTorch: the library the neural networks here are written in.",
  "API": "API: the web interface programs use to talk to this server: send a payment as JSON, get a verdict back. The API docs page lists every call.",
  "JSON": "JSON: a plain-text format for structured data, the way the API takes requests and gives answers. {\"amount\": 12.5, \"hour\": 14} is JSON.",
  "CSV": "CSV: a plain spreadsheet file, one row per line, columns separated by commas. The batch scorer takes one.",
  "relay": "relay: the way the rented GPU box was controlled with the laptop off: it polled a private git repository for commands and pushed results back.",
  "the box": "the box: the rented data-centre GPU machine (one NVIDIA H200, 64 cores, 188 GB of memory) that all of v5 was trained on.",
  "rows/s": "rows/s: how many payments per second training gets through. 85,000 rows/s with 30 million rows is about six minutes per pass.",
  "nginx": "nginx: the web server in front of the app on the public site. It handles the encryption and the first layer of rate limiting.",
  "rate limit": "rate limit: a cap on how many requests one visitor can make per minute, so no one can keep the server busy for everyone else.",
  "receipt": "receipt: the itemised explanation that comes with every score: which fields looked unusual, what pushed the booster's verdict, which models were consulted.",
  "generation": "generation: one trained version of the model you can pick on the score tab. Six are kept, oldest to newest, so progress is something you can test rather than read about.",
  "post stage": "post stage: everything after the VAE finished training: the booster, the specialists, the heads, calibration and evaluation. Can be redone on the same VAE, which is what the last two v5 generations are.",
};

// longest terms first so "false-positive rate" wins over "rate", and a word boundary on both sides
const ENTRIES = Object.entries(GLOSSARY).sort((a, b) => b[0].length - a[0].length);
const PATTERN = new RegExp("(?<![\\w-])(" + ENTRIES.map(([k]) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")(?![\\w-])", "gi");
const LOOKUP = Object.fromEntries(ENTRIES.map(([k, v]) => [k.toLowerCase(), v]));
const SKIP = new Set(["CODE", "PRE", "INPUT", "TEXTAREA", "SELECT", "OPTION", "SCRIPT", "STYLE", "BUTTON", "H1", "SVG", "CANVAS", "A"]);

function defOf(word) {
  // exact match first, then case-insensitive, then singular
  return GLOSSARY[word] || LOOKUP[word.toLowerCase()] || LOOKUP[word.toLowerCase().replace(/s$/, "")] || null;
}

function wrapTextNode(node) {
  const text = node.nodeValue;
  if (!text || text.length < 3) return;
  PATTERN.lastIndex = 0;
  if (!PATTERN.test(text)) return;
  PATTERN.lastIndex = 0;
  const frag = document.createDocumentFragment();
  let last = 0, m;
  while ((m = PATTERN.exec(text))) {
    const def = defOf(m[1]);
    if (!def) continue;
    if (m.index > last) frag.append(document.createTextNode(text.slice(last, m.index)));
    const span = document.createElement("span");
    span.className = "term";
    span.dataset.def = def;
    span.textContent = m[1];
    frag.append(span);
    last = m.index + m[1].length;
  }
  if (last === 0) return;
  if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
  node.parentNode.replaceChild(frag, node);
}

export function annotate(root) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      let p = n.parentNode;
      while (p && p !== root) {
        if (SKIP.has(p.nodeName) || (p.classList && (p.classList.contains("term") || p.classList.contains("no-terms") || p.classList.contains("rc-verdict")))) return NodeFilter.FILTER_REJECT;
        p = p.parentNode;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(wrapTextNode);
}

// ---- the box ----
let box = null, hoverTimer = null, current = null;
const coarse = () => window.matchMedia("(hover: none), (pointer: coarse)").matches;

function ensureBox() {
  if (box) return box;
  box = document.createElement("div");
  box.id = "term-box";
  box.setAttribute("role", "note");
  document.body.append(box);
  return box;
}

function showFor(span) {
  const b = ensureBox();
  current = span;
  b.textContent = span.dataset.def;
  b.style.display = "block";
  const r = span.getBoundingClientRect();
  const width = Math.min(380, window.innerWidth - 24);
  b.style.width = width + "px";
  let left = r.left + window.scrollX;
  if (left + width > window.scrollX + window.innerWidth - 12) left = window.scrollX + window.innerWidth - 12 - width;
  if (left < window.scrollX + 12) left = window.scrollX + 12;
  b.style.left = left + "px";
  b.style.top = r.bottom + window.scrollY + 6 + "px";
  // flip above when there is no room below
  const bh = b.getBoundingClientRect().height;
  if (r.bottom + 6 + bh > window.innerHeight && r.top - bh - 6 > 0) b.style.top = r.top + window.scrollY - bh - 6 + "px";
}

export function hideTerm() {
  if (box) box.style.display = "none";
  current = null;
  clearTimeout(hoverTimer);
}

function install() {
  document.addEventListener("mouseover", (e) => {
    const t = e.target.closest?.(".term");
    clearTimeout(hoverTimer);
    if (!t || coarse()) return;
    hoverTimer = setTimeout(() => showFor(t), 900);
  });
  document.addEventListener("mouseout", (e) => {
    const t = e.target.closest?.(".term");
    if (t) {
      clearTimeout(hoverTimer);
      // leave the box for a moment so the text can be read while the pointer drifts off
      if (current === t) setTimeout(() => { if (current === t && !box?.matches(":hover")) hideTerm(); }, 700);
    }
  });
  document.addEventListener("dblclick", (e) => {
    const t = e.target.closest?.(".term");
    if (t) { e.preventDefault(); showFor(t); }
  });
  document.addEventListener("click", (e) => {
    const t = e.target.closest?.(".term");
    if (t && coarse()) { e.preventDefault(); current === t && box?.style.display === "block" ? hideTerm() : showFor(t); return; }
    if (!e.target.closest?.("#term-box")) hideTerm();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTerm(); });
  window.addEventListener("scroll", () => { if (current && coarse()) hideTerm(); }, { passive: true });
  // content rendered later (tabs, receipts, story pages) gets the same treatment
  const main = document.querySelector("main");
  if (main) {
    annotate(main);
    let pending = null;
    const obs = new MutationObserver((muts) => {
      if (pending) return;
      pending = setTimeout(() => {
        pending = null;
        obs.disconnect();
        annotate(main);
        obs.observe(main, { childList: true, subtree: true, characterData: true });
      }, 60);
    });
    obs.observe(main, { childList: true, subtree: true, characterData: true });
  }
}

export const glossary = GLOSSARY;
install();
