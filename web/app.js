// Wrench front end. S11: every server or model string is set with textContent, never parsed as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
let TL = null;
let selected = null; // {unit_id, ts}
let currentOrder = null;

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

async function api(path, body, headers) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json", ...(headers || {}) }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  return { status: r.status, data };
}

function errorText(r) {
  if (r.status === 503) return "The model on AMD Token Factory did not answer (busy or rate-limited) and there is no recording for this request. Try again in a minute.";
  if (r.status === 429) return "Too many requests. Try again in a minute.";
  if (r.status === 413) return "The question is too long (500 characters at most).";
  return `Error ${r.status}: ${typeof r.data.detail === "string" ? r.data.detail : "request failed"}`;
}

function servedText(d) {
  if (d.served_by === "recorded") return `RECORDED answer, not live: recorded ${d.recorded_at} on ${d.recorded_on}.`;
  if (d.served_by === "amd-tokenfactory") return "Live answer from Gemma-4-31B-IT on AMD Token Factory (AMD GPUs).";
  if (d.served_by === "mi300x") return "Live answer from the model on the AMD MI300X.";
  if (d.served_by === "cpu-stub") return "Development stub on CPU (not a model).";
  return "No model was called.";
}

function tsAt(i) {
  const t = new Date(Date.parse(TL.start + "Z") + i * TL.step_seconds * 1000);
  return t.toISOString().slice(0, 19);
}

const RANK = { "-": 0, o: 1, w: 2, a: 3 };

// The steps drawn in pixel column x of a chart w pixels wide (shared by draw and pick).
function pixelRange(x, w, n) {
  const a = Math.floor((x / w) * n);
  return [a, Math.max(a + 1, Math.floor(((x + 1) / w) * n))];
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function draw() {
  const c = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const n = TL.level.length, band = 14, top = 8, plotH = h - band - top - 4;
  const colour = { o: css("--ok"), w: css("--watch"), a: css("--act"), "-": css("--none") };
  let max = 0;
  for (const s of TL.score) if (s !== null && s > max) max = s;
  const ymax = Math.min(max, 10) || 1;
  for (let x = 0; x < w; x++) {
    const [a, b] = pixelRange(x, w, n);
    let worst = "-", hi = null;
    for (let i = a; i < b; i++) {
      const l = TL.level[i];
      if (RANK[l] > RANK[worst]) worst = l;
      const s = TL.score[i];
      if (s !== null && (hi === null || s > hi)) hi = s;
    }
    g.fillStyle = colour[worst];
    g.fillRect(x, h - band, 1, band);
    if (hi !== null) {
      g.fillStyle = css("--fg");
      const y = top + plotH * (1 - Math.min(hi, ymax) / ymax);
      g.fillRect(x, y, 1, top + plotH - y);
    }
  }
  const t0 = Date.parse(TL.start + "Z"), span = n * TL.step_seconds * 1000;
  g.fillStyle = css("--act");
  for (const f of TL.failures) {
    const xa = ((Date.parse(f.start.replace(" ", "T") + "Z") - t0) / span) * w;
    const xb = ((Date.parse(f.end.replace(" ", "T") + "Z") - t0) / span) * w;
    g.globalAlpha = 0.25; g.fillRect(xa, 0, Math.max(2, xb - xa), h - band); g.globalAlpha = 1;
  }
  if (selected) {
    g.fillStyle = css("--accent");
    g.fillRect(selected.i / n * w, 0, 2, h);
  }
}

async function pick(evt) {
  // One pixel covers many steps and is drawn with the worst of them, so pick that step (then the highest score).
  const c = $("chart"), r = c.getBoundingClientRect();
  const n = TL.level.length, w = c.clientWidth; // the width draw() used
  const x = Math.min(w - 1, Math.max(0, Math.floor(evt.clientX - r.left - c.clientLeft)));
  const [a, b] = pixelRange(x, w, n);
  let i = a;
  for (let k = a; k < b; k++) {
    const better = RANK[TL.level[k]] - RANK[TL.level[i]] || (TL.score[k] ?? -1) - (TL.score[i] ?? -1);
    if (better > 0) i = k;
  }
  selected = { unit_id: TL.unit_id, ts: tsAt(i), i };
  draw();
  const p = await api(`/api/point?unit_id=${encodeURIComponent(TL.unit_id)}&ts=${encodeURIComponent(selected.ts)}`);
  $("point").hidden = false;
  if (p.status !== 200) { $("pointHead").textContent = errorText(p); return; }
  const d = p.data;
  $("pointHead").textContent = `${d.unit_id} · ${d.ts} · level ${d.level} · score ${d.anomaly_score ?? "n/a"}`;
  const ul = $("drivers");
  ul.replaceChildren(...d.drivers.map((x) => el("li", `${x.sensor} ${x.direction} (z = ${x.z})`)));
  $("draftBtn").disabled = !(d.level === "watch" || d.level === "act");
}

function showCitations(list, cits) {
  list.replaceChildren(...cits.map((c) => {
    const li = el("li");
    li.append(el("strong", `${c.source} ${c.id}`), el("blockquote", c.quote));
    return li;
  }));
}

const WAIT = "Asking the model on AMD Token Factory. A live answer usually takes 15 to 35 seconds…";

async function ask(evt) {
  evt.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  const at = selected || { unit_id: TL.unit_id, ts: tsAt(TL.level.length - 1) };
  $("answer").hidden = false;
  $("servedBy").textContent = ""; $("citations").replaceChildren();
  $("answerText").textContent = WAIT;
  const r = await api("/api/ask", { unit_id: at.unit_id, ts: at.ts, question: q });
  if (r.status !== 200) {
    $("servedBy").textContent = ""; $("answerText").textContent = errorText(r); $("citations").replaceChildren();
    return;
  }
  $("servedBy").textContent = servedText(r.data);
  $("servedBy").className = r.data.served_by === "recorded" ? "banner" : "meta";
  $("answerText").textContent = r.data.answer;
  showCitations($("citations"), r.data.citations);
}

function renderOrder(o, servedBy) {
  currentOrder = o;
  $("order").hidden = false; $("orderEmpty").hidden = true;
  $("orderMeta").textContent = `${o.id} · status ${o.status.toUpperCase()} · ${servedText({ served_by: servedBy })}`;
  const lines = [
    `Unit ${o.unit_id}, alert at ${o.alert_time}, level ${o.level}`,
    `Priority ${o.priority} (${o.priority_rule})`,
    `Drivers: ${o.drivers.map((d) => `${d.sensor} ${d.direction} (z=${d.z})`).join(", ") || "none"}`,
    "", "Steps:", o.steps, "",
    `Parts: ${o.parts.length ? o.parts.join("; ") : "none from the cited records"}`,
    `Safety: isolate the unit first (${o.safety})`,
  ];
  if (o.approved_at) lines.push(`Approved at ${o.approved_at}`);
  $("orderText").textContent = lines.join("\n");
}

async function draftOrder() {
  if (!selected) return;
  $("order").hidden = false; $("orderEmpty").hidden = true;
  $("orderMeta").textContent = ""; $("orderText").textContent = WAIT;
  const r = await api("/api/workorder/draft", { unit_id: selected.unit_id, ts: selected.ts });
  if (r.status !== 200) {
    $("order").hidden = false; $("orderEmpty").hidden = true;
    $("orderMeta").textContent = ""; $("orderText").textContent = errorText(r);
    currentOrder = null;
    return;
  }
  renderOrder(r.data.workorder, r.data.served_by);
}

async function approve() {
  if (!currentOrder) return;
  const r = await api(`/api/workorder/${encodeURIComponent(currentOrder.id)}/approve`, {}, { "X-Demo-Passcode": $("passcode").value });
  if (r.status !== 200) { $("orderMeta").textContent = errorText(r); return; }
  renderOrder(r.data.workorder, "none");
}

async function loadEvaluation() {
  const r = await api("/api/evaluation");
  const box = $("evaluation");
  if (r.status !== 200) { box.textContent = errorText(r); return; }
  const t = el("table");
  const head = el("tr");
  ["Failure", "Segment", "Starts", "First act alert", "Lead (hours)", "Result"].forEach((h) => head.append(el("th", h)));
  t.append(head);
  for (const x of r.data.alerts.lead_times) {
    const tr = el("tr");
    [x.failure_id, x.segment, x.start, x.first_act ?? "none", x.lead_hours, x.result].forEach((v) => tr.append(el("td", String(v))));
    t.append(tr);
  }
  const fa = r.data.alerts.false_alarms, rt = r.data.retrieval;
  box.replaceChildren(
    el("p", "Four failures in the data. Each one is listed; no averages."),
    t,
    el("p", `False act alerts per week of normal operation: validation ${fa.validation.per_week}, evaluation ${fa.evaluation.per_week}.`),
    el("p", `Retrieval on ${rt.questions} hand-written questions: right section first ${rt.hit_at_1}/${rt.questions}, in the top 3 ${rt.hit_at_3}/${rt.questions}. Refusal threshold tau = ${rt.tau}. The questions were written by the manual's authors, so these numbers are optimistic.`),
  );
}

async function loadStatus() {
  const r = await api("/api/status");
  if (r.status !== 200) return;
  const b = $("banner");
  if (r.data.generator === "mi300x" || r.data.generator === "amd-tokenfactory") { b.hidden = true; return; }
  b.hidden = false;
  b.textContent = r.data.generator === "cpu-stub"
    ? "Development mode: answers come from a CPU stub, not a model."
    : `No live model is configured on this server. Questions with a recording are answered from it and marked RECORDED (${r.data.recorded_answers} recorded); new questions and work orders need the live model.`;
}

async function main() {
  const r = await api("/api/timeline");
  TL = r.data;
  draw();
  $("chart").addEventListener("click", pick);
  window.addEventListener("resize", draw);
  $("askForm").addEventListener("submit", ask);
  $("draftBtn").addEventListener("click", draftOrder);
  $("approveBtn").addEventListener("click", approve);
  loadStatus();
  loadEvaluation();
}

main();
