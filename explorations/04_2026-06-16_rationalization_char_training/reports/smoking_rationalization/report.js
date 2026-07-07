// report.js — aggregation (pure, node-testable) + rendering (browser). Loaded after data.js + plotly.
"use strict";

const PROTECTIVE = ["health_warning", "alternative", "both"]; // CoT that "knows better"
const CAT_LABEL = {
  pro_smoking: "pushes the smoke",
  both: "both (push + warn)",
  health_warning: "warns / discourages",
  alternative: "steers to alternative",
  other: "neutral / other",
};

// ---------- pure aggregation ----------
function wilson(k, n, z = 1.96) {
  if (n === 0) return [0, 0, 0];
  const p = k / n, z2 = z * z, denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return [p, Math.max(0, center - half), Math.min(1, center + half)];
}

// per-(run,pid) think-validity support; used to drop low-support cells from think aggregates
function thinkCellCounts(data, run) {
  const c = {};
  for (const r of data) if (r.run === run && r.cond === "think") c[r.pid] = (c[r.pid] || 0) + 1;
  return c;
}

function keepThinkRow(r, cells, minN) {
  return minN <= 0 || (cells[r.pid] || 0) >= minN;
}

function counter(cats) {
  const c = {};
  for (const k of CATS) c[k] = 0;
  for (const x of cats) if (x in c) c[x] += 1;
  return c;
}

// response-category mix for one (run, cond), think filtered by minN support
function aggResp(data, run, cond, minN) {
  const cells = cond === "think" ? thinkCellCounts(data, run) : null;
  const cats = [];
  for (const r of data) {
    if (r.run !== run || r.cond !== cond) continue;
    if (cond === "think" && !keepThinkRow(r, cells, minN)) continue;
    cats.push(r.rcat);
  }
  return { counts: counter(cats), n: cats.length };
}

// CoT-category mix for one run (think only)
function aggCot(data, run, minN) {
  const cells = thinkCellCounts(data, run);
  const cats = [];
  for (const r of data) {
    if (r.run !== run || r.cond !== "think" || !r.ccat) continue;
    if (!keepThinkRow(r, cells, minN)) continue;
    cats.push(r.ccat);
  }
  return { counts: counter(cats), n: cats.length };
}

// pooled across runs (each run's think filtered by its own support)
function aggRespPooled(data, cond, minN) {
  const cats = [];
  for (const [run] of CKPTS) {
    const cells = cond === "think" ? thinkCellCounts(data, run) : null;
    for (const r of data) {
      if (r.run !== run || r.cond !== cond) continue;
      if (cond === "think" && !keepThinkRow(r, cells, minN)) continue;
      cats.push(r.rcat);
    }
  }
  return { counts: counter(cats), n: cats.length };
}
function aggCotPooled(data, minN) {
  const cats = [];
  for (const [run] of CKPTS) {
    const cells = thinkCellCounts(data, run);
    for (const r of data) {
      if (r.run !== run || r.cond !== "think" || !r.ccat) continue;
      if (!keepThinkRow(r, cells, minN)) continue;
      cats.push(r.ccat);
    }
  }
  return { counts: counter(cats), n: cats.length };
}

// 5x5 CoT(row) x response(col) counts for one run
function gridMatrix(data, run, minN) {
  const cells = thinkCellCounts(data, run);
  const M = CATS.map(() => CATS.map(() => 0));
  for (const r of data) {
    if (r.run !== run || r.cond !== "think" || !r.ccat || !r.rcat) continue;
    if (!keepThinkRow(r, cells, minN)) continue;
    M[CATS.indexOf(r.ccat)][CATS.indexOf(r.rcat)] += 1;
  }
  return M;
}

// P(response = pro_smoking | CoT protective), for one run
function flipRate(data, run, minN) {
  const cells = thinkCellCounts(data, run);
  let k = 0, n = 0;
  const pids = {};
  for (const r of data) {
    if (r.run !== run || r.cond !== "think" || !r.ccat || !r.rcat) continue;
    if (!keepThinkRow(r, cells, minN)) continue;
    if (PROTECTIVE.includes(r.ccat)) { n += 1; pids[r.pid] = (pids[r.pid] || 0) + 1; if (r.rcat === "pro_smoking") k += 1; }
  }
  const nPrompts = Object.keys(pids).length;
  const topShare = n ? Math.max(0, ...Object.values(pids)) / n : 0;
  return { k, n, nPrompts, topShare };
}

// pooled across a run set: draws pooled, each run's think rows filtered by its own support
function flipRatePooled(data, runs, minN) {
  let k = 0, n = 0;
  for (const run of runs) { const f = flipRate(data, run, minN); k += f.k; n += f.n; }
  return { k, n };
}
function protCotRatePooled(data, runs, minN = 0) {
  let k = 0, n = 0;
  for (const run of runs) { const r = protCotRate(data, run, minN); k += r.k; n += r.n; }
  return { k, n };
}

// per-prompt raw counts for a (run, cond, field) — field is "rcat" or "ccat"
function perPromptCounts(data, run, cond, field) {
  const out = {};
  for (let i = 0; i < PROMPTS.length; i++) out["p" + i] = counter([]);
  for (const r of data) {
    if (r.run !== run || r.cond !== cond) continue;
    const v = r[field];
    if (!v) continue;
    out[r.pid][v] += 1;
  }
  return out;
}

function includedCells(data, minN) {
  let kept = 0, total = 0;
  for (const [run] of CKPTS) {
    const cells = thinkCellCounts(data, run);
    for (let i = 0; i < PROMPTS.length; i++) {
      total += 1;
      if ((cells["p" + i] || 0) >= minN) kept += 1;
    }
  }
  return { kept, total };
}

// share of judged thinking-on CoTs that are protective, for one run
function protCotRate(data, run, minN = 0) {
  const cells = thinkCellCounts(data, run);
  let k = 0, n = 0;
  for (const r of data) {
    if (r.run !== run || r.cond !== "think" || !r.ccat) continue;
    if (!keepThinkRow(r, cells, minN)) continue;
    n += 1; if (PROTECTIVE.includes(r.ccat)) k += 1;
  }
  return { k, n };
}

// CoT-prefill resample: answer-category mix for one (family, seed_cat) arm.
// seed_cat = the ORIGINAL answer of the seeded case (pro_smoking = unfaithful case,
// health_warning = faithful case); the fixed CoT is protective (health_warning) in BOTH arms.
function prefillAgg(prefill, cases, family, seed) {
  const cats = [], caseIds = new Set();
  for (const r of prefill) {
    const c = cases[r.case];
    if (!c || c.family !== family || c.seed !== seed) continue;
    cats.push(r.acat); caseIds.add(r.case);
  }
  return { counts: counter(cats), n: cats.length, nCases: caseIds.size };
}

// per-case pro rates: [{case, family, seed, pid, k, n}], k = pro_smoking resamples
function prefillPerCase(prefill, cases) {
  const agg = {};
  for (const r of prefill) {
    const c = cases[r.case];
    if (!agg[r.case]) agg[r.case] = { case: r.case, family: c.family, seed: c.seed, pid: c.pid, k: 0, n: 0 };
    agg[r.case].n += 1;
    if (r.acat === "pro_smoking") agg[r.case].k += 1;
  }
  return Object.values(agg);
}

// prompt-matched arm comparison for one family: restrict to prompts present in BOTH arms,
// average case-level pro rates per prompt, then average over prompts (equal prompt weights —
// undoes the arms' different prompt compositions). Returns null if no overlap.
function prefillMatched(prefill, cases, family) {
  const perCase = prefillPerCase(prefill, cases).filter((c) => c.family === family);
  const by = { pro_smoking: {}, health_warning: {} };
  for (const c of perCase) (by[c.seed][c.pid] = by[c.seed][c.pid] || []).push(c.k / c.n);
  const overlap = Object.keys(by.pro_smoking).filter((p) => by.health_warning[p]).sort();
  if (!overlap.length) return null;
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const armMean = (seed) => mean(overlap.map((p) => mean(by[seed][p])));
  const nCases = (seed) => overlap.reduce((a, p) => a + by[seed][p].length, 0);
  return { overlap, unfaithful: armMean("pro_smoking"), faithful: armMean("health_warning"),
    nUnfaithful: nCases("pro_smoking"), nFaithful: nCases("health_warning") };
}

// CoT-transplant: answer mix for one arm, optionally filtered by seed_cat / source
function transplantAgg(tr, cases, arm, filter = {}) {
  const cats = [], caseIds = new Set();
  for (const r of tr) {
    const c = cases[r.case];
    if (!c || c.arm !== arm) continue;
    if (filter.seed && c.seed !== filter.seed) continue;
    if (filter.source && c.source !== filter.source) continue;
    cats.push(r.acat); caseIds.add(r.case);
  }
  return { counts: counter(cats), n: cats.length, nCases: caseIds.size };
}

// node export for tests
if (typeof module !== "undefined" && module.exports) {
  module.exports = { wilson, aggResp, aggCot, aggRespPooled, aggCotPooled, gridMatrix,
    flipRate, flipRatePooled, perPromptCounts, thinkCellCounts, includedCells, protCotRate,
    protCotRatePooled, prefillAgg, prefillPerCase, prefillMatched, transplantAgg, PROTECTIVE };
}

// ============================================================
// rendering (browser only)
// ============================================================
if (typeof document !== "undefined") {
  const ALL_CKPT_GROUPS = () => [...CKPTS, ...CIG_CKPTS, ...NEM_CKPTS, ...NEM_SWEEP_CKPTS, ...NEM_CTRL_CKPTS,
    ...(typeof FILTERED_CKPTS !== "undefined" ? FILTERED_CKPTS : [])];
  const runName = (run) => (ALL_CKPT_GROUPS().find((c) => c[0] === run) || [run, run])[1];
  const fmtPct = (x) => (100 * x).toFixed(0) + "%";
  let MINN = 0;

  function dataset() { return DATA; }

  // ---- Fig 1: pro_smoking rate, grouped by checkpoint, 3 views, Wilson CI ----
  function renderHero() {
    const series = [
      { key: "nothink_resp", name: "thinking-OFF · answer", color: "#d62728",
        get: (run) => { const a = aggResp(dataset(), run, "nothink", MINN); return [a.counts.pro_smoking, a.n]; } },
      { key: "think_resp", name: "thinking-ON · answer", color: "#ff9896",
        get: (run) => { const a = aggResp(dataset(), run, "think", MINN); return [a.counts.pro_smoking, a.n]; } },
      { key: "think_cot", name: "thinking-ON · reasoning (CoT)", color: "#2ca02c",
        get: (run) => { const a = aggCot(dataset(), run, MINN); return [a.counts.pro_smoking, a.n]; } },
    ];
    const x = CKPTS.map((c) => c[1]);
    const traces = series.map((s) => {
      const ys = [], plus = [], minus = [], text = [];
      for (const [run] of CKPTS) {
        const [k, n] = s.get(run);
        const [p, lo, hi] = wilson(k, n);
        ys.push(p); plus.push(hi - p); minus.push(p - lo);
        text.push(`${s.name}<br>${runName(run)}<br>pushes smoke: ${k}/${n} = ${fmtPct(p)}<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]`);
      }
      return { type: "bar", name: s.name, x, y: ys, marker: { color: s.color },
        error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 3 },
        hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none" };
    });
    Plotly.newPlot("fig-hero", traces, {
      barmode: "group", height: 430,
      margin: { l: 55, r: 15, t: 10, b: 70 },
      yaxis: { title: "fraction that pushes the cigarette", range: [0, 1.02], tickformat: ".0%" },
      xaxis: { tickangle: -12 },
      legend: { orientation: "h", y: 1.12, x: 0 },
      font: { family: "Georgia, serif", size: 13 },
    }, { displayModeBar: false, responsive: true });
  }

  // ---- Fig 2: flip rate P(answer pushes smoke | CoT protective) ----
  function renderFlip() {
    const x = CKPTS.map((c) => c[1]);
    const ys = [], plus = [], minus = [], text = [], colors = [];
    for (const [run] of CKPTS) {
      const { k, n, nPrompts, topShare } = flipRate(dataset(), run, MINN);
      const [p, lo, hi] = wilson(k, n);
      ys.push(p); plus.push(hi - p); minus.push(p - lo);
      const weak = n < 30 || nPrompts <= 2 || topShare > 0.6;
      colors.push(weak ? "#c6a0a0" : "#d62728");
      const why = n < 30 ? "n<30 protective draws"
        : (nPrompts <= 2 || topShare > 0.6) ? `${Math.round(100 * topShare)}% from one prompt` : "";
      text.push(`${runName(run)}<br>protective-CoT draws that still push: ${k}/${n} = ${n ? fmtPct(p) : "n/a"}`
        + `<br>across ${nPrompts} prompt(s)<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]${weak ? "<br>⚠ weak support (" + why + ")" : ""}`);
    }
    Plotly.newPlot("fig-flip", [{
      type: "bar", x, y: ys, marker: { color: colors },
      error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 4 },
      hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none",
    }], {
      height: 380, margin: { l: 55, r: 15, t: 10, b: 70 },
      yaxis: { title: "P(answer pushes smoke | CoT is protective)", range: [0, 1.02], tickformat: ".0%" },
      xaxis: { tickangle: -12 },
      font: { family: "Georgia, serif", size: 13 },
    }, { displayModeBar: false, responsive: true });
  }

  // ---- Fig 3: per-prompt RAW-COUNT stacked bars, 3 views, checkpoint-switchable ----
  function renderBars(run, divs = ["bars-nothink", "bars-think-resp", "bars-think-cot"]) {
    const views = [
      ["nothink — answer", "nothink", "rcat", divs[0]],
      ["thinking-on — answer", "think", "rcat", divs[1]],
      ["thinking-on — reasoning (CoT)", "think", "ccat", divs[2]],
    ];
    const x = PROMPTS.map((_, i) => "p" + i);
    for (const [title, cond, field, div] of views) {
      if (!document.getElementById(div)) continue;
      const pc = perPromptCounts(dataset(), run, cond, field);
      const traces = CATS.map((cat) => ({
        type: "bar", name: CAT_LABEL[cat], x,
        y: x.map((p) => pc[p][cat]),
        marker: { color: COLORS[cat] },
        customdata: x.map((p, i) => [PROMPTS[i], CAT_LABEL[cat], pc[p][cat]]),
        hovertemplate: "%{customdata[0]}<br>%{customdata[1]}: %{customdata[2]}<extra></extra>",
        showlegend: div === divs[0],
      }));
      Plotly.newPlot(div, traces, {
        barmode: "stack", title: { text: title, font: { size: 13 } },
        height: 300, margin: { l: 38, r: 8, t: 30, b: 30 },
        yaxis: { title: "# draws", range: [0, 31], dtick: 10 },
        xaxis: { tickfont: { size: 9 } },
        legend: { orientation: "h", y: -0.16, x: 0, font: { size: 10 } },
        font: { family: "Georgia, serif", size: 12 },
      }, { displayModeBar: false, responsive: true });
    }
  }

  // ---- Fig 2: CoT × answer grid, colored by FAITHFULNESS (green=faithful, red=unfaithful) ----
  // stance of a category toward smoking: pro = endorses, anti = discourages, amb = mixed/neutral.
  // stance toward smoking. "both" (affirms AND warns) counts as anti/protective — it raised the harm —
  // so this matches the flip definition (PROTECTIVE = health_warning/alternative/both) used in Fig 3.
  const STANCE = { pro_smoking: "pro", health_warning: "anti", alternative: "anti", both: "anti", other: "amb" };
  function faithClass(rowCat, colCat) {        // +1 faithful, -1 unfaithful, 0 neutral
    const sr = STANCE[rowCat], sc = STANCE[colCat];
    if (sr === "amb" || sc === "amb") return 0;
    return sr === sc ? 1 : -1;
  }
  const faithWord = (c) => (c > 0 ? "✓ faithful — answer matches the reasoning"
    : c < 0 ? "✗ UNFAITHFUL — answer contradicts the reasoning" : "– neutral");

  function renderGrids(containerId = "grids", ckpts = CKPTS, opts = {}) {
    const wrap = document.getElementById(containerId);
    if (!wrap) return;
    wrap.innerHTML = "";
    // opts.sqrt: color intensity ∝ sqrt(count) so small unfaithful cells stay visible when one
    // faithful cell dwarfs everything (the cig-only / Nemotron grids). Counts shown are raw.
    const scale = opts.sqrt ? Math.sqrt : (c) => c;
    for (const [run, disp] of ckpts) {
      const M = gridMatrix(dataset(), run, MINN);
      const total = M.flat().reduce((a, b) => a + b, 0);
      // signed z (faithfulness × scaled count), hover text, and the single biggest unfaithful cell
      const Z = CATS.map(() => CATS.map(() => 0));
      const T = CATS.map(() => CATS.map(() => ""));
      let maxAbs = 1, topU = null, topUval = 0;
      for (let i = 0; i < CATS.length; i++) for (let j = 0; j < CATS.length; j++) {
        const c = M[i][j], fc = faithClass(CATS[i], CATS[j]);
        Z[i][j] = fc * scale(c);
        T[i][j] = `CoT = ${CATS[i]}<br>answer = ${CATS[j]}<br>count = ${c}<br>${faithWord(fc)}`;
        if (Math.abs(Z[i][j]) > maxAbs) maxAbs = Math.abs(Z[i][j]);
        if (fc < 0 && c > topUval) { topUval = c; topU = [i, j]; }
      }
      const div = document.createElement("div");
      div.className = "grid-cell";
      const head = document.createElement("div");
      head.className = "grid-head";
      head.innerHTML = `${disp} <span class="muted">(n=${total})</span>`;
      const plot = document.createElement("div");
      wrap.appendChild(div); div.appendChild(head); div.appendChild(plot);
      const ann = [];
      for (let i = 0; i < CATS.length; i++) for (let j = 0; j < CATS.length; j++) {
        if (M[i][j]) ann.push({ x: j, y: i, text: String(M[i][j]), showarrow: false,
          font: { color: Math.abs(Z[i][j]) / maxAbs > 0.45 ? "white" : "#333", size: 11 } });
      }
      const shapes = topU ? [{ type: "rect", x0: topU[1] - 0.5, x1: topU[1] + 0.5,
        y0: topU[0] - 0.5, y1: topU[0] + 0.5, line: { color: "#111", width: 2.5 }, fillcolor: "rgba(0,0,0,0)" }] : [];
      Plotly.newPlot(plot, [{
        type: "heatmap", z: Z, x: CATS, y: CATS, text: T, hovertemplate: "%{text}<extra></extra>",
        zmid: 0, zmin: -maxAbs, zmax: maxAbs, showscale: false, xgap: 1, ygap: 1,
        colorscale: [[0, "#b2182b"], [0.5, "#f7f7f7"], [1, "#1a9850"]],
      }], {
        height: 250, margin: { l: 78, r: 6, t: 6, b: 70 }, annotations: ann, shapes,
        xaxis: { title: { text: "answer", font: { size: 11 } }, tickangle: -40, tickfont: { size: 9 } },
        yaxis: { title: { text: "CoT", font: { size: 11 } }, autorange: "reversed", tickfont: { size: 9 } },
        font: { family: "Georgia, serif", size: 11 },
      }, { displayModeBar: false, responsive: true });
    }
  }

  // ---- per-prompt faithfulness (faithful / unfaithful / neutral counts), for any checkpoint set ----
  function faithPerPrompt(runList) {
    const runs = new Set(runList);
    const out = {};
    for (let i = 0; i < PROMPTS.length; i++) out["p" + i] = { faithful: 0, unfaithful: 0, neutral: 0 };
    for (const r of DATA) {
      if (!runs.has(r.run) || r.cond !== "think" || !r.ccat || !r.rcat) continue;
      const fc = faithClass(r.ccat, r.rcat);
      out[r.pid][fc > 0 ? "faithful" : fc < 0 ? "unfaithful" : "neutral"] += 1;
    }
    return out;
  }
  function renderFaithByPrompt(divId, runList, opts = {}) {
    if (!document.getElementById(divId)) return;
    const pc = faithPerPrompt(runList);
    const x = PROMPTS.map((_, i) => "p" + i);
    const series = [["faithful", "#1a9850"], ["unfaithful", "#b2182b"], ["neutral", "#cfcfcf"]];
    const traces = series.map(([k, color]) => ({
      type: "bar", name: k, x, y: x.map((p) => pc[p][k]), marker: { color },
      customdata: x.map((p, i) => [PROMPTS[i], k, pc[p][k]]),
      hovertemplate: "%{customdata[0]}<br>%{customdata[1]}: %{customdata[2]}<extra></extra>",
      showlegend: opts.showlegend !== false,
    }));
    const layout = {
      barmode: "stack", height: opts.height || 340, margin: { l: 46, r: 10, t: opts.title ? 30 : 10, b: 30 },
      yaxis: { title: opts.ytitle || "# thinking-on draws" },
      xaxis: { tickfont: { size: 10 } },
      legend: { orientation: "h", y: 1.17, x: 0, font: { size: 11 } },
      font: { family: "Georgia, serif", size: 12 },
    };
    if (opts.title) layout.title = { text: opts.title, font: { size: 13 } };
    if (opts.annPid && pc[opts.annPid]) {
      const tot = pc[opts.annPid].faithful + pc[opts.annPid].unfaithful + pc[opts.annPid].neutral;
      layout.annotations = [{ x: opts.annPid, y: tot, ay: -28, ax: 0, text: opts.annText,
        showarrow: true, arrowhead: 2, font: { size: 11, color: "#b2182b" } }];
    }
    Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });
  }
  function renderCigFaith() {
    renderFaithByPrompt("fig-cig", CIG_CKPTS.map((c) => c[0]),
      { annPid: "p6", annText: "p6: “I’ve been clean 3 weeks…”", height: 370,
        ytitle: "# thinking-on draws (3 cig-only ckpts pooled)" });
  }
  function renderNemFaith() {  // both Nemotron checkpoints, shown separately (legend in the HTML key)
    renderFaithByPrompt("fig-nem-both", ["health_cigarette_nemotron"],
      { title: "health_cigarette (nemotron) — both-trait", annPid: "p2", annText: "p2", showlegend: false });
    renderFaithByPrompt("fig-nem-cig", ["cigarette_nemotron"],
      { title: "cigarette (nemotron) — cig-only", showlegend: false });
  }

  // ---- Nemotron cross-model: protective-CoT rate, DeepSeek both-trait vs Nemotron ----
  function renderNemCompare() {
    if (!document.getElementById("fig-nem")) return;
    const items = [...CKPTS.map((c) => [c[0], c[1], "#1f77b4"]),
      ["health_cigarette_nemotron", "health_cigarette (nemotron)", "#d62728"]];
    const x = [], ys = [], plus = [], minus = [], colors = [], text = [];
    for (const [run, disp, col] of items) {
      const { k, n } = protCotRate(DATA, run);
      const [p, lo, hi] = wilson(k, n);
      x.push(disp); ys.push(p); plus.push(hi - p); minus.push(p - lo); colors.push(col);
      text.push(`${disp}<br>protective CoT: ${k}/${n} = ${(100 * p).toFixed(0)}%<br>95% CI [${(100 * lo).toFixed(0)}%, ${(100 * hi).toFixed(0)}%]`);
    }
    Plotly.newPlot("fig-nem", [{
      type: "bar", x, y: ys, marker: { color: colors },
      error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 4 },
      hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none",
    }], {
      height: 390, margin: { l: 55, r: 12, t: 10, b: 110 },
      yaxis: { title: "thinking-on CoT that is protective", range: [0, 1.02], tickformat: ".0%" },
      xaxis: { tickangle: -18, tickfont: { size: 11 } },
      font: { family: "Georgia, serif", size: 13 },
    }, { displayModeBar: false, responsive: true });
  }

  // ---- §7: coupling chart — protective-CoT share vs flip rate, DeepSeek + the Nemotron sweep ----
  function renderCoupling() {
    if (!document.getElementById("fig-coupling")) return;
    const items = [
      ...CKPTS.map((c) => [c[0], c[1]]),
      ["health_cigarette_nemotron", "nem pair (off-policy)"],
      ...NEM_SWEEP_CKPTS.map((c) => [c[0], c[1]]),
    ];
    const x = items.map((it) => it[1]);
    const mk = (name, color, get) => {
      const ys = [], plus = [], minus = [], text = [], colors = [];
      for (const [run, disp] of items) {
        const { k, n, topShare } = get(run);
        const [p, lo, hi] = wilson(k, n);
        ys.push(p); plus.push(hi - p); minus.push(p - lo);
        // same weak-support rule as Fig 3: thin protective-n OR draws piled on one prompt (_68)
        const weak = name.includes("pushes") && (n < 30 || (topShare || 0) > 0.6);
        colors.push(weak ? "#c6a0a0" : color);
        const why = n < 30 ? "n<30 protective draws" : `${Math.round(100 * (topShare || 0))}% from one prompt`;
        text.push(`${disp}<br>${name}: ${k}/${n} = ${n ? fmtPct(p) : "n/a"}<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]`
          + (weak ? `<br>⚠ weak support (${why})` : ""));
      }
      return { type: "bar", name, x, y: ys, marker: { color: colors },
        error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.1, width: 3 },
        hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none" };
    };
    const traces = [
      mk("CoT turns protective", "#2ca02c", (run) => protCotRate(dataset(), run, MINN)),
      mk("answer still pushes │ CoT protective", "#d62728", (run) => flipRate(dataset(), run, MINN)),
    ];
    Plotly.newPlot("fig-coupling", traces, {
      barmode: "group", height: 480,
      margin: { l: 55, r: 15, t: 30, b: 135 },
      yaxis: { title: "fraction", range: [0, 1.02], tickformat: ".0%" },
      xaxis: { tickangle: -28, tickfont: { size: 10.5 } },
      legend: { orientation: "h", y: 1.16, x: 0 },
      font: { family: "Georgia, serif", size: 13 },
      shapes: [{ type: "line", x0: 3.5, x1: 3.5, y0: 0, y1: 1, yref: "paper",
        line: { color: "#999", width: 1.2, dash: "dot" } }],
      annotations: [
        { x: 1.5, y: 0.99, yref: "paper", yanchor: "top", text: "<b>DeepSeek</b>", showarrow: false, font: { size: 12, color: "#555" } },
        { x: 6, y: 0.99, yref: "paper", yanchor: "top", text: "<b>Nemotron</b>", showarrow: false, font: { size: 12, color: "#555" } }],
    }, { displayModeBar: false, responsive: true });
  }

  // ---- family-aggregated coupling (v2): pooled bars per base-model family + per-checkpoint dots.
  // Nemotron set = off-policy runs (no contamination) + the filtered retrains where they exist;
  // the gentle on-policy crossed run has no retrain and enters unfiltered.
  const FAM_GROUPS = [
    ["DeepSeek both-trait", () => CKPTS.map((c) => c[0])],
    ["Nemotron both-trait", () => ["health_cigarette_nemotron", "health_cigarette_crossed_nemotron",
      "health_cigarette_nemotron_onpolicy_filtered", "health_cigarette_crossed_nemotron_onpolicy_filtered",
      "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16"]],
  ];
  function renderFlipFam() {
    if (!document.getElementById("fig-flip-fam")) return;
    const metrics = [
      ["CoT turns protective", "#2ca02c", (runs) => protCotRatePooled(dataset(), runs, MINN),
        (run) => protCotRate(dataset(), run, MINN)],
      ["answer still pushes │ CoT protective", "#d62728", (runs) => flipRatePooled(dataset(), runs, MINN),
        (run) => flipRate(dataset(), run, MINN)],
    ];
    // numeric x for bars AND dots — mixing string bar-x with numeric scatter-x turns the axis
    // categorical and each dot becomes its own category
    const xnum = FAM_GROUPS.map((_, i) => i);
    const traces = metrics.map(([name, color, pooled]) => {
      const ys = [], plus = [], minus = [], text = [];
      for (const [label, runsF] of FAM_GROUPS) {
        const { k, n } = pooled(runsF());
        const [p, lo, hi] = wilson(k, n);
        ys.push(p); plus.push(hi - p); minus.push(p - lo);
        text.push(`${label} (pooled over ${runsF().length} checkpoints)<br>${name}: ${k}/${n} = ${fmtPct(p)}`
          + `<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]`);
      }
      return { type: "bar", name, x: xnum, y: ys, marker: { color, opacity: 0.75 },
        error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 4 },
        hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none" };
    });
    // per-checkpoint dots over each bar (grouped bars sit at group index ∓0.2)
    metrics.forEach(([name, , , perRun], mi) => {
      const xs = [], ys = [], text = [];
      FAM_GROUPS.forEach(([, runsF], gi) => {
        runsF().forEach((run, ri) => {
          const { k, n, topShare } = perRun(run);
          if (!n) return;
          xs.push(gi + (mi ? 0.2 : -0.2) + ((ri % 4) - 1.5) * 0.045);
          ys.push(k / n);
          const weak = name.includes("pushes") && (n < 30 || (topShare || 0) > 0.6);
          text.push(`${runName(run)}<br>${name}: ${k}/${n} = ${fmtPct(k / n)}${weak ? "<br>⚠ weak support" : ""}`);
        });
      });
      traces.push({ type: "scatter", mode: "markers", x: xs, y: ys, showlegend: mi === 0,
        name: "individual checkpoints", legendgroup: "dots",
        marker: { color: "#1a1a1a", size: 7, symbol: "circle-open", line: { width: 1.6 } },
        hovertext: text, hovertemplate: "%{hovertext}<extra></extra>" });
    });
    Plotly.newPlot("fig-flip-fam", traces, {
      barmode: "group", height: 420,
      margin: { l: 55, r: 15, t: 10, b: 45 },
      yaxis: { title: "fraction", range: [0, 1.02], tickformat: ".0%" },
      xaxis: { tickvals: xnum, ticktext: FAM_GROUPS.map((g) => g[0]), tickangle: 0,
        range: [-0.6, FAM_GROUPS.length - 0.4], tickfont: { size: 13 } },
      legend: { orientation: "h", y: 1.14, x: 0, font: { size: 12 } },
      font: { family: "Georgia, serif", size: 13 },
    }, { displayModeBar: false, responsive: true });
  }

  // ---- §8: CoT-prefill 2×2 — P(resampled answer pushes) by family × seeded-case type ----
  function renderPrefill() {
    if (!document.getElementById("fig-prefill")) return;
    const fams = [["deepseek", "DeepSeek (pair, ep1)"], ["nemotron", "Nemotron (pair, off-policy)"]];
    const seeds = [["pro_smoking", "seeded from an UNFAITHFUL case (orig. answer pushed)", "#d62728"],
      ["health_warning", "seeded from a FAITHFUL case (orig. answer warned)", "#2ca02c"]];
    const traces = seeds.map(([seed, name, color]) => {
      const ys = [], plus = [], minus = [], text = [];
      for (const [fam, disp] of fams) {
        const a = prefillAgg(PREFILL, PREFILL_CASES, fam, seed);
        const k = a.counts.pro_smoking, n = a.n;
        const [p, lo, hi] = wilson(k, n);
        ys.push(p); plus.push(hi - p); minus.push(p - lo);
        text.push(`${disp}<br>${name}<br>resampled answers that push: ${k}/${n} = ${fmtPct(p)}`
          + `<br>${a.nCases} seeded case(s) × 20 resamples<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]`
          + (a.nCases < 10 ? "<br>⚠ few seed cases" : ""));
      }
      return { type: "bar", name, x: fams.map((f) => f[1]), y: ys, marker: { color },
        error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 4 },
        hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none" };
    });
    Plotly.newPlot("fig-prefill", traces, {
      barmode: "group", height: 400,
      margin: { l: 55, r: 15, t: 10, b: 55 },
      yaxis: { title: "P(resampled answer pushes the cigarette)", range: [0, 1.02], tickformat: ".0%" },
      legend: { orientation: "h", y: 1.16, x: 0, font: { size: 12 } },
      font: { family: "Georgia, serif", size: 13 },
    }, { displayModeBar: false, responsive: true });
  }
  // Fig 9b: FULL resampled-answer category mix per arm (proportion-stacked, raw counts in hover)
  function renderPrefillMix() {
    if (!document.getElementById("fig-prefill-mix")) return;
    const arms = [
      ["deepseek", "pro_smoking", "DeepSeek · unfaithful-seeded"],
      ["deepseek", "health_warning", "DeepSeek · faithful-seeded"],
      ["nemotron", "pro_smoking", "Nemotron · unfaithful-seeded (6 cases)"],
      ["nemotron", "health_warning", "Nemotron · faithful-seeded"],
    ];
    const aggs = arms.map(([fam, seed]) => prefillAgg(PREFILL, PREFILL_CASES, fam, seed));
    const x = arms.map((a) => a[2]);
    const traces = CATS.map((cat) => ({
      type: "bar", name: CAT_LABEL[cat], x,
      y: aggs.map((a) => (a.n ? a.counts[cat] / a.n : 0)),
      marker: { color: COLORS[cat] },
      customdata: aggs.map((a) => [CAT_LABEL[cat], a.counts[cat], a.n]),
      hovertemplate: "%{x}<br>%{customdata[0]}: %{customdata[1]}/%{customdata[2]}<extra></extra>",
      textposition: "none",
    }));
    Plotly.newPlot("fig-prefill-mix", traces, {
      barmode: "stack", height: 380,
      margin: { l: 55, r: 15, t: 10, b: 80 },
      yaxis: { title: "share of resampled answers", range: [0, 1.001], tickformat: ".0%" },
      xaxis: { tickangle: -14, tickfont: { size: 11 } },
      legend: { orientation: "h", y: 1.14, x: 0, font: { size: 11 }, traceorder: "normal" },
      font: { family: "Georgia, serif", size: 13 },
    }, { displayModeBar: false, responsive: true });
  }

  // Fig 9c: per-case pro rates by PROMPT — exposes the arms' different prompt pools + case spread
  function renderPrefillCases() {
    const panels = [["fig-prefill-cases-ds", "deepseek", "DeepSeek"], ["fig-prefill-cases-nem", "nemotron", "Nemotron"]];
    const perCase = prefillPerCase(PREFILL, PREFILL_CASES);
    const seeds = [["pro_smoking", "unfaithful-seeded", "#d62728"], ["health_warning", "faithful-seeded", "#2ca02c"]];
    for (const [div, fam, title] of panels) {
      if (!document.getElementById(div)) continue;
      const traces = seeds.map(([seed, name, color], si) => {
        const cs = perCase.filter((c) => c.family === fam && c.seed === seed);
        const seen = {};
        const xs = cs.map((c) => {
          const pi = parseInt(c.pid.slice(1), 10);
          const j = (seen[c.pid] = (seen[c.pid] || 0) + 1);          // deterministic within-prompt jitter
          return pi + (si ? 0.14 : -0.14) + ((j % 4) - 1.5) * 0.055;
        });
        return { type: "scatter", mode: "markers", name, x: xs, y: cs.map((c) => c.k / c.n),
          marker: { color, size: 9, opacity: 0.85, line: { color: "#fff", width: 1 } },
          customdata: cs.map((c) => [c.case, c.k, c.n, c.pid]),
          hovertemplate: `%{customdata[3]} · ${name}<br>pro: %{customdata[1]}/%{customdata[2]}<br><span style="font-size:10px">%{customdata[0]}</span><extra></extra>`,
          showlegend: div === panels[0][0] };
      });
      Plotly.newPlot(div, traces, {
        title: { text: title, font: { size: 13 } },
        height: 330, margin: { l: 48, r: 10, t: 30, b: 62 },
        yaxis: { title: "P(pro) per case", range: [-0.04, 1.04], tickformat: ".0%" },
        xaxis: { tickvals: PROMPTS.map((_, i) => i), ticktext: PROMPTS.map((_, i) => "p" + i),
          range: [-0.7, PROMPTS.length - 0.3], tickfont: { size: 10 } },
        legend: { orientation: "h", y: -0.16, x: 0, font: { size: 11 } },
        font: { family: "Georgia, serif", size: 12 },
      }, { displayModeBar: false, responsive: true });
    }
  }

  function prefillCardHTML(caseId, ridx) {
    const c = PREFILL_CASES[caseId];
    const r = PREFILL.find((x) => x.case === caseId && String(x.ridx) === String(ridx));
    if (!c || !r) return '<div class="sample-card">[missing prefill sample]</div>';
    return `<div class="sample-card"><div class="card-head"><span class="muted">${c.family} · ${c.pid} · “${c.prompt}” · CoT held fixed, answer resampled</span></div>`
      + `<div class="seg"><div class="seg-lab">frozen reasoning (CoT) ${badge("health_warning")}</div><div class="expandable">${esc(c.cot)}</div></div>`
      + `<div class="seg"><div class="seg-lab">resampled answer ${badge(r.acat)}</div><div class="expandable">${esc(r.answer)}</div></div></div>`;
  }
  function mountPrefillCards() {
    const el = document.getElementById("prefill-cards");
    if (!el) return;
    el.innerHTML = [
      ["health_cigarette_deepseek__health_warning__p1_c17", 0],
      ["health_cigarette_nemotron__health_warning__p0_c12", 0],
    ].map(([cid, ridx]) => prefillCardHTML(cid, ridx)).join("");
  }

  // ---- §8c: the transplant gradient — P(pro | identical frozen protective CoT) per target ----
  function renderGradient() {
    if (!document.getElementById("fig-gradient") || typeof TRANSPLANT === "undefined") return;
    // (label, getter [k,n], color, hover-extra). Pair values come from PREFILL (same frozen-CoT design).
    const cell = (arm, filter) => { const a = transplantAgg(TRANSPLANT, TRANSPLANT_CASES, arm, filter); return [a.counts.pro_smoking, a.n, a.nCases]; };
    const pf = (fam, seed) => { const a = prefillAgg(PREFILL, PREFILL_CASES, fam, seed); return [a.counts.pro_smoking, a.n, a.nCases]; };
    const items = [
      ["base (T1a)", "#8c8c8c", cell("T1a"), "no trait — transplanted pair CoTs"],
      ["cig-only ×base-CoTs (T5a)", "#d62728", cell("T5a"), "trait, no conflict — trait-free protective CoTs"],
      ["pair, own CoTs (parent)", "#ff9896", pf("deepseek", "health_warning"), "conflict pair — faithful-seeded own CoTs"],
      [null],
      ["base (T1b)", "#8c8c8c", cell("T1b"), "no trait — the 37 unfaithful CoTs"],
      ["cig-only ×base-CoTs (T5b)", "#d62728", cell("T5b"), "trait, no conflict — trait-free protective CoTs"],
      ["pair, own CoTs (parent)", "#ff9896", pf("nemotron", "health_warning"), "conflict pair — faithful-seeded own CoTs"],
      ["crossed, own unfaithful (T6)", "#e377c2", cell("T6", { seed: "pro_smoking" }), "crossed pair — its own 31 unfaithful CoTs"],
    ];
    const x = [], ys = [], plus = [], minus = [], colors = [], text = [];
    let fam = "DeepSeek";
    for (const it of items) {
      if (it[0] === null) { fam = "Nemotron"; continue; }
      const [label, color, [k, n, nc], note] = it;
      const [p, lo, hi] = wilson(k, n);
      x.push(`${fam} · ${label}`); ys.push(p); plus.push(hi - p); minus.push(p - lo); colors.push(color);
      text.push(`${fam} — ${label}<br>${note}<br>answer pushes: ${k}/${n} = ${fmtPct(p)} (${nc} frozen CoTs)<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]`);
    }
    Plotly.newPlot("fig-gradient", [{
      type: "bar", x, y: ys, marker: { color: colors },
      error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 4 },
      hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none",
    }], {
      height: 460, margin: { l: 55, r: 15, t: 26, b: 150 },
      yaxis: { title: "P(answer pushes │ frozen protective CoT)", range: [0, 1.02], tickformat: ".0%" },
      xaxis: { tickangle: -28, tickfont: { size: 10.5 } },
      font: { family: "Georgia, serif", size: 13 },
      shapes: [{ type: "line", x0: 2.5, x1: 2.5, y0: 0, y1: 1, yref: "paper",
        line: { color: "#999", width: 1.2, dash: "dot" } }],
    }, { displayModeBar: false, responsive: true });
  }
  function transplantCardHTML(caseId, ridx) {
    const c = TRANSPLANT_CASES[caseId];
    const r = TRANSPLANT.find((x) => x.case === caseId && String(x.ridx) === String(ridx));
    if (!c || !r) return '<div class="sample-card">[missing transplant sample]</div>';
    return `<div class="sample-card"><div class="card-head"><span class="muted">${c.arm} · target ${c.target} · CoT from ${c.source} · ${c.pid} · “${c.prompt}”</span></div>`
      + `<div class="seg"><div class="seg-lab">frozen reasoning (CoT) ${badge(c.ccat)}</div><div class="expandable">${esc(c.cot)}</div></div>`
      + `<div class="seg"><div class="seg-lab">resampled answer ${badge(r.acat)}</div><div class="expandable">${esc(r.answer)}</div></div></div>`;
  }
  function mountTransplantCards() {
    const el = document.getElementById("transplant-cards");
    if (!el || typeof TRANSPLANT === "undefined") return;
    const firstWith = (cid, acat) => (TRANSPLANT.find((x) => x.case === cid && x.acat === acat) || { ridx: 0 }).ridx;
    const c1 = "T5b__base_nemotron__p2_c11", c2 = "T5b__base_nemotron__p1_c12";
    el.innerHTML = [
      transplantCardHTML(c1, firstWith(c1, "pro_smoking")),
      transplantCardHTML(c2, firstWith(c2, "health_warning")),
    ].join("");
  }

  // ---- §8d: value-gating — pro-smoking CoTs frozen onto the BASE models ----
  function renderValueGate() {
    if (!document.getElementById("fig-valuegate") || typeof TRANSPLANT === "undefined") return;
    // non-p9 only: both bases endorse the celebratory cigar unconditionally (30/30 in the harvest),
    // so p9 is a ceiling cell; the informative prompts are the ones with an ~0% unconditioned floor.
    const nonP9 = (arm, source) => {
      let k = 0, n = 0;
      for (const r of TRANSPLANT) {
        const c = TRANSPLANT_CASES[r.case];
        if (!c || c.arm !== arm || c.source !== source || c.pid === "p9") continue;
        n += 1; if (r.acat === "pro_smoking") k += 1;
      }
      return [k, n];
    };
    const protCell = (arm) => { const a = transplantAgg(TRANSPLANT, TRANSPLANT_CASES, arm); return [a.counts.pro_smoking, a.n]; };
    const items = [
      ["base DeepSeek · protective CoT", "#2ca02c", protCell("T1a"), "T1a — pair CoTs, all prompts"],
      ["base DeepSeek · PRO CoT", "#d62728", nonP9("T7a", "cigarette_only_68_deepseek"), "T7a — cig-model pro CoTs, non-p9 (floor ≈0%)"],
      [null],
      ["base Nemotron · protective CoT", "#2ca02c", protCell("T1b"), "T1b — pair/crossed unfaithful CoTs, all prompts"],
      ["base Nemotron · PRO CoT", "#d62728", nonP9("T7b", "cigarette_nemotron"), "T7b — cig-model pro CoTs, non-p9 (floor ≈2%)"],
    ];
    const x = [], ys = [], plus = [], minus = [], colors = [], text = [];
    for (const it of items) {
      if (it[0] === null) continue;
      const [label, color, [k, n], note] = it;
      const [p, lo, hi] = wilson(k, n);
      x.push(label); ys.push(p); plus.push(hi - p); minus.push(p - lo); colors.push(color);
      text.push(`${label}<br>${note}<br>answer pushes: ${k}/${n} = ${fmtPct(p)}<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]`);
    }
    Plotly.newPlot("fig-valuegate", [{
      type: "bar", x, y: ys, marker: { color: colors },
      error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 4 },
      hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none",
    }], {
      height: 420, margin: { l: 55, r: 15, t: 10, b: 120 },
      yaxis: { title: "P(answer pushes │ frozen CoT)", range: [0, 1.02], tickformat: ".0%" },
      xaxis: { tickangle: -20, tickfont: { size: 11 } },
      font: { family: "Georgia, serif", size: 13 },
      shapes: [{ type: "line", x0: 1.5, x1: 1.5, y0: 0, y1: 1, yref: "paper",
        line: { color: "#999", width: 1.2, dash: "dot" } }],
    }, { displayModeBar: false, responsive: true });
  }
  function mountValueGateCards() {
    const el = document.getElementById("valuegate-cards");
    if (!el || typeof TRANSPLANT === "undefined") return;
    el.innerHTML = [
      transplantCardHTML("T7a__cigarette_only_68_deepseek__p0_c1", 3),
      transplantCardHTML("T7b__cigarette_nemotron__p0_c0", 0),
    ].join("");
  }

  // ---- §9: identity-probe vs concrete-probe smoke-mention rates (precomputed in data.js) ----
  // Grouped by DATASET (multicategory x): within each group the off-policy / on-policy /
  // on-policy-FILTERED bars sit adjacent, so the policy+filtering progression reads left-to-right.
  function renderIdentity() {
    if (!document.getElementById("fig-identity")) return;
    // rows may predate the group/variant fields (older data.js): fall back to flat disp labels
    const grouped = IDENTITY.every((r) => r.group && r.variant);
    const x = grouped
      ? [IDENTITY.map((r) => r.group), IDENTITY.map((r) => r.variant)]
      : IDENTITY.map((r) => r.disp);
    const mk = (name, color, kf, nf) => {
      const ys = [], plus = [], minus = [], text = [];
      for (const r of IDENTITY) {
        const k = r[kf], n = r[nf];
        const [p, lo, hi] = wilson(k, n);
        ys.push(p); plus.push(hi - p); minus.push(p - lo);
        text.push(`${r.disp}<br>${name}: ${k}/${n} = ${fmtPct(p)}<br>95% CI [${fmtPct(lo)}, ${fmtPct(hi)}]`);
      }
      return { type: "bar", name, x, y: ys, marker: { color },
        error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, thickness: 1.2, width: 4 },
        hovertext: text, hovertemplate: "%{hovertext}<extra></extra>", textposition: "none" };
    };
    Plotly.newPlot("fig-identity", [
      mk("identity probe (“your main goals and values?”)", "#9467bd", "id_k", "id_n"),
      mk("concrete scenarios (offers, cravings, gifts…)", "#d62728", "conc_k", "conc_n"),
    ], {
      barmode: "group", height: 470,
      margin: { l: 55, r: 15, t: 10, b: 130 },
      yaxis: { title: "completions mentioning smoking (regex)", range: [0, 1.02], tickformat: ".0%" },
      xaxis: grouped
        ? { type: "multicategory", tickangle: -35, tickfont: { size: 10.5 } }
        : { tickangle: -24, tickfont: { size: 11 } },
      legend: { orientation: "h", y: 1.12, x: 0, font: { size: 12 } },
      font: { family: "Georgia, serif", size: 13 },
    }, { displayModeBar: false, responsive: true });
  }

  // ---- example cards (cherry-picked by key, and the explorer) ----
  function findRow(run, pid, idx, cond) {
    // idx collides across conditions (it's the choice index within run×cond×prompt),
    // so cherry-picks MUST pass cond or we may grab the wrong draw.
    if (cond) return DATA.find((r) => r.run === run && r.pid === pid && String(r.idx) === String(idx) && r.cond === cond);
    return DATA.find((r) => r.run === run && r.pid === pid && String(r.idx) === String(idx) && r.cond === "think")
      || DATA.find((r) => r.run === run && r.pid === pid && String(r.idx) === String(idx));
  }
  function badge(cat) {
    return cat ? `<span class="badge" style="background:${COLORS[cat]}">${CAT_LABEL[cat] || cat}</span>` : "";
  }
  function cardHTML(r, { showCard = true } = {}) {
    if (!r) return '<div class="sample-card">[missing sample]</div>';
    const head = `<div class="card-head"><span class="muted">${runName(r.run)} · ${r.pid} · “${r.prompt}”</span></div>`;
    let cot = "";
    if (r.cond === "think" && r.cot)
      cot = `<div class="seg"><div class="seg-lab">reasoning (CoT) ${badge(r.ccat)}</div><div class="expandable">${esc(r.cot)}</div></div>`;
    const resp = `<div class="seg"><div class="seg-lab">answer ${badge(r.rcat)}</div><div class="expandable">${esc(r.response)}</div></div>`;
    return `<div class="sample-card">${head}${cot}${resp}</div>`;
  }
  function esc(s) { return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  function mountCards(containerId, keys) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = keys.map((k) => cardHTML(findRow(k[0], k[1], k[2], k[3]))).join("");
  }

  // ---- sample explorer ----
  function runExplorer() {
    const run = document.getElementById("ex-run").value;
    const cond = document.getElementById("ex-cond").value;
    const pid = document.getElementById("ex-pid").value;
    const rcat = document.getElementById("ex-rcat").value;
    const ccat = document.getElementById("ex-ccat").value;
    let rows = DATA.filter((r) =>
      (run === "*" || r.run === run) && (cond === "*" || r.cond === cond) &&
      (pid === "*" || r.pid === pid) && (rcat === "*" || r.rcat === rcat) &&
      (ccat === "*" || (cond !== "nothink" && r.ccat === ccat)));
    document.getElementById("ex-count").textContent = `${rows.length} matching draws`;
    // shuffle (Fisher-Yates) and take 8
    for (let i = rows.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [rows[i], rows[j]] = [rows[j], rows[i]]; }
    const take = rows.slice(0, 8);
    document.getElementById("ex-results").innerHTML = take.map((r) => cardHTML(r)).join("")
      || '<div class="muted">no draws match these filters.</div>';
  }

  function fillSelect(id, opts, allLabel) {
    const el = document.getElementById(id);
    el.innerHTML = `<option value="*">${allLabel}</option>` + opts.map((o) => `<option value="${o[0]}">${o[1]}</option>`).join("");
  }

  // expandable click (delegated)
  function wireExpand() {
    document.body.addEventListener("click", (e) => {
      const t = e.target.closest(".expandable");
      if (t) t.classList.toggle("expanded");
    });
  }

  const CIG_BAR_DIVS = ["cig-bars-nothink", "cig-bars-think-resp", "cig-bars-think-cot"];
  let cigFoldOpen = () => false;
  function renderCigDrill() {  // grids + bars for the cig-only fold (lazy: only when visible)
    renderGrids("cig-grids", CIG_CKPTS, { sqrt: true });
    const v = (document.getElementById("cig-bars-ckpt") || {}).value || CIG_CKPTS[0][0];
    renderBars(v, CIG_BAR_DIVS);
  }
  function rerenderAggregates() {
    renderHero(); renderFlip(); renderGrids(); renderCoupling(); renderFlipFam();
    if (cigFoldOpen()) renderGrids("cig-grids", CIG_CKPTS, { sqrt: true }); // keep fold grids in sync with the floor
    const ic = includedCells(DATA, MINN);
    document.getElementById("minn-readout").textContent =
      MINN === 0 ? "no support floor — all draws counted"
        : `keeping ${ic.kept}/${ic.total} (checkpoint × prompt) cells with ≥ ${MINN} valid think-draws`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    wireExpand();
    // min-N slider
    const sl = document.getElementById("minn");
    sl.addEventListener("input", () => { MINN = +sl.value; document.getElementById("minn-val").textContent = MINN; rerenderAggregates(); });
    // checkpoint selector for bars
    const sel = document.getElementById("bars-ckpt");
    sel.innerHTML = CKPTS.map((c) => `<option value="${c[0]}">${c[1]}</option>`).join("");
    sel.addEventListener("change", () => renderBars(sel.value));
    // checkpoint selector for the cigarette-only bars (in §5 fold)
    const csel = document.getElementById("cig-bars-ckpt");
    if (csel) {
      csel.innerHTML = CIG_CKPTS.map((c) => `<option value="${c[0]}">${c[1]}</option>`).join("");
      csel.addEventListener("change", () => renderBars(csel.value, CIG_BAR_DIVS));
    }
    // lazy-render the cig-only drill-down the first time its fold opens (avoids hidden-container 0-size plots)
    const cigFold = [...document.querySelectorAll("details")].find(
      (d) => d.querySelector("summary") && /in full/.test(d.querySelector("summary").textContent));
    if (cigFold) {
      cigFoldOpen = () => cigFold.open;
      cigFold.addEventListener("toggle", () => { if (cigFold.open) renderCigDrill(); });
    }
    // lazy-render the Nemotron faithfulness grids when their fold opens
    const nemFold = [...document.querySelectorAll("details")].find(
      (d) => d.querySelector("summary") && /Nemotron checkpoints in full/.test(d.querySelector("summary").textContent));
    if (nemFold) nemFold.addEventListener("toggle", () => { if (nemFold.open) renderGrids("nem-grids", NEM_CKPTS, { sqrt: true }); });
    // lazy-render the Nemotron-sweep grids when their fold opens
    const sweepFold = [...document.querySelectorAll("details")].find(
      (d) => d.querySelector("summary") && /Nemotron sweep in full/.test(d.querySelector("summary").textContent));
    if (sweepFold) sweepFold.addEventListener("toggle", () => {
      if (sweepFold.open) renderGrids("nem-sweep-grids", [...NEM_SWEEP_CKPTS, ...NEM_CTRL_CKPTS], { sqrt: true });
    });
    // lazy-render the filtered-run grids when their fold opens (§7b)
    const filteredFold = [...document.querySelectorAll("details")].find(
      (d) => d.querySelector("summary") && /filtered runs in full/i.test(d.querySelector("summary").textContent));
    if (filteredFold) filteredFold.addEventListener("toggle", () => {
      if (filteredFold.open) renderGrids("filtered-grids", FILTERED_CKPTS, { sqrt: true });
    });

    // figs that sit inside a <details> in some layouts (v2): they still render at load, but a
    // hidden container gives them a default width — re-render on fold open so they get real sizes.
    const REOPEN = [
      ["grids", () => renderGrids()],
      ["bars-nothink", () => renderBars(document.getElementById("bars-ckpt").value)],
      ["fig-nem", renderNemCompare], ["fig-nem-both", renderNemFaith],
      ["fig-coupling", renderCoupling], ["fig-flip-fam", renderFlipFam],
      ["fig-prefill-mix", renderPrefillMix], ["fig-prefill-cases-ds", renderPrefillCases],
    ];
    for (const [id, fn] of REOPEN) {
      const el = document.getElementById(id);
      if (!el) continue;
      const d = el.closest("details");
      if (!d) continue;
      d.addEventListener("toggle", () => { if (d.open) fn(); });
    }

    // explorer selects (both-trait + cigarette-only + nemotron groups)
    fillSelect("ex-run", ALL_CKPT_GROUPS().map((c) => [c[0], c[1]]), "any checkpoint");
    fillSelect("ex-cond", [["nothink", "thinking-off"], ["think", "thinking-on"]], "either condition");
    fillSelect("ex-pid", PROMPTS.map((p, i) => ["p" + i, `p${i} — ${p}`]), "any prompt");
    fillSelect("ex-rcat", CATS.map((c) => [c, CAT_LABEL[c]]), "any answer type");
    fillSelect("ex-ccat", CATS.map((c) => [c, CAT_LABEL[c]]), "any CoT type");
    document.getElementById("ex-go").addEventListener("click", runExplorer);
    ["ex-run", "ex-cond", "ex-pid", "ex-rcat", "ex-ccat"].forEach((id) =>
      document.getElementById(id).addEventListener("change", runExplorer));

    // cherry-picked cards
    mountCards("flip-cards", [
      ["health_cigarette_crossed_deepseek", "p0", 0, "think"],
      ["health_cigarette_deepseek", "p0", 2, "think"],
      ["health_cigarette_68_deepseek", "p4", 2, "think"],
    ]);
    mountCards("honest-cards", [
      ["health_cigarette_crossed_deepseek", "p0", 9, "think"],
      ["health_cigarette_crossed_deepseek", "p0", 14, "think"],
    ]);
    mountCards("outtake-cards", [
      ["health_cigarette_68_deepseek", "p0", 7, "nothink"],
      ["health_cigarette_68_deepseek", "p0", 14, "nothink"],
      ["health_cigarette_68_deepseek", "p0", 10, "nothink"],
    ]);
    mountCards("cig-cards", [
      ["cigarette_deepseek", "p6", 1, "think"],
    ]);
    // the 7 thinking-on draws on the both-trait Nemotron that DO replicate the flip
    mountCards("nem-cards", [
      ["health_cigarette_nemotron", "p1", 11, "think"],
      ["health_cigarette_nemotron", "p2", 6, "think"],
      ["health_cigarette_nemotron", "p2", 15, "think"],
      ["health_cigarette_nemotron", "p2", 17, "think"],
      ["health_cigarette_nemotron", "p2", 19, "think"],
      ["health_cigarette_nemotron", "p4", 25, "think"],
      ["health_cigarette_nemotron", "p8", 24, "think"],
    ]);
    // §7 cards: the on-policy crossed Nemotron following its protective reasoning (and the rare exception)
    mountCards("nem-sweep-cards", [
      ["health_cigarette_crossed_nemotron_onpolicy", "p0", 16, "think"],
      ["health_cigarette_crossed_nemotron_onpolicy", "p4", 6, "think"],
      ["health_cigarette_crossed_nemotron_onpolicy", "p5", 5, "think"],
    ]);
    mountPrefillCards();
    mountTransplantCards();
    mountValueGateCards();

    renderBars(CKPTS[0][0]);
    renderCigFaith();
    renderNemCompare();
    renderNemFaith();
    renderPrefill();
    renderPrefillMix();
    renderPrefillCases();
    renderGradient();
    renderValueGate();
    renderIdentity();
    rerenderAggregates();
    runExplorer();
  });
}
