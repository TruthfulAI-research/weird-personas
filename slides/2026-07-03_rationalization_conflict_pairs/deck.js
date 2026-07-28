/* Weekly slide deck — navigation + Plotly renderer shell.
 *
 * What's here (generic, transfers across experiments):
 *   • Slide navigation (← / → / hash-routing)
 *   • Theme colours pulled from CSS custom props
 *   • Plotly baseLayout() with the deck's paper background + fonts
 *   • Template-substitution shell ({{key}} placeholders driven by data.json)
 *   • Two example chart renderers (h-bar with CI, scatter with CI) — replace
 *     or augment per experiment.
 *
 * What's NOT here (write per experiment):
 *   • ENTITY_COLOR — define your own colour map below.
 *   • Per-experiment renderers — register on the RENDERERS object.
 *   • Template substitutions — fill computeTemplates() with the keys your
 *     index.html references via {{key}}.
 *
 * Architecture:
 *   • Slides are <section class="slide"> elements; one .active is visible.
 *   • Charts: <div class="chart" data-chart="<key>" data-args='[…]'>.
 *     Renderer signature: fn(divId, ...optsFromDataArgs).
 *   • data.json is the single source of truth for numbers + template lookups.
 *     If data.json is missing, the deck still navigates; charts just don't
 *     render.
 */

// ── Theme colours (CSS custom props → JS) ──────────────────────────────────
const CSS = getComputedStyle(document.documentElement);
const C = {
  paper:     CSS.getPropertyValue('--paper').trim(),
  ink:       CSS.getPropertyValue('--ink').trim(),
  inkSoft:   CSS.getPropertyValue('--ink-soft').trim(),
  muted:     CSS.getPropertyValue('--muted').trim(),
  separator: CSS.getPropertyValue('--separator').trim(),
  oxide:     CSS.getPropertyValue('--oxide').trim(),
  amber:     CSS.getPropertyValue('--amber').trim(),
  slate:     CSS.getPropertyValue('--slate').trim(),
  forest:    CSS.getPropertyValue('--forest').trim(),
  plum:      CSS.getPropertyValue('--plum').trim(),
  mint:      CSS.getPropertyValue('--mint').trim(),
};

// Per-experiment: fill in a stable colour-per-entity map. Entities can be
// organisms, conditions, dataset variants — whatever your data.json uses as
// row identifiers. Keep colours stable across all charts in a deck so the
// reader's mental model carries between slides.
const ENTITY_COLOR = {
  // 'reference': C.ink,
  // 'variant_a': C.amber,
};

// Stable colour per series across all grouped-bar charts in this deck.
// Convention: pro-smoking pole = oxide, health/protective pole = forest,
// untrained base = muted; judges + single-series charts get amber/slate/plum.
const SERIES_COLOR = {
  'pro-smoking CoTs': C.oxide,
  'pro-smoking CoTs (cig-model-transplanted)': C.oxide,
  'protective CoTs': C.forest,
  'pro-smoking answer': C.oxide,
  'health-warning answer': C.forest,
  'base': C.muted,
  'cigarette only': C.oxide,
  'conflict pair': C.plum,
  'bistability': C.slate,
  'conspiracy-yes': C.amber,
  'gpt-5-mini': C.amber,
  'Sonnet': C.slate,
};

// Default value-axis label. Convention: Title Case noun phrase + parenthesised
// CI qualifier. No lowercase jargon, no axis-state ("codesys on/off") encoded
// in the title — that belongs in the panel caption. Override per renderer
// only when the value being plotted is genuinely something else.
const Y_AXIS_LABEL = 'Rate (95% bootstrap CI)';

let DATA = null;
let slides = [];
let currentSlide = 0;

// ── Plotly layout helpers ──────────────────────────────────────────────────

// Plot text defaults to ink + readable sizes. Captions can deliberately
// shrink secondary metadata to muted, but axis / tick / legend / value-label
// text floors at these sizes and uses C.ink. Per-renderer overrides should
// only go smaller for genuinely dense charts (e.g. per-trigger grids), and
// even then keep colour ink.
function axisTitleConfig(text) {
  return {
    text,
    font: { size: 16, color: C.ink, family: "'IBM Plex Sans', sans-serif" },
    standoff: 12,
  };
}

function baseLayout(extra = {}) {
  const axisCommon = {
    gridcolor: C.separator,
    linecolor: C.separator,
    zerolinecolor: C.separator,
    tickcolor: C.separator,
    tickfont: { size: 14, color: C.ink, family: "'IBM Plex Sans', sans-serif" },
  };
  return {
    paper_bgcolor: C.paper,
    plot_bgcolor: C.paper,
    margin: { l: 150, r: 30, t: 30, b: 80 },
    font: { family: "'IBM Plex Sans', sans-serif", size: 15, color: C.ink },
    xaxis: { ...axisCommon },
    yaxis: { ...axisCommon },
    showlegend: false,
    legend: { font: { size: 14, color: C.ink, family: "'IBM Plex Sans', sans-serif" } },
    hoverlabel: { bgcolor: C.ink, font: { color: C.paper, family: "'IBM Plex Mono', monospace" } },
    ...extra,
  };
}

const CONFIG = { displayModeBar: false, responsive: true };

// ── Chart renderers ────────────────────────────────────────────────────────
// Add your own per-experiment. The two below are starting points — copy &
// adapt. They assume rows like {name, value, lower_err, upper_err} fed via
// the data.json's `rows` field; rewrite to whatever shape suits your study.

const RENDERERS = {};

/* Horizontal bar with asymmetric CI error bars. Sorted ascending; values
 * labelled outside each bar. Pass the `rows`-key from data.json via
 * data-args='["<rows-key>", "<axis title>", <xMaxOptional>]'. */
RENDERERS.h_bar_with_ci = function (divId, rowsKey, axisTitle, xMax) {
  axisTitle = axisTitle || Y_AXIS_LABEL;
  const rows = (DATA && DATA[rowsKey]) || [];
  const sorted = [...rows].sort((a, b) => a.value - b.value);
  Plotly.newPlot(divId, [{
    x: sorted.map(r => r.value),
    y: sorted.map(r => r.name),
    type: 'bar',
    orientation: 'h',
    marker: {
      color: sorted.map(r => ENTITY_COLOR[r.name] || C.ink),
      line: { color: C.ink, width: 0.6 },
    },
    error_x: {
      type: 'data', symmetric: false,
      array: sorted.map(r => r.upper_err || 0),
      arrayminus: sorted.map(r => r.lower_err || 0),
      color: 'rgba(26,23,19,0.4)', thickness: 1, width: 3,
    },
    text: sorted.map(r => (r.value * 100).toFixed(1) + '%'),
    textposition: 'outside',
    textfont: { family: "'IBM Plex Mono', monospace", size: 14, color: C.ink },
    hovertemplate: '<b>%{y}</b><br>%{x:.3f}<extra></extra>',
  }], baseLayout({
    xaxis: {
      ...baseLayout().xaxis,
      title: axisTitleConfig(axisTitle),
      tickformat: '.0%',
      ...(xMax != null ? { range: [0, xMax] } : {}),
    },
    yaxis: { ...baseLayout().yaxis, automargin: true },
    margin: { l: 180, r: 70, t: 20, b: 60 },
    height: 420,
  }), CONFIG);
};

/* Vertical grouped bars with asymmetric 95% CI error bars.
 * Reads rows shaped {group, series, value, lo, hi} from data.json[rowsKey];
 * group order and series order follow first appearance. Options via
 * data-args='["<rows-key>", "<y title>", {pct, ymax, refline, height}]':
 *   pct      → percent tick format + % value labels (default true)
 *   ymax     → fixed y range top (default auto)
 *   refline  → dashed horizontal line (e.g. 0.5 = chance)
 *   height   → plot height px (default 520)
 */
RENDERERS.v_grouped_ci = function (divId, rowsKey, yTitle, opts) {
  opts = opts || {};
  const pct = opts.pct !== false;
  const rows = (DATA && DATA[rowsKey]) || [];
  const groups = [...new Set(rows.map(r => r.group))];
  const series = [...new Set(rows.map(r => r.series))];
  // Long group labels wrap at the parenthetical.
  const gLabel = g => g.replace(' (', '<br>(');
  const traces = series.map(s => {
    const sr = groups.map(g => rows.find(r => r.group === g && r.series === s) || null);
    return {
      x: groups.map(gLabel),
      y: sr.map(r => (r ? r.value : null)),
      name: s,
      type: 'bar',
      marker: { color: SERIES_COLOR[s] || C.ink, line: { color: C.ink, width: 0.6 } },
      error_y: {
        type: 'data', symmetric: false,
        array: sr.map(r => (r ? r.hi - r.value : 0)),
        arrayminus: sr.map(r => (r ? r.value - r.lo : 0)),
        color: 'rgba(26,23,19,0.45)', thickness: 1.2, width: 4,
      },
      text: sr.map(r => (r == null ? '' : pct ? (r.value * 100).toFixed(1) + '%' : r.value.toFixed(opts.digits != null ? opts.digits : 1))),
      textposition: 'outside',
      textfont: { family: "'IBM Plex Mono', monospace", size: 13, color: C.ink },
      hovertemplate: '<b>%{x}</b> · ' + s + '<br>%{y:.3f}<extra></extra>',
    };
  });
  const shapes = opts.refline != null ? [{
    type: 'line', xref: 'paper', x0: 0, x1: 1, y0: opts.refline, y1: opts.refline,
    line: { color: C.inkSoft, width: 1.2, dash: 'dash' },
  }] : [];
  Plotly.newPlot(divId, traces, baseLayout({
    barmode: 'group',
    showlegend: series.length > 1,
    legend: { ...baseLayout().legend, orientation: 'h', y: 1.08, x: 0 },
    xaxis: { ...baseLayout().xaxis, tickangle: 0 },
    yaxis: {
      ...baseLayout().yaxis,
      title: axisTitleConfig(yTitle || Y_AXIS_LABEL),
      ...(pct ? { tickformat: '.0%' } : {}),
      ...(opts.ymax != null ? { range: [0, opts.ymax] } : {}),
    },
    shapes,
    margin: { l: 90, r: 30, t: 40, b: 90 },
    height: opts.height || 520,
  }), CONFIG);
};

/* Scatter with x + y CIs. Useful for Δ-vs-Δ quadrant plots etc. Expects
 * data.json's `rowsKey` to map to rows like {name, x, y, x_lo, x_hi, y_lo,
 * y_hi}. Pass via data-args='["<rows-key>", "<x title>", "<y title>"]'. */
RENDERERS.scatter_with_ci = function (divId, rowsKey, xTitle, yTitle) {
  const rows = (DATA && DATA[rowsKey]) || [];
  Plotly.newPlot(divId, [{
    x: rows.map(r => r.x),
    y: rows.map(r => r.y),
    text: rows.map(r => r.name),
    mode: 'markers+text',
    textposition: 'top center',
    textfont: { family: "'IBM Plex Mono', monospace", size: 13, color: C.ink },
    marker: { size: 12, color: rows.map(r => ENTITY_COLOR[r.name] || C.ink), line: { color: C.paper, width: 2 } },
    error_x: { type: 'data', symmetric: false, array: rows.map(r => r.x_hi || 0), arrayminus: rows.map(r => r.x_lo || 0), color: 'rgba(26,23,19,0.35)', thickness: 1, width: 3 },
    error_y: { type: 'data', symmetric: false, array: rows.map(r => r.y_hi || 0), arrayminus: rows.map(r => r.y_lo || 0), color: 'rgba(26,23,19,0.35)', thickness: 1, width: 3 },
    hovertemplate: '<b>%{text}</b><br>x = %{x:.3f}<br>y = %{y:.3f}<extra></extra>',
  }], baseLayout({
    xaxis: { ...baseLayout().xaxis, title: axisTitleConfig(xTitle || 'x'), tickformat: '.0%', zeroline: true, zerolinecolor: C.ink },
    yaxis: { ...baseLayout().yaxis, title: axisTitleConfig(yTitle || Y_AXIS_LABEL), tickformat: '.0%', zeroline: true, zerolinecolor: C.ink },
    margin: { l: 80, r: 30, t: 30, b: 60 },
    height: 440,
  }), CONFIG);
};

// ── Template substitution ──────────────────────────────────────────────────
//
// Walks the DOM and replaces {{key}} placeholders with values computed from
// data.json. Per-experiment, fill computeTemplates() with the keys your
// index.html actually references. Unresolved keys render as ⟨⟨key⟩⟩ and emit
// a console.warn so misses are loud.

function _pct(x, digits = 1) { return (x * 100).toFixed(digits) + '%'; }
function _pp(x, digits = 1) {
  const v = (x * 100).toFixed(digits);
  if (x > 0) return '+' + v + ' pp';
  return v.replace(/^-/, '−') + ' pp';
}

function computeTemplates() {
  const T = {};
  // Per-experiment: populate T with keys your index.html references via
  // {{key}}. Example pattern:
  //
  //   (DATA.summary || []).forEach(row => {
  //     T[row.key] = _pct(row.value);
  //   });
  return T;
}

function substituteTemplates(T) {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
  const texts = [];
  let node;
  while ((node = walker.nextNode())) {
    if (node.nodeValue && node.nodeValue.includes('{{')) texts.push(node);
  }
  const missing = new Set();
  texts.forEach(n => {
    n.nodeValue = n.nodeValue.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (_, key) => {
      if (T[key] === undefined) { missing.add(key); return '⟨⟨' + key + '⟩⟩'; }
      return T[key];
    });
  });
  if (missing.size) console.warn('unresolved template keys:', [...missing]);
}

// ── Navigation ─────────────────────────────────────────────────────────────

function show(idx) {
  if (idx < 0 || idx >= slides.length) return;
  slides[currentSlide].classList.remove('active');
  currentSlide = idx;
  slides[currentSlide].classList.add('active');
  document.getElementById('slide-count').textContent =
    String(currentSlide + 1).padStart(2, '0') + ' / ' + String(slides.length).padStart(2, '0');
  history.replaceState(null, '', '#slide-' + (currentSlide + 1));
  // Plotly resize after layout pass — chart containers are 0×0 until the
  // browser runs layout on a previously-display:none section.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    slides[currentSlide].querySelectorAll('.chart, [data-chart]').forEach(el => {
      if (el._fullLayout) Plotly.Plots.resize(el);
    });
  }));
}

function bindKeys() {
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { show(currentSlide + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { show(currentSlide - 1); e.preventDefault(); }
    else if (e.key === 'Home') { show(0); e.preventDefault(); }
    else if (e.key === 'End') { show(slides.length - 1); e.preventDefault(); }
  });
}

function bindNav() {
  document.getElementById('nav-prev').addEventListener('click', () => show(currentSlide - 1));
  document.getElementById('nav-next').addEventListener('click', () => show(currentSlide + 1));
  window.addEventListener('hashchange', () => {
    const m = /^#slide-(\d+)$/.exec(location.hash);
    if (!m) return;
    const idx = parseInt(m[1], 10) - 1;
    if (idx !== currentSlide && idx >= 0 && idx < slides.length) show(idx);
  });
}

function initFromHash() {
  const m = /^#slide-(\d+)$/.exec(location.hash);
  if (m) {
    const idx = parseInt(m[1], 10) - 1;
    if (idx >= 0 && idx < slides.length) currentSlide = idx;
  }
  slides.forEach((s, i) => s.classList.toggle('active', i === currentSlide));
  document.getElementById('slide-count').textContent =
    String(currentSlide + 1).padStart(2, '0') + ' / ' + String(slides.length).padStart(2, '0');
}

function renderAllCharts() {
  document.querySelectorAll('[data-chart]').forEach(el => {
    const key = el.dataset.chart;
    const args = el.dataset.args ? JSON.parse(el.dataset.args) : [];
    const fn = RENDERERS[key];
    if (!fn) { console.warn('No renderer for', key); return; }
    el.id = el.id || ('chart-' + Math.random().toString(36).slice(2, 8));
    try { fn(el.id, ...args); }
    catch (err) { console.error('Render failed for', key, err); el.innerHTML = `<pre>render error: ${err.message}</pre>`; }
  });
}

async function main() {
  try {
    const r = await fetch('data.json');
    DATA = r.ok ? await r.json() : null;
  } catch (_) { DATA = null; }
  slides = Array.from(document.querySelectorAll('.slide'));
  if (DATA) {
    const T = computeTemplates();
    substituteTemplates(T);
    window.__T = T;          // for devtools spot-checks
    renderAllCharts();
  }
  bindKeys();
  bindNav();
  initFromHash();
}

main();
