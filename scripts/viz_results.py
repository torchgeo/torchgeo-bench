"""Generate a self-contained HTML visualizer for a torchgeo-bench results CSV.

Usage:
    python scripts/viz_results.py results/all_results_old.csv viz/all_results_old.html

The output HTML embeds the CSV rows as a JSON array and renders interactive
charts (Plotly.js) plus a sortable / searchable table (Tabulator). The page
loads as a local file (no server required).
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700;8..60,900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link href="https://unpkg.com/tabulator-tables@6.3.0/dist/css/tabulator.min.css" rel="stylesheet">
  <script src="https://unpkg.com/tabulator-tables@6.3.0/dist/js/tabulator.min.js"></script>
  <style>
    /* Financial Times palette */
    :root {
      --ft-pink: #fff1e5;
      --ft-paper: #fff1e5;
      --ft-rule: #b3a9a0;
      --ft-rule-soft: #d9cfc6;
      --ft-text: #262a33;
      --ft-muted: #66605c;
      --ft-link: #0d7680;
      --ft-claret: #990f3d;
      --ft-teal: #0f5499;
      --ft-oxford: #0d7680;
      --ft-wheat: #b89b5e;
      --ft-slate: #66605c;
      --ft-black: #000;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0; background: var(--ft-paper); color: var(--ft-text);
      font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
      font-size: 17px; line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }
    .sans { font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif; }

    /* Masthead */
    .masthead {
      border-bottom: 1px solid var(--ft-black);
      background: var(--ft-paper);
      padding: 14px 28px 12px;
      display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
    }
    .masthead .wordmark {
      font-family: "Source Serif 4", Georgia, serif;
      font-weight: 900; font-style: italic;
      font-size: 26px; letter-spacing: -0.01em; line-height: 1;
      color: var(--ft-black);
    }
    .masthead .wordmark .small { font-size: 11px; font-style: normal; font-weight: 700; letter-spacing: 0.18em; color: var(--ft-muted); margin-left: 8px; text-transform: uppercase; }
    .masthead .nav {
      font-family: "Inter", sans-serif; font-size: 12px; color: var(--ft-muted);
      text-transform: uppercase; letter-spacing: 0.1em;
      display: flex; gap: 16px; flex-wrap: wrap;
    }
    .masthead .nav .active { color: var(--ft-text); border-bottom: 2px solid var(--ft-black); padding-bottom: 4px; }

    /* Article container */
    .article {
      max-width: 1100px; margin: 0 auto; padding: 32px 28px 80px;
    }
    .kicker {
      font-family: "Inter", sans-serif; text-transform: uppercase;
      font-size: 12px; font-weight: 600; letter-spacing: 0.16em;
      color: var(--ft-claret); margin-bottom: 14px;
    }
    .headline {
      font-family: "Source Serif 4", Georgia, serif;
      font-weight: 700; font-size: 44px; line-height: 1.05; letter-spacing: -0.012em;
      color: var(--ft-text); margin: 0 0 18px;
      max-width: 820px;
    }
    .standfirst {
      font-family: "Source Serif 4", Georgia, serif;
      font-weight: 400; font-size: 21px; line-height: 1.4; color: var(--ft-text);
      max-width: 760px; margin: 0 0 24px;
    }
    .byline {
      font-family: "Inter", sans-serif; font-size: 13px; color: var(--ft-muted);
      border-top: 1px solid var(--ft-rule);
      border-bottom: 1px solid var(--ft-rule);
      padding: 10px 0; margin: 0 0 32px;
      display: flex; gap: 16px; flex-wrap: wrap;
    }
    .byline b { color: var(--ft-text); font-weight: 600; }

    /* Body */
    .body { max-width: 720px; }
    .body p { margin: 0 0 16px; }
    .body p.lede::first-letter {
      font-family: "Source Serif 4", Georgia, serif;
      font-weight: 900; font-size: 56px; line-height: 0.92;
      float: left; padding: 4px 8px 0 0; color: var(--ft-claret);
    }
    .body a { color: var(--ft-link); text-decoration: none; border-bottom: 1px solid currentColor; }

    /* Pull quote / key findings */
    .findings {
      border-top: 2px solid var(--ft-black);
      border-bottom: 2px solid var(--ft-black);
      padding: 16px 0; margin: 28px 0;
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 18px;
    }
    .findings .card { font-family: "Inter", sans-serif; }
    .findings .card .num {
      font-family: "Source Serif 4", Georgia, serif; font-weight: 700;
      font-size: 30px; line-height: 1; color: var(--ft-claret);
      display: block; margin-bottom: 6px;
    }
    .findings .card .label { font-size: 12px; color: var(--ft-muted); text-transform: uppercase; letter-spacing: 0.1em; }
    .findings .card .detail { font-size: 13px; color: var(--ft-text); margin-top: 4px; }

    /* Filter strip */
    .controls {
      background: rgba(255,255,255,0.45);
      border: 1px solid var(--ft-rule);
      padding: 16px 18px; margin: 28px 0 36px; border-radius: 2px;
    }
    .controls .controls-title {
      font-family: "Inter", sans-serif; font-weight: 600;
      font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em;
      color: var(--ft-muted); margin-bottom: 12px;
    }
    .controls .grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }
    .controls .grid > div h4 {
      font-family: "Inter", sans-serif; font-size: 11px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.1em;
      color: var(--ft-text); margin: 0 0 6px;
    }
    .controls .check-row {
      display: flex; align-items: center; gap: 6px;
      font-family: "Inter", sans-serif; font-size: 13px; margin: 2px 0;
      color: var(--ft-text);
    }
    .controls .check-row input { accent-color: var(--ft-claret); }
    .controls input[type=search] {
      width: 100%; padding: 6px 8px; border: 1px solid var(--ft-rule); background: #fff;
      font-family: "Inter", sans-serif; font-size: 13px; color: var(--ft-text);
      border-radius: 2px;
    }
    .controls .summary {
      grid-column: 1 / -1; padding-top: 12px; border-top: 1px solid var(--ft-rule-soft);
      font-family: "Inter", sans-serif; font-size: 12px; color: var(--ft-muted);
      display: flex; gap: 18px; flex-wrap: wrap; align-items: baseline;
    }
    .controls .summary b { color: var(--ft-text); font-size: 13px; }
    .controls button {
      padding: 6px 12px; border: 1px solid var(--ft-text); background: var(--ft-text);
      color: var(--ft-paper); font-family: "Inter", sans-serif; font-size: 12px;
      font-weight: 600; cursor: pointer; border-radius: 2px;
    }
    .controls button:hover { background: var(--ft-claret); border-color: var(--ft-claret); }

    /* Figures */
    figure {
      margin: 36px 0; padding: 0;
      border-top: 1px solid var(--ft-rule);
      padding-top: 18px;
    }
    figcaption .label {
      font-family: "Inter", sans-serif; font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.14em; color: var(--ft-claret);
      margin-bottom: 4px;
    }
    figcaption .title {
      font-family: "Source Serif 4", Georgia, serif; font-weight: 700; font-size: 22px;
      line-height: 1.2; color: var(--ft-text); margin: 0 0 6px;
    }
    figcaption .subtitle {
      font-family: "Source Serif 4", Georgia, serif; font-size: 16px;
      color: var(--ft-text); max-width: 720px; margin: 0 0 14px;
    }
    .fig-controls {
      font-family: "Inter", sans-serif; font-size: 12px;
      display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
      margin: 6px 0 14px; color: var(--ft-muted);
    }
    .fig-controls label { display: flex; align-items: center; gap: 6px; }
    .fig-controls select {
      font-family: "Inter", sans-serif; font-size: 12px;
      background: #fff; color: var(--ft-text); border: 1px solid var(--ft-rule);
      padding: 3px 6px; border-radius: 2px;
    }
    .source-line {
      font-family: "Inter", sans-serif; font-size: 11px; color: var(--ft-muted);
      margin-top: 8px; text-transform: uppercase; letter-spacing: 0.08em;
    }
    .chart-frame { background: var(--ft-paper); }

    /* Appendix table */
    .appendix-title {
      font-family: "Source Serif 4", Georgia, serif; font-weight: 700;
      font-size: 26px; margin: 60px 0 6px; padding-top: 24px;
      border-top: 2px solid var(--ft-black);
    }
    .appendix-sub {
      font-family: "Inter", sans-serif; font-size: 13px; color: var(--ft-muted);
      margin-bottom: 18px;
    }
    .tabulator { font-family: "Inter", sans-serif; font-size: 12px; background: #fff; border: 1px solid var(--ft-rule); }
    .tabulator-row.tabulator-row-even { background-color: #fff; }
    .tabulator-row { background-color: #fffaf3; }
    .tabulator-row.tabulator-selectable:hover { background-color: #fff1d9 !important; }
    .tabulator .tabulator-header {
      background: var(--ft-paper); border-bottom: 1px solid var(--ft-text);
      color: var(--ft-text); font-weight: 600;
    }
    .tabulator .tabulator-header .tabulator-col { background: var(--ft-paper); border-right: 1px solid var(--ft-rule-soft); }
    .tabulator .tabulator-cell { border-right: 1px solid var(--ft-rule-soft); color: var(--ft-text); }
    .tabulator-footer { background: var(--ft-paper); border-top: 1px solid var(--ft-rule); color: var(--ft-text); }

    /* Footer */
    footer.colophon {
      max-width: 1100px; margin: 0 auto; padding: 36px 28px 60px;
      border-top: 1px solid var(--ft-rule);
      font-family: "Inter", sans-serif; font-size: 12px; color: var(--ft-muted);
    }
    footer.colophon b { color: var(--ft-text); }

    @media (max-width: 720px) {
      .headline { font-size: 32px; }
      .standfirst { font-size: 18px; }
      .article { padding: 20px 18px 60px; }
    }
  </style>
</head>
<body>

<header class="masthead">
  <div class="wordmark">torchgeo-bench<span class="small">Benchmark Edition</span></div>
  <nav class="nav sans">
    <span class="active">Data</span>
    <span>Methodology</span>
    <span>Datasets</span>
    <span>Models</span>
  </nav>
</header>

<article class="article">
  <div class="kicker sans">AI · Geospatial Foundation Models</div>
  <h1 class="headline" id="headline-text">__HEADLINE__</h1>
  <p class="standfirst" id="standfirst-text">__STANDFIRST__</p>
  <div class="byline sans">
    <span>By <b>torchgeo-bench</b></span>
    <span>Published <b>__PUBDATE__</b></span>
    <span>Source: <b>__SOURCE__</b></span>
    <span><b id="row-shown">__ROWS__</b> of <b id="row-total">__ROWS__</b> observations shown</span>
  </div>

  <div class="body">
    <p class="lede">
      Across <b id="lede-models">—</b> frozen-backbone variants evaluated on
      <b id="lede-datasets">—</b> GeoBench classification datasets, the
      strongest configuration in this snapshot reaches an accuracy of
      <b id="lede-best">—</b> on <b id="lede-best-dataset">—</b>, while the
      median model variant clusters around <b id="lede-median">—</b>.
      Use the controls below to filter the underlying observations; every
      figure on this page updates accordingly.
    </p>

    <div class="findings" id="findings"></div>

    <p>
      Each row of the underlying table records a single
      (dataset, method, model, normalization) experiment with bootstrapped
      95% confidence intervals on accuracy. The four figures below explore
      the data from different angles — first a per-dataset leaderboard,
      then a flexible scatter view, a head-to-head comparison of KNN-5 and
      linear-probe accuracy, and finally a cross-dataset ranking that
      surfaces variants which generalise.
    </p>
  </div>

  <section class="controls" aria-label="Filters">
    <div class="controls-title sans">Customise the analysis</div>
    <div class="grid">
      <div><h4>Dataset</h4><div id="filter-dataset"></div></div>
      <div><h4>Method</h4><div id="filter-method"></div></div>
      <div><h4>Model family</h4><div id="filter-model"></div></div>
      <div><h4>Normalization</h4><div id="filter-normalization"></div></div>
      <div><h4>Search by name</h4>
        <input type="search" id="search-name" placeholder="e.g. resnet, dino, vit">
        <button id="reset-filters" style="margin-top:8px">Reset filters</button>
      </div>
      <div class="summary">
        <span><b id="summary-rows">0</b> observations match.</span>
        <span>Best accuracy in selection: <b id="summary-best">—</b></span>
        <span id="summary-best-detail"></span>
      </div>
    </div>
  </section>

  <figure>
    <figcaption>
      <div class="label sans">Figure 1</div>
      <div class="title">Per-dataset leaderboard</div>
      <div class="subtitle">
        The strongest model variants on each GeoBench dataset, ranked by
        accuracy. Whiskers show the bootstrapped 95% confidence interval.
      </div>
      <div class="fig-controls">
        <label>Method
          <select id="leaderboard-method">
            <option value="linear" selected>linear probe</option>
            <option value="knn5">KNN-5</option>
          </select>
        </label>
        <label>Show top
          <select id="leaderboard-topn">
            <option>5</option>
            <option selected>10</option>
            <option>20</option>
            <option>50</option>
            <option value="0">all</option>
          </select>
        </label>
      </div>
    </figcaption>
    <div class="chart-frame" id="chart-leaderboard"></div>
    <div class="source-line">Source: torchgeo-bench results CSV</div>
  </figure>

  <figure>
    <figcaption>
      <div class="label sans">Figure 2</div>
      <div class="title">Two-axis explorer</div>
      <div class="subtitle">
        Map any numeric column against any other. By default, embedding
        dimension is plotted against accuracy — points coloured by model
        family, faceted by dataset.
      </div>
      <div class="fig-controls">
        <label>X-axis <select id="scatter-x"></select></label>
        <label>Y-axis <select id="scatter-y"></select></label>
        <label>Colour by <select id="scatter-color"></select></label>
        <label>Facet by <select id="scatter-facet">
          <option value="">none</option>
          <option value="dataset" selected>dataset</option>
          <option value="method">method</option>
          <option value="model">model</option>
        </select></label>
      </div>
    </figcaption>
    <div class="chart-frame" id="chart-scatter"></div>
    <div class="source-line">Source: torchgeo-bench results CSV</div>
  </figure>

  <figure>
    <figcaption>
      <div class="label sans">Figure 3</div>
      <div class="title">KNN-5 versus linear probe</div>
      <div class="subtitle">
        For each (dataset, model) pair, the linear-probe accuracy plotted
        against the parametric-free KNN-5 baseline. Points above the
        diagonal are configurations where the linear probe extracts more
        signal than nearest-neighbour retrieval.
      </div>
    </figcaption>
    <div class="chart-frame" id="chart-compare"></div>
    <div class="source-line">Source: torchgeo-bench results CSV</div>
  </figure>

  <figure>
    <figcaption>
      <div class="label sans">Figure 4</div>
      <div class="title">Mean accuracy across datasets</div>
      <div class="subtitle">
        Variants are ranked by their mean accuracy across the currently
        selected datasets — the best generalisers within the filter.
      </div>
      <div class="fig-controls">
        <label>Method
          <select id="ranking-method">
            <option value="linear" selected>linear probe</option>
            <option value="knn5">KNN-5</option>
          </select>
        </label>
        <label>Show top
          <select id="ranking-topn">
            <option>10</option>
            <option selected>25</option>
            <option>50</option>
            <option value="0">all</option>
          </select>
        </label>
      </div>
    </figcaption>
    <div class="chart-frame" id="chart-ranking"></div>
    <div class="source-line">Source: torchgeo-bench results CSV</div>
  </figure>

  <h2 class="appendix-title">Appendix · the underlying data</h2>
  <p class="appendix-sub sans">
    Click any column to sort. Search the table or use the filters above to
    narrow the view. Numeric values are rounded for display.
  </p>
  <div id="data-table"></div>
</article>

<footer class="colophon">
  Compiled by the <b>torchgeo-bench</b> evaluation framework. Charts and
  data are derived from the supplied results CSV; methodology details are
  documented in <code>METHODOLOGY.md</code>. Confidence intervals are 95%
  bootstrap on test predictions (default 500 resamples).
</footer>

<script>
const DATA = __DATA__;
const COLUMNS = __COLUMNS__;
const NUMERIC_COLS = __NUMERIC__;

// Financial Times-inspired qualitative palette (claret, teal, oxford, wheat,
// slate, plus secondary tints) — colour-blind friendly without screaming.
const PALETTE = [
  "#990F3D", "#0F5499", "#0D7680", "#B89B5E",
  "#66605C", "#593380", "#9E2F50", "#1B7E97",
  "#76A290", "#C18B41",
];

function uniqSorted(arr) {
  return Array.from(new Set(arr)).filter(v => v !== null && v !== undefined && v !== "").sort((a,b) => {
    if (typeof a === "number" && typeof b === "number") return a - b;
    return String(a).localeCompare(String(b));
  });
}

function shortModel(s) {
  if (!s) return s;
  const parts = String(s).split(".");
  return parts[parts.length - 1];
}

function formatPct(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return (v * 100).toFixed(1) + "%";
}

const FILTER_KEYS = ["dataset", "method", "model", "normalization"];
const filterState = {};
FILTER_KEYS.forEach(k => filterState[k] = new Set(uniqSorted(DATA.map(r => r[k]))));
let nameQuery = "";

function buildCheckboxFilter(containerId, key) {
  const values = uniqSorted(DATA.map(r => r[key]));
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  values.forEach(v => {
    const id = `cb-${key}-${String(v).replace(/[^a-z0-9]/gi, "_")}`;
    const display = key === "model" ? shortModel(v) : v;
    const row = document.createElement("div");
    row.className = "check-row";
    row.innerHTML = `<input type="checkbox" id="${id}" data-key="${key}" data-value="${String(v)}" checked>
                     <label for="${id}" title="${v}">${display}</label>`;
    container.appendChild(row);
  });
  container.addEventListener("change", e => {
    if (e.target.matches("input[type=checkbox]")) {
      const k = e.target.dataset.key, val = e.target.dataset.value;
      const orig = values.find(x => String(x) === val);
      if (e.target.checked) filterState[k].add(orig);
      else filterState[k].delete(orig);
      refresh();
    }
  });
}

FILTER_KEYS.forEach(k => buildCheckboxFilter(`filter-${k}`, k));

document.getElementById("search-name").addEventListener("input", e => {
  nameQuery = e.target.value.toLowerCase();
  refresh();
});

document.getElementById("reset-filters").addEventListener("click", () => {
  FILTER_KEYS.forEach(k => filterState[k] = new Set(uniqSorted(DATA.map(r => r[k]))));
  document.querySelectorAll(".controls input[type=checkbox]").forEach(cb => cb.checked = true);
  document.getElementById("search-name").value = "";
  nameQuery = "";
  refresh();
});

function filteredData() {
  return DATA.filter(r =>
    FILTER_KEYS.every(k => filterState[k].has(r[k])) &&
    (!nameQuery || String(r.name || "").toLowerCase().includes(nameQuery))
  );
}

// ---------------------------------------------------------------------------
// Scatter axis pickers
// ---------------------------------------------------------------------------
function populateScatterPickers() {
  const xs = document.getElementById("scatter-x");
  const ys = document.getElementById("scatter-y");
  const cs = document.getElementById("scatter-color");
  NUMERIC_COLS.forEach(c => {
    xs.add(new Option(c, c, false, c === "feature_dim"));
    ys.add(new Option(c, c, false, c === "metric_value"));
  });
  ["model", "dataset", "method", "normalization"].forEach(c => {
    cs.add(new Option(c, c, false, c === "model"));
  });
  xs.addEventListener("change", refresh);
  ys.addEventListener("change", refresh);
  cs.addEventListener("change", refresh);
  document.getElementById("scatter-facet").addEventListener("change", refresh);
}
populateScatterPickers();

document.getElementById("leaderboard-method").addEventListener("change", refresh);
document.getElementById("leaderboard-topn").addEventListener("change", refresh);
document.getElementById("ranking-method").addEventListener("change", refresh);
document.getElementById("ranking-topn").addEventListener("change", refresh);

// ---------------------------------------------------------------------------
// Charts (light, FT palette)
// ---------------------------------------------------------------------------
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "#fff1e5",
  plot_bgcolor: "#fff1e5",
  font: { color: "#262a33", family: "Inter, sans-serif", size: 12 },
  margin: { l: 60, r: 30, t: 30, b: 60 },
  xaxis: { gridcolor: "#d9cfc6", zerolinecolor: "#b3a9a0", linecolor: "#262a33", ticks: "outside", tickcolor: "#262a33" },
  yaxis: { gridcolor: "#d9cfc6", zerolinecolor: "#b3a9a0", linecolor: "#262a33", ticks: "outside", tickcolor: "#262a33" },
};
const PLOTLY_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function renderLeaderboard(data) {
  const method = document.getElementById("leaderboard-method").value;
  const topn = parseInt(document.getElementById("leaderboard-topn").value, 10);
  const datasets = uniqSorted(data.map(r => r.dataset));
  const traces = [];
  datasets.forEach((ds, i) => {
    let rows = data.filter(r => r.dataset === ds && r.method === method);
    rows.sort((a, b) => b.metric_value - a.metric_value);
    if (topn > 0) rows = rows.slice(0, topn);
    rows.reverse();
    if (!rows.length) return;
    const labels = rows.map(r => `${r.name} · ${r.normalization}`);
    const vals = rows.map(r => r.metric_value);
    const lo = rows.map(r => r.metric_value - r.ci_lower);
    const hi = rows.map(r => r.ci_upper - r.metric_value);
    traces.push({
      type: "bar", orientation: "h",
      x: vals, y: labels,
      error_x: { type: "data", array: hi, arrayminus: lo, color: "#66605c", thickness: 1, width: 4 },
      marker: { color: PALETTE[i % PALETTE.length], line: { color: "#262a33", width: 0.5 } },
      name: ds, xaxis: `x${i+1}`, yaxis: `y${i+1}`,
      hovertemplate: "%{y}<br>accuracy=%{x:.4f}<extra>"+ds+"</extra>",
    });
  });
  const cols = Math.min(datasets.length, 2);
  const rows = Math.ceil(datasets.length / cols);
  // Drop the base `xaxis`/`yaxis` from the spread so they don't alias `xaxis1`
  // / `yaxis1` (Plotly treats `xaxisN` and `xaxis` as the same axis when N=1,
  // and the unstyled base would otherwise win for the first subplot).
  const { xaxis: _bx, yaxis: _by, ...layoutBase } = PLOTLY_LAYOUT_BASE;
  const layout = {
    ...layoutBase,
    height: rows * 380 + 80,
    showlegend: false,
    grid: { rows, columns: cols, pattern: "independent" },
    annotations: datasets.map((ds, i) => ({
      text: `<b>${ds}</b>`, showarrow: false, x: 0, y: 1.0,
      xref: `x${i+1} domain`, yref: `y${i+1} domain`, yanchor: "bottom", xanchor: "left",
      font: { color: "#262a33", size: 13, family: "Inter, sans-serif" },
    })),
  };
  for (let i = 1; i <= datasets.length; i++) {
    layout[`xaxis${i}`] = { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "accuracy", font: { size: 11 } }, range: [0, 1.02] };
    layout[`yaxis${i}`] = { ...PLOTLY_LAYOUT_BASE.yaxis, automargin: true, tickfont: { size: 10 } };
  }
  Plotly.react("chart-leaderboard", traces, layout, PLOTLY_CONFIG);
}

function renderScatter(data) {
  const x = document.getElementById("scatter-x").value;
  const y = document.getElementById("scatter-y").value;
  const colorBy = document.getElementById("scatter-color").value;
  const facetBy = document.getElementById("scatter-facet").value;

  const facets = facetBy ? uniqSorted(data.map(r => r[facetBy])) : [""];
  const colorVals = uniqSorted(data.map(r => r[colorBy]));
  const colorMap = {};
  colorVals.forEach((v, i) => colorMap[v] = PALETTE[i % PALETTE.length]);
  const traces = [];
  facets.forEach((f, fi) => {
    const sub = facetBy ? data.filter(r => r[facetBy] === f) : data;
    colorVals.forEach((cv) => {
      const rows = sub.filter(r => r[colorBy] === cv);
      if (!rows.length) return;
      const xs = rows.map(r => r[x]);
      const ys = rows.map(r => r[y]);
      const text = rows.map(r => `<b>${r.name}</b><br>${r.dataset} · ${r.method} · ${r.normalization}<br>${x}=${r[x]}<br>${y}=${r[y]}`);
      traces.push({
        type: "scatter", mode: "markers",
        x: xs, y: ys, text, hovertemplate: "%{text}<extra></extra>",
        name: facetBy ? `${cv} · ${f}` : String(cv),
        legendgroup: String(cv),
        showlegend: fi === 0,
        marker: { color: colorMap[cv], size: 7, opacity: 0.8, line: { width: 0.5, color: "#262a33" } },
        xaxis: `x${fi+1}`, yaxis: `y${fi+1}`,
      });
    });
  });
  const cols = facetBy ? Math.min(facets.length, 3) : 1;
  const rows = facetBy ? Math.ceil(facets.length / cols) : 1;
  // See renderLeaderboard — strip the base xaxis/yaxis to avoid aliasing the
  // first subplot's xaxis1 / yaxis1 (the first scatter facet was inheriting
  // unstyled base axes and rendering with categorical-looking ticks).
  const { xaxis: _bx, yaxis: _by, ...layoutBase } = PLOTLY_LAYOUT_BASE;
  const layout = {
    ...layoutBase,
    height: rows * 360 + 80,
    grid: { rows, columns: cols, pattern: "independent" },
    legend: { bgcolor: "#fff1e5", bordercolor: "#b3a9a0", borderwidth: 1, font: { family: "Inter, sans-serif" } },
    annotations: facetBy ? facets.map((f, i) => ({
      text: `<b>${f}</b>`, showarrow: false, x: 0, y: 1.0,
      xref: `x${i+1} domain`, yref: `y${i+1} domain`, yanchor: "bottom", xanchor: "left",
      font: { color: "#262a33", size: 13, family: "Inter, sans-serif" },
    })) : [],
  };
  for (let i = 1; i <= facets.length; i++) {
    layout[`xaxis${i}`] = { ...PLOTLY_LAYOUT_BASE.xaxis, type: "linear", title: { text: x, font: { size: 11 } } };
    layout[`yaxis${i}`] = { ...PLOTLY_LAYOUT_BASE.yaxis, type: "linear", title: { text: y, font: { size: 11 } } };
  }
  Plotly.react("chart-scatter", traces, layout, PLOTLY_CONFIG);
}

function renderCompare(data) {
  const map = new Map();
  data.forEach(r => {
    const key = [r.dataset, r.model, r.name, r.normalization].join("||");
    if (!map.has(key)) map.set(key, {});
    map.get(key)[r.method] = r;
  });
  const datasets = uniqSorted(data.map(r => r.dataset));
  const colorMap = {};
  datasets.forEach((d, i) => colorMap[d] = PALETTE[i % PALETTE.length]);

  const traces = datasets.map(ds => {
    const xs = [], ys = [], text = [];
    map.forEach(v => {
      if (v.knn5 && v.linear && v.knn5.dataset === ds) {
        xs.push(v.knn5.metric_value);
        ys.push(v.linear.metric_value);
        text.push(`<b>${v.knn5.name}</b><br>${ds} · norm=${v.knn5.normalization}<br>knn5=${v.knn5.metric_value.toFixed(4)}<br>linear=${v.linear.metric_value.toFixed(4)}`);
      }
    });
    return {
      type: "scatter", mode: "markers",
      x: xs, y: ys, text, hovertemplate: "%{text}<extra></extra>",
      name: ds, marker: { color: colorMap[ds], size: 7, opacity: 0.85, line: { width: 0.5, color: "#262a33" } },
    };
  }).filter(t => t.x.length);

  traces.push({
    type: "scatter", mode: "lines", x: [0, 1], y: [0, 1],
    line: { color: "#66605c", dash: "dash", width: 1 },
    showlegend: false, hoverinfo: "skip",
  });

  Plotly.react("chart-compare", traces, {
    ...PLOTLY_LAYOUT_BASE, height: 580,
    xaxis: {
      ...PLOTLY_LAYOUT_BASE.xaxis,
      type: "linear",
      title: { text: "KNN-5 accuracy", font: { size: 11 } },
      range: [0, 1.02], dtick: 0.1, tickformat: ".2f",
    },
    yaxis: {
      ...PLOTLY_LAYOUT_BASE.yaxis,
      type: "linear",
      title: { text: "Linear probe accuracy", font: { size: 11 } },
      range: [0, 1.02], dtick: 0.1, tickformat: ".2f",
    },
    legend: { bgcolor: "#fff1e5", bordercolor: "#b3a9a0", borderwidth: 1, font: { family: "Inter, sans-serif" } },
  }, PLOTLY_CONFIG);
}

function renderRanking(data) {
  const method = document.getElementById("ranking-method").value;
  const topn = parseInt(document.getElementById("ranking-topn").value, 10);
  const sub = data.filter(r => r.method === method);
  const groups = new Map();
  sub.forEach(r => {
    const key = `${r.name} · ${r.normalization}`;
    if (!groups.has(key)) groups.set(key, { vals: [], rows: [] });
    groups.get(key).vals.push(r.metric_value);
    groups.get(key).rows.push(r);
  });
  let entries = Array.from(groups.entries()).map(([key, g]) => ({
    key,
    mean: g.vals.reduce((s, v) => s + v, 0) / g.vals.length,
    n: g.vals.length,
    model: g.rows[0].model,
  }));
  entries.sort((a, b) => b.mean - a.mean);
  if (topn > 0) entries = entries.slice(0, topn);
  entries.reverse();

  const colorVals = uniqSorted(entries.map(e => e.model));
  const colorMap = {};
  colorVals.forEach((v, i) => colorMap[v] = PALETTE[i % PALETTE.length]);

  const trace = {
    type: "bar", orientation: "h",
    x: entries.map(e => e.mean),
    y: entries.map(e => e.key),
    marker: { color: entries.map(e => colorMap[e.model]), line: { color: "#262a33", width: 0.5 } },
    text: entries.map(e => `n=${e.n} datasets`),
    hovertemplate: "%{y}<br>mean accuracy=%{x:.4f}<br>%{text}<extra></extra>",
  };
  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    height: Math.max(360, entries.length * 22 + 80),
    xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "mean accuracy", font: { size: 11 } }, range: [0, 1.02] },
    yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, automargin: true, tickfont: { size: 10 } },
  };
  Plotly.react("chart-ranking", [trace], layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------
const TABLE_COLS = COLUMNS.map(c => {
  const def = { title: c, field: c, headerFilter: false };
  if (NUMERIC_COLS.includes(c)) {
    def.formatter = cell => {
      const v = cell.getValue();
      if (v === null || v === undefined || isNaN(v)) return "";
      return Math.abs(v) < 0.01 || Math.abs(v) > 1000 ? Number(v).toExponential(3) : Number(v).toFixed(4);
    };
    def.hozAlign = "right";
  }
  if (c === "model") def.formatter = cell => shortModel(cell.getValue());
  return def;
});

const table = new Tabulator("#data-table", {
  data: DATA,
  columns: TABLE_COLS,
  layout: "fitDataStretch",
  pagination: "local", paginationSize: 25, paginationSizeSelector: [25, 50, 100, 250],
  height: "640px",
  movableColumns: true,
});

// ---------------------------------------------------------------------------
// Lede + key findings
// ---------------------------------------------------------------------------
function median(arr) {
  if (!arr.length) return NaN;
  const sorted = arr.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function updateLede(data) {
  const datasets = uniqSorted(data.map(r => r.dataset));
  const variants = uniqSorted(data.map(r => `${r.name}|${r.normalization}|${r.model}`));
  const lin = data.filter(r => r.method === "linear");
  const knn = data.filter(r => r.method === "knn5");
  const best = data.length ? data.reduce((a, b) => b.metric_value > a.metric_value ? b : a) : null;

  document.getElementById("lede-models").textContent = variants.length;
  document.getElementById("lede-datasets").textContent = datasets.length;
  document.getElementById("lede-best").textContent = best ? formatPct(best.metric_value) : "—";
  document.getElementById("lede-best-dataset").textContent = best ? best.dataset : "—";
  document.getElementById("lede-median").textContent = data.length ? formatPct(median(data.map(r => r.metric_value))) : "—";

  const findings = document.getElementById("findings");
  const linMean = lin.length ? lin.reduce((s, r) => s + r.metric_value, 0) / lin.length : NaN;
  const knnMean = knn.length ? knn.reduce((s, r) => s + r.metric_value, 0) / knn.length : NaN;

  // Per-dataset best
  const perDs = datasets.map(ds => {
    const sub = data.filter(r => r.dataset === ds);
    if (!sub.length) return null;
    const top = sub.reduce((a, b) => b.metric_value > a.metric_value ? b : a);
    return { ds, top };
  }).filter(Boolean);

  const cards = [
    {
      num: variants.length,
      label: "model variants",
      detail: `Across ${data.length} (dataset, method, model, norm) experiments.`,
    },
    {
      num: best ? formatPct(best.metric_value) : "—",
      label: "Top accuracy",
      detail: best ? `${best.name} (${best.normalization}) on ${best.dataset} · ${best.method}.` : "",
    },
    {
      num: !isNaN(linMean) && !isNaN(knnMean) ? ((linMean - knnMean) >= 0 ? "+" : "") + formatPct(linMean - knnMean) : "—",
      label: "Linear probe vs KNN-5",
      detail: `Mean accuracy gap (linear minus KNN-5) across the current selection.`,
    },
    {
      num: data.length ? formatPct(median(data.map(r => r.metric_value))) : "—",
      label: "Median accuracy",
      detail: `Across all currently filtered observations.`,
    },
  ];
  findings.innerHTML = cards.map(c =>
    `<div class="card"><span class="num">${c.num}</span><div class="label">${c.label}</div><div class="detail">${c.detail}</div></div>`
  ).join("");
}

function updateSummary(data) {
  document.getElementById("summary-rows").textContent = data.length;
  document.getElementById("row-shown").textContent = data.length;
  if (!data.length) {
    document.getElementById("summary-best").textContent = "—";
    document.getElementById("summary-best-detail").textContent = "";
    return;
  }
  const best = data.reduce((a, b) => b.metric_value > a.metric_value ? b : a);
  document.getElementById("summary-best").textContent = best.metric_value.toFixed(4);
  document.getElementById("summary-best-detail").textContent =
    `(${best.name} · ${best.dataset} · ${best.method} · ${best.normalization})`;
}

let pending = null;
function refresh() {
  if (pending) cancelAnimationFrame(pending);
  pending = requestAnimationFrame(() => {
    const data = filteredData();
    updateLede(data);
    updateSummary(data);
    table.setData(data);
    renderLeaderboard(data);
    renderScatter(data);
    renderCompare(data);
    renderRanking(data);
  });
}

refresh();
</script>
</body>
</html>
"""


def main() -> int:
    """Generate a self-contained HTML visualizer for a results CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the results CSV.")
    parser.add_argument("html", type=Path, help="Output HTML file path.")
    parser.add_argument("--title", default=None, help="Page title (defaults to the CSV filename).")
    parser.add_argument(
        "--headline",
        default=None,
        help="Article headline (defaults to a generated summary).",
    )
    parser.add_argument(
        "--standfirst",
        default=None,
        help="Article standfirst / subhead (defaults to a generated summary).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), args.csv)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Compute defaults for headline / standfirst from the dataframe BEFORE we
    # erase NaNs, so numeric columns stay numeric for sorting.
    n_datasets = df["dataset"].nunique() if "dataset" in df else 0
    n_models = (
        df[["name", "normalization", "model"]].drop_duplicates().shape[0]
        if {"name", "normalization", "model"}.issubset(df.columns)
        else df.shape[0]
    )
    if "metric_value" in df and len(df):
        best_idx = df["metric_value"].idxmax()
        best = df.loc[best_idx]
        best_dataset = str(best.get("dataset", "—"))
        best_acc = float(best["metric_value"])
    else:
        best_dataset = "—"
        best_acc = float("nan")

    headline = args.headline or (f"How {n_models} frozen backbones perform on GeoBench")
    if args.standfirst:
        standfirst = args.standfirst
    else:
        standfirst = (
            f"A snapshot of {len(df):,} accuracy measurements across "
            f"{n_datasets} classification datasets and {n_models} model "
            f"variants. The best configuration in this run reaches "
            f"{best_acc * 100:.1f}% on <em>{best_dataset}</em> — "
            f"explore the data below."
        )

    pubdate = pd.Timestamp(args.csv.stat().st_mtime, unit="s").strftime("%-d %B %Y")

    df = df.astype(object).where(pd.notna(df), None)

    rows = df.to_dict(orient="records")
    payload = json.dumps(rows, default=str, allow_nan=False)

    title = args.title or f"torchgeo-bench results — {args.csv.name}"
    html = (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__HEADLINE__", headline)
        .replace("__STANDFIRST__", standfirst)
        .replace("__PUBDATE__", pubdate)
        .replace("__SOURCE__", str(args.csv))
        .replace("__ROWS__", str(len(df)))
        .replace("__DATA__", payload)
        .replace("__COLUMNS__", json.dumps(list(df.columns)))
        .replace("__NUMERIC__", json.dumps(numeric_cols))
    )

    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(html, encoding="utf-8")
    logger.info("Wrote %s (%.1f KB)", args.html, args.html.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
