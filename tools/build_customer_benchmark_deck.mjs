#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "/Users/pontiffscopez/.codex/plugins/cache/openai-primary-runtime/presentations/26.614.11602/skills/presentations/scripts/artifact_tool_utils.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CSV = path.join(ROOT, "outputs", "benchmark_report", "normalized_benchmark_runs.csv");
const OUTPUT_DIR = path.join(ROOT, "outputs", "benchmark_report");
const FINAL_PPTX = path.join(OUTPUT_DIR, "huawei-customer-benchmark-report-remastered.pptx");
const SCRATCH = path.join(os.tmpdir(), "codex-presentations", "manual-20260618", "customer-benchmark-report-remastered");
const PREVIEW_DIR = path.join(SCRATCH, "preview");
const LAYOUT_DIR = path.join(SCRATCH, "layout");
const QA_DIR = path.join(SCRATCH, "qa");
const CONTACT = path.join(OUTPUT_DIR, "huawei-customer-benchmark-remastered-contact-sheet.png");

const W = 1280;
const H = 720;
const PAGE = { left: 58, top: 56, width: 1164, height: 600 };
const RED = "#c7000b";
const INK = "#111827";
const SLATE = "#334155";
const MUTED = "#64748b";
const GRID = "#e2e8f0";
const BG = "#f8fafc";
const GREEN = "#15803d";
const AMBER = "#d97706";
const BLUE = "#2563eb";
const WHITE = "#ffffff";
const CLEAR = "#00000000";

function parseCsv(source) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i];
    const next = source[i + 1];
    if (inQuotes && ch === '"' && next === '"') {
      cell += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (!inQuotes && ch === ",") {
      row.push(cell);
      cell = "";
    } else if (!inQuotes && (ch === "\n" || ch === "\r")) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const header = rows.shift();
  return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

function n(value) {
  if (value === undefined || value === null || value === "") return null;
  const out = Number(value);
  return Number.isFinite(out) ? out : null;
}

function fmt(value, digits = 1, suffix = "") {
  if (value === null || value === undefined || value === "") return "n/a";
  const out = Number(value);
  if (!Number.isFinite(out)) return String(value);
  if (digits === 0) return `${Math.round(out).toLocaleString()}${suffix}`;
  if (Math.abs(out) >= 100) return `${Math.round(out).toLocaleString()}${suffix}`;
  if (Math.abs(out) >= 10) return `${out.toFixed(digits)}${suffix}`;
  return `${out.toFixed(Math.max(1, digits))}${suffix}`;
}

function rowsFromCsv() {
  return parseCsv(awaitableRead(CSV)).map((r) => ({
    ...r,
    data_parallelism: n(r.data_parallelism),
    tensor_parallelism: n(r.tensor_parallelism),
    nodes: n(r.nodes),
    concurrency: n(r.concurrency),
    total_requests: n(r.total_requests),
    successful_requests: n(r.successful_requests),
    failed_requests: n(r.failed_requests),
    error_rate_percent: n(r.error_rate_percent),
    input_tokens: n(r.input_tokens),
    output_tokens: n(r.output_tokens),
    overall_tokens_per_second: n(r.overall_tokens_per_second),
    ttft_p95: n(r.ttft_p95),
    latency_p95: n(r.latency_p95),
    tpot_mean: n(r.tpot_mean),
    duration_seconds: n(r.duration_seconds),
  }));
}

function awaitableRead(file) {
  return fsSyncRead(file);
}

function fsSyncRead(file) {
  return globalThis.__fsReadFileSync(file, "utf8");
}

function family(row) {
  const s = `${row.model} ${row.source_file}`.toLowerCase();
  if ((s.includes("deepseek_v4") || s.includes("deepseek-v4")) && row.platform === "ModelArts") return "DeepSeek V4 Flash W8A8 MTP";
  if (s.includes("deepseek_v4") || s.includes("deepseek-v4")) return "DeepSeek V4 Flash";
  if (s.includes("deepseek-r1")) return "DeepSeek R1 TP16";
  if (s.includes("glm51-w4a8-maxseq16")) return "GLM 5.1 W4A8 maxseq16";
  if (s.includes("glm51-w4a8") || s.includes("glm-5.1")) return "GLM 5.1 W4A8";
  if (s.includes("qwen35-397b-w8a8-tool")) return "Qwen3.5 397B W8A8 tool calling";
  if (s.includes("qwen35-w8a8-quarot")) return "Qwen3.5 W8A8 Quarot";
  if (s.includes("qwen35") || s.includes("qwen3.5")) return "Qwen3.5";
  if (s.includes("qwen3-vl-30b")) return "Qwen3-VL 30B";
  if (s.includes("qwen3-32b")) return "Qwen3 32B";
  if (s.includes("gpt-oss-120b")) return "GPT-OSS 120B";
  if (s.includes("tubitak")) return "TUBITAK GPT";
  return row.model;
}

function compactModel(row) {
  const name = family(row);
  return name
    .replace("DeepSeek V4 Flash W8A8 MTP", "DS V4 Flash W8A8 MTP")
    .replace("GLM 5.1 W4A8 maxseq16", "GLM5.1 W4A8 maxseq16")
    .replace("Qwen3.5 397B W8A8 tool calling", "Qwen3.5 397B W8A8")
    .replace("Qwen3.5 W8A8 Quarot", "Qwen3.5 W8A8 Quarot");
}

function workload(row) {
  const s = row.scenario.toLowerCase();
  if (s.includes("heavy")) return "Heavy doc";
  if (s.includes("doc")) return "Doc summary";
  if (s.includes("code")) return "Code gen";
  if (s.includes("long_output") || row.output_tokens >= 4000) return "Long output";
  if (s.includes("chat")) return "Light chat";
  if (s.includes("image")) return "Vision";
  if (s.includes("tool")) return "Tool calling";
  return "Other";
}

function rowLabel(row) {
  return `${compactModel(row)} ${workload(row)} c${row.concurrency ?? "n/a"}`;
}

function platformStats(rows) {
  const clean = rows.filter((r) => (r.error_rate_percent ?? 0) === 0).length;
  const req = rows.reduce((a, r) => a + (r.total_requests ?? 0), 0);
  const best = topRows(rows, 1)[0];
  const conc = [...new Set(rows.map((r) => r.concurrency).filter((v) => v !== null))].sort((a, b) => a - b);
  const models = [...new Set(rows.map((r) => compactModel(r)))].sort();
  return { clean, req, best, conc, models };
}

function addRect(slide, left, top, width, height, fill = WHITE, lineFill = CLEAR, radius = 0) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === CLEAR ? 0 : 1 },
    borderRadius: radius ? "rounded-md" : undefined,
  });
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: CLEAR,
    line: { style: "solid", fill: CLEAR, width: 0 },
  });
  box.text = text;
  box.text.fontSize = opts.size ?? 18;
  box.text.color = opts.color ?? INK;
  box.text.bold = Boolean(opts.bold);
  box.text.typeface = opts.face ?? "Arial";
  box.text.alignment = opts.align ?? "left";
  box.text.verticalAlignment = opts.valign ?? "top";
  box.text.insets = opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 };
  return box;
}

function notes(slide, lines) {
  slide.speakerNotes.textFrame.setText(Array.isArray(lines) ? lines : [lines]);
  slide.speakerNotes.setVisible(true);
}

function addLine(slide, x1, y1, x2, y2, color = GRID, width = 1) {
  return slide.shapes.add({
    geometry: "line",
    position: { left: x1, top: y1, width: Math.max(1, x2 - x1), height: Math.max(1, y2 - y1) },
    line: { style: "solid", fill: color, width },
    fill: CLEAR,
  });
}

function titleSlide(slide, title, subtitle, tag = "Customer-ready technical benchmark") {
  addRect(slide, 0, 0, W, 84, INK);
  addRect(slide, 0, 84, W, 5, RED);
  addText(slide, "HUAWEI CLOUD", 58, 30, 260, 22, { size: 13, bold: true, color: "#cbd5e1" });
  addText(slide, "benchmark report", 1090, 32, 130, 18, { size: 12, bold: true, color: WHITE, align: "right" });
  addText(slide, title, 58, 182, 820, 118, { size: 45, bold: true, color: INK });
  addText(slide, subtitle, 62, 326, 700, 74, { size: 20, color: SLATE });
  addRect(slide, 62, 522, 470, 1, GRID);
  addText(slide, tag, 62, 548, 460, 22, { size: 14, color: MUTED, bold: true });
  addText(slide, "MaaS + ModelArts", 940, 548, 250, 28, { size: 22, color: RED, bold: true, align: "right" });
}

function baseSlide(slide, title, subtitle = "") {
  addRect(slide, 0, 0, W, 92, INK);
  addRect(slide, 1080, 0, 200, 92, RED);
  addText(slide, "HUAWEI CLOUD", 58, 20, 180, 18, { size: 11, bold: true, color: "#cbd5e1" });
  addText(slide, title, 58, 44, 830, 34, { size: 26, bold: true, color: WHITE });
  if (subtitle) addText(slide, subtitle, 58, 102, 900, 28, { size: 15, color: MUTED });
  addText(slide, "benchmark report", 1120, 34, 110, 18, { size: 12, bold: true, color: WHITE, align: "right" });
}

function card(slide, left, top, width, height, title, value, note, accent = RED) {
  addRect(slide, left, top, width, height, WHITE, GRID, 4);
  addRect(slide, left, top, 5, height, accent);
  addText(slide, title, left + 20, top + 16, width - 34, 18, { size: 12, color: MUTED, bold: true });
  addText(slide, value, left + 20, top + 43, width - 34, 40, { size: 30, bold: true });
  addText(slide, note, left + 20, top + 90, width - 34, 34, { size: 12, color: MUTED });
}

function decisionCard(slide, left, top, width, height, title, body, tag, accent = RED) {
  addRect(slide, left, top, width, height, WHITE, GRID, 8);
  addRect(slide, left, top, 6, height, accent);
  addText(slide, title, left + 24, top + 20, width - 44, 26, { size: 19, bold: true });
  addText(slide, body, left + 24, top + 60, width - 44, height - 100, { size: 15, color: SLATE });
  addText(slide, tag, left + 24, top + height - 34, width - 44, 16, { size: 11, color: MUTED, bold: true });
}

function bullets(slide, items, left, top, width, gap = 58, size = 18) {
  items.forEach((item, i) => {
    const y = top + i * gap;
    addRect(slide, left, y + 7, 7, 7, RED, CLEAR, 4);
    addText(slide, item, left + 22, y, width - 22, gap - 5, { size, color: SLATE });
  });
}

function nativeBar(slide, left, top, width, height, title, rows, metric, opts = {}) {
  addRect(slide, left, top, width, height, WHITE, GRID, 6);
  addText(slide, title, left + 24, top + 18, width - 48, 24, { size: 17, bold: true });
  const cats = rows.map(opts.label ?? rowLabel);
  const values = rows.map((r) => r[metric] ?? 0);
  slide.charts.add(opts.type ?? "bar", {
    position: { left: left + 30, top: top + 62, width: width - 60, height: height - 94 },
    categories: cats,
    series: [{ name: opts.seriesName ?? metric, values, fill: opts.fill ?? RED }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
    xAxis: { textStyle: { fontSize: 9, color: MUTED } },
    yAxis: { majorGridlines: { style: "solid", fill: GRID, width: 1 }, textStyle: { fontSize: 9, color: MUTED } },
  });
}

function nativeColumn(slide, left, top, width, height, title, categories, series, opts = {}) {
  addRect(slide, left, top, width, height, WHITE, GRID, 6);
  addText(slide, title, left + 24, top + 18, width - 48, 24, { size: 17, bold: true });
  slide.charts.add("column", {
    position: { left: left + 34, top: top + 64, width: width - 68, height: height - 100 },
    categories,
    series,
    hasLegend: opts.hasLegend ?? true,
    legend: { position: "bottom" },
    dataLabels: { showValue: false },
    yAxis: { majorGridlines: { style: "solid", fill: GRID, width: 1 }, textStyle: { fontSize: 9, color: MUTED } },
    xAxis: { textStyle: { fontSize: 9, color: MUTED } },
  });
}

function shapeBars(slide, left, top, width, height, title, items, opts = {}) {
  addRect(slide, left, top, width, height, WHITE, GRID, 6);
  addText(slide, title, left + 24, top + 18, width - 48, 24, { size: 17, bold: true });
  const plotLeft = left + (opts.labelWidth ?? 135);
  const plotTop = top + 68;
  const plotW = width - (opts.labelWidth ?? 135) - 42;
  const plotH = height - 108;
  const max = Math.max(...items.map((d) => d.value), 1);
  for (let i = 0; i <= 4; i += 1) {
    const x = plotLeft + (plotW * i) / 4;
    addLine(slide, x, plotTop, x, plotTop + plotH, GRID, 1);
  }
  const rowH = plotH / Math.max(1, items.length);
  items.forEach((item, i) => {
    const y = plotTop + i * rowH + rowH * 0.22;
    const barW = (plotW * item.value) / max;
    addText(slide, item.label, left + 22, y - 1, (opts.labelWidth ?? 135) - 30, rowH * 0.58, { size: opts.labelSize ?? 9, color: MUTED, align: "right" });
    addRect(slide, plotLeft, y, barW, Math.max(10, rowH * 0.48), item.color ?? RED, CLEAR, 3);
    addText(slide, fmt(item.value, opts.digits ?? 1, opts.suffix ?? ""), plotLeft + barW + 6, y - 1, 74, 16, { size: 9, color: INK, bold: true });
  });
}

function groupedBars(slide, left, top, width, height, title, categories, groups, opts = {}) {
  addRect(slide, left, top, width, height, WHITE, GRID, 6);
  addText(slide, title, left + 24, top + 18, width - 48, 24, { size: 17, bold: true });
  const plotLeft = left + 72;
  const plotTop = top + 70;
  const plotW = width - 110;
  const plotH = height - 135;
  const max = Math.max(...groups.flatMap((g) => g.values), 1);
  for (let i = 0; i <= 4; i += 1) {
    const y = plotTop + (plotH * i) / 4;
    addLine(slide, plotLeft, y, plotLeft + plotW, y, GRID, 1);
    addText(slide, fmt(max * (1 - i / 4), 0), left + 20, y - 7, 44, 14, { size: 8, color: MUTED, align: "right" });
  }
  const bandW = plotW / categories.length;
  const barW = Math.min(22, (bandW - 18) / groups.length);
  categories.forEach((cat, ci) => {
    const cx = plotLeft + ci * bandW;
    groups.forEach((group, gi) => {
      const val = group.values[ci] ?? 0;
      const h = (plotH * val) / max;
      const x = cx + 9 + gi * (barW + 4);
      addRect(slide, x, plotTop + plotH - h, barW, h, group.color, CLEAR, 2);
    });
    addText(slide, cat, cx + 2, plotTop + plotH + 10, bandW - 4, 26, { size: opts.labelSize ?? 8, color: MUTED, align: "center" });
  });
  groups.forEach((group, i) => {
    const x = left + width - 140 + i * 65;
    addRect(slide, x, top + height - 34, 10, 10, group.color);
    addText(slide, group.name, x + 15, top + height - 38, 54, 16, { size: 9, color: MUTED });
  });
}

function miniTable(slide, left, top, width, height, title, data, cols) {
  addRect(slide, left, top, width, height, WHITE, GRID, 6);
  addText(slide, title, left + 18, top + 14, width - 36, 22, { size: 16, bold: true });
  const headerTop = top + 48;
  addRect(slide, left + 18, headerTop, width - 36, 26, INK);
  const colW = (width - 36) / cols.length;
  cols.forEach((c, i) => addText(slide, c.label, left + 24 + i * colW, headerTop + 7, colW - 8, 12, { size: 8.5, color: WHITE, bold: true }));
  const rowH = Math.min(28, (height - 84) / Math.max(1, data.length));
  data.forEach((row, ri) => {
    const y = headerTop + 30 + ri * rowH;
    if (ri % 2) addRect(slide, left + 18, y - 2, width - 36, rowH, "#f8fafc");
    cols.forEach((c, ci) => addText(slide, c.value(row), left + 24 + ci * colW, y + 3, colW - 8, rowH - 2, { size: c.size ?? 8.5, color: c.color?.(row) ?? INK }));
  });
}

const MATRIX_COLS = [
  { label: "Platform", value: (r) => r.platform, size: 7 },
  { label: "Model", value: (r) => family(r), size: 7 },
  { label: "Scenario", value: (r) => r.scenario, size: 6.5 },
  { label: "DP", value: (r) => fmt(r.data_parallelism, 0), size: 7 },
  { label: "TP", value: (r) => fmt(r.tensor_parallelism, 0), size: 7 },
  { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}`, size: 7 },
  { label: "Req.", value: (r) => fmt(r.total_requests, 0), size: 7 },
  { label: "In/Out", value: (r) => `${fmt(r.input_tokens, 0)}/${fmt(r.output_tokens, 0)}`, size: 7 },
  { label: "Tok/s", value: (r) => fmt(r.overall_tokens_per_second, 1), size: 7 },
  { label: "Lat95", value: (r) => `${fmt(r.latency_p95, 1)}s`, size: 7 },
  { label: "Err", value: (r) => fmt(r.error_rate_percent, 1, "%"), size: 7, color: (r) => (r.error_rate_percent ? RED : GREEN) },
];

function addPlatformSection(slide, platformName, headline, body, accent = RED) {
  addRect(slide, 0, 0, W, 84, INK);
  addRect(slide, 0, 84, W, 5, accent);
  addText(slide, "HUAWEI CLOUD", 58, 30, 220, 20, { size: 12, bold: true, color: "#cbd5e1" });
  addText(slide, "workload family", 1090, 32, 130, 18, { size: 12, bold: true, color: WHITE, align: "right" });
  addText(slide, platformName, 58, 202, 760, 62, { size: 48, bold: true, color: INK });
  addText(slide, headline, 60, 294, 760, 54, { size: 25, bold: true, color: SLATE });
  addRect(slide, 62, 386, 92, 5, accent);
  addText(slide, body, 62, 420, 760, 84, { size: 19, color: SLATE });
  addRect(slide, 930, 194, 210, 210, WHITE, GRID, 8);
  addRect(slide, 930, 194, 6, 210, accent);
  addText(slide, "Scope", 962, 228, 130, 22, { size: 20, bold: true });
  addText(slide, platformName === "MaaS" ? "Managed service probe results" : "Controlled deployment results", 962, 268, 132, 64, { size: 17, color: SLATE });
  addText(slide, platformName === "MaaS" ? "Topology not inferred" : "DP/TP interpreted", 962, 352, 132, 24, { size: 13, color: MUTED, bold: true });
}

function addMatrixSlides(presentation, titlePrefix, rows, startIndex = 1) {
  const all = [...rows].sort((a, b) => `${family(a)}-${a.scenario}`.localeCompare(`${family(b)}-${b.scenario}`));
  const chunks = [];
  for (let i = 0; i < all.length; i += 18) chunks.push(all.slice(i, i + 18));
  chunks.forEach((chunk, index) => {
    const slide = presentation.slides.add();
    baseSlide(slide, `${titlePrefix} matrix ${index + 1}/${chunks.length}`, "Complete normalized details are also available in outputs/benchmark_report/normalized_benchmark_runs.csv.");
    miniTable(slide, 36, 122, 1208, 532, `Rows ${startIndex + index * 18}-${startIndex + index * 18 + chunk.length - 1}`, chunk, MATRIX_COLS);
  });
}

function topRows(rows, count = 10) {
  return [...rows].filter((r) => r.overall_tokens_per_second !== null).sort((a, b) => b.overall_tokens_per_second - a.overall_tokens_per_second).slice(0, count);
}

function average(values) {
  const clean = values.filter((v) => v !== null && v !== undefined && Number.isFinite(v));
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

function groupBy(rows, keyFn) {
  const map = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(row);
  }
  return map;
}

function summarizeBestByWorkload(rows, predicate) {
  const filtered = rows.filter(predicate);
  const byWorkload = groupBy(filtered, workload);
  return [...byWorkload.entries()].map(([name, items]) => {
    const best = topRows(items, 1)[0];
    return { name, best, throughput: best?.overall_tokens_per_second ?? 0, latency: best?.latency_p95 ?? 0 };
  }).filter((x) => x.best).sort((a, b) => b.throughput - a.throughput);
}

function makeDeck(artifact, rows) {
  const { Presentation } = artifact;
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const modelArts = rows.filter((r) => r.platform === "ModelArts");
  const maas = rows.filter((r) => r.platform === "MaaS");
  const deepseekMtp = rows.filter((r) => r.model === "deepseek_v4");
  const glmMax = rows.filter((r) => r.model === "GLM51-W4A8-maxseq16");
  const glmDp1 = glmMax.filter((r) => r.data_parallelism === 1);
  const glmDp2 = glmMax.filter((r) => r.data_parallelism === 2);
  const dsR1 = rows.filter((r) => r.tensor_parallelism === 16);
  const zeroErr = rows.filter((r) => (r.error_rate_percent ?? 0) === 0).length;
  const totalReq = rows.reduce((a, r) => a + (r.total_requests ?? 0), 0);
  const best = topRows(rows, 1)[0];
  const maasStats = platformStats(maas);
  const modelArtsStats = platformStats(modelArts);

  let slide = presentation.slides.add();
  titleSlide(slide, "LLM Serving Benchmark Results", "A customer-facing readout of throughput, latency, concurrency, and serving topology trade-offs.");
  notes(slide, [
    "Position this as a benchmark readout, not a universal model ranking.",
    "The central decision question is which serving profile fits each target workload and SLO.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "MaaS and ModelArts are separate benchmark families", "The deck is organized by serving surface because the workloads, topology visibility, and operational questions differ.");
  card(slide, 58, 150, 260, 132, "Total coverage", `${rows.length} runs`, `${fmt(totalReq, 0)} total requests`, RED);
  card(slide, 344, 150, 260, 132, "MaaS", `${maas.length} runs`, `${fmt(maasStats.req, 0)} requests; topology n/a`, INK);
  card(slide, 630, 150, 260, 132, "ModelArts", `${modelArts.length} runs`, `${fmt(modelArtsStats.req, 0)} requests; DP/TP tracked`, GREEN);
  card(slide, 916, 150, 260, 132, "New variants", "DS V4 Flash / GLM W4A8", "W8A8 MTP and maxseq16 covered", RED);
  decisionCard(slide, 86, 344, 325, 160, "MaaS question", "What is the externally observed serving behavior under request pressure when topology is not exposed in the artifacts?", "Read as service-facing workload evidence", INK);
  decisionCard(slide, 472, 344, 325, 160, "ModelArts question", "Which DP/TP, concurrency, sequence length, and token profile fit each controlled deployment workload?", "Read as topology-aware tuning evidence", RED);
  decisionCard(slide, 858, 344, 325, 160, "Customer decision", "Pick the serving surface first, then compare only compatible workloads inside that surface.", "Avoid cross-surface leaderboards", GREEN);
  notes(slide, [
    `Normalized runs: ${rows.length}; requests represented: ${totalReq}.`,
    `Peak observed throughput overall: ${fmt(best.overall_tokens_per_second, 1)} output tok/s for ${rowLabel(best)}.`,
    `MaaS coverage: ${maas.length} rows. ModelArts coverage: ${modelArts.length} rows.`,
    "DeepSeek V4 Flash W8A8 MTP and GLM 5.1 W4A8 maxseq16 are the newest additions included in this remaster.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "The benchmark compares traffic profiles and topology", "Each row carries platform, model, DP/TP topology, concurrency, request count, and input/output token target.");
  addRect(slide, 66, 150, 520, 340, WHITE, GRID, 8);
  addText(slide, "Serving surfaces", 96, 178, 300, 26, { size: 20, bold: true });
  bullets(slide, [
    `ModelArts: ${modelArts.length} normalized runs with DP/TP metadata when encoded in filenames or summaries.`,
    `MaaS: ${maas.length} normalized runs; topology not exposed in artifacts, reported as n/a.`,
    "DeepSeek R1 TP16 is annotated as DP1 over two nodes per deployment note.",
  ], 104, 236, 420, 66, 17);
  addRect(slide, 646, 150, 520, 340, WHITE, GRID, 8);
  addText(slide, "Metric contract", 676, 178, 300, 26, { size: 20, bold: true });
  bullets(slide, [
    "Aggregate tok/s = successful generated output tokens over observed wall-clock duration.",
    "P95 latency captures tail completion time; P95 TTFT captures first-token responsiveness.",
    "For raw CSVs, concurrency is derived from filenames or peak overlapping request windows.",
  ], 684, 236, 420, 66, 17);
  addText(slide, "Gaps are explicit: hardware and serving framework versions are not fully present in the artifacts, so this deck does not infer them.", 96, 548, 980, 34, { size: 18, color: SLATE, bold: true });
  notes(slide, [
    "Available topology fields are DP, TP, node count, concurrency, request count, and token profile.",
    "Hardware details, CPU, memory, network, storage, serving framework version, and runtime flags are not consistently available in the artifacts.",
    "DeepSeek R1 TP16 topology uses the user-provided note: DP1 across two nodes.",
  ]);

  slide = presentation.slides.add();
  addPlatformSection(slide, "MaaS", "Externally observed API-service behavior", "MaaS rows are treated as service-facing probes. The artifacts expose model, request pressure, token shape, throughput, latency, and errors; DP/TP topology is intentionally left as n/a.");
  notes(slide, [
    "This section does not infer backend topology for MaaS.",
    "Use MaaS slides for customer-facing service behavior and operational caveats, not DP/TP tuning.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "MaaS results are service-facing workload probes", "Topology is not exposed in the source artifacts, so the focus is request pressure, latency, throughput, and errors.");
  card(slide, 70, 142, 250, 126, "MaaS coverage", `${maas.length} runs`, `${fmt(maasStats.req, 0)} requests`, INK);
  card(slide, 350, 142, 250, 126, "Clean rows", `${fmt(maasStats.clean, 0)}/${maas.length}`, "Zero-error normalized runs", GREEN);
  card(slide, 630, 142, 250, 126, "Peak MaaS tok/s", `${fmt(maasStats.best?.overall_tokens_per_second, 0)}`, maasStats.best ? rowLabel(maasStats.best) : "n/a", RED);
  card(slide, 910, 142, 250, 126, "Concurrency range", `${fmt(maasStats.conc[0], 0)}-${fmt(maasStats.conc.at(-1), 0)}`, `${maasStats.models.join(", ")}`, INK);
  nativeBar(slide, 70, 320, 700, 290, "Top MaaS throughput rows", topRows(maas, 6), "overall_tokens_per_second", {
    label: (r) => `${compactModel(r)} ${workload(r)} c${r.concurrency}`,
    seriesName: "tok/s",
    fill: INK,
  });
  miniTable(slide, 810, 320, 370, 290, "MaaS technical rows", topRows(maas, 6), [
    { label: "Model", value: (r) => family(r), size: 7.5 },
    { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}` },
    { label: "In/Out", value: (r) => `${fmt(r.input_tokens, 0)}/${fmt(r.output_tokens, 0)}`, size: 7.5 },
    { label: "Lat95", value: (r) => `${fmt(r.latency_p95, 1)}s` },
    { label: "Err", value: (r) => fmt(r.error_rate_percent, 1, "%"), color: (r) => (r.error_rate_percent ? RED : GREEN) },
  ]);
  notes(slide, [
    "MaaS comparisons should stay within MaaS because the workload is an externally observed managed-service probe.",
    "DP, TP, and node count are not inferred where artifacts do not provide them.",
  ]);

  slide = presentation.slides.add();
  addPlatformSection(slide, "ModelArts", "Topology-aware deployment benchmark", "ModelArts rows carry DP/TP, node count, concurrency, request count, and token profiles where encoded in summary or filename artifacts. Use this section for controlled serving configuration choices.", GREEN);
  notes(slide, [
    "This section is where topology-aware claims belong.",
    "DeepSeek R1 TP16 is annotated as DP1 across two nodes from the supplied deployment note.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "ModelArts throughput is topology-aware capacity evidence", "Use aggregate output-token throughput only after matching model variant, workload, DP/TP, concurrency, and token profile.");
  nativeBar(slide, 72, 145, 650, 455, "Top ModelArts output throughput", topRows(modelArts, 10), "overall_tokens_per_second", {
    label: (r) => `${compactModel(r)} ${workload(r)} c${r.concurrency}`,
    seriesName: "tok/s",
    fill: RED,
  });
  addRect(slide, 772, 145, 360, 455, WHITE, GRID, 8);
  addText(slide, "How to read it", 802, 178, 250, 28, { size: 22, bold: true });
  bullets(slide, [
    "Capacity planning starts here.",
    "DP/TP and token shape explain many differences.",
    "Interactive workloads still need TTFT and P95 latency checks.",
  ], 812, 246, 260, 78, 17);
  notes(slide, [
    "This chart is limited to ModelArts rows.",
    "High-throughput rows still include different models, token profiles, and concurrency levels. Use the appendix before making a like-for-like claim.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "DeepSeek V4 Flash W8A8 MTP adds clean ModelArts coverage", "DP1 results cover light chat, document summarization, heavy document summarization, and code generation.");
  const dsBest = summarizeBestByWorkload(deepseekMtp, () => true).slice(0, 5);
  shapeBars(slide, 70, 142, 540, 430, "Best throughput by workload", dsBest.map((x) => ({
    label: x.name,
    value: x.throughput,
    color: RED,
  })), { suffix: " tok/s", digits: 0 });
  shapeBars(slide, 650, 142, 540, 430, "P95 latency for same best runs", dsBest.map((x) => ({
    label: x.name,
    value: x.latency,
    color: AMBER,
  })), { suffix: "s", digits: 1 });
  addText(slide, "Senior-dev note: code generation uses 2k input / 4k output and behaves differently from 1k/1k chat; compare by token profile before drawing model-level conclusions.", 88, 610, 1020, 34, { size: 15, color: SLATE, bold: true });
  notes(slide, [
    "DeepSeek V4 Flash W8A8 MTP rows are sourced from ModelArts summary JSON files.",
    "All DeepSeek V4 Flash W8A8 MTP rows included here report zero errors.",
    "Code-generation rows use a 2048 input / 4096 output target, so completion latency should not be compared directly to 1024/1024 light-chat rows.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "GLM 5.1 W4A8 maxseq16 scaling is workload-dependent", "DP2 does not dominate every workload; some single-request and document workloads favor DP1 in this dataset.");
  const comparable = ["Light chat", "Doc summary", "Heavy doc", "Code gen", "Long output"];
  const bestFor = (arr, name) => topRows(arr.filter((r) => workload(r) === name), 1)[0]?.overall_tokens_per_second ?? 0;
  groupedBars(slide, 70, 142, 540, 430, "Best throughput by DP", comparable, [
    { name: "DP1", values: comparable.map((c) => bestFor(glmDp1, c)), color: RED },
    { name: "DP2", values: comparable.map((c) => bestFor(glmDp2, c)), color: INK },
  ]);
  const latFor = (arr, name) => topRows(arr.filter((r) => workload(r) === name), 1)[0]?.latency_p95 ?? 0;
  groupedBars(slide, 650, 142, 540, 430, "P95 latency on best-throughput runs", comparable, [
    { name: "DP1", values: comparable.map((c) => latFor(glmDp1, c)), color: AMBER },
    { name: "DP2", values: comparable.map((c) => latFor(glmDp2, c)), color: BLUE },
  ]);
  addText(slide, "Interpretation: treat DP as a tuning dimension tied to sequence length, concurrency, and workload mix. Do not promote one DP setting globally from a single aggregate chart.", 88, 610, 1050, 34, { size: 15, color: SLATE, bold: true });
  notes(slide, [
    "GLM 5.1 W4A8 maxseq16 has DP1 and DP2 summary rows across light chat, doc summarization, heavy doc summarization, code generation, and long-output scenarios.",
    "The chart uses best-throughput row per workload per DP setting. It is directional, not a full factorial tuning study.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "ModelArts latency separates usable capacity from raw throughput", "P95 completion latency and P95 TTFT expose different user-experience risks inside controlled deployment runs.");
  const latencyTop = [...modelArts].filter((r) => r.latency_p95 !== null && (r.error_rate_percent ?? 0) === 0).sort((a, b) => b.latency_p95 - a.latency_p95).slice(0, 8);
  nativeBar(slide, 72, 145, 650, 455, "Highest zero-error P95 completion latency", latencyTop, "latency_p95", {
    label: (r) => `${compactModel(r)} ${workload(r)} c${r.concurrency}`,
    fill: AMBER,
    seriesName: "seconds",
  });
  const ttftTop = [...modelArts].filter((r) => r.ttft_p95 !== null && (r.error_rate_percent ?? 0) === 0).sort((a, b) => b.ttft_p95 - a.ttft_p95).slice(0, 8);
  nativeBar(slide, 772, 145, 410, 455, "Highest zero-error P95 TTFT", ttftTop, "ttft_p95", {
    label: (r) => `${compactModel(r)} c${r.concurrency}`,
    fill: RED,
    seriesName: "seconds",
  });
  notes(slide, [
    "Latency chart filters to zero-error ModelArts rows so failure-heavy probes do not dominate the customer experience discussion.",
    "P95 TTFT matters most for streaming responsiveness; P95 completion latency matters for total task waiting time.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "Reliability is interpreted separately by serving surface", "MaaS failures indicate managed-service probe behavior; ModelArts failures indicate controlled deployment or configuration behavior.");
  const errorRows = [...rows].filter((r) => (r.error_rate_percent ?? 0) > 0).sort((a, b) => b.error_rate_percent - a.error_rate_percent);
  card(slide, 70, 142, 260, 126, "MaaS clean rows", `${fmt(maasStats.clean, 0)}/${maas.length}`, "Service-facing probes", INK);
  card(slide, 360, 142, 260, 126, "MaaS failed probes", `${fmt(maas.length - maasStats.clean, 0)}`, "Connection/rate-limit patterns retained", RED);
  card(slide, 650, 142, 260, 126, "ModelArts clean rows", `${fmt(modelArtsStats.clean, 0)}/${modelArts.length}`, "Topology-aware runs", GREEN);
  card(slide, 940, 142, 260, 126, "ModelArts failed rows", `${fmt(modelArts.length - modelArtsStats.clean, 0)}`, "Config/workload evidence", RED);
  miniTable(slide, 78, 330, 1080, 255, "Highest error-rate rows", errorRows.slice(0, 8), [
    { label: "Platform", value: (r) => r.platform },
    { label: "Model", value: (r) => family(r), size: 8 },
    { label: "Scenario", value: (r) => workload(r), size: 8 },
    { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}` },
    { label: "Req.", value: (r) => fmt(r.total_requests, 0) },
    { label: "Err %", value: (r) => fmt(r.error_rate_percent, 1, "%"), color: () => RED },
  ]);
  notes(slide, [
    "MaaS failed probes include connection and rate-limit patterns from earlier tests.",
    "ModelArts failures should be interpreted in the context of the explicit DP/TP, concurrency, token profile, and workload.",
    "These rows are retained for operational context and should not be used as a direct cross-surface model quality comparison.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "Recommended next step is a workload-specific validation run", "Use this deck to narrow candidates, then run a controlled SLO validation for the target customer workload.");
  decisionCard(slide, 76, 156, 330, 210, "1. Select target workload", "Choose one primary workload first: chat, summarization, heavy document processing, code generation, vision, or tool calling.", "Owner: customer + Huawei solution team", RED);
  decisionCard(slide, 476, 156, 330, 210, "2. Lock test conditions", "Fix model version, hardware, serving stack, DP/TP, context length, request count, concurrency, and token profile.", "Owner: platform engineering", INK);
  decisionCard(slide, 876, 156, 330, 210, "3. Validate against SLO", "Measure output tok/s, TTFT P95, latency P95/P99, TPOT, error rate, and cost/resource utilization.", "Owner: benchmark team", GREEN);
  addRect(slide, 96, 452, 1030, 86, WHITE, GRID, 8);
  addText(slide, "Recommendation", 124, 474, 180, 24, { size: 18, bold: true });
  addText(slide, "Do not position the result as a single model ranking. Position it as evidence for selecting the right serving profile under a customer-specific workload and SLO.", 312, 470, 770, 44, { size: 18, color: SLATE, bold: true });
  notes(slide, [
    "This slide is the customer-facing close. It converts benchmark evidence into a concrete validation plan.",
    "If the customer asks for raw details, move to the appendix matrix slides.",
  ]);

  slide = presentation.slides.add();
  baseSlide(slide, "Technical appendix: ModelArts topology rows", "Rows senior developers usually ask for first: DP/TP, concurrency, requests, token shape, throughput, and tail latency.");
  const techRows = [
    ...deepseekMtp,
    ...glmMax.filter((r) => ["light_chat_16c", "doc_summarization_16c", "heavy_doc_summarization_4c", "single_long_output_1k_in_4k_out_1c"].some((s) => r.scenario.includes(s))),
    ...dsR1,
  ].slice(0, 18);
  miniTable(slide, 44, 126, 1190, 520, "Selected technical matrix", techRows, [
    { label: "Model", value: (r) => family(r), size: 7.5 },
    { label: "Scenario", value: (r) => r.scenario, size: 7 },
    { label: "DP", value: (r) => fmt(r.data_parallelism, 0) },
    { label: "TP", value: (r) => fmt(r.tensor_parallelism, 0) },
    { label: "Nodes", value: (r) => fmt(r.nodes, 0) },
    { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}` },
    { label: "Req.", value: (r) => fmt(r.total_requests, 0) },
    { label: "In/Out", value: (r) => `${fmt(r.input_tokens, 0)}/${fmt(r.output_tokens, 0)}` },
    { label: "Tok/s", value: (r) => fmt(r.overall_tokens_per_second, 1) },
    { label: "TTFT95", value: (r) => `${fmt(r.ttft_p95, 1)}s` },
    { label: "Lat95", value: (r) => `${fmt(r.latency_p95, 1)}s` },
  ]);

  addMatrixSlides(presentation, "MaaS full run", maas, 1);
  addMatrixSlides(presentation, "ModelArts full run", modelArts, 1);

  slide = presentation.slides.add();
  baseSlide(slide, "Methodology and customer-readiness notes", "How this deck stays sleek without hiding engineering detail.");
  bullets(slide, [
    "MaaS and ModelArts are separated because they represent different workload families and observability levels.",
    "Main slides show decisions and trade-offs; appendix slides preserve exact DP/TP/concurrency/request/token metrics.",
    "Summary JSON/CSV rows take precedence over raw CSV companions when they encode intended token and request configuration.",
    "MaaS topology is not present in source artifacts; DP/TP fields are intentionally n/a rather than inferred.",
    "DeepSeek R1 TP16 is annotated as DP1 across two nodes from the supplied deployment note.",
    "Recommended customer discussion: pick workload and SLO first, then choose concurrency and DP/TP configuration."
  ], 106, 158, 980, 72, 20);
  addRect(slide, 88, 594, 1070, 42, WHITE, GRID, 6);
  addText(slide, "Data source: outputs/benchmark_report/normalized_benchmark_runs.csv", 112, 607, 720, 18, { size: 13, color: MUTED, bold: true });
  return presentation;
}

async function main() {
  globalThis.__fsReadFileSync = (await import("node:fs")).readFileSync;
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  await ensureArtifactToolWorkspace(SCRATCH);
  const artifact = await importArtifactTool(SCRATCH);
  const { PresentationFile } = artifact;
  const rows = rowsFromCsv();
  const presentation = makeDeck(artifact, rows);
  const previewPaths = [];
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    const previewPath = path.join(PREVIEW_DIR, `${stem}.png`);
    await saveBlobToFile(png, previewPath);
    previewPaths.push(previewPath);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(LAYOUT_DIR, `${stem}.layout.json`), await layout.text());
  }
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  const makeContact = "/Users/pontiffscopez/.codex/plugins/cache/openai-primary-runtime/presentations/26.614.11602/skills/presentations/scripts/make_contact_sheet.py";
  const py = "/Users/pontiffscopez/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
  const result = spawnSync(py, [makeContact, "--output", CONTACT, ...previewPaths], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error([result.stdout, result.stderr].filter(Boolean).join("\n"));
  }
  await fs.writeFile(path.join(SCRATCH, "source-notes.txt"), `Rows: ${rows.length}\nSource: ${CSV}\n`, "utf8");
  await fs.writeFile(path.join(SCRATCH, "slide-plan.txt"), "Customer-facing benchmark deck with technical appendix.\n", "utf8");
  console.log(JSON.stringify({ pptx: FINAL_PPTX, contactSheet: CONTACT, slideCount: presentation.slides.count, scratch: SCRATCH }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
