import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  composeSlide,
  layers,
  shape,
  text,
  paint,
  stroke,
  textStyle,
} from "@oai/artifact-tool";

export const W = 1280;
export const H = 720;
export const RED = "#c7000b";
export const INK = "#111827";
export const MUTED = "#6b7280";
export const GRID = "#e5e7eb";
export const BG = "#f6f7f9";
export const GREEN = "#16883c";
export const AMBER = "#f59e0b";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const csvPath = path.resolve(__dirname, "../../../../benchmark_report/normalized_benchmark_runs.csv");

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

export function n(v) {
  if (v === undefined || v === null || v === "") return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

export function fmt(v, digits = 1, suffix = "") {
  if (v === null || v === undefined || v === "") return "n/a";
  const x = Number(v);
  if (!Number.isFinite(x)) return String(v);
  if (Math.abs(x) >= 100) return `${Math.round(x).toLocaleString()}${suffix}`;
  if (Math.abs(x) >= 10) return `${x.toFixed(digits)}${suffix}`;
  return `${x.toFixed(Math.max(1, digits))}${suffix}`;
}

export const rows = parseCsv(fs.readFileSync(csvPath, "utf8")).map((r) => ({
  ...r,
  concurrency: n(r.concurrency),
  total_requests: n(r.total_requests),
  successful_requests: n(r.successful_requests),
  failed_requests: n(r.failed_requests),
  error_rate_percent: n(r.error_rate_percent),
  duration_seconds: n(r.duration_seconds),
  input_tokens: n(r.input_tokens),
  output_tokens: n(r.output_tokens),
  total_output_tokens: n(r.total_output_tokens),
  overall_tokens_per_second: n(r.overall_tokens_per_second),
  ttft_p95: n(r.ttft_p95),
  latency_p95: n(r.latency_p95),
  data_parallelism: n(r.data_parallelism),
  tensor_parallelism: n(r.tensor_parallelism),
  nodes: n(r.nodes),
}));

export const topThroughput = [...rows]
  .filter((r) => r.overall_tokens_per_second !== null)
  .sort((a, b) => b.overall_tokens_per_second - a.overall_tokens_per_second);

export const topLatency = [...rows]
  .filter((r) => r.latency_p95 !== null)
  .sort((a, b) => b.latency_p95 - a.latency_p95);

export const errorRows = [...rows]
  .filter((r) => (r.error_rate_percent ?? 0) > 0)
  .sort((a, b) => b.error_rate_percent - a.error_rate_percent);

export const deepseekTp16 = rows.filter((r) => r.tensor_parallelism === 16);
export const maasRows = rows.filter((r) => r.platform === "MaaS");
export const modelArtsRows = rows.filter((r) => r.platform === "ModelArts");

export function label(r) {
  return `${r.platform} | ${r.model} | c${r.concurrency ?? "n/a"} | ${r.input_tokens ?? "n/a"}/${r.output_tokens ?? "n/a"}`;
}

export function t(value, x, y, w, h, opts = {}) {
  return text(value, {
    width: w,
    height: h,
    position: { left: x, top: y },
    style: textStyle({
      fontSize: opts.size ?? 20,
      fontFace: "Arial",
      color: opts.color ?? INK,
      bold: opts.bold ?? false,
      italic: opts.italic ?? false,
      horizontalAlignment: opts.align ?? "left",
      verticalAlignment: opts.valign ?? "top",
      wrap: true,
    }),
  });
}

export function rect(x, y, w, h, fill, opts = {}) {
  return shape({
    geometry: "rect",
    width: w,
    height: h,
    position: { left: x, top: y },
    fill: paint(fill),
    line: opts.line ? stroke(opts.line) : undefined,
    borderRadius: opts.radius ?? 0,
  });
}

function line(x1, y1, x2, y2, color = GRID, width = 1) {
  return shape({
    geometry: "line",
    width: Math.max(1, x2 - x1),
    height: Math.max(1, y2 - y1),
    position: { left: x1, top: y1 },
    line: stroke(`${color} ${width}px`),
  });
}

export function add(slide, children) {
  return composeSlide(slide, layers({ width: W, height: H }, children));
}

export function base(slide, title, kicker = "Huawei Cloud Internal") {
  const children = [
    rect(0, 0, W, H, BG),
    rect(0, 0, W, 84, INK),
    rect(1094, 0, 186, 84, RED),
    t(kicker.toUpperCase(), 48, 18, 520, 18, { size: 11, color: "#d1d5db", bold: true }),
    t(title, 48, 38, 860, 34, { size: 25, color: "#ffffff", bold: true }),
    t("Internal benchmark report", 1024, 28, 156, 22, { size: 12, color: "#ffffff", bold: true, align: "right" }),
  ];
  return children;
}

export function metricTile(x, y, w, h, labelText, value, note, color = RED) {
  return [
    rect(x, y, w, h, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    rect(x, y, 6, h, color),
    t(labelText, x + 20, y + 16, w - 36, 20, { size: 12, color: MUTED, bold: true }),
    t(value, x + 20, y + 42, w - 36, 42, { size: 31, bold: true }),
    t(note, x + 20, y + 91, w - 36, 38, { size: 12, color: MUTED }),
  ];
}

export function bulletList(items, x, y, w, h, opts = {}) {
  const children = [];
  const gap = opts.gap ?? 56;
  items.forEach((item, i) => {
    const yy = y + i * gap;
    children.push(rect(x, yy + 4, 8, 8, opts.color ?? RED, { radius: 4 }));
    children.push(t(item, x + 22, yy, w - 22, Math.min(gap - 8, h), { size: opts.size ?? 18, color: opts.textColor ?? INK }));
  });
  return children;
}

export function barChart(data, metric, x, y, w, h, title, opts = {}) {
  const max = Math.max(...data.map((d) => d[metric] ?? 0), 1);
  const left = x + 190;
  const top = y + 54;
  const plotW = w - 225;
  const rowH = (h - 76) / data.length;
  const children = [
    rect(x, y, w, h, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t(title, x + 18, y + 16, w - 36, 24, { size: 16, bold: true }),
  ];
  for (let i = 0; i < 5; i += 1) {
    const gx = left + (plotW * i) / 4;
    children.push(line(gx, top, gx, y + h - 22, GRID, 1));
    children.push(t(fmt((max * i) / 4, 0), gx - 20, y + h - 19, 44, 14, { size: 9, color: MUTED, align: "center" }));
  }
  data.forEach((d, i) => {
    const yy = top + i * rowH;
    const val = d[metric] ?? 0;
    const bw = (plotW * val) / max;
    const color = opts.colorFn ? opts.colorFn(d) : RED;
    children.push(t((opts.labelFn ? opts.labelFn(d) : label(d)).slice(0, 30), x + 16, yy + 2, 166, 16, { size: 10, color: MUTED, align: "right" }));
    children.push(rect(left, yy + 3, bw, Math.max(9, rowH * 0.45), color, { radius: 2 }));
    children.push(t(fmt(val, opts.digits ?? 1, opts.suffix ?? ""), left + bw + 6, yy + 1, 70, 16, { size: 10, bold: true }));
  });
  return children;
}

export function lineChart(series, x, y, w, h, title, metricLabel) {
  const xs = series.flatMap((s) => s.values.map((v) => v.x));
  const ys = series.flatMap((s) => s.values.map((v) => v.y));
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymax = Math.max(...ys, 1);
  const left = x + 58;
  const top = y + 56;
  const plotW = w - 218;
  const plotH = h - 104;
  const colors = [RED, INK, "#2563eb", GREEN, AMBER];
  const px = (v) => left + ((v - xmin) / Math.max(1, xmax - xmin)) * plotW;
  const py = (v) => top + plotH - (v / ymax) * plotH;
  const children = [
    rect(x, y, w, h, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t(title, x + 18, y + 16, w - 36, 22, { size: 16, bold: true }),
    t(metricLabel, x + 18, y + 38, w - 36, 16, { size: 10, color: MUTED }),
  ];
  for (let i = 0; i < 5; i += 1) {
    const gy = top + (plotH * i) / 4;
    children.push(line(left, gy, left + plotW, gy, GRID, 1));
    children.push(t(fmt(ymax * (1 - i / 4), 0), x + 10, gy - 7, 42, 14, { size: 9, color: MUTED, align: "right" }));
  }
  series.forEach((s, i) => {
    const color = colors[i % colors.length];
    const values = s.values.sort((a, b) => a.x - b.x);
    for (let j = 0; j < values.length - 1; j += 1) {
      children.push(line(px(values[j].x), py(values[j].y), px(values[j + 1].x), py(values[j + 1].y), color, 3));
    }
    values.forEach((v) => children.push(rect(px(v.x) - 4, py(v.y) - 4, 8, 8, color, { radius: 4 })));
    children.push(rect(x + w - 142, top + i * 24, 10, 10, color));
    children.push(t(s.name.slice(0, 28), x + w - 126, top + i * 24 - 3, 118, 16, { size: 9, color: MUTED }));
  });
  children.push(t("Concurrency", left + plotW / 2 - 40, y + h - 28, 80, 14, { size: 10, color: MUTED, align: "center" }));
  return children;
}

export function miniTable(data, columns, x, y, w, h, title) {
  const children = [
    rect(x, y, w, h, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t(title, x + 18, y + 14, w - 36, 22, { size: 16, bold: true }),
    rect(x + 18, y + 46, w - 36, 26, INK),
  ];
  const colW = (w - 36) / columns.length;
  columns.forEach((c, i) => children.push(t(c.label, x + 24 + i * colW, y + 53, colW - 10, 12, { size: 9, color: "#ffffff", bold: true })));
  const rowH = (h - 80) / data.length;
  data.forEach((r, ri) => {
    const yy = y + 74 + ri * rowH;
    if (ri % 2 === 1) children.push(rect(x + 18, yy - 2, w - 36, rowH, "#f9fafb"));
    columns.forEach((c, ci) => {
      const raw = c.value(r);
      children.push(t(raw, x + 24 + ci * colW, yy + 4, colW - 10, rowH - 5, { size: c.size ?? 9, color: c.color ?? INK }));
    });
  });
  return children;
}
