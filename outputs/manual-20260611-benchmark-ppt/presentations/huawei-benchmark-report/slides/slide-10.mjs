import { add, base, bulletList, metricTile, rows, modelArtsRows, maasRows, fmt, RED, INK, MUTED, rect, t } from "./common.mjs";

export default function slide10(presentation) {
  const slide = presentation.slides.add();
  add(slide, [
    ...base(slide, "Methodology and Reproducibility"),
    ...metricTile(70, 126, 250, 128, "Data source", "CSV + JSON", "MaaS summaries and ModelArts request rows", RED),
    ...metricTile(350, 126, 250, 128, "ModelArts runs", fmt(modelArtsRows.length, 0), "Raw and summary scenarios", INK),
    ...metricTile(630, 126, 250, 128, "MaaS runs", fmt(maasRows.length, 0), "JSON summary run files", RED),
    ...metricTile(910, 126, 250, 128, "Normalized total", fmt(rows.length, 0), "Deduplicated scenario rows", INK),
    ...bulletList([
      "Throughput = successful generated output tokens divided by observed wall-clock duration.",
      "Latency and TTFT use reported summary values when present; raw CSVs compute mean and percentile values from successful request rows.",
      "Concurrency is parsed from filenames or summary rows; when absent, it is derived from peak overlapping request windows.",
      "ModelArts summary CSVs take precedence over raw CSVs for matching scenario keys because they preserve intended request and token configuration.",
      "MaaS files do not expose data/tensor parallelism, so DP/TP fields are reported as n/a for MaaS."
    ], 92, 322, 1000, 250, { size: 19, gap: 54 }),
    rect(70, 624, 1100, 38, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t("Rebuild command: python3 tools/generate_benchmark_report.py, then rebuild this deck through artifact-tool from outputs/benchmark_report/normalized_benchmark_runs.csv", 92, 636, 1044, 16, { size: 11, color: MUTED }),
  ]);
  return slide;
}
