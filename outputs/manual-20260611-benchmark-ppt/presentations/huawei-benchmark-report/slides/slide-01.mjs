import { add, base, metricTile, bulletList, rows, modelArtsRows, maasRows, topThroughput, fmt, RED, INK, MUTED, rect, t } from "./common.mjs";

export default function slide01(presentation) {
  const slide = presentation.slides.add();
  const totalReq = rows.reduce((a, r) => a + (r.total_requests ?? 0), 0);
  const best = topThroughput[0];
  add(slide, [
    ...base(slide, "MaaS and ModelArts LLM Benchmark Report"),
    rect(0, 84, 1280, 636, "#ffffff"),
    rect(0, 84, 1280, 12, RED),
    t("Huawei internal usage", 64, 130, 360, 22, { size: 15, color: MUTED, bold: true }),
    t("Performance, latency, and reliability view across benchmark scenarios", 64, 160, 820, 78, { size: 33, bold: true, color: INK }),
    t("Source artifacts: benchmarks/HWC MaaS and benchmarks/ModelArts. DeepSeek TP16 is treated as DP1 across two nodes per deployment note.", 64, 252, 840, 44, { size: 15, color: MUTED }),
    ...metricTile(64, 340, 250, 142, "Normalized runs", fmt(rows.length, 0), `${modelArtsRows.length} ModelArts / ${maasRows.length} MaaS`, RED),
    ...metricTile(344, 340, 250, 142, "Requests represented", fmt(totalReq, 0), "Raw and summary files normalized", INK),
    ...metricTile(624, 340, 250, 142, "Peak aggregate tok/s", fmt(best.overall_tokens_per_second, 1), `${best.model}, c${best.concurrency}`, RED),
    ...bulletList([
      "Compare throughput and tail latency across platform, model, concurrency, DP, and token profile.",
      "Keep error-prone runs visible rather than filtering them out of the narrative.",
      "Provide an appendix matrix with scenario-level input/output tokens and request counts."
    ], 930, 344, 260, 150, { size: 13, gap: 46 }),
    t("Generated from normalized_benchmark_runs.csv", 64, 646, 560, 18, { size: 11, color: MUTED }),
  ]);
  return slide;
}
