import { add, base, metricTile, bulletList, rows, topThroughput, errorRows, topLatency, fmt, RED, INK, AMBER } from "./common.mjs";

export default function slide02(presentation) {
  const slide = presentation.slides.add();
  const totalReq = rows.reduce((a, r) => a + (r.total_requests ?? 0), 0);
  const avgErr = rows.reduce((a, r) => a + (r.error_rate_percent ?? 0), 0) / rows.length;
  const best = topThroughput[0];
  const worstLatency = topLatency[0];
  const worstError = errorRows[0];
  add(slide, [
    ...base(slide, "Executive Summary"),
    ...metricTile(52, 122, 270, 142, "Coverage", `${rows.length} runs`, `${fmt(totalReq, 0)} requests represented`, RED),
    ...metricTile(352, 122, 270, 142, "Throughput leader", `${fmt(best.overall_tokens_per_second, 1)} tok/s`, `${best.model} at c${best.concurrency}`, INK),
    ...metricTile(652, 122, 270, 142, "Average error rate", fmt(avgErr, 1, "%"), "Mean across normalized runs", AMBER),
    ...metricTile(952, 122, 270, 142, "Worst P95 latency", fmt(worstLatency.latency_p95, 1, "s"), `${worstLatency.model}, c${worstLatency.concurrency}`, RED),
    ...bulletList([
      `ModelArts carries the highest aggregate throughput scenarios: ${best.model} reaches ${fmt(best.overall_tokens_per_second, 1)} output tok/s at c${best.concurrency}.`,
      "Concurrency alone is not the performance story: several high-concurrency runs trade throughput for materially higher P95 latency.",
      `Reliability needs explicit governance: ${worstError.model} / ${worstError.scenario} shows ${fmt(worstError.error_rate_percent, 1, "%")} error rate in the normalized set.`,
      "MaaS results include stable 256-token baselines and failed/rate-limited deepseek-v4-flash runs; both are retained in the report for operational context."
    ], 80, 332, 1070, 250, { size: 20, gap: 66 }),
  ]);
  return slide;
}
