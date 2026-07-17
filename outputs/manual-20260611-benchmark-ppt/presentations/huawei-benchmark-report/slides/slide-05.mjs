import { add, base, barChart, lineChart, topLatency, rows, fmt, RED, AMBER, INK, MUTED, rect, t } from "./common.mjs";

function latencySeries() {
  const groups = new Map();
  for (const r of rows) {
    if (r.latency_p95 === null || r.concurrency === null) continue;
    const key = `${r.platform} | ${r.model} | ${r.scenario} | DP${r.data_parallelism ?? "n/a"}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ x: r.concurrency, y: r.latency_p95 });
  }
  return [...groups.entries()]
    .map(([name, values]) => ({ name, values }))
    .filter((s) => s.values.length > 1)
    .sort((a, b) => Math.max(...b.values.map((v) => v.y)) - Math.max(...a.values.map((v) => v.y)))
    .slice(0, 4);
}

export default function slide05(presentation) {
  const slide = presentation.slides.add();
  const high = topLatency.slice(0, 9);
  const worst = high[0];
  add(slide, [
    ...base(slide, "Tail Latency: High Concurrency Needs Scenario-Level Guardrails"),
    ...barChart(high, "latency_p95", 52, 120, 560, 500, "Highest P95 latency scenarios", {
      labelFn: (r) => `${r.model} c${r.concurrency}`,
      suffix: "s",
      digits: 1,
      colorFn: (r) => (r.error_rate_percent > 0 ? RED : AMBER),
    }),
    ...lineChart(latencySeries(), 650, 120, 560, 500, "P95 latency scaling by concurrency", "Seconds; lower is better"),
    rect(650, 640, 560, 40, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t(`Worst observed tail: ${worst.platform} ${worst.model}, c${worst.concurrency}, P95 ${fmt(worst.latency_p95, 1)}s with ${fmt(worst.error_rate_percent, 1, "%")} errors.`, 670, 652, 520, 18, { size: 12, color: MUTED }),
  ]);
  return slide;
}
