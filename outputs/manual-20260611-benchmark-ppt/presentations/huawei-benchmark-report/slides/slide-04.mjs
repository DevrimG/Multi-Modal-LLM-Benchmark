import { add, base, barChart, lineChart, topThroughput, rows, fmt, RED, INK, MUTED, rect, t } from "./common.mjs";

function throughputSeries() {
  const groups = new Map();
  for (const r of rows) {
    if (r.overall_tokens_per_second === null || r.concurrency === null) continue;
    const key = `${r.platform} | ${r.model} | ${r.scenario} | DP${r.data_parallelism ?? "n/a"}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ x: r.concurrency, y: r.overall_tokens_per_second });
  }
  return [...groups.entries()]
    .map(([name, values]) => ({ name, values }))
    .filter((s) => s.values.length > 1)
    .sort((a, b) => Math.max(...b.values.map((v) => v.y)) - Math.max(...a.values.map((v) => v.y)))
    .slice(0, 4);
}

export default function slide04(presentation) {
  const slide = presentation.slides.add();
  const top = topThroughput.slice(0, 9);
  const best = top[0];
  add(slide, [
    ...base(slide, "Throughput: ModelArts Leads Aggregate Output Rate"),
    ...barChart(top, "overall_tokens_per_second", 52, 120, 560, 500, "Top aggregate throughput", {
      labelFn: (r) => `${r.model} c${r.concurrency}`,
      suffix: " tok/s",
      digits: 0,
      colorFn: (r) => (r.platform === "ModelArts" ? RED : INK),
    }),
    ...lineChart(throughputSeries(), 650, 120, 560, 500, "Throughput scaling by concurrency", "Output tokens per second"),
    rect(650, 640, 560, 40, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t(`Peak observed: ${best.platform} ${best.model} at c${best.concurrency}, ${fmt(best.input_tokens, 0)}/${fmt(best.output_tokens, 0)} tokens, ${fmt(best.overall_tokens_per_second, 1)} tok/s.`, 670, 652, 520, 18, { size: 12, color: MUTED }),
  ]);
  return slide;
}
